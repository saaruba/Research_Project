#!/usr/bin/env python3
"""
V2 APPROACH-POSE DATASET  (additive - v1 is not modified)

    python3 scripts/build_approach_pose_dataset_v2.py

Writes: dataset/processed/approach_pose_dataset_v2.csv
v1 (approach_pose_dataset.csv) is left exactly as it is.

============================================================================
WHY V2 EXISTS: THE V1 SEGMENTATION CAPTURED ADJUSTMENTS, NOT APPROACHES
============================================================================
Measured on the v1 dataset (462 events, 70,555 rows):

    median distance from event start to the final stop pose   0.17 m
    events starting more than 1.0 m from the stop             53 / 462  (11%)
    events starting more than 2.0 m from the stop             13 / 462  (3%)
    rows within 0.5 m of the stop                             72%
    median event duration                                     2.5 s

So the typical "approach demonstration" was a 2.5-second shuffle ending 17 cm
from where it began. The models were therefore trained almost entirely on the
terminal adjustment phase, which is exactly why their predictions are sensible
within a metre of a group and unreliable further out.

The cause is in v1's find_approach_events(). It pairs each moving segment with
the stop that follows it, where "moving" is any 1.0 s bin above a 0.05 speed
threshold. A real human walk toward a group contains pauses, turns and
hesitations, so it is chopped into many fragments and only the final fragment
- the last shuffle - is labelled.

V2 FIX: ANCHOR ON THE STOP AND EXTEND BACKWARDS
-----------------------------------------------
Rather than pairing single segments, v2 finds each sustained stop near a group
detection and walks BACKWARDS in time, absorbing brief pauses, until one of:

    - MIN_APPROACH_M of travel has been accumulated, and then some margin
    - a stop longer than LONG_STOP_S is met (a genuinely separate activity)
    - MAX_LOOKBACK_S is exhausted

Events that never accumulate MIN_APPROACH_M of travel are discarded outright:
a demonstration that never approached anything cannot teach approaching.

============================================================================
FEATURE DESIGN: EVERYTHING IN RADIANS OR METRES, NOTHING IN PIXELS
============================================================================
v1's group_scale_norm is a normalised pixel width. In simulation the same
quantity has to be reconstructed from metric depth, which is a documented
domain shift between training and deployment.

Every v2 feature is therefore expressed as an angle (radians) or a distance
(metres). Pixel coordinates are converted to bearings through the camera FOV
before use, so the SAME quantity can be computed offline from the recordings
and online from the simulated depth camera without a change of units.

New features, all derived from detected_people_individual.csv:

    group_bearing_rad         bearing to the group centre          (v1, kept)
    group_span_rad            angular width of the whole group
    nearest_person_span_rad   angular width of the largest person box
                              (apparent size -> proximity, FOV-normalised)
    gap_bearing_rad           bearing to the widest gap between adjacent
                              members - the P-space opening the robot aims for
    gap_width_rad             angular width of that gap
    person_spacing_rad        mean angular spacing between adjacent members
    lidar_min_range           metres                               (v1, kept)
    lidar_mean_range          metres                               (v1, kept)
    linear_x_prev             action history                       (v1, kept)
    angular_z_prev            action history                       (v1, kept)
    num_people                                                     (v1, kept)

gap_bearing_rad and gap_width_rad are the closest thing in the recordings to
the decision the policy actually makes at run time, which is to stand in the
free opening of an F-formation rather than at its geometric centre.

An event_id column is written so that training can weight or subsample by
approach event instead of by row - one long approach otherwise contributes
hundreds of near-identical rows and dominates the loss.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"

# --- segmentation ----------------------------------------------------------
BIN_SEC = 0.5
MOVE_THRESHOLD = 0.05        # same speed threshold as v1, for comparability
MIN_STOP_BINS = 2            # >= 1.0 s of stillness marks the end of an approach
LONG_STOP_S = 3.0            # a stop this long separates two distinct activities
MAX_LOOKBACK_S = 20.0        # never absorb more than this much history
# An event must contain at least this much travel. 1.0 m is the default, but it
# is the knob that trades EVENT COUNT against EVENT QUALITY: raising it keeps
# only unambiguous approaches and leaves fewer of them, lowering it admits
# shorter movements and recovers sample count. scripts/sweep_dataset_v2.py
# sweeps it, so it is settable from the environment rather than being edited.
MIN_APPROACH_M = float(os.environ.get("MIN_APPROACH_M", "1.0"))
OUTPUT_NAME = os.environ.get("V2_OUTPUT", "approach_pose_dataset_v2.csv")
GROUP_TIME_TOLERANCE = 2.0   # seconds, as v1

# --- camera ----------------------------------------------------------------
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0
CAMERA_HORIZONTAL_FOV_RAD = math.radians(58.0)  # documented assumption, uncalibrated


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def px_to_bearing(x_px: np.ndarray | float) -> np.ndarray | float:
    """Horizontal pixel coordinate -> bearing in radians, +ve to the left."""
    return -((np.asarray(x_px, dtype=float) / IMAGE_WIDTH) - 0.5) * CAMERA_HORIZONTAL_FOV_RAD


def px_span_to_rad(width_px: np.ndarray | float) -> np.ndarray | float:
    """Pixel width -> angular width in radians."""
    return (np.asarray(width_px, dtype=float) / IMAGE_WIDTH) * CAMERA_HORIZONTAL_FOV_RAD


# ---------------------------------------------------------------- segmentation
def find_approach_events(session_df: pd.DataFrame, groups_df: pd.DataFrame) -> list[dict]:
    """Anchor on each sustained stop near a group, then extend backwards."""
    s = session_df.sort_values("timestamp").reset_index(drop=True)
    speed = s["linear_x"].abs() + s["angular_z"].abs()
    s = s.assign(speed=speed)
    s["bin"] = ((s["timestamp"] - s["timestamp"].iloc[0]) // BIN_SEC).astype(int)

    binned = s.groupby("bin").agg(
        t=("timestamp", "first"), speed=("speed", "mean"),
        x=("robot_x", "last"), y=("robot_y", "last"), yaw=("robot_yaw", "last"),
    ).reset_index(drop=True)
    if len(binned) < 3:
        return []

    binned["moving"] = binned["speed"] > MOVE_THRESHOLD
    seg_id = (binned["moving"] != binned["moving"].shift()).cumsum()
    segs = binned.groupby(seg_id).agg(
        start=("t", "first"), end=("t", "last"), moving=("moving", "first"),
        n=("t", "size"), first_idx=("t", lambda v: v.index[0]),
        stop_x=("x", "last"), stop_y=("y", "last"), stop_yaw=("yaw", "last"),
    ).reset_index(drop=True)

    largest = (groups_df[groups_df["is_largest_group"]].sort_values("timestamp")
               if not groups_df.empty else groups_df)
    if largest.empty:
        return []

    max_lookback_bins = int(MAX_LOOKBACK_S / BIN_SEC)
    long_stop_bins = int(LONG_STOP_S / BIN_SEC)

    events = []
    for i, stop_seg in segs.iterrows():
        # The event must END in a sustained stop.
        if stop_seg["moving"] or stop_seg["n"] < MIN_STOP_BINS:
            continue
        # ...and that stop must coincide with a group being visible.
        if (largest["timestamp"] - stop_seg["start"]).abs().min() > GROUP_TIME_TOLERANCE:
            continue

        # Walk backwards through preceding segments, absorbing brief pauses.
        j = i - 1
        bins_used = 0
        move_start = None
        while j >= 0 and bins_used < max_lookback_bins:
            seg = segs.iloc[j]
            # A long stop means the previous activity was something else.
            if not seg["moving"] and seg["n"] >= long_stop_bins:
                break
            bins_used += int(seg["n"])
            move_start = seg["start"]
            j -= 1

        if move_start is None:
            continue

        # Require real travel. This is what v1 never checked, and it is the
        # difference between an approach and a shuffle.
        span = binned[(binned["t"] >= move_start) & (binned["t"] <= stop_seg["start"])]
        if len(span) < 2:
            continue
        travel = float(np.hypot(np.diff(span["x"].values), np.diff(span["y"].values)).sum())
        if travel < MIN_APPROACH_M:
            continue

        events.append({
            "move_start": move_start,
            "stop_start": stop_seg["start"],
            "stop_x": stop_seg["stop_x"],
            "stop_y": stop_seg["stop_y"],
            "stop_yaw": stop_seg["stop_yaw"],
            "travel_m": travel,
        })

    return events


# -------------------------------------------------------------- group geometry
def group_geometry(individual_df: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp angular geometry of the detected people.

    Returns one row per timestamp with bearings and angular widths in radians,
    so nothing downstream ever sees a pixel.
    """
    if individual_df.empty:
        return pd.DataFrame()

    rows = []
    for ts, frame in individual_df.groupby("timestamp"):
        f = frame.sort_values("center_x")
        bearings = np.asarray(px_to_bearing(f["center_x"].values), dtype=float)
        widths = px_span_to_rad((f["bbox_max_x"] - f["bbox_min_x"]).values)
        heights_px = (f["bbox_max_y"] - f["bbox_min_y"]).values

        # Bearings are +ve to the left, so sorting by pixel x gives descending
        # bearing. Sort ascending for gap analysis.
        order = np.argsort(bearings)
        b_sorted = bearings[order]

        if len(b_sorted) >= 2:
            gaps = np.diff(b_sorted)
            k = int(np.argmax(gaps))
            gap_width = float(gaps[k])
            gap_bearing = float((b_sorted[k] + b_sorted[k + 1]) / 2.0)
            spacing = float(np.mean(gaps))
            span = float(b_sorted[-1] - b_sorted[0])
        else:
            # A single person has no gap between members. The approach opening
            # is then the direction the person is NOT occupying; using their own
            # bearing keeps the feature defined without inventing structure.
            gap_width = 0.0
            gap_bearing = float(b_sorted[0]) if len(b_sorted) else 0.0
            spacing = 0.0
            span = float(widths[0]) if len(widths) else 0.0

        # Apparent size of the largest person box -> proximity proxy, in radians
        # of vertical extent expressed through the same FOV scaling.
        nearest_span = float(px_span_to_rad(np.max(heights_px) * IMAGE_WIDTH / IMAGE_HEIGHT)) \
            if len(heights_px) else 0.0

        rows.append({
            "timestamp": ts,
            "group_bearing_rad_v2": float(np.mean(bearings)),
            "group_span_rad": span,
            "nearest_person_span_rad": nearest_span,
            "gap_bearing_rad": gap_bearing,
            "gap_width_rad": gap_width,
            "person_spacing_rad": spacing,
            "n_detected": int(len(f)),
        })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------------ per session
def build_session_rows(session_df: pd.DataFrame, groups_df: pd.DataFrame,
                       individual_df: pd.DataFrame, session_id: int,
                       event_offset: int) -> tuple[pd.DataFrame, int]:
    s = session_df.sort_values("timestamp").reset_index(drop=True)
    events = find_approach_events(s, groups_df)
    if not events:
        return pd.DataFrame(), event_offset

    s = s.copy()
    s["linear_x_prev"] = s["linear_x"].shift(1).fillna(0.0)
    s["angular_z_prev"] = s["angular_z"].shift(1).fillna(0.0)

    geo = group_geometry(individual_df)

    # Group context, merged by nearest detection timestamp (as v1).
    if not groups_df.empty:
        g = groups_df[groups_df["is_largest_group"]].sort_values("timestamp")
        g = g[["timestamp", "num_people", "group_center_x", "avg_bbox_width"]]
        merged = pd.merge_asof(s, g, on="timestamp", direction="nearest",
                               tolerance=GROUP_TIME_TOLERANCE)
    else:
        merged = s.assign(num_people=np.nan, group_center_x=np.nan, avg_bbox_width=np.nan)

    if not geo.empty:
        merged = pd.merge_asof(merged, geo, on="timestamp", direction="nearest",
                               tolerance=GROUP_TIME_TOLERANCE)
    else:
        for c in ["group_bearing_rad_v2", "group_span_rad", "nearest_person_span_rad",
                  "gap_bearing_rad", "gap_width_rad", "person_spacing_rad", "n_detected"]:
            merged[c] = np.nan

    # v1-compatible columns, computed exactly as v1 computes them. These are
    # carried so that the v1 MODELS can be scored on the v2 rows - without them
    # the two versions could only be compared on different evaluation sets,
    # which would confound the segmentation change with a change of test data.
    merged["num_people"] = merged["num_people"].fillna(0)
    merged["group_bearing_rad"] = ((0.5 - (merged["group_center_x"] / IMAGE_WIDTH).fillna(0.5))
                                   * CAMERA_HORIZONTAL_FOV_RAD)
    merged["group_scale_norm"] = (merged["avg_bbox_width"] / IMAGE_WIDTH).fillna(0.0)

    labelled = []
    for ev in events:
        window = merged[(merged["timestamp"] >= ev["move_start"])
                        & (merged["timestamp"] <= ev["stop_start"])].copy()
        if window.empty:
            continue

        # Label: the stop pose, expressed in the robot's CURRENT frame. Same
        # convention as v1 - room-independent, so it generalises across sessions.
        dx_world = ev["stop_x"] - window["robot_x"]
        dy_world = ev["stop_y"] - window["robot_y"]
        cos_yaw = np.cos(-window["robot_yaw"])
        sin_yaw = np.sin(-window["robot_yaw"])
        window["target_dx"] = dx_world * cos_yaw - dy_world * sin_yaw
        window["target_dy"] = dx_world * sin_yaw + dy_world * cos_yaw
        window["target_dyaw"] = (ev["stop_yaw"] - window["robot_yaw"]).apply(wrap_angle)

        window["session_id"] = session_id
        window["event_id"] = event_offset
        window["event_travel_m"] = ev["travel_m"]
        event_offset += 1
        labelled.append(window)

    if not labelled:
        return pd.DataFrame(), event_offset

    result = pd.concat(labelled, ignore_index=True)
    columns = [
        "session_id", "event_id", "timestamp",
        # v1 features, kept so v1-vs-v2 isolates the segmentation change
        "lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev",
        "num_people", "group_bearing_rad", "group_scale_norm",
        # v2 features, all radians
        "group_span_rad", "nearest_person_span_rad", "gap_bearing_rad",
        "gap_width_rad", "person_spacing_rad", "people_visible",
        # bookkeeping + labels
        "event_travel_m", "target_dx", "target_dy", "target_dyaw",
    ]
    # About 9% of rows have no individual-person detection within tolerance -
    # nobody was visible at that instant. That is information, not missing data,
    # so it is encoded explicitly rather than dropped: the geometry features go
    # to zero (as v1 does for group_scale_norm) and a flag tells the model which
    # rows those are. Without the flag, "no one visible" and "a gap of zero
    # width" would be indistinguishable.
    geo_cols = ["group_span_rad", "nearest_person_span_rad", "gap_bearing_rad",
                "gap_width_rad", "person_spacing_rad"]
    result["people_visible"] = (~result[geo_cols].isna().any(axis=1)).astype(float)
    result[geo_cols] = result[geo_cols].fillna(0.0)

    result = result[[c for c in columns if c in result.columns]]
    return result.dropna(subset=["target_dx", "target_dy", "target_dyaw"]), event_offset


def main() -> None:
    combined_path = PROCESSED_DIR / "combined_training_table.csv"
    if not combined_path.exists():
        raise SystemExit(f"missing {combined_path} - run extract_training_table.py first")

    combined = pd.read_csv(combined_path)
    all_rows = []
    event_offset = 0

    for session_id, session_df in combined.groupby("session_id"):
        sid = int(session_id)
        groups_path = DATASET_DIR / str(sid) / "detected_groups.csv"
        indiv_path = DATASET_DIR / str(sid) / "detected_people_individual.csv"
        groups_df = pd.read_csv(groups_path) if groups_path.exists() else pd.DataFrame()
        indiv_df = pd.read_csv(indiv_path) if indiv_path.exists() else pd.DataFrame()

        rows, event_offset = build_session_rows(
            session_df, groups_df, indiv_df, sid, event_offset)
        if not rows.empty:
            all_rows.append(rows)
            n_ev = rows["event_id"].nunique()
            print(f"  session {sid:3d}: {len(rows):6d} rows across {n_ev:3d} event(s)")
        else:
            print(f"  session {sid:3d}:      0 rows (no qualifying approach)")

    if not all_rows:
        raise SystemExit("No qualifying approach events found - relax MIN_APPROACH_M.")

    final = pd.concat(all_rows, ignore_index=True)
    out = PROCESSED_DIR / OUTPUT_NAME
    final.to_csv(out, index=False)

    dist = np.hypot(final["target_dx"], final["target_dy"])
    starts = final.groupby("event_id").apply(
        lambda x: math.hypot(x["target_dx"].iloc[0], x["target_dy"].iloc[0]))

    print("\n" + "=" * 64)
    print(f"  rows            : {len(final)}")
    print(f"  events          : {final['event_id'].nunique()}")
    print(f"  sessions        : {final['session_id'].nunique()}")
    print(f"  median rows/event: {final.groupby('event_id').size().median():.0f}")
    print(f"\n  distance-to-go, median : {dist.median():.3f} m")
    print(f"  rows within 0.5 m      : {100 * (dist < 0.5).mean():.1f}%")
    print(f"  rows beyond 2.0 m      : {100 * (dist > 2.0).mean():.1f}%")
    print(f"\n  event start distance, median : {starts.median():.2f} m")
    print(f"  event travel, median         : {final.groupby('event_id')['event_travel_m'].first().median():.2f} m")
    print("=" * 64)
    print(f"Written: {out}")
    print("v1 (approach_pose_dataset.csv) is unchanged.")


if __name__ == "__main__":
    main()
