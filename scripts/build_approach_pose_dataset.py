"""
Phase F, step 0: build the actual training set for approach-POSE prediction
(x, y, yaw), instead of raw cmd_vel - this is the target your proposal
actually evaluates against, and nothing has produced it before now.

THE LABELLING PROBLEM THIS SOLVES
----------------------------------
The recorded sessions never say "this is a demonstrated group-approach".
We have to infer it from the teleoperation trace itself. The approach used
here: find moments where the robot was MOVING, then came to a genuine
STOP, AND a group was detected nearby around that stop - that stop is
treated as "the human decided this was a good place to be near this
group", i.e. a demonstrated approach pose. Every row leading up to that
stop (the movement that led to it) is labelled with that stop's pose as
its target - the model is trained to answer "given what I can see right
now, where will I end up stopping near this group?"

Why binning first: cmd_vel in this dataset is very spiky (isolated
single-sample nonzero values at ~40Hz, not sustained speed) - a raw
sample-by-sample moving/stopped test finds almost no segments longer than
one sample. Averaging speed into 0.5s bins smooths this out into genuine
moving/stopped periods.

WHY THE TARGET IS RELATIVE, NOT ABSOLUTE (x, y)
-------------------------------------------------
train_baseline_model.py already found that absolute robot (x, y) as a
FEATURE made predictions worse, because raw position doesn't generalise
across sessions recorded in different rooms. The same logic applies even
harder to a TARGET: "stop at map coordinate (3.2, -1.1)" is meaningless
for a session recorded in a different room. So the label here is the
STOP POSE RELATIVE TO THE ROBOT'S OWN POSITION/HEADING at the time of
prediction (dx, dy in the robot's own forward/left frame, plus dyaw) -
this is the same quantity regardless of which room a session was recorded
in, so it should generalise the way LiDAR/action-history features did.

FEATURE UPDATE (6 Aug 2026): raw normalised pixel position
(group_center_x_norm/y_norm) has been replaced with group_bearing_rad - the
angle of the group relative to the robot's forward direction, computed from
the pixel x-position assuming a horizontal camera field of view (58 degrees,
a typical value for the RGB cameras used on TIAGo - not calibrated, a
documented assumption). This is a real, physically meaningful geometric
quantity (radians off forward), unlike a raw normalised pixel coordinate,
which is an arbitrary unit tied to image resolution and carries no
directly interpretable "how far off-centre" information. The vertical pixel
position (group_center_y_norm) is dropped - it mostly reflects camera tilt
and the person's height in-frame, not information useful for a ground
navigation decision.

Output: dataset/processed/approach_pose_dataset.csv, one row per
(session, timestamp) that falls inside a labelled movement segment, with:
    session_id, timestamp,
    lidar_min_range, lidar_mean_range, linear_x_prev, angular_z_prev,
    num_people, group_bearing_rad (radians, + = group is to the robot's left),
    group_scale_norm (avg_bbox_width / image width - a "how close" proxy),
    target_dx, target_dy   (metres, in the robot's own frame, at prediction time)
    target_dyaw             (radians, wrapped to [-pi, pi])

Run this AFTER cluster_groups.py has produced detected_groups.csv for all
sessions.

Usage:
    python3 scripts/build_approach_pose_dataset.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"

BIN_SEC = 0.5
MOVE_THRESHOLD = 0.05       # mean |linear_x| + |angular_z| per bin, above this = "moving"
MIN_MOVE_BINS = 2           # >= 1.0s of sustained movement before a stop counts
MIN_STOP_BINS = 2           # >= 1.0s of sustained stop after movement counts
GROUP_TIME_TOLERANCE = 2.0  # seconds - how close a group detection must be to the stop
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0
CAMERA_HORIZONTAL_FOV_RAD = math.radians(58.0)  # documented assumption, not calibrated


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def find_approach_events(session_df: pd.DataFrame, groups_df: pd.DataFrame) -> list[dict]:
    """Find (moving-segment -> stop) events near a detected group, per session."""
    s = session_df.sort_values("timestamp").reset_index(drop=True)
    speed = s["linear_x"].abs() + s["angular_z"].abs()
    s = s.assign(speed=speed)
    s["bin"] = ((s["timestamp"] - s["timestamp"].iloc[0]) // BIN_SEC).astype(int)

    binned = s.groupby("bin").agg(
        t=("timestamp", "first"), speed=("speed", "mean"),
        x=("robot_x", "last"), y=("robot_y", "last"), yaw=("robot_yaw", "last"),
    ).reset_index(drop=True)

    moving = binned["speed"] > MOVE_THRESHOLD
    seg_id = (moving != moving.shift()).cumsum()
    binned["moving"] = moving

    segs = binned.groupby(seg_id).agg(
        start=("t", "first"), end=("t", "last"), moving=("moving", "first"), n=("t", "size"),
        stop_x=("x", "last"), stop_y=("y", "last"), stop_yaw=("yaw", "last"),
    ).reset_index(drop=True)

    largest = groups_df[groups_df["is_largest_group"]].sort_values("timestamp") if not groups_df.empty else groups_df

    events = []
    for i in range(len(segs) - 1):
        move_seg = segs.iloc[i]
        stop_seg = segs.iloc[i + 1]
        if not move_seg["moving"] or move_seg["n"] < MIN_MOVE_BINS:
            continue
        if stop_seg["moving"] or stop_seg["n"] < MIN_STOP_BINS:
            continue
        if largest.empty:
            continue
        diffs = (largest["timestamp"] - stop_seg["start"]).abs()
        if diffs.min() > GROUP_TIME_TOLERANCE:
            continue

        events.append({
            "move_start": move_seg["start"],
            "stop_start": stop_seg["start"],
            "stop_x": stop_seg["stop_x"],
            "stop_y": stop_seg["stop_y"],
            "stop_yaw": stop_seg["stop_yaw"],
        })

    return events


def build_session_rows(session_df: pd.DataFrame, groups_df: pd.DataFrame, session_id: int) -> pd.DataFrame:
    s = session_df.sort_values("timestamp").reset_index(drop=True)
    s["linear_x_prev"] = s["linear_x"].shift(1)
    s["angular_z_prev"] = s["angular_z"].shift(1)

    events = find_approach_events(s, groups_df)
    if not events:
        return pd.DataFrame()

    # Current-group-context features, merged onto every row by nearest detection timestamp.
    largest = groups_df[groups_df["is_largest_group"]].sort_values("timestamp").copy() if not groups_df.empty else pd.DataFrame()
    if not largest.empty:
        merged = pd.merge_asof(
            s, largest[["timestamp", "num_people", "group_center_x", "group_center_y", "avg_bbox_width"]],
            on="timestamp", direction="nearest", tolerance=1.0,
        )
    else:
        merged = s.copy()
        for col in ["num_people", "group_center_x", "group_center_y", "avg_bbox_width"]:
            merged[col] = np.nan

    merged["num_people"] = merged["num_people"].fillna(0)
    group_center_x_norm = (merged["group_center_x"] / IMAGE_WIDTH).fillna(0.5)
    # bearing = 0 when the group is dead-centre in frame; positive = group is to the robot's left
    merged["group_bearing_rad"] = (0.5 - group_center_x_norm) * CAMERA_HORIZONTAL_FOV_RAD
    merged["group_scale_norm"] = (merged["avg_bbox_width"] / IMAGE_WIDTH).fillna(0.0)

    labelled_rows = []
    for event in events:
        window = merged[(merged["timestamp"] >= event["move_start"]) & (merged["timestamp"] <= event["stop_start"])]
        if window.empty:
            continue

        dx_world = event["stop_x"] - window["robot_x"]
        dy_world = event["stop_y"] - window["robot_y"]
        cos_yaw = np.cos(-window["robot_yaw"])
        sin_yaw = np.sin(-window["robot_yaw"])

        window = window.copy()
        window["target_dx"] = dx_world * cos_yaw - dy_world * sin_yaw
        window["target_dy"] = dx_world * sin_yaw + dy_world * cos_yaw
        window["target_dyaw"] = (event["stop_yaw"] - window["robot_yaw"]).apply(wrap_angle)
        window["session_id"] = session_id

        labelled_rows.append(window)

    if not labelled_rows:
        return pd.DataFrame()

    result = pd.concat(labelled_rows, ignore_index=True)
    keep_cols = [
        "session_id", "timestamp", "lidar_min_range", "lidar_mean_range",
        "linear_x_prev", "angular_z_prev", "num_people",
        "group_bearing_rad", "group_scale_norm",
        "target_dx", "target_dy", "target_dyaw",
    ]
    return result[keep_cols].dropna(subset=["lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev"])


def main() -> None:
    combined_path = PROCESSED_DIR / "combined_training_table.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"{combined_path} not found - run extract_training_table.py / the master merge first")

    combined = pd.read_csv(combined_path)
    all_rows = []
    total_events = 0

    for session_id, session_df in combined.groupby("session_id"):
        groups_path = DATASET_DIR / str(int(session_id)) / "detected_groups.csv"
        groups_df = pd.read_csv(groups_path) if groups_path.exists() else pd.DataFrame()

        rows = build_session_rows(session_df, groups_df, int(session_id))
        n_events = rows["timestamp"].nunique() if not rows.empty else 0
        print(f"  session {int(session_id)}: {len(rows)} labelled rows")
        if not rows.empty:
            all_rows.append(rows)

    if not all_rows:
        print("No labelled approach events found across any session - check thresholds.")
        return

    final = pd.concat(all_rows, ignore_index=True)
    output_path = PROCESSED_DIR / "approach_pose_dataset.csv"
    final.to_csv(output_path, index=False)

    print(f"\nTotal labelled rows: {len(final)}")
    print(f"Sessions represented: {sorted(final['session_id'].unique())}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
