#!/usr/bin/env python3
"""
DOES MORE TRAINING DATA HELP?  (additive - nothing is overwritten)

    python3 scripts/build_approach_pose_dataset_v2.py            # 1.0 m
    MIN_APPROACH_M=0.5  V2_OUTPUT=approach_pose_dataset_v2_0.5.csv  \
        python3 scripts/build_approach_pose_dataset_v2.py
    MIN_APPROACH_M=0.25 V2_OUTPUT=approach_pose_dataset_v2_0.25.csv \
        python3 scripts/build_approach_pose_dataset_v2.py
    python3 scripts/sweep_dataset_v2.py

============================================================================
THE QUESTION
============================================================================
The v2 study found that re-segmenting the recordings into genuine approaches
made predictions WORSE, and attributed that to sample count: requiring 1 m of
travel cut 462 events to 182, of which 120 are in the training sessions.

If that attribution is right, then recovering sample count should recover
performance. This script tests it three ways, none of which requires new
recordings:

  1. MIN_APPROACH_M sweep. The 1.0 m threshold was a judgement call. Lowering
     it admits shorter movements, trading event quality for event count:
         1.0 m -> 120 training events, median start 1.37 m
         0.5 m -> 178 training events, median start 0.98 m
        0.25 m -> 216 training events, median start 0.74 m

  2. Mirror augmentation. Approach geometry is left-right symmetric: a group
     30 deg to the left, approached by stepping left, is a valid demonstration
     of a group 30 deg to the right approached by stepping right. Reflecting
     every event doubles the training set at no cost and adds no assumption
     beyond the symmetry of the plane. Bearings, lateral displacement, yaw and
     angular velocity are negated; ranges, counts and angular WIDTHS are not.

  3. Folding the validation sessions into training. Hyper-parameters are fixed
     (carried from the v1 grid search), so a validation split earns nothing
     here, and 4 more sessions is a ~24% increase in sessions.

============================================================================
WHAT MAKES THE COMPARISON VALID
============================================================================
The TEST SET NEVER CHANGES. Every configuration is scored on the full-rate
rows of sessions 5, 9 and 59 from the 1.0 m dataset - the same rows used for
Panels B, C and D of the v2 study - so these numbers sit directly alongside:

    v1 random forest (shipped)   0.656 m    23.5% within both thresholds
    v2seg random forest          0.722 m    15.5%

Only the RandomForest is swept. It was the strongest v2 model, and holding the
learner fixed keeps the comparison about DATA, which is the question.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROCESSED / "models"

TRAIN_SESSIONS = [1, 3, 7, 8, 11, 12, 26, 27, 28, 30, 31, 49, 51, 52, 55, 58, 60]
VAL_SESSIONS = [10, 14, 15, 54]
TEST_SESSIONS = [5, 9, 59]

V1_FEATURES = ["lidar_min_range", "lidar_mean_range", "linear_x_prev",
               "angular_z_prev", "num_people", "group_bearing_rad", "group_scale_norm"]
TARGETS = ["target_dx", "target_dy", "target_dyaw"]

POSITION_THRESHOLD_M = 0.4
ORIENTATION_THRESHOLD_DEG = 20.0
DECIMATE_HZ = 10.0

RF_PARAMS = dict(n_estimators=150, max_depth=8, min_samples_leaf=20,
                 n_jobs=-1, random_state=42)

# Quantities that flip sign under a left-right reflection. Everything else -
# ranges, counts, forward velocity, and all angular WIDTHS - is invariant.
MIRROR_NEGATE = ["group_bearing_rad", "gap_bearing_rad", "angular_z_prev",
                 "target_dy", "target_dyaw"]


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def decimate(df: pd.DataFrame, hz: float = DECIMATE_HZ) -> pd.DataFrame:
    keep = []
    for _, ev in df.groupby("event_id"):
        ev = ev.sort_values("timestamp")
        last, mask = -np.inf, []
        for t in ev["timestamp"].values:
            take = (t - last) >= (1.0 / hz)
            mask.append(take)
            if take:
                last = t
        keep.append(ev[np.array(mask)])
    return pd.concat(keep, ignore_index=True)


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Left-right reflection of every row, as a second copy of the data."""
    m = df.copy()
    for col in MIRROR_NEGATE:
        if col in m.columns:
            m[col] = -m[col]
    if "target_dyaw" in m.columns:
        m["target_dyaw"] = wrap_angle(m["target_dyaw"].values)
    m["event_id"] = m["event_id"] + df["event_id"].max() + 1     # distinct events
    return pd.concat([df, m], ignore_index=True)


def event_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("event_id")["event_id"].transform("size").values
    w = 1.0 / counts
    return w * (len(df) / w.sum())


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pos = np.hypot(y_true[:, 0] - y_pred[:, 0], y_true[:, 1] - y_pred[:, 1])
    ori = np.degrees(np.abs(wrap_angle(y_true[:, 2] - y_pred[:, 2])))
    return {
        "mean_position_error_m": float(pos.mean()),
        "mean_orientation_error_deg": float(ori.mean()),
        "pct_within_position_threshold": float((pos < POSITION_THRESHOLD_M).mean() * 100),
        "pct_within_orientation_threshold": float((ori < ORIENTATION_THRESHOLD_DEG).mean() * 100),
        "pct_within_both_thresholds": float(
            ((pos < POSITION_THRESHOLD_M) & (ori < ORIENTATION_THRESHOLD_DEG)).mean() * 100),
    }


def main() -> None:
    # The one fixed evaluation set, shared by every configuration.
    base = pd.read_csv(PROCESSED / "approach_pose_dataset_v2.csv")
    test = base[base.session_id.isin(TEST_SESSIONS)]
    y_test = test[TARGETS].values
    X_test = test[V1_FEATURES].values

    print("=" * 78)
    print(f"  Fixed test set: sessions {TEST_SESSIONS}  "
          f"{len(test):,} rows, {test.event_id.nunique()} events")
    print("=" * 78, flush=True)

    configs = []
    for thresh, fname in [(1.0, "approach_pose_dataset_v2.csv"),
                          (0.5, "approach_pose_dataset_v2_0.5.csv"),
                          (0.25, "approach_pose_dataset_v2_0.25.csv")]:
        path = PROCESSED / fname
        if not path.exists():
            print(f"  (skipping {fname} - not built)")
            continue
        df = pd.read_csv(path)
        configs.append((f"min_travel={thresh}m", df, TRAIN_SESSIONS, False))

    # Augmentation and extra sessions, applied to the most data-rich threshold.
    richest = configs[-1] if configs else None
    if richest is not None:
        label, df, _, _ = richest
        configs.append((f"{label} + mirror", df, TRAIN_SESSIONS, True))
        configs.append((f"{label} + mirror + val", df, TRAIN_SESSIONS + VAL_SESSIONS, True))

    rows = []
    for label, df, sessions, do_mirror in configs:
        train = decimate(df[df.session_id.isin(sessions)])
        n_ev_before = train.event_id.nunique()
        if do_mirror:
            train = mirror(train)
        w = event_weights(train)

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(train[V1_FEATURES].values, train[TARGETS].values, sample_weight=w)
        s = score(y_test, rf.predict(X_test))
        s.update(label=label, train_rows=len(train), train_events=train.event_id.nunique(),
                 source_events=n_ev_before, sessions=len(sessions))
        rows.append(s)
        print(f"  {label:<32} events={s['train_events']:4d}  "
              f"pos={s['mean_position_error_m']:.3f} m  "
              f"ori={s['mean_orientation_error_deg']:.2f} deg  "
              f"both={s['pct_within_both_thresholds']:.1f}%", flush=True)

    # Reference lines from the models already on disk.
    print("\n" + "-" * 78)
    print("  REFERENCE (same test rows)")
    print("-" * 78)
    for fname, label in [("approach_pose_random_forest_tuned.joblib", "v1 random forest (SHIPPED)"),
                         ("approach_pose_random_forest_v2seg.joblib", "v2seg random forest")]:
        p = MODEL_DIR / fname
        if p.exists():
            s = score(y_test, joblib.load(p).predict(X_test))
            print(f"  {label:<32}              "
                  f"pos={s['mean_position_error_m']:.3f} m  "
                  f"ori={s['mean_orientation_error_deg']:.2f} deg  "
                  f"both={s['pct_within_both_thresholds']:.1f}%")
            rows.append({**s, "label": label, "train_rows": None,
                         "train_events": None, "source_events": None, "sessions": None})

    # ------------------------------------------------------------------------
    # SAMPLING-RATE SWEEP
    # ------------------------------------------------------------------------
    # Training rows are decimated to DECIMATE_HZ within each event. The
    # recordings are ~60 Hz, so the obvious question is whether keeping more of
    # them helps. It should not: consecutive rows are ~17 ms apart, the robot
    # has moved about a centimetre and the group has not moved at all, so the
    # extra rows are near-duplicates carrying no new information.
    #
    # There is also a structural reason to expect no effect. Each row is
    # weighted by 1/(rows in its event), so the TOTAL weight of an event is
    # constant however finely it is sampled. Decimation therefore changes how
    # many rows carry an event's weight, not how much weight it has.
    #
    # This sweep measures that rather than assuming it. Same fixed test set.
    rate_path = PROCESSED / "approach_pose_dataset_v2_0.25.csv"
    if rate_path.exists():
        print("\n" + "-" * 78)
        print("  SAMPLING-RATE SWEEP  (min_travel=0.25m, train sessions only)")
        print("-" * 78, flush=True)
        rate_df = pd.read_csv(rate_path)
        rate_df = rate_df[rate_df.session_id.isin(TRAIN_SESSIONS)]
        for hz in [5.0, 10.0, 20.0, 60.0]:
            train = decimate(rate_df, hz)
            w = event_weights(train)
            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(train[V1_FEATURES].values, train[TARGETS].values, sample_weight=w)
            s = score(y_test, rf.predict(X_test))
            s.update(label=f"decimate={hz:.0f}Hz", train_rows=len(train),
                     train_events=train.event_id.nunique(), source_events=None,
                     sessions=len(TRAIN_SESSIONS))
            rows.append(s)
            print(f"  {hz:>4.0f} Hz   rows={len(train):>7,}   "
                  f"pos={s['mean_position_error_m']:.3f} m  "
                  f"ori={s['mean_orientation_error_deg']:.2f} deg  "
                  f"both={s['pct_within_both_thresholds']:.1f}%", flush=True)

    out = MODEL_DIR / "dataset_volume_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(MODEL_DIR / "dataset_volume_sweep.csv", index=False)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
