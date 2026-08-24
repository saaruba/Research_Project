#!/usr/bin/env python3
"""
DOES USING MORE FRAMES PER SECOND HELP?  (additive - nothing is overwritten)

    python3 scripts/sweep_decimation.py

============================================================================
THE QUESTION
============================================================================
Training rows are decimated to DECIMATE_HZ within each approach event. The
recordings run at ~33 Hz (median inter-sample interval 0.030 s), so at the
default 10 Hz roughly two of every three rows are discarded. The obvious
question is whether keeping them helps.

There is a reason to expect it will not. Rows are weighted by 1/(rows in the
event), so every approach contributes the same total influence regardless of
how many rows represent it. Raising the rate therefore adds rows that are
near-duplicates of their neighbours - at 33 Hz the robot has moved about a
centimetre between consecutive samples and the group has not moved at all -
without adding either new behaviour or new weight.

That is an argument, not evidence, so this measures it. The test set, the
learner and the source dataset are all held fixed; only the training frame
rate varies.

Test set: full-rate rows of sessions 5, 9, 59 from approach_pose_dataset_v2.csv
- the same rows used for Panels B-D of the v2 study and for the volume sweep,
so every number here is directly comparable to:

    v1 random forest (shipped)   0.656 m    23.5% within both thresholds
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROCESSED / "models"

TRAIN_SESSIONS = [1, 3, 7, 8, 11, 12, 26, 27, 28, 30, 31, 49, 51, 52, 55, 58, 60]
TEST_SESSIONS = [5, 9, 59]

V1_FEATURES = ["lidar_min_range", "lidar_mean_range", "linear_x_prev",
               "angular_z_prev", "num_people", "group_bearing_rad", "group_scale_norm"]
TARGETS = ["target_dx", "target_dy", "target_dyaw"]

POSITION_THRESHOLD_M = 0.4
ORIENTATION_THRESHOLD_DEG = 20.0
RF_PARAMS = dict(n_estimators=150, max_depth=8, min_samples_leaf=20,
                 n_jobs=-1, random_state=42)

# The richest source dataset, so frame rate is the only thing varying.
SOURCE = "approach_pose_dataset_v2_0.25.csv"
RATES = [2.0, 5.0, 10.0, 20.0, 60.0]     # 60 exceeds the native rate: keeps all rows


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def decimate(df: pd.DataFrame, hz: float) -> pd.DataFrame:
    if hz >= 60.0:
        return df                      # native rate; nothing to drop
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
        "pct_within_both_thresholds": float(
            ((pos < POSITION_THRESHOLD_M) & (ori < ORIENTATION_THRESHOLD_DEG)).mean() * 100),
    }


def main() -> None:
    base = pd.read_csv(PROCESSED / "approach_pose_dataset_v2.csv")
    test = base[base.session_id.isin(TEST_SESSIONS)]
    y_test, X_test = test[TARGETS].values, test[V1_FEATURES].values

    src = pd.read_csv(PROCESSED / SOURCE)
    train_full = src[src.session_id.isin(TRAIN_SESSIONS)]

    print("=" * 74)
    print(f"  source     : {SOURCE}")
    print(f"  train pool : {len(train_full):,} rows, "
          f"{train_full.event_id.nunique()} events (native ~33 Hz)")
    print(f"  test       : {len(test):,} rows, {test.event_id.nunique()} events (fixed)")
    print("=" * 74)
    print(f"  {'rate':>8}  {'train rows':>11}  {'events':>7}  "
          f"{'pos m':>7}  {'ori deg':>8}  {'both':>7}", flush=True)

    rows = []
    for hz in RATES:
        train = decimate(train_full, hz)
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(train[V1_FEATURES].values, train[TARGETS].values,
               sample_weight=event_weights(train))
        s = score(y_test, rf.predict(X_test))
        s.update(rate_hz=hz, train_rows=len(train),
                 train_events=int(train.event_id.nunique()))
        rows.append(s)
        label = "all (33Hz)" if hz >= 60 else f"{hz:g} Hz"
        print(f"  {label:>8}  {len(train):>11,}  {s['train_events']:>7}  "
              f"{s['mean_position_error_m']:>7.3f}  "
              f"{s['mean_orientation_error_deg']:>8.2f}  "
              f"{s['pct_within_both_thresholds']:>6.1f}%", flush=True)

    out = MODEL_DIR / "decimation_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    best = min(rows, key=lambda r: r["mean_position_error_m"])
    worst = max(rows, key=lambda r: r["mean_position_error_m"])
    spread = worst["mean_position_error_m"] - best["mean_position_error_m"]
    print("\n" + "-" * 74)
    print(f"  spread across a 30x change in frame rate: {spread:.4f} m "
          f"({100 * spread / best['mean_position_error_m']:.1f}%)")
    print(f"  reference - v1 random forest (shipped):   0.656 m, 23.5% both")
    print("-" * 74)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
