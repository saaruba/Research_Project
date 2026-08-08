#!/usr/bin/env python3
"""
Phase F: train the actual approach-POSE model - the target your proposal
evaluates against (x, y, yaw), not raw cmd_vel. Uses the labelled dataset
from build_approach_pose_dataset.py, with group-level features (from
Phase C's clustering) instead of raw scattered person positions.

Target is (target_dx, target_dy, target_dyaw): the demonstrated stop pose,
expressed RELATIVE to the robot's own position/heading at prediction time
(metres/radians in the robot's own frame) - not absolute map coordinates,
for the same generalisation reason documented in train_baseline_model.py
(absolute position doesn't transfer across sessions recorded in different
rooms; relative quantities do).

Splits by the SAME session-level train/val/test assignment as
split_dataset.py (read directly from the existing table files) - no
leakage, and directly comparable to the earlier cmd_vel-target models.

Run AFTER build_approach_pose_dataset.py.

Example:
    python3 scripts/train_approach_pose_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "dataset" / "processed"
MODEL_DIR = PROCESSED_DIR / "models"

FEATURE_COLUMNS = [
    "lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev",
    "num_people", "group_bearing_rad", "group_scale_norm",
]
TARGET_COLUMNS = ["target_dx", "target_dy", "target_dyaw"]


def session_split() -> dict[str, set[int]]:
    split = {}
    for name in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{name}_table.csv"
        sessions = set(pd.read_csv(path, usecols=["session_id"])["session_id"].unique())
        split[name] = sessions
    return split


def evaluate(model: RandomForestRegressor, df: pd.DataFrame, train_targets: np.ndarray, name: str) -> dict:
    X = df[FEATURE_COLUMNS].values
    y_true = df[TARGET_COLUMNS].values
    y_pred = model.predict(X)

    mae = {col: float(mean_absolute_error(y_true[:, i], y_pred[:, i])) for i, col in enumerate(TARGET_COLUMNS)}
    naive_pred = np.tile(train_targets.mean(axis=0), (len(y_true), 1))
    naive_mae = {col: float(mean_absolute_error(y_true[:, i], naive_pred[:, i])) for i, col in enumerate(TARGET_COLUMNS)}

    print(f"\n{name} set ({len(df)} rows):")
    for col in TARGET_COLUMNS:
        unit = "rad" if col == "target_dyaw" else "m"
        print(f"  {col:15s} MAE: {mae[col]:.4f} {unit}  (naive baseline: {naive_mae[col]:.4f} {unit})")

    return {"rows": int(len(df)), "mae": mae, "naive_mae": naive_mae}


def main() -> None:
    dataset_path = PROCESSED_DIR / "approach_pose_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found - run build_approach_pose_dataset.py first")

    df = pd.read_csv(dataset_path)
    split = session_split()

    train_df = df[df["session_id"].isin(split["train"])].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    val_df = df[df["session_id"].isin(split["val"])].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    test_df = df[df["session_id"].isin(split["test"])].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)

    print(f"Train: {len(train_df)} rows ({sorted(train_df['session_id'].unique())})")
    print(f"Val:   {len(val_df)} rows ({sorted(val_df['session_id'].unique())})")
    print(f"Test:  {len(test_df)} rows ({sorted(test_df['session_id'].unique())})")

    if train_df.empty or test_df.empty:
        print("\nTrain or test split is empty for the approach-pose dataset - check session coverage.")
        return

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values

    print(f"\nFeatures: {FEATURE_COLUMNS}")
    print(f"Targets:  {TARGET_COLUMNS}")
    print("\nTraining RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=150, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    print("Training complete.")

    metrics = {"train": evaluate(model, train_df, y_train, "Train")}
    if not val_df.empty:
        metrics["val"] = evaluate(model, val_df, y_train, "Validation")
    metrics["test"] = evaluate(model, test_df, y_train, "Test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "approach_pose_random_forest.joblib")
    with (MODEL_DIR / "approach_pose_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n" + "=" * 70)
    print("APPROACH-POSE MODEL SAVED")
    print(f"Model:   {MODEL_DIR / 'approach_pose_random_forest.joblib'}")
    print(f"Metrics: {MODEL_DIR / 'approach_pose_metrics.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
