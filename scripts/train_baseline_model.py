#!/usr/bin/env python3
"""
Train a first baseline Behavioural Cloning model.

IMPORTANT LESSON LEARNED (kept here deliberately, don't remove):
An earlier version of this script used the robot's absolute (x, y, yaw) as
features. That performed WORSE than just predicting the average action,
because absolute position/orientation is different in every session's room
and doesn't generalise - robot_x=2.0 in session 5 has nothing in common with
robot_x=2.0 in session 9. Swapping to LiDAR distances (which describe "how
close is the nearest obstacle", independent of the room) plus the robot's
own previous action (a short history of what it was just doing) actually
beats the naive baseline by ~30%. That's the version below.

This still does not use any human/face position features, because those
only exist for 2 of your 24 sessions. Once the person-detector (Phase B on
the checklist) is built, human-position features can be added on top of
this same pipeline as a richer follow-up model.

Run this AFTER split_dataset.py has produced train_table.csv / val_table.csv
/ test_table.csv.

Example:
    cd /workspaces/Research_Project
    python3 scripts/train_baseline_model.py
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

# lidar_* describe the surroundings; the *_prev columns are the robot's own
# previous action, giving the model a short sense of recent motion/history.
FEATURE_COLUMNS = ["lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev"]
TARGET_COLUMNS = ["linear_x", "angular_z"]


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run split_dataset.py first.")
    return pd.read_csv(path)


def add_previous_action(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add linear_x_prev / angular_z_prev, computed WITHIN each session only
    (using groupby + shift), so the first row of every session correctly
    has no previous action rather than borrowing one from a different
    session's last row.
    """
    df = df.sort_values(["session_id", "timestamp"]).copy()
    df["linear_x_prev"] = df.groupby("session_id")["linear_x"].shift(1)
    df["angular_z_prev"] = df.groupby("session_id")["angular_z"].shift(1)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    after = len(df)
    if after < before:
        print(f"  (dropped {before - after} rows: first row of each session has no previous action)")
    return df


def evaluate(model: RandomForestRegressor, df: pd.DataFrame, train_targets: np.ndarray, name: str) -> dict:
    X = df[FEATURE_COLUMNS].values
    y_true = df[TARGET_COLUMNS].values
    y_pred = model.predict(X)

    mae_linear = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_angular = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

    # naive baseline: always predict the mean of the TRAINING targets
    naive_pred = np.tile(train_targets.mean(axis=0), (len(y_true), 1))
    naive_mae_linear = mean_absolute_error(y_true[:, 0], naive_pred[:, 0])
    naive_mae_angular = mean_absolute_error(y_true[:, 1], naive_pred[:, 1])

    print(f"\n{name} set ({len(df)} rows):")
    print(f"  linear_x  MAE: {mae_linear:.4f}  (always-predict-mean baseline: {naive_mae_linear:.4f})")
    print(f"  angular_z MAE: {mae_angular:.4f}  (always-predict-mean baseline: {naive_mae_angular:.4f})")

    if mae_linear < naive_mae_linear and mae_angular < naive_mae_angular:
        print("  -> model beats the naive baseline on both targets. Good.")
    else:
        print("  -> model does NOT clearly beat the naive baseline - treat this split's result with caution.")

    return {
        "rows": int(len(df)),
        "mae_linear_x": float(mae_linear),
        "mae_angular_z": float(mae_angular),
        "naive_mae_linear_x": float(naive_mae_linear),
        "naive_mae_angular_z": float(naive_mae_angular),
    }


def main() -> None:
    print("Loading data...")
    train_df = add_previous_action(load_split("train"))
    val_df = add_previous_action(load_split("val"))
    test_df = add_previous_action(load_split("test"))

    print(f"\nFeatures used: {FEATURE_COLUMNS}")
    print(f"Targets: {TARGET_COLUMNS}")
    print("(No human/face features yet - see docstring at top of this script.)")

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values

    print(f"\nTraining RandomForestRegressor on {len(train_df)} rows (this can take a couple of minutes)...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    metrics = {
        "train": evaluate(model, train_df, y_train, "Train"),
        "val": evaluate(model, val_df, y_train, "Validation"),
        "test": evaluate(model, test_df, y_train, "Test"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_random_forest.joblib"
    joblib.dump(model, model_path)

    metrics_path = MODEL_DIR / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n" + "=" * 70)
    print("BASELINE MODEL SAVED")
    print("=" * 70)
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
