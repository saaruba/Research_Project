#!/usr/bin/env python3
"""
Train the "socially aware" Behavioural Cloning model: adds human-position
features (from facial_landmarks_uniface.csv for sessions 1/3, or
detected_people.csv everywhere else) on top of the robot-only baseline from
train_baseline_model.py.

This is the direct comparison your dissertation needs: does knowing where
people are actually improve the model, or not? Both results are useful -
if this version wins, that is your core finding; if it doesn't, that is
still a legitimate, reportable result (and likely means the group/O-space
features from Phase C are the piece that will move the needle, not raw
person position alone).

New features on top of the existing lidar_min_range, lidar_mean_range,
linear_x_prev, angular_z_prev:
    num_faces          - how many people detected (0 = none, never NaN)
    face_center_x_norm - horizontal position of the person(s) in the camera
                          image, normalised to 0-1 (0.5 = straight ahead,
                          filled with 0.5 when no one detected - a neutral
                          "nothing off to one side" default)
    face_center_y_norm - same, vertical position, 0-1
    face_area_norm     - how much of the image the person's bounding box
                          covers, normalised 0-1 (a rough "how close/large
                          they appear" proxy - filled with 0 when no one
                          detected, since no person means no footprint)

Run this AFTER split_dataset.py has produced train_table.csv / val_table.csv
/ test_table.csv (with the corrected FACE_TOLERANCE_SEC = 0.6 in
extract_training_table.py already applied).

Example:
    cd /workspaces/Research_Project
    python3 scripts/train_social_baseline_model.py
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

IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0

FEATURE_COLUMNS = [
    "lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev",
    "num_faces", "face_center_x_norm", "face_center_y_norm", "face_area_norm",
]
TARGET_COLUMNS = ["linear_x", "angular_z"]


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run split_dataset.py first.")
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["session_id", "timestamp"]).copy()
    df["linear_x_prev"] = df.groupby("session_id")["linear_x"].shift(1)
    df["angular_z_prev"] = df.groupby("session_id")["angular_z"].shift(1)

    df["face_center_x_norm"] = (df["face_center_x"] / IMAGE_WIDTH).fillna(0.5)
    df["face_center_y_norm"] = (df["face_center_y"] / IMAGE_HEIGHT).fillna(0.5)

    box_width = (df["face_bbox_max_x"] - df["face_bbox_min_x"]).clip(lower=0)
    box_height = (df["face_bbox_max_y"] - df["face_bbox_min_y"]).clip(lower=0)
    df["face_area_norm"] = ((box_width * box_height) / (IMAGE_WIDTH * IMAGE_HEIGHT)).fillna(0.0)

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

    naive_pred = np.tile(train_targets.mean(axis=0), (len(y_true), 1))
    naive_mae_linear = mean_absolute_error(y_true[:, 0], naive_pred[:, 0])
    naive_mae_angular = mean_absolute_error(y_true[:, 1], naive_pred[:, 1])

    print(f"\n{name} set ({len(df)} rows):")
    print(f"  linear_x  MAE: {mae_linear:.4f}  (naive baseline: {naive_mae_linear:.4f})")
    print(f"  angular_z MAE: {mae_angular:.4f}  (naive baseline: {naive_mae_angular:.4f})")

    return {
        "rows": int(len(df)),
        "mae_linear_x": float(mae_linear),
        "mae_angular_z": float(mae_angular),
        "naive_mae_linear_x": float(naive_mae_linear),
        "naive_mae_angular_z": float(naive_mae_angular),
    }


def compare_to_previous_baseline(new_metrics: dict) -> None:
    old_metrics_path = MODEL_DIR / "baseline_metrics.json"
    if not old_metrics_path.exists():
        print("\n(No previous baseline_metrics.json found to compare against - run train_baseline_model.py first for a side-by-side comparison.)")
        return

    with old_metrics_path.open("r", encoding="utf-8") as handle:
        old_metrics = json.load(handle)

    print("\n" + "=" * 70)
    print("COMPARISON: robot-only baseline  vs  socially-aware model  (test set)")
    print("=" * 70)
    for target in ["linear_x", "angular_z"]:
        old_mae = old_metrics["test"][f"mae_{target}"]
        new_mae = new_metrics["test"][f"mae_{target}"]
        change_pct = (old_mae - new_mae) / old_mae * 100
        direction = "improved" if new_mae < old_mae else "got worse"
        print(f"  {target}: {old_mae:.4f} -> {new_mae:.4f}  ({direction} by {abs(change_pct):.1f}%)")


def main() -> None:
    print("Loading and engineering features...")
    train_df = engineer_features(load_split("train"))
    val_df = engineer_features(load_split("val"))
    test_df = engineer_features(load_split("test"))

    print(f"\nFeatures used: {FEATURE_COLUMNS}")
    print(f"Targets: {TARGET_COLUMNS}")

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values

    print(f"\nTraining RandomForestRegressor on {len(train_df)} rows (this can take a couple of minutes)...")
    model = RandomForestRegressor(n_estimators=150, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    print("Training complete.")

    metrics = {
        "train": evaluate(model, train_df, y_train, "Train"),
        "val": evaluate(model, val_df, y_train, "Validation"),
        "test": evaluate(model, test_df, y_train, "Test"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "social_random_forest.joblib")
    with (MODEL_DIR / "social_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    compare_to_previous_baseline(metrics)

    print("\n" + "=" * 70)
    print("SOCIALLY-AWARE MODEL SAVED")
    print(f"Model:   {MODEL_DIR / 'social_random_forest.joblib'}")
    print(f"Metrics: {MODEL_DIR / 'social_metrics.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
