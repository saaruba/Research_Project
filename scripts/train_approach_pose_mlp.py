#!/usr/bin/env python3
"""
Phase F: train the MLP (neural network) approach-pose model - this is the
PRIMARY model per the proposal/supervisor-feedback architecture decision
(2 hidden layers, 128/64 units), with the Random Forest as the documented
comparison. Uses the exact same dataset, features, target, and session
split as train_approach_pose_model.py so the two are directly comparable.

MLPs are sensitive to feature scale (unlike Random Forest), so inputs are
standardised (zero mean, unit variance) before the network - this is
inside an sklearn Pipeline so the same fitted scaler is reused at
prediction time automatically.

Run AFTER build_approach_pose_dataset.py (and ideally after
train_approach_pose_model.py, so both metrics files exist for comparison).

Example:
    python3 scripts/train_approach_pose_mlp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def evaluate(model: Pipeline, df: pd.DataFrame, train_targets: np.ndarray, name: str) -> dict:
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


def compare_to_random_forest(mlp_metrics: dict) -> None:
    rf_path = MODEL_DIR / "approach_pose_metrics.json"
    if not rf_path.exists():
        print("\n(No approach_pose_metrics.json found - run train_approach_pose_model.py for a side-by-side comparison.)")
        return

    with rf_path.open("r", encoding="utf-8") as handle:
        rf_metrics = json.load(handle)

    print("\n" + "=" * 70)
    print("COMPARISON: Random Forest  vs  MLP  (test set, held-out sessions)")
    print("=" * 70)
    for target in TARGET_COLUMNS:
        rf_mae = rf_metrics["test"]["mae"][target]
        mlp_mae = mlp_metrics["test"]["mae"][target]
        direction = "MLP better" if mlp_mae < rf_mae else "Random Forest better"
        change_pct = abs(rf_mae - mlp_mae) / rf_mae * 100 if rf_mae > 0 else 0.0
        print(f"  {target:15s} RF={rf_mae:.4f}  MLP={mlp_mae:.4f}  ({direction} by {change_pct:.1f}%)")


def main() -> None:
    dataset_path = PROCESSED_DIR / "approach_pose_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found - run build_approach_pose_dataset.py first")

    df = pd.read_csv(dataset_path)
    split = session_split()

    train_df = df[df["session_id"].isin(split["train"])].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    val_df = df[df["session_id"].isin(split["val"])].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    test_df = df[df["session_id"].isin(split["test"])].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    if train_df.empty or test_df.empty:
        print("\nTrain or test split is empty for the approach-pose dataset - check session coverage.")
        return

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values

    print(f"\nFeatures: {FEATURE_COLUMNS}")
    print(f"Targets:  {TARGET_COLUMNS}")
    print("\nTraining MLPRegressor (2 hidden layers, 128/64 units)...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,           # L2 regularisation - some help against overfitting
            max_iter=500,
            early_stopping=True,  # holds out part of train internally to stop before overfitting
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    n_iter = model.named_steps["mlp"].n_iter_
    print(f"Training complete ({n_iter} iterations, early stopping {'triggered' if n_iter < 500 else 'not triggered'}).")

    metrics = {"train": evaluate(model, train_df, y_train, "Train")}
    if not val_df.empty:
        metrics["val"] = evaluate(model, val_df, y_train, "Validation")
    metrics["test"] = evaluate(model, test_df, y_train, "Test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "approach_pose_mlp.joblib")
    with (MODEL_DIR / "approach_pose_mlp_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    compare_to_random_forest(metrics)

    print("\n" + "=" * 70)
    print("MLP APPROACH-POSE MODEL SAVED")
    print(f"Model:   {MODEL_DIR / 'approach_pose_mlp.joblib'}")
    print(f"Metrics: {MODEL_DIR / 'approach_pose_mlp_metrics.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
