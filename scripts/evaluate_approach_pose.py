"""
Phase H (partial): evaluate and COMPARE all approach-pose policies on the
held-out test sessions, against the proposal's own Objective 4 thresholds.

WHAT THIS MEASURES (and what it deliberately does not)
--------------------------------------------------------
Your proposal lists 8 evaluation metrics. They split cleanly into two
groups, and it matters which is which:

  MEASURABLE OFFLINE (this script):
    - approach-position error (metres)   <- Objective 4 threshold: < 0.4 m
    - approach-orientation error (deg)   <- Objective 4 threshold: < 20 deg
  These are computable because the demonstrated stop pose is known from
  the robot's own odometry (real metres/radians), and each policy predicts
  a pose in the same robot-relative frame. This is the CORE Objective 4
  comparison - "does the learned model predict where a human would have
  stopped, better than a rule does?"

  REQUIRES THE RUNNING SIMULATION (NOT this script):
    - O-space intrusion rate, min distance to group, group cut-through rate
      -> need group positions in METRES; this project only has 2D pixel
         positions from uncalibrated video, so these cannot be computed
         honestly from the recorded dataset.
    - collision-free rate, task success rate, path length, navigation time
      -> these are properties of an executed trajectory, so they require
         Nav2 actually driving TIAGo in Gazebo (Phase D/G).
  Do not fabricate these from recorded data - state in the write-up that
  they are measured in simulation, and report them once Phase G runs.

POLICIES COMPARED
-----------------
  1. naive        - always predict the training-set mean pose. The "did we
                    learn anything at all?" floor. Any model that can't beat
                    this has learned nothing useful.
  2. rule_based   - the Phase E rule, applied offline: turn to face the
                    group, stop STANDOFF_DISTANCE metres short of it.
                    Distance-to-group is approximated by lidar_min_range
                    (the nearest obstacle ahead is, in these sessions,
                    usually the group being approached) - a documented
                    approximation, since no metric group distance exists.
                    This is the Objective 4 comparison point.
  3. random_forest- trained model (approach_pose_random_forest.joblib)
  4. mlp          - trained model (approach_pose_mlp.joblib), the
                    proposal's primary architecture

Reported per policy: mean + median position error, mean + median
orientation error, and the % of test rows meeting each proposal threshold.
Median is reported alongside mean because a handful of very large errors
can dominate a mean and hide otherwise-reasonable typical behaviour.

Usage:
    python3 scripts/evaluate_approach_pose.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "dataset" / "processed"
MODEL_DIR = PROCESSED_DIR / "models"

FEATURE_COLUMNS = [
    "lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev",
    "num_people", "group_bearing_rad", "group_scale_norm",
]
TARGET_COLUMNS = ["target_dx", "target_dy", "target_dyaw"]

STANDOFF_DISTANCE = 1.2          # metres - same value as the Phase E Nav2 node
POSITION_THRESHOLD_M = 0.4       # proposal Objective 4
ORIENTATION_THRESHOLD_DEG = 20.0  # proposal Objective 4


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def predict_rule_based(df: pd.DataFrame) -> np.ndarray:
    """
    Phase E rule, offline: face the group, stop STANDOFF_DISTANCE short of it.
    Distance to group approximated by lidar_min_range (documented limitation).
    """
    bearing = df["group_bearing_rad"].values
    distance_to_group = df["lidar_min_range"].values
    # Never predict driving backwards through the group if already too close.
    forward_travel = np.maximum(distance_to_group - STANDOFF_DISTANCE, 0.0)

    pred_dx = forward_travel * np.cos(bearing)
    pred_dy = forward_travel * np.sin(bearing)
    pred_dyaw = bearing  # turn to face the group centre
    return np.column_stack([pred_dx, pred_dy, pred_dyaw])


def score(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    position_error = np.hypot(y_true[:, 0] - y_pred[:, 0], y_true[:, 1] - y_pred[:, 1])
    orientation_error = np.abs(wrap_angle(y_true[:, 2] - y_pred[:, 2]))
    orientation_error_deg = np.degrees(orientation_error)

    result = {
        "policy": name,
        "mean_position_error_m": float(position_error.mean()),
        "median_position_error_m": float(np.median(position_error)),
        "mean_orientation_error_deg": float(orientation_error_deg.mean()),
        "median_orientation_error_deg": float(np.median(orientation_error_deg)),
        "pct_within_position_threshold": float((position_error < POSITION_THRESHOLD_M).mean() * 100),
        "pct_within_orientation_threshold": float((orientation_error_deg < ORIENTATION_THRESHOLD_DEG).mean() * 100),
        "pct_within_both_thresholds": float(
            ((position_error < POSITION_THRESHOLD_M) & (orientation_error_deg < ORIENTATION_THRESHOLD_DEG)).mean() * 100
        ),
    }
    return result


def main() -> None:
    dataset_path = PROCESSED_DIR / "approach_pose_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found - run build_approach_pose_dataset.py first")

    df = pd.read_csv(dataset_path)

    test_sessions = set(pd.read_csv(PROCESSED_DIR / "test_table.csv", usecols=["session_id"])["session_id"].unique())
    train_sessions = set(pd.read_csv(PROCESSED_DIR / "train_table.csv", usecols=["session_id"])["session_id"].unique())

    train_df = df[df["session_id"].isin(train_sessions)].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    test_df = df[df["session_id"].isin(test_sessions)].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)

    if test_df.empty:
        print("Test split is empty - nothing to evaluate.")
        return

    print(f"Evaluating on {len(test_df)} held-out test rows "
          f"(sessions {sorted(int(s) for s in test_df['session_id'].unique())})\n")

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values

    results = []

    # 1. Naive: always predict the training-set mean pose.
    naive_pred = np.tile(y_train.mean(axis=0), (len(y_test), 1))
    results.append(score(y_test, naive_pred, "naive (predict mean)"))

    # 2. Rule-based (Phase E logic, offline).
    results.append(score(y_test, predict_rule_based(test_df), "rule_based (Phase E)"))

    # 3 & 4. Trained models.
    for model_file, label in [
        ("approach_pose_random_forest.joblib", "random_forest (untuned)"),
        ("approach_pose_mlp.joblib", "mlp (untuned)"),
        ("approach_pose_random_forest_tuned.joblib", "random_forest (TUNED)"),
        ("approach_pose_mlp_tuned.joblib", "mlp (TUNED, primary)"),
    ]:
        model_path = MODEL_DIR / model_file
        if not model_path.exists():
            print(f"(skipping {label} - {model_path.name} not found)")
            continue
        model = joblib.load(model_path)
        results.append(score(y_test, model.predict(X_test), label))

    results_df = pd.DataFrame(results)

    print("=" * 100)
    print("APPROACH-POSE EVALUATION - held-out test sessions")
    print("=" * 100)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print()
    print(f"Proposal Objective 4 thresholds: position < {POSITION_THRESHOLD_M} m, "
          f"orientation < {ORIENTATION_THRESHOLD_DEG} deg")
    print()

    best_position = results_df.loc[results_df["mean_position_error_m"].idxmin(), "policy"]
    best_orientation = results_df.loc[results_df["mean_orientation_error_deg"].idxmin(), "policy"]
    print(f"Lowest mean position error:    {best_position}")
    print(f"Lowest mean orientation error: {best_orientation}")

    output_path = MODEL_DIR / "approach_pose_evaluation.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    results_df.to_csv(MODEL_DIR / "approach_pose_evaluation.csv", index=False)

    print(f"\nWritten to: {output_path}")
    print(f"Written to: {MODEL_DIR / 'approach_pose_evaluation.csv'}")


if __name__ == "__main__":
    main()
