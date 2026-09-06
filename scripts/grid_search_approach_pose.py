"""
Hyper-parameter grid search for both approach-pose models.

WHY THIS EXISTS
---------------
Your proposal (Research Methods section) explicitly commits to:
    "Hyper-parameters (hidden layer sizes, learning rate, L2 regularisation
     for the MLP; tree depth and count for the Random Forest) are tuned via
     grid search on the validation split, with the test split held out
     entirely until final evaluation to avoid information leakage."

Until now both models used hand-picked defaults, so that promise was
unfulfilled. This script fulfils it exactly as written: every candidate is
scored ONLY on the validation split, the test split is never touched during
selection, and only the single winning configuration per model family is
finally evaluated on test.

This is also the last remaining lever that could plausibly improve the
Objective 4 numbers - so treat the outcome either way as a real result. If
tuning doesn't help, that strengthens (not weakens) the conclusion that the
limitation is the data, not the model configuration.

SELECTION METRIC
----------------
Candidates are ranked by mean validation POSITION error in metres
(sqrt(dx_err^2 + dy_err^2)), which is the quantity Objective 4 actually
sets a threshold on (<0.4 m). Orientation error is reported alongside but
not used for ranking, because a model that nails orientation while
stopping in the wrong place is not useful for this task.

RESUMABLE
---------
Fitting every candidate can take a while, so this script checkpoints after
each configuration to grid_search_results.csv and skips any configuration
already present in that file. If it is interrupted (or you stop it), just
run the same command again and it picks up where it left off. Delete
grid_search_results.csv to start the search over from scratch.

Usage:
    python3 scripts/grid_search_approach_pose.py
    python3 scripts/grid_search_approach_pose.py --quick   # smaller grid
"""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    position_error = np.hypot(y_true[:, 0] - y_pred[:, 0], y_true[:, 1] - y_pred[:, 1])
    orientation_error_deg = np.degrees(np.abs(wrap_angle(y_true[:, 2] - y_pred[:, 2])))
    return float(position_error.mean()), float(orientation_error_deg.mean())


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_path = PROCESSED_DIR / "approach_pose_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found - run build_approach_pose_dataset.py first")
    df = pd.read_csv(dataset_path)

    splits = {}
    for name in ["train", "val", "test"]:
        sessions = set(pd.read_csv(PROCESSED_DIR / f"{name}_table.csv", usecols=["session_id"])["session_id"].unique())
        splits[name] = df[df["session_id"].isin(sessions)].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)

    return splits["train"], splits["val"], splits["test"]


def load_checkpoint() -> tuple[list[dict], set[str]]:
    """Return previously-evaluated results and the set of config keys already done."""
    path = MODEL_DIR / "grid_search_results.csv"
    if not path.exists():
        return [], set()
    done_df = pd.read_csv(path)
    results = done_df.to_dict("records")
    keys = {f"{r['family']}|{r.get('config_key', '')}" for r in results}
    print(f"Resuming: {len(results)} configuration(s) already evaluated (skipping those).\n")
    return results, keys


def checkpoint(all_results: list[dict]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).sort_values("val_position_error_m").to_csv(
        MODEL_DIR / "grid_search_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a smaller grid (faster)")
    args = parser.parse_args()

    train_df, val_df, test_df = load_splits()
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows (held out)\n")

    X_train, y_train = train_df[FEATURE_COLUMNS].values, train_df[TARGET_COLUMNS].values
    X_val, y_val = val_df[FEATURE_COLUMNS].values, val_df[TARGET_COLUMNS].values
    X_test, y_test = test_df[FEATURE_COLUMNS].values, test_df[TARGET_COLUMNS].values

    all_results, done_keys = load_checkpoint()

    # ---------------- Random Forest grid ----------------
    if args.quick:
        rf_grid = {"n_estimators": [150], "max_depth": [8, 14, None], "min_samples_leaf": [1, 20]}
    else:
        rf_grid = {
            "n_estimators": [100, 300],
            "max_depth": [6, 10, 14, 20, None],
            "min_samples_leaf": [1, 5, 20, 50],
        }

    rf_keys = list(rf_grid)
    rf_combos = list(itertools.product(*(rf_grid[k] for k in rf_keys)))
    print(f"Random Forest: {len(rf_combos)} configurations")

    for combo in rf_combos:
        params = dict(zip(rf_keys, combo))
        config_key = "|".join(f"{k}={params[k]}" for k in rf_keys)
        if f"random_forest|{config_key}" in done_keys:
            continue
        model = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
        model.fit(X_train, y_train)
        pos_err, orient_err = score_predictions(y_val, model.predict(X_val))
        all_results.append({"family": "random_forest", "config_key": config_key,
                            **{k: str(v) for k, v in params.items()},
                            "val_position_error_m": pos_err, "val_orientation_error_deg": orient_err})
        checkpoint(all_results)
        print(f"  {params} -> val pos {pos_err:.4f} m, orient {orient_err:.2f} deg")

    # ---------------- MLP grid ----------------
    if args.quick:
        mlp_grid = {"hidden_layer_sizes": [(128, 64), (32, 16)], "alpha": [1e-3, 1e-1],
                    "learning_rate_init": [1e-3]}
    else:
        mlp_grid = {
            "hidden_layer_sizes": [(128, 64), (64, 32), (32, 16), (256, 128)],
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "learning_rate_init": [1e-3, 1e-2],
        }

    mlp_keys = list(mlp_grid)
    mlp_combos = list(itertools.product(*(mlp_grid[k] for k in mlp_keys)))
    print(f"\nMLP: {len(mlp_combos)} configurations")

    for combo in mlp_combos:
        params = dict(zip(mlp_keys, combo))
        config_key = "|".join(f"{k}={params[k]}" for k in mlp_keys)
        if f"mlp|{config_key}" in done_keys:
            continue
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(activation="relu", solver="adam", max_iter=500,
                                  early_stopping=True, random_state=42, **params)),
        ])
        model.fit(X_train, y_train)
        pos_err, orient_err = score_predictions(y_val, model.predict(X_val))
        all_results.append({"family": "mlp", "config_key": config_key,
                            **{k: str(v) for k, v in params.items()},
                            "val_position_error_m": pos_err, "val_orientation_error_deg": orient_err})
        checkpoint(all_results)
        print(f"  {params} -> val pos {pos_err:.4f} m, orient {orient_err:.2f} deg")

    # ---------------- Refit the winners and evaluate on the held-out test split ----------------
    results_df = pd.DataFrame(all_results)
    rf_rows = results_df[results_df["family"] == "random_forest"]
    mlp_rows = results_df[results_df["family"] == "mlp"]
    if rf_rows.empty or mlp_rows.empty:
        print("\nGrid search incomplete - run the script again to finish the remaining configurations.")
        return

    best_rf_row = rf_rows.loc[rf_rows["val_position_error_m"].idxmin()]
    best_mlp_row = mlp_rows.loc[mlp_rows["val_position_error_m"].idxmin()]

    def parse_value(text):
        """Rebuild a hyper-parameter value from its CSV string form.

        Note: pandas may read integer columns back as floats (e.g. "8.0")
        when the column also contains "None", so integral floats are
        converted back to int - sklearn rejects max_depth=8.0.
        """
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return None
        text = str(text).strip()
        if text in {"None", "nan", ""}:
            return None
        if text.startswith("("):
            return tuple(int(float(p)) for p in text.strip("()").replace(" ", "").split(",") if p)
        value = float(text)
        return int(value) if value.is_integer() else value

    best_rf_params = {k: parse_value(best_rf_row[k]) for k in rf_keys}
    best_mlp_params = {k: parse_value(best_mlp_row[k]) for k in mlp_keys}
    best_rf_score = float(best_rf_row["val_position_error_m"])
    best_mlp_score = float(best_mlp_row["val_position_error_m"])

    print("\n" + "=" * 78)
    print("BEST CONFIGURATIONS (selected on VALIDATION only)")
    print("=" * 78)
    print(f"Random Forest: {best_rf_params}  (val position error {best_rf_score:.4f} m)")
    print(f"MLP:           {best_mlp_params}  (val position error {best_mlp_score:.4f} m)")

    best_rf = RandomForestRegressor(n_jobs=-1, random_state=42, **best_rf_params).fit(X_train, y_train)
    best_mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(activation="relu", solver="adam", max_iter=500,
                              early_stopping=True, random_state=42, **best_mlp_params)),
    ]).fit(X_train, y_train)

    rf_test_pos, rf_test_orient = score_predictions(y_test, best_rf.predict(X_test))
    mlp_test_pos, mlp_test_orient = score_predictions(y_test, best_mlp.predict(X_test))

    print("\n" + "=" * 78)
    print("FINAL TEST-SET PERFORMANCE (test touched only now)")
    print("=" * 78)
    print(f"Random Forest: position {rf_test_pos:.4f} m | orientation {rf_test_orient:.2f} deg")
    print(f"MLP:           position {mlp_test_pos:.4f} m | orientation {mlp_test_orient:.2f} deg")
    print("\nProposal Objective 4 thresholds: position < 0.4 m, orientation < 20 deg")
    print("Reference (pre-tuning, from evaluate_approach_pose.py):")
    print("  Random Forest 0.401 m / 29.27 deg | MLP 0.465 m / 42.22 deg | rule-based 0.305 m / 29.13 deg")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, MODEL_DIR / "approach_pose_random_forest_tuned.joblib")
    joblib.dump(best_mlp, MODEL_DIR / "approach_pose_mlp_tuned.joblib")

    checkpoint(all_results)

    summary = {
        "random_forest": {"best_params": {k: str(v) for k, v in best_rf_params.items()},
                          "val_position_error_m": best_rf_score,
                          "test_position_error_m": rf_test_pos,
                          "test_orientation_error_deg": rf_test_orient},
        "mlp": {"best_params": {k: str(v) for k, v in best_mlp_params.items()},
                "val_position_error_m": best_mlp_score,
                "test_position_error_m": mlp_test_pos,
                "test_orientation_error_deg": mlp_test_orient},
    }
    with (MODEL_DIR / "grid_search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nWritten: {MODEL_DIR / 'grid_search_results.csv'}")
    print(f"Written: {MODEL_DIR / 'grid_search_summary.json'}")


if __name__ == "__main__":
    main()
