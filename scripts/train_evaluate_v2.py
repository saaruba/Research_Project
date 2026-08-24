#!/usr/bin/env python3
"""
TRAIN AND EVALUATE THE V2 MODELS  (additive - no v1 artefact is overwritten)

    python3 scripts/train_evaluate_v2.py

Writes:
    dataset/processed/models/*_v2.joblib
    dataset/processed/models/v1_vs_v2_evaluation.json
    dataset/processed/models/v1_vs_v2_evaluation.csv

============================================================================
THE COMPARISON THIS MAKES, AND WHY IT IS BUILT THIS WAY
============================================================================
v1 and v2 segment the recordings differently, so they contain different rows.
Comparing "v1 model on v1 rows" against "v2 model on v2 rows" would confound
the change of model with a change of test set, and would prove nothing.

The v2 dataset therefore carries the v1 feature columns unchanged. Every model
is scored on the SAME v2 test rows, which lets the improvement be decomposed:

    A. v1 model, v1 test rows   - reproduces the existing Table 5.1 (sanity)
    B. v1 model, v2 test rows   - how the shipped models do on real approaches
    C. v2 model, v1 features    - the effect of RE-SEGMENTATION alone
    D. v2 model, v2 features    - the effect of re-segmentation PLUS geometry

B -> C isolates the segmentation fix. C -> D isolates the new features. If D
is not better than B there is no case for changing anything, and that is a
publishable result in itself.

EVENT-LEVEL WEIGHTING
---------------------
One approach event contributes hundreds of near-identical consecutive rows, so
row-level training lets a few long events dominate the loss. Each row is
weighted by 1/(rows in its event), making every approach count equally
regardless of how long it took. Sessions - not rows - define the split, so no
event is ever spread across train and test.

TEMPORAL DECIMATION
-------------------
The recordings are ~60 Hz, so consecutive rows within an event are nearly
identical: the robot has moved about a centimetre and the group has not moved
at all. Those duplicates inflate the apparent sample size without adding
information - which is precisely the "70,555 rows but only 462 independent
behaviours" problem. Rows are therefore decimated to DECIMATE_HZ within each
event before training. This is a statement about the real information content,
not a shortcut; the held-out TEST rows are left at full rate so the evaluation
is unaffected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROCESSED / "models"

# Same session split as v1. Hard-coded so the comparison cannot drift.
TRAIN_SESSIONS = [1, 3, 7, 8, 11, 12, 26, 27, 28, 30, 31, 49, 51, 52, 55, 58, 60]
VAL_SESSIONS = [10, 14, 15, 54]
TEST_SESSIONS = [5, 9, 59]

V1_FEATURES = ["lidar_min_range", "lidar_mean_range", "linear_x_prev",
               "angular_z_prev", "num_people", "group_bearing_rad", "group_scale_norm"]
V2_EXTRA = ["group_span_rad", "nearest_person_span_rad", "gap_bearing_rad",
            "gap_width_rad", "person_spacing_rad", "people_visible"]
V2_FEATURES = V1_FEATURES + V2_EXTRA
TARGETS = ["target_dx", "target_dy", "target_dyaw"]

STANDOFF_DISTANCE = 1.2
POSITION_THRESHOLD_M = 0.4
ORIENTATION_THRESHOLD_DEG = 20.0
DECIMATE_HZ = 10.0          # training rows only; test rows stay at full rate


def decimate(df: pd.DataFrame, hz: float = DECIMATE_HZ) -> pd.DataFrame:
    """Keep at most `hz` rows per second within each event."""
    if "event_id" not in df.columns:
        return df
    keep = []
    for _, ev in df.groupby("event_id"):
        ev = ev.sort_values("timestamp")
        last = -np.inf
        mask = []
        for t in ev["timestamp"].values:
            take = (t - last) >= (1.0 / hz)
            mask.append(take)
            if take:
                last = t
        keep.append(ev[np.array(mask)])
    return pd.concat(keep, ignore_index=True)

# Tuned hyper-parameters carried over from grid_search_summary.json. The MLP is
# deliberately kept at 32/16 with alpha=0.1: the earlier tuning showed the
# smaller regularised network beat 128/64, so capacity is not the constraint.
RF_PARAMS = dict(n_estimators=150, max_depth=8, min_samples_leaf=20,
                 n_jobs=-1, random_state=42)
MLP_PARAMS = dict(hidden_layer_sizes=(32, 16), alpha=0.1, learning_rate_init=0.001,
                  max_iter=600, early_stopping=True, n_iter_no_change=20,
                  random_state=42)
GB_PARAMS = dict(max_iter=300, learning_rate=0.06, max_depth=6,
                 min_samples_leaf=40, l2_regularization=1.0, random_state=42)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def score(y_true: np.ndarray, y_pred: np.ndarray, name: str, n_events: int) -> dict:
    pos = np.hypot(y_true[:, 0] - y_pred[:, 0], y_true[:, 1] - y_pred[:, 1])
    ori = np.degrees(np.abs(wrap_angle(y_true[:, 2] - y_pred[:, 2])))
    return {
        "policy": name,
        "n_rows": int(len(y_true)),
        "n_events": int(n_events),
        "mean_position_error_m": float(pos.mean()),
        "median_position_error_m": float(np.median(pos)),
        "mean_orientation_error_deg": float(ori.mean()),
        "median_orientation_error_deg": float(np.median(ori)),
        "pct_within_position_threshold": float((pos < POSITION_THRESHOLD_M).mean() * 100),
        "pct_within_orientation_threshold": float((ori < ORIENTATION_THRESHOLD_DEG).mean() * 100),
        "pct_within_both_thresholds": float(
            ((pos < POSITION_THRESHOLD_M) & (ori < ORIENTATION_THRESHOLD_DEG)).mean() * 100),
    }


def predict_rule(df: pd.DataFrame) -> np.ndarray:
    """Phase E rule applied offline - identical to evaluate_approach_pose.py."""
    bearing = df["group_bearing_rad"].values
    forward = np.maximum(df["lidar_min_range"].values - STANDOFF_DISTANCE, 0.0)
    return np.column_stack([forward * np.cos(bearing),
                            forward * np.sin(bearing),
                            bearing])


def event_weights(df: pd.DataFrame) -> np.ndarray:
    """1/(rows in event), so every approach counts equally."""
    if "event_id" not in df.columns:
        return np.ones(len(df))
    counts = df.groupby("event_id")["event_id"].transform("size").values
    w = 1.0 / counts
    return w * (len(df) / w.sum())          # keep the mean weight at 1.0


def fit_models(train: pd.DataFrame, features: list[str], tag: str,
               weights: np.ndarray | None) -> dict:
    """Fit the three regressors, reusing anything already on disk.

    Each model is saved as soon as it is fitted and reloaded on a later run, so
    an interrupted session resumes instead of starting over.
    """
    X = train[features].values
    Y = train[TARGETS].values
    models = {}

    def cached(name: str, build):
        path = MODEL_DIR / f"approach_pose_{name}.joblib"
        if path.exists():
            print(f"    {name}: reusing {path.name}", flush=True)
            return joblib.load(path)
        print(f"    {name}: fitting...", flush=True)
        model = build()
        joblib.dump(model, path)
        return model

    def build_rf():
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X, Y, sample_weight=weights)
        return m

    models[f"random_forest_{tag}"] = cached(f"random_forest_{tag}", build_rf)

    def build_mlp():
        # MLPRegressor has no sample_weight, so events are balanced by
        # resampling instead: draw rows with probability proportional to the
        # same weights.
        if weights is not None:
            rng = np.random.default_rng(42)
            p = weights / weights.sum()
            idx = rng.choice(len(X), size=len(X), replace=True, p=p)
            Xm, Ym = X[idx], Y[idx]
        else:
            Xm, Ym = X, Y
        m = Pipeline([("scale", StandardScaler()), ("mlp", MLPRegressor(**MLP_PARAMS))])
        m.fit(Xm, Ym)
        return m

    def build_gb():
        m = MultiOutputRegressor(HistGradientBoostingRegressor(**GB_PARAMS))
        m.fit(X, Y, sample_weight=weights)
        return m

    models[f"mlp_{tag}"] = cached(f"mlp_{tag}", build_mlp)
    models[f"gradient_boosting_{tag}"] = cached(f"gradient_boosting_{tag}", build_gb)
    return models


def main() -> None:
    v1 = pd.read_csv(PROCESSED / "approach_pose_dataset.csv")
    v2 = pd.read_csv(PROCESSED / "approach_pose_dataset_v2.csv")

    v1_test = v1[v1.session_id.isin(TEST_SESSIONS)]
    v2_train_full = v2[v2.session_id.isin(TRAIN_SESSIONS)]
    v2_train = decimate(v2_train_full)
    v2_test = v2[v2.session_id.isin(TEST_SESSIONS)]   # full rate, untouched

    print("=" * 74, flush=True)
    print(f"  v1: {len(v1):,} rows   test {len(v1_test):,}")
    print(f"  v2: {len(v2):,} rows   test {len(v2_test):,} "
          f"({v2_test.event_id.nunique()} events)")
    print(f"  v2 train: {len(v2_train_full):,} rows -> {len(v2_train):,} "
          f"after {DECIMATE_HZ:.0f} Hz decimation "
          f"({v2_train.event_id.nunique()} events)")
    print("=" * 74, flush=True)

    w = event_weights(v2_train)
    print(f"\nTraining v2 models (event-weighted, {v2_train.event_id.nunique()} events)...")
    models = {}
    models.update(fit_models(v2_train, V1_FEATURES, "v2seg", w))
    print("  v1-feature models trained (segmentation effect)", flush=True)
    models.update(fit_models(v2_train, V2_FEATURES, "v2full", w))
    print("  v2-feature models trained (segmentation + geometry)", flush=True)

    print(f"  {len(models)} model(s) ready in {MODEL_DIR}", flush=True)

    results = {"panel_A_v1_models_v1_rows": [], "panel_B_v1_models_v2_rows": [],
               "panel_C_v2seg_models_v2_rows": [], "panel_D_v2full_models_v2_rows": []}

    n_ev_v2 = v2_test.event_id.nunique()

    # ---- Panel A: reproduce the existing evaluation, as a sanity check -------
    yA = v1_test[TARGETS].values
    naive_v1 = np.tile(v1[v1.session_id.isin(TRAIN_SESSIONS)][TARGETS].values.mean(axis=0),
                       (len(v1_test), 1))
    results["panel_A_v1_models_v1_rows"].append(score(yA, naive_v1, "naive", 0))
    results["panel_A_v1_models_v1_rows"].append(score(yA, predict_rule(v1_test), "rule_based", 0))
    for f, label in [("approach_pose_random_forest_tuned.joblib", "random_forest_tuned"),
                     ("approach_pose_mlp_tuned.joblib", "mlp_tuned")]:
        p = MODEL_DIR / f
        if p.exists():
            results["panel_A_v1_models_v1_rows"].append(
                score(yA, joblib.load(p).predict(v1_test[V1_FEATURES].values), label, 0))

    # ---- Panels B, C, D: everything scored on the SAME v2 test rows ----------
    yB = v2_test[TARGETS].values
    naive_v2 = np.tile(v2_train[TARGETS].values.mean(axis=0), (len(v2_test), 1))
    results["panel_B_v1_models_v2_rows"].append(score(yB, naive_v2, "naive", n_ev_v2))
    results["panel_B_v1_models_v2_rows"].append(score(yB, predict_rule(v2_test), "rule_based", n_ev_v2))
    for f, label in [("approach_pose_random_forest_tuned.joblib", "v1_random_forest_tuned"),
                     ("approach_pose_mlp_tuned.joblib", "v1_mlp_tuned")]:
        p = MODEL_DIR / f
        if p.exists():
            results["panel_B_v1_models_v2_rows"].append(
                score(yB, joblib.load(p).predict(v2_test[V1_FEATURES].values), label, n_ev_v2))

    for tag, panel, feats in [("v2seg", "panel_C_v2seg_models_v2_rows", V1_FEATURES),
                              ("v2full", "panel_D_v2full_models_v2_rows", V2_FEATURES)]:
        for kind in ["random_forest", "mlp", "gradient_boosting"]:
            m = models[f"{kind}_{tag}"]
            results[panel].append(
                score(yB, m.predict(v2_test[feats].values), f"{kind}_{tag}", n_ev_v2))

    # ---- Report -------------------------------------------------------------
    titles = {
        "panel_A_v1_models_v1_rows": "A. v1 models on v1 test rows  (reproduces Table 5.1)",
        "panel_B_v1_models_v2_rows": "B. v1 models on v2 test rows  (real approaches)",
        "panel_C_v2seg_models_v2_rows": "C. v2 models, v1 features     (segmentation only)",
        "panel_D_v2full_models_v2_rows": "D. v2 models, v2 features     (segmentation + geometry)",
    }
    rows = []
    for key, title in titles.items():
        print("\n" + "=" * 74)
        print(f"  {title}")
        print("=" * 74)
        print(f"  {'policy':<26} {'pos m':>8} {'ori deg':>9} {'<0.4m':>7} {'<20deg':>8} {'both':>7}")
        print("  " + "-" * 70)
        for r in results[key]:
            print(f"  {r['policy']:<26} {r['mean_position_error_m']:>8.3f} "
                  f"{r['mean_orientation_error_deg']:>9.2f} "
                  f"{r['pct_within_position_threshold']:>6.1f}% "
                  f"{r['pct_within_orientation_threshold']:>7.1f}% "
                  f"{r['pct_within_both_thresholds']:>6.1f}%")
            rows.append({**r, "panel": key})

    (MODEL_DIR / "v1_vs_v2_evaluation.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(MODEL_DIR / "v1_vs_v2_evaluation.csv", index=False)

    # ---- The decision --------------------------------------------------------
    def best(panel, metric="mean_position_error_m"):
        learned = [r for r in results[panel] if r["policy"] not in ("naive", "rule_based")]
        return min(learned, key=lambda r: r[metric]) if learned else None

    b, c, d = best("panel_B_v1_models_v2_rows"), best("panel_C_v2seg_models_v2_rows"), \
        best("panel_D_v2full_models_v2_rows")
    print("\n" + "=" * 74)
    print("  DECISION GATE - best learned model on the v2 test rows")
    print("=" * 74)
    for lbl, r in [("B  shipped v1 model ", b), ("C  re-segmented     ", c),
                   ("D  + new geometry   ", d)]:
        if r:
            print(f"  {lbl} {r['policy']:<24} {r['mean_position_error_m']:.3f} m  "
                  f"{r['mean_orientation_error_deg']:.2f} deg  both {r['pct_within_both_thresholds']:.1f}%")
    if b and d:
        dp = 100 * (b["mean_position_error_m"] - d["mean_position_error_m"]) / b["mean_position_error_m"]
        do = 100 * (b["mean_orientation_error_deg"] - d["mean_orientation_error_deg"]) / b["mean_orientation_error_deg"]
        print(f"\n  position error {dp:+.1f}%   orientation error {do:+.1f}%   (positive = v2 better)")
    print("=" * 74)
    print("v1 models and datasets are unchanged on disk.")


if __name__ == "__main__":
    main()
