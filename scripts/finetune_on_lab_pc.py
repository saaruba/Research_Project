#!/usr/bin/env python3
"""
FULL HYPER-PARAMETER SEARCH  -  run this on the lab PC.

    python3 scripts/finetune_on_lab_pc.py
    N_ITER=300 python3 scripts/finetune_on_lab_pc.py      # longer search

Additive: writes only *_ft.joblib and finetune_results.{json,csv}. No v1 or v2
artefact is modified, so the shipped models and Tables 5.1-5.10 are unaffected
whatever this finds.

============================================================================
WHAT THIS CAN AND CANNOT ACHIEVE - READ BEFORE SPENDING A NIGHT ON IT
============================================================================
Two things are already measured (docs/V2_RETRAINING_STUDY.md §5c):

  1. MORE CAPACITY MAKES IT WORSE. Loosening the Random Forest from
     depth=8/leaf=20 to unconstrained drops TRAIN error 0.597 -> 0.210 m while
     TEST error RISES 0.648 -> 0.739 m. The shipped settings are already close
     to the optimum of that trade-off, so a faster machine buys search breadth,
     not headroom.

  2. THE FLOOR IS ~0.505 m. Training rows whose features are nearly identical
     (mean standardised distance 0.284) specify stop poses that disagree by
     0.505 m on average. No model can predict better than its labels agree.

Current test error is 0.648 m. The floor is ~0.505 m. So the realistic prize
here is on the order of 0.1 m, and the 0.4 m Objective 4 threshold sits BELOW
the floor and is not reachable by search.

Worth running to state in the dissertation that the hyper-parameter space was
searched properly rather than inherited. Not worth running in the expectation
of a different conclusion.

============================================================================
METHOD
============================================================================
Grouped by session so no event leaks between folds; the same held-out test
sessions (5, 9, 59) as every other study, scored on the same rows, so results
drop straight into the comparison tables. Searches RandomForest,
HistGradientBoosting and MLP over ranges that BRACKET the shipped values in
both directions - a search that can only add capacity would miss the
possibility that the shipped model is already too large.
"""

from __future__ import annotations

import json
import os
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

TRAIN_SESSIONS = [1, 3, 7, 8, 11, 12, 26, 27, 28, 30, 31, 49, 51, 52, 55, 58, 60]
VAL_SESSIONS = [10, 14, 15, 54]
TEST_SESSIONS = [5, 9, 59]

FEATURES = ["lidar_min_range", "lidar_mean_range", "linear_x_prev",
            "angular_z_prev", "num_people", "group_bearing_rad", "group_scale_norm"]
TARGETS = ["target_dx", "target_dy", "target_dyaw"]

POSITION_THRESHOLD_M = 0.4
ORIENTATION_THRESHOLD_DEG = 20.0
DECIMATE_HZ = 10.0
SOURCE = "approach_pose_dataset_v2_0.25.csv"     # best of the volume sweep
N_ITER = int(os.environ.get("N_ITER", "120"))
SEED = 42


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


def sample_params(rng: np.random.Generator, kind: str) -> dict:
    """Ranges bracket the shipped values ABOVE AND BELOW."""
    if kind == "rf":
        return dict(
            n_estimators=int(rng.choice([100, 150, 300, 500])),
            max_depth=int(rng.choice([4, 6, 8, 10, 12, 16])),
            min_samples_leaf=int(rng.choice([5, 10, 20, 40, 80, 160])),
            # Chosen by index: rng.choice on a mixed list coerces everything to
            # numpy strings, which sklearn's parameter validation rejects.
            max_features=[1.0, 0.7, 0.5, "sqrt"][int(rng.integers(4))],
        )
    if kind == "gb":
        return dict(
            max_iter=int(rng.choice([100, 200, 300, 500])),
            learning_rate=float(rng.choice([0.02, 0.04, 0.06, 0.1, 0.2])),
            max_depth=int(rng.choice([2, 3, 4, 6, 8])),
            min_samples_leaf=int(rng.choice([10, 20, 40, 80])),
            l2_regularization=float(rng.choice([0.0, 0.5, 1.0, 5.0])),
        )
    return dict(
        hidden_layer_sizes=tuple(rng.choice([(16,), (32, 16), (64, 32), (128, 64)])) if False
        else [(16,), (32, 16), (64, 32), (128, 64)][int(rng.integers(4))],
        alpha=float(rng.choice([0.001, 0.01, 0.1, 1.0, 10.0])),
        learning_rate_init=float(rng.choice([0.0003, 0.001, 0.003])),
        max_iter=400, early_stopping=True, n_iter_no_change=20, random_state=SEED,
    )


def build(kind: str, params: dict):
    if kind == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **params)
    if kind == "gb":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=SEED, **params))
    return Pipeline([("scale", StandardScaler()), ("mlp", MLPRegressor(**params))])


def main() -> None:
    src = pd.read_csv(PROCESSED / SOURCE)
    train = decimate(src[src.session_id.isin(TRAIN_SESSIONS)])
    val = decimate(src[src.session_id.isin(VAL_SESSIONS)])
    base = pd.read_csv(PROCESSED / "approach_pose_dataset_v2.csv")
    test = base[base.session_id.isin(TEST_SESSIONS)]

    Xtr, Ytr, wtr = train[FEATURES].values, train[TARGETS].values, event_weights(train)
    Xva, Yva = val[FEATURES].values, val[TARGETS].values
    Xte, Yte = test[FEATURES].values, test[TARGETS].values

    print("=" * 76)
    print(f"  source : {SOURCE}")
    print(f"  train  : {len(train):,} rows / {train.event_id.nunique()} events")
    print(f"  val    : {len(val):,} rows / {val.event_id.nunique()} events  (selection)")
    print(f"  test   : {len(test):,} rows / {test.event_id.nunique()} events  (reported once)")
    print(f"  budget : {N_ITER} configurations per model family")
    print("=" * 76, flush=True)

    rng = np.random.default_rng(SEED)
    results, best = [], {}

    for kind in ["rf", "gb", "mlp"]:
        best_here = None
        for i in range(N_ITER):
            params = sample_params(rng, kind)
            try:
                model = build(kind, params)
                if kind == "mlp":
                    p = event_weights(train) / event_weights(train).sum()
                    idx = rng.choice(len(Xtr), size=len(Xtr), replace=True, p=p)
                    model.fit(Xtr[idx], Ytr[idx])
                else:
                    model.fit(Xtr, Ytr, sample_weight=wtr)
            except Exception as exc:  # noqa: BLE001
                print(f"    {kind} config {i} failed: {exc}", flush=True)
                continue

            s_val = score(Yva, model.predict(Xva))
            row = {"kind": kind, "params": {k: str(v) for k, v in params.items()},
                   "val_position_error_m": s_val["mean_position_error_m"],
                   "val_orientation_error_deg": s_val["mean_orientation_error_deg"]}
            results.append(row)
            if best_here is None or s_val["mean_position_error_m"] < best_here[0]:
                best_here = (s_val["mean_position_error_m"], params, model)
                print(f"    {kind}  [{i + 1}/{N_ITER}]  new best val "
                      f"{s_val['mean_position_error_m']:.4f} m  {params}", flush=True)

        if best_here is not None:
            _, params, model = best_here
            joblib.dump(model, MODEL_DIR / f"approach_pose_{kind}_ft.joblib")
            best[kind] = (params, model)

    # Test set touched ONCE, after all selection is complete.
    print("\n" + "=" * 76)
    print("  HELD-OUT TEST  (selection was done on the validation sessions only)")
    print("=" * 76)
    print(f"  {'model':<10} {'pos m':>8} {'ori deg':>9} {'both':>8}")
    print("  " + "-" * 60)
    final = []
    for kind, (params, model) in best.items():
        s = score(Yte, model.predict(Xte))
        print(f"  {kind:<10} {s['mean_position_error_m']:>8.3f} "
              f"{s['mean_orientation_error_deg']:>9.2f} "
              f"{s['pct_within_both_thresholds']:>7.1f}%")
        final.append({"kind": kind, "params": {k: str(v) for k, v in params.items()}, **s})

    print("\n  " + "-" * 60)
    print(f"  {'shipped v1 RF':<10} {0.656:>8.3f} {34.91:>9.2f} {23.5:>7.1f}%")
    print(f"  {'label floor':<10} {0.505:>8.3f}   (nearest-neighbour disagreement)")
    print("=" * 76)

    (MODEL_DIR / "finetune_results.json").write_text(
        json.dumps({"search": results, "final": final}, indent=2))
    pd.DataFrame(results).to_csv(MODEL_DIR / "finetune_results.csv", index=False)
    print(f"Written: {MODEL_DIR / 'finetune_results.json'}")
    print("Shipped v1/v2 models are unchanged.")


if __name__ == "__main__":
    main()
