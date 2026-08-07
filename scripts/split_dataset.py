#!/usr/bin/env python3
"""
Split the combined training table into train / validation / test sets,
splitting by whole session so no session's rows leak across splits.

Run this AFTER check_dataset_readiness.py has produced
dataset/processed/combined_training_table.csv.

Example:
    cd /workspaces/Research_Project
    python3 scripts/split_dataset.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "dataset" / "processed"
COMBINED_PATH = PROCESSED_DIR / "combined_training_table.csv"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# whatever's left goes to test
RANDOM_SEED = 42

# Sessions 1 and 3 are the only ones with real face/human-position data.
# We deliberately keep both in the training set for now, since with only 2
# such sessions there isn't enough to hold one out and still learn anything
# from it. This is a limitation worth stating plainly in your report.
FORCE_INTO_TRAIN = {1, 3}


def main() -> None:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(
            f"{COMBINED_PATH} not found. Run check_dataset_readiness.py first."
        )

    df = pd.read_csv(COMBINED_PATH)
    all_sessions = sorted(df["session_id"].unique().tolist())

    remaining = [s for s in all_sessions if s not in FORCE_INTO_TRAIN]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(remaining)

    n_total = len(all_sessions)
    n_train_target = round(n_total * TRAIN_FRACTION)
    n_val_target = round(n_total * VAL_FRACTION)

    train_sessions = list(FORCE_INTO_TRAIN)
    n_more_train_needed = max(0, n_train_target - len(train_sessions))
    train_sessions += remaining[:n_more_train_needed]
    remaining = remaining[n_more_train_needed:]

    val_sessions = remaining[:n_val_target]
    test_sessions = remaining[n_val_target:]

    train_df = df[df["session_id"].isin(train_sessions)]
    val_df = df[df["session_id"].isin(val_sessions)]
    test_df = df[df["session_id"].isin(test_sessions)]

    train_df.to_csv(PROCESSED_DIR / "train_table.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val_table.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_table.csv", index=False)

    print("\n" + "=" * 70)
    print("DATASET SPLIT (by whole session, no leakage)")
    print("=" * 70)
    print(f"Train sessions ({len(train_sessions)}): {sorted(train_sessions)}")
    print(f"  rows: {len(train_df)}")
    print(f"Val sessions   ({len(val_sessions)}): {sorted(val_sessions)}")
    print(f"  rows: {len(val_df)}")
    print(f"Test sessions  ({len(test_sessions)}): {sorted(test_sessions)}")
    print(f"  rows: {len(test_df)}")
    print("\nNote: sessions 1 and 3 (the only ones with face/human-position")
    print("data) are both kept in the training set on purpose - see comment")
    print("in this script for why.")
    print("\nOutput written to:")
    print(f"  {PROCESSED_DIR / 'train_table.csv'}")
    print(f"  {PROCESSED_DIR / 'val_table.csv'}")
    print(f"  {PROCESSED_DIR / 'test_table.csv'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
