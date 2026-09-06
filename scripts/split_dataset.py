"""
SPLIT THE DATA INTO TRAIN / VALIDATION / TEST - by whole session.

    python3 scripts/split_dataset.py

Run this AFTER check_dataset_readiness.py has produced
dataset/processed/combined_training_table.csv.

============================================================================
WHY SPLITTING MATTERS, IF THIS IS NEW TO YOU
============================================================================
A model that is tested on data it was trained on will look excellent and be
useless, because it can simply memorise. So the data is divided three ways:

    TRAIN       the model learns from these
    VALIDATION  used to choose settings (how deep a tree, how big a network)
    TEST        touched ONCE, at the very end, to report the honest result

If the test set is consulted while tuning, its numbers stop being honest -
choices have then been made to suit it.

============================================================================
THE IMPORTANT DECISION: SPLIT BY SESSION, NOT BY ROW
============================================================================
Rows in this dataset are sampled about 30 times a second, so consecutive rows
are nearly identical. Splitting randomly by row would put one moment in
training and the moment 33 milliseconds later in test. The model would score
brilliantly by recognising almost the same instant twice - a leak that hides a
total failure to generalise.

Splitting by WHOLE SESSION prevents it. Every row of a session goes to exactly
one split, so the test set contains rooms, people and lighting the model has
never encountered. That is a much harder and much more meaningful test, and it
is why the reported errors are larger than a row-wise split would produce.

    17 sessions -> train        44,190 rows
     4 sessions -> validation   14,444 rows
     3 sessions -> test         11,921 rows

Sessions 1 and 3 are forced into TRAIN because they are the only two carrying
face and gaze annotations - holding one out would remove capability from the
training set for very little validation gain.
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
