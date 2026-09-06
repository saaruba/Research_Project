"""
Check whether the extracted training tables are actually ready to train on.

Run this AFTER running extract_training_table.py for every session you care about.

Example:
    cd /workspaces/Research_Project
    python3 scripts/check_dataset_readiness.py

What it does:
    1. Scans dataset/processed/session_*/training_table.csv for every session
       that has been extracted so far.
    2. For each one, reports row count, session duration, how much of each
       column is missing (NaN %), and whether the robot actually moved
       (based on the spread of robot_x / robot_y).
    3. Concatenates every available session into one
       dataset/processed/combined_training_table.csv with a session_id
       column added, ready to split into train/val/test.
    4. Prints a plain-English readiness verdict per session and overall.

This does not train anything. It only tells you what you actually have.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "dataset" / "processed"

# Columns we actually plan to feed a model. Adjust this list as your feature
# set gets finalised (see Phase A of the master checklist).
CORE_COLUMNS = ["robot_x", "robot_y", "robot_yaw", "linear_x", "angular_z"]
# num_faces is deliberately NOT in this list: the extraction script fills it
# with 0 whenever face data is missing, so it is never NaN and cannot be used
# to detect whether face data actually exists. Use the position/bbox columns
# instead, which stay genuinely blank when there is no face data.
FACE_COLUMNS = [
    "num_faces",
    "face_center_x",
    "face_center_y",
    "face_bbox_min_x",
    "face_bbox_min_y",
    "face_bbox_max_x",
    "face_bbox_max_y",
]
FACE_PRESENCE_COLUMNS = [
    "face_center_x",
    "face_center_y",
    "face_bbox_min_x",
    "face_bbox_min_y",
    "face_bbox_max_x",
    "face_bbox_max_y",
]
LIDAR_COLUMNS = ["lidar_min_range", "lidar_mean_range"]

MOVEMENT_STD_THRESHOLD_M = 0.15  # below this, the robot is judged as "barely moved"


def find_session_dirs() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(
        [p for p in PROCESSED_DIR.iterdir() if p.is_dir() and p.name.startswith("session_")],
        key=lambda p: p.name,
    )


def check_one_session(session_dir: Path) -> dict:
    session_name = session_dir.name.replace("session_", "")
    table_path = session_dir / "training_table.csv"

    result = {
        "session_name": session_name,
        "extracted": table_path.exists(),
        "row_count": 0,
        "duration_sec": None,
        "nan_pct": {},
        "moved": None,
        "verdict": "",
    }

    if not table_path.exists():
        result["verdict"] = "NOT YET EXTRACTED - run extract_training_table.py for this session."
        return result

    df = pd.read_csv(table_path)
    result["row_count"] = len(df)

    if df.empty:
        result["verdict"] = "EXTRACTED BUT EMPTY - check the bag/csv inputs for this session."
        return result

    if "timestamp" in df.columns:
        result["duration_sec"] = round(float(df["timestamp"].max() - df["timestamp"].min()), 2)

    for column in CORE_COLUMNS + FACE_COLUMNS + LIDAR_COLUMNS:
        if column in df.columns:
            nan_pct = float(df[column].isna().mean() * 100)
            result["nan_pct"][column] = round(nan_pct, 1)

    if "robot_x" in df.columns and "robot_y" in df.columns:
        spread = float(np.hypot(df["robot_x"].std(skipna=True), df["robot_y"].std(skipna=True)))
        result["moved"] = spread >= MOVEMENT_STD_THRESHOLD_M

    # Build a plain-English verdict
    notes = []
    core_nan = [c for c in CORE_COLUMNS if result["nan_pct"].get(c, 100) > 5]
    if core_nan:
        notes.append(f"core columns have missing data: {core_nan}")

    face_all_missing = all(result["nan_pct"].get(c, 100) >= 99 for c in FACE_PRESENCE_COLUMNS)
    if face_all_missing:
        notes.append("no usable face/human-position data (expected for non-session-1/3 sessions)")

    if result["moved"] is False:
        notes.append("robot barely moved - check this session before using it for training")

    if not notes:
        result["verdict"] = "OK - usable for training with full feature set."
    elif not core_nan and result["moved"] is not False:
        result["verdict"] = "USABLE, WITHOUT FACE FEATURES - " + "; ".join(notes)
    else:
        result["verdict"] = "NEEDS REVIEW - " + "; ".join(notes)

    return result


def build_combined_table(session_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for session_dir in session_dirs:
        session_name = session_dir.name.replace("session_", "")
        table_path = session_dir / "training_table.csv"
        if not table_path.exists():
            continue
        df = pd.read_csv(table_path)
        if df.empty:
            continue
        df.insert(0, "session_id", session_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    session_dirs = find_session_dirs()

    if not session_dirs:
        print(f"No processed sessions found under {PROCESSED_DIR}")
        return

    print("\n" + "=" * 90)
    print("DATASET READINESS REPORT")
    print("=" * 90)

    results = []
    for session_dir in session_dirs:
        result = check_one_session(session_dir)
        results.append(result)

        print(f"\nSession {result['session_name']}:")
        if not result["extracted"]:
            print(f"  {result['verdict']}")
            continue

        print(f"  rows: {result['row_count']}, duration: {result['duration_sec']} sec")
        print(f"  robot moved meaningfully: {result['moved']}")
        if result["nan_pct"]:
            print("  missing data (%):")
            for column, pct in result["nan_pct"].items():
                print(f"    - {column}: {pct}%")
        print(f"  verdict: {result['verdict']}")

    combined = build_combined_table(session_dirs)
    output_path = PROCESSED_DIR / "combined_training_table.csv"

    if combined.empty:
        print("\nNo sessions have been extracted yet, so no combined table was written.")
    else:
        combined.to_csv(output_path, index=False)
        print(f"\nCombined table written to: {output_path}")
        print(f"Total rows across all sessions: {len(combined)}")
        print(f"Sessions included: {sorted(combined['session_id'].unique().tolist())}")

    extracted_count = sum(1 for r in results if r["extracted"])
    print("\n" + "=" * 90)
    print(f"SUMMARY: {extracted_count}/{len(results)} sessions extracted")
    ok_count = sum(1 for r in results if r["verdict"].startswith("OK") or r["verdict"].startswith("USABLE"))
    print(f"Usable for a first training pass: {ok_count}/{len(results)}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()