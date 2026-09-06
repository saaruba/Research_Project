"""
Build one master summary table from all processed session_summary.json files.

Default input directory:
    dataset/processed

Outputs:
    dataset/processed/master_session_summary.csv
    dataset/processed/master_session_summary.json

Example:
    python3 scripts/build_master_session_summary.py
    python3 scripts/build_master_session_summary.py --processed-dir /workspaces/Research_Project/dataset/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "session_name",
    "session_path",
    "bag_file",
    "bag_format",
    "has_rgb",
    "has_depth",
    "has_lidar",
    "has_robot_pose",
    "has_cmd_vel",
    "has_tf",
    "has_face_landmarks",
    "is_suitable_for_navigation_learning",
    "recommended_next_step",
]

REQUIRED_BOOL_FIELDS = [
    "has_rgb",
    "has_depth",
    "has_lidar",
    "has_robot_pose",
    "has_cmd_vel",
    "has_tf",
    "has_face_landmarks",
]


def find_session_summary_files(processed_dir: Path) -> list[Path]:
    """Find all session_summary.json files inside session_* folders."""
    json_files = []

    if not processed_dir.exists():
        print(f"Warning: processed directory does not exist: {processed_dir}")
        return json_files

    session_dirs = sorted(path for path in processed_dir.glob("session_*") if path.is_dir())

    if not session_dirs:
        print(f"Warning: no session_* folders found in: {processed_dir}")
        return json_files

    for session_dir in session_dirs:
        json_path = session_dir / "session_summary.json"
        if json_path.exists():
            json_files.append(json_path)
        else:
            print(f"Warning: missing session_summary.json in {session_dir}")

    return json_files


def load_session_summary(json_path: Path) -> dict[str, Any] | None:
    """Load one session_summary.json safely."""
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            print(f"Warning: JSON is not an object in {json_path}")
            return None

        return data

    except json.JSONDecodeError as error:
        print(f"Warning: malformed JSON in {json_path}: {error}")
        return None
    except Exception as error:
        print(f"Warning: could not read {json_path}: {error}")
        return None


def calculate_suitability(summary: dict[str, Any]) -> bool:
    """A session is suitable only if all required boolean fields are True."""
    return all(bool(summary.get(field, False)) for field in REQUIRED_BOOL_FIELDS)


def find_missing_required_fields(summary: dict[str, Any]) -> list[str]:
    """Return the important required fields that are missing or False."""
    missing = []
    for field in REQUIRED_BOOL_FIELDS:
        if not bool(summary.get(field, False)):
            missing.append(field)
    return missing


def normalise_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only the required columns and calculate suitability."""
    row = {
        "session_name": summary.get("session_name"),
        "session_path": summary.get("session_path"),
        "bag_file": summary.get("bag_file"),
        "bag_format": summary.get("bag_format"),
        "has_rgb": bool(summary.get("has_rgb", False)),
        "has_depth": bool(summary.get("has_depth", False)),
        "has_lidar": bool(summary.get("has_lidar", False)),
        "has_robot_pose": bool(summary.get("has_robot_pose", False)),
        "has_cmd_vel": bool(summary.get("has_cmd_vel", False)),
        "has_tf": bool(summary.get("has_tf", False)),
        "has_face_landmarks": bool(summary.get("has_face_landmarks", False)),
        "recommended_next_step": summary.get("recommended_next_step", ""),
    }

    row["is_suitable_for_navigation_learning"] = calculate_suitability(row)
    return row


def build_master_dataframe(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the combined master dataframe."""
    rows = [normalise_summary(summary) for summary in summaries]
    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        dataframe = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        dataframe = dataframe[REQUIRED_COLUMNS]

    return dataframe


def write_outputs(dataframe: pd.DataFrame, processed_dir: Path) -> tuple[Path, Path]:
    """Write master CSV and JSON outputs into dataset/processed."""
    csv_path = processed_dir / "master_session_summary.csv"
    json_path = processed_dir / "master_session_summary.json"

    dataframe.to_csv(csv_path, index=False)

    records = dataframe.to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    return csv_path, json_path


def print_terminal_summary(dataframe: pd.DataFrame) -> None:
    """Print a clean terminal summary."""
    total_sessions = len(dataframe)

    if total_sessions == 0:
        print("\nMaster Session Summary Created\n")
        print("Processed sessions found: 0")
        print("Suitable sessions: 0")
        print("Not suitable sessions: 0")
        print("\nNo valid session summaries were available.\n")
        return

    suitable_df = dataframe[dataframe["is_suitable_for_navigation_learning"] == True]
    unsuitable_df = dataframe[dataframe["is_suitable_for_navigation_learning"] == False]

    suitable_names = suitable_df["session_name"].dropna().astype(str).tolist()

    print("\nMaster Session Summary Created\n")
    print(f"Processed sessions found: {total_sessions}")
    print(f"Suitable sessions: {len(suitable_df)}")
    print(f"Not suitable sessions: {len(unsuitable_df)}")

    print("\nSuitable session names:")
    if suitable_names:
        print(", ".join(suitable_names))
    else:
        print("None")

    print("\nNot suitable sessions:")
    if unsuitable_df.empty:
        print("None")
    else:
        for _, row in unsuitable_df.iterrows():
            missing_fields = find_missing_required_fields(row.to_dict())
            session_name = row.get("session_name", "unknown_session")
            print(f"{session_name} -> missing: {', '.join(missing_fields)}")

    print()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a master session summary from processed session_summary.json files."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Path to the processed sessions directory (default: dataset/processed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    processed_dir = args.processed_dir.expanduser().resolve()

    if not processed_dir.exists():
        print(f"Warning: processed directory does not exist: {processed_dir}")
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        print_terminal_summary(empty_df)
        return

    json_files = find_session_summary_files(processed_dir)
    summaries = []

    for json_path in json_files:
        summary = load_session_summary(json_path)
        if summary is not None:
            summaries.append(summary)

    dataframe = build_master_dataframe(summaries)
    csv_path, json_path = write_outputs(dataframe, processed_dir)

    print_terminal_summary(dataframe)
    print(f"CSV written to: {csv_path}")
    print(f"JSON written to: {json_path}")


if __name__ == "__main__":
    main()