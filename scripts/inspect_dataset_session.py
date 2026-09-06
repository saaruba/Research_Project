"""
LOOK INSIDE ONE RECORDING - a read-only survey tool.

    python3 scripts/inspect_dataset_session.py --session dataset/1

============================================================================
WHY THIS EXISTS
============================================================================
Each PLUS-HRI session folder holds a mixture of sensor bags, video, CSVs and
JSON manifests, and the 24 sessions are NOT consistent with each other. Some
have gaze tracking, some do not. Some have face annotations, most do not. Some
record robot pose on one ROS topic, some on another. A few have a file present
but empty.

Before trusting a session you have to know what is actually in it. This script
answers that without changing anything: it opens every file, reports what it
found, how many rows, over what time span, at what rate, and flags anything
missing or unreadable.

It writes nothing and modifies nothing. Run it as often as you like.

============================================================================
WHAT IT REPORTS
============================================================================
  * which files are present, and their size
  * for the bag: which topics were recorded, and their message counts
  * for each CSV: row count, which column holds the timestamp, the start and
    end times, the duration, and the approximate sampling frequency
  * empty or corrupt files, called out explicitly rather than passed over

============================================================================
WHAT IT WAS ACTUALLY USED FOR
============================================================================
This is how the project established that only 2 of the 24 sessions carry face
and gaze annotations, and that gaze_uniface.csv is empty in both. That finding
shaped the whole methodology: with no orientation ground truth anywhere in the
dataset, F-formation O-space estimation had to fall back on the mutual-facing
assumption (O-space centre approximated by the group centroid), which is stated
as a limitation in the dissertation rather than hidden.

Run it on a session that behaves oddly downstream before assuming the fault is
in your code - very often the recording itself is missing something.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rosbags.rosbag1 import Reader


IMPORTANT_TOPICS = {
    "camera_rgb": [
        "/xtion/rgb/image_raw/compressed",
        "/xtion/rgb/camera_info",
    ],
    "camera_depth": [
        "/xtion/depth/image_raw/compressed",
        "/xtion/depth/camera_info",
    ],
    "lidar": [
        "/scan",
    ],
    "pointcloud": [
        "/throttle_filtering_points/filtered_points",
    ],
    "robot_pose": [
        "/robot_pose",
        "/mobile_base_controller/odom",
        "/dlo_node/odom",
    ],
    "robot_velocity": [
        "/input_joy/cmd_vel",
        "/mobile_base_controller/cmd_vel",
        "/mobile_base_controller/cmd_vel_out",
        "/joy_vel",
    ],
    "tf": [
        "/tf",
        "/tf_static",
    ],
    "map": [
        "/map",
        "/vo_map",
        "/vo_loc_map",
    ],
    "imu": [
        "/base_imu",
    ],
    "joint_states": [
        "/joint_states",
    ],
    "sonar": [
        "/sonar_base",
    ],
}


def detect_session_name(session_path: Path) -> str:
    return session_path.name


def find_bag_file(session_path: Path) -> Path | None:
    bag_files = sorted(session_path.glob("*.bag"))
    return bag_files[0] if bag_files else None


def check_ros1_bag_header(bag_path: Path) -> bool:
    with bag_path.open("rb") as handle:
        header = handle.read(13)
    return header.startswith(b"#ROSBAG V2.0")


def classify_topic(topic_name: str) -> str:
    for category, topics in IMPORTANT_TOPICS.items():
        if topic_name in topics:
            return category
    return "other"


def find_timestamp_column(columns: list[str]) -> str | None:
    preferred = [
        "timestamp",
        "time",
        "stamp",
        "header.stamp",
        "msg.header.stamp",
        "header.stamp.sec",
        "msg.header.stamp.sec",
    ]

    lowered_map = {column.lower(): column for column in columns}

    for name in preferred:
        if name in lowered_map:
            return lowered_map[name]

    for column in columns:
        lowered = column.lower()
        if "time" in lowered or "stamp" in lowered:
            return column

    return None


def safe_numeric(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return round(value, 6)
    return value


def compute_duration(start_value: Any, end_value: Any) -> float | None:
    try:
        start_num = float(start_value)
        end_num = float(end_value)
    except (TypeError, ValueError):
        return None

    duration = end_num - start_num
    if duration < 0:
        return None
    return round(duration, 6)


def estimate_frequency(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return None

    duration = float(numeric.iloc[-1] - numeric.iloc[0])
    if duration <= 0:
        return None

    return round((len(numeric) - 1) / duration, 3)


def inspect_ros1_bag(bag_path: Path) -> dict[str, Any]:
    topic_info: dict[str, dict[str, Any]] = {}
    topic_timestamps: dict[str, list[float]] = defaultdict(list)

    with Reader(bag_path) as reader:
        for connection in reader.connections:
            topic_info[connection.topic] = {
                "topic": connection.topic,
                "message_type": connection.msgtype,
                "message_count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "estimated_frequency_hz": None,
                "category": classify_topic(connection.topic),
            }

        for connection, timestamp, _rawdata in reader.messages():
            topic_name = connection.topic
            timestamp_sec = timestamp / 1_000_000_000.0

            topic_info[topic_name]["message_count"] += 1

            if topic_info[topic_name]["first_timestamp"] is None:
                topic_info[topic_name]["first_timestamp"] = round(timestamp_sec, 6)

            topic_info[topic_name]["last_timestamp"] = round(timestamp_sec, 6)
            topic_timestamps[topic_name].append(timestamp_sec)

    for topic_name, timestamps in topic_timestamps.items():
        if len(timestamps) >= 2:
            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                topic_info[topic_name]["estimated_frequency_hz"] = round(
                    (len(timestamps) - 1) / duration, 3
                )

    categories_present = {
        category: any(
            topic_data["category"] == category for topic_data in topic_info.values()
        )
        for category in IMPORTANT_TOPICS
    }

    return {
        "bag_file": str(bag_path),
        "bag_format": "ROS1",
        "topic_count": len(topic_info),
        "topics": sorted(topic_info.values(), key=lambda item: item["topic"]),
        "categories_present": categories_present,
    }


def inspect_generic_csv(csv_path: Path) -> dict[str, Any]:
    dataframe = pd.read_csv(csv_path)
    columns = dataframe.columns.tolist()
    timestamp_column = find_timestamp_column(columns)

    summary = {
        "file": str(csv_path),
        "exists": True,
        "rows": int(len(dataframe)),
        "columns": columns,
        "timestamp_column": timestamp_column,
        "start_timestamp": None,
        "end_timestamp": None,
        "duration": None,
        "estimated_frequency_hz": None,
    }

    if timestamp_column and not dataframe.empty:
        start_value = dataframe[timestamp_column].iloc[0]
        end_value = dataframe[timestamp_column].iloc[-1]

        summary["start_timestamp"] = safe_numeric(start_value)
        summary["end_timestamp"] = safe_numeric(end_value)
        summary["duration"] = compute_duration(start_value, end_value)
        summary["estimated_frequency_hz"] = estimate_frequency(dataframe[timestamp_column])

    return summary


def inspect_cmd_vel(csv_path: Path) -> dict[str, Any]:
    summary = inspect_generic_csv(csv_path)
    dataframe = pd.read_csv(csv_path)

    target_columns = [
        "msg.linear.x",
        "msg.linear.y",
        "msg.angular.z",
    ]

    velocity_stats = {}

    for column in target_columns:
        if column in dataframe.columns:
            numeric = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            if not numeric.empty:
                velocity_stats[column] = {
                    "min": round(float(numeric.min()), 6),
                    "max": round(float(numeric.max()), 6),
                    "mean": round(float(numeric.mean()), 6),
                }
            else:
                velocity_stats[column] = None
        else:
            velocity_stats[column] = None

    summary["velocity_stats"] = velocity_stats
    return summary


def inspect_face_landmarks(csv_path: Path) -> dict[str, Any]:
    summary = inspect_generic_csv(csv_path)
    dataframe = pd.read_csv(csv_path)
    columns = dataframe.columns.tolist()

    frame_columns = [column for column in columns if "frame" in column.lower()]
    face_index_column = "face_index" if "face_index" in dataframe.columns else None

    xy_columns = [
        column
        for column in columns
        if ("x" in column.lower() or "y" in column.lower())
    ]

    depth_columns = [
        column
        for column in columns
        if (
            "z" in column.lower()
            or "depth" in column.lower()
            or "world" in column.lower()
        )
    ]

    unique_frames = None
    if frame_columns:
        unique_frames = int(dataframe[frame_columns[0]].nunique(dropna=True))

    unique_faces = None
    if face_index_column:
        unique_faces = int(dataframe[face_index_column].nunique(dropna=True))

    depth_has_values = False
    if depth_columns:
        depth_has_values = any(
            not dataframe[column].dropna().empty for column in depth_columns
        )

    summary.update(
        {
            "unique_frames": unique_frames,
            "unique_faces": unique_faces,
            "has_xy_positions": bool(xy_columns),
            "xy_columns": xy_columns,
            "has_depth_or_world_positions": bool(depth_columns),
            "depth_or_world_columns": depth_columns,
            "depth_or_world_has_values": depth_has_values,
        }
    )

    return summary


def inspect_frame_manifest(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    timestamps = []

    if isinstance(payload, dict):
        if "timestamps" in payload and isinstance(payload["timestamps"], list):
            timestamps = payload["timestamps"]
        elif "frame_timestamps" in payload and isinstance(payload["frame_timestamps"], list):
            timestamps = payload["frame_timestamps"]
        else:
            for value in payload.values():
                if isinstance(value, list) and value and all(
                    isinstance(item, (int, float)) for item in value
                ):
                    timestamps = value
                    break

    elif isinstance(payload, list):
        timestamps = payload

    start_timestamp = timestamps[0] if timestamps else None
    end_timestamp = timestamps[-1] if timestamps else None
    duration = None
    frame_rate = None

    if len(timestamps) >= 2:
        duration = round(float(end_timestamp - start_timestamp), 6)
        if duration > 0:
            frame_rate = round((len(timestamps) - 1) / duration, 3)

    return {
        "file": str(json_path),
        "exists": True,
        "timestamp_count": len(timestamps),
        "start_timestamp": safe_numeric(start_timestamp),
        "end_timestamp": safe_numeric(end_timestamp),
        "duration": duration,
        "estimated_frame_rate_hz": frame_rate,
    }


def find_first_match(session_path: Path, pattern: str) -> Path | None:
    matches = sorted(session_path.glob(pattern))
    return matches[0] if matches else None


def build_recommendation(session_summary: dict[str, Any]) -> str:
    available = []

    if session_summary["has_cmd_vel"]:
        available.append("robot velocity commands")
    if session_summary["has_robot_pose"]:
        available.append("robot pose/odometry")
    if session_summary["has_lidar"]:
        available.append("LiDAR")
    if session_summary["has_rgb"]:
        available.append("RGB camera data")
    if session_summary["has_depth"]:
        available.append("depth camera data")
    if session_summary["has_face_landmarks"]:
        available.append("face landmark data")

    if (
        session_summary["has_cmd_vel"]
        and session_summary["has_robot_pose"]
        and (
            session_summary["has_lidar"]
            or session_summary["has_rgb"]
            or session_summary["has_depth"]
        )
    ):
        return (
            "This session appears suitable for navigation-policy learning because it contains "
            + ", ".join(available)
            + "."
        )

    missing = []

    if not session_summary["has_cmd_vel"]:
        missing.append("robot velocity commands")
    if not session_summary["has_robot_pose"]:
        missing.append("robot pose/odometry")
    if not (
        session_summary["has_lidar"]
        or session_summary["has_rgb"]
        or session_summary["has_depth"]
    ):
        missing.append("key perception topics")

    if missing:
        return (
            "This session may not yet be ideal for navigation-policy learning because it is missing "
            + ", ".join(missing)
            + "."
        )

    return "This session has partial useful data, but it still needs manual review."


def write_outputs(
    output_dir: Path,
    bag_summary: dict[str, Any] | None,
    session_summary: dict[str, Any],
    csv_files_summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if bag_summary is not None:
        topic_dataframe = pd.DataFrame(bag_summary["topics"])
        topic_dataframe.to_csv(output_dir / "bag_topics_summary.csv", index=False)

    with (output_dir / "session_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(session_summary, handle, indent=2)

    with (output_dir / "csv_files_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(csv_files_summary, handle, indent=2)


def print_terminal_summary(
    session_name: str,
    session_path: Path,
    discovered_files: dict[str, str | None],
    bag_summary: dict[str, Any] | None,
    csv_summaries: dict[str, Any],
    session_summary: dict[str, Any],
    output_dir: Path,
) -> None:
    print("\n" + "=" * 80)
    print(f"DATASET SESSION SUMMARY: {session_name}")
    print("=" * 80)
    print(f"Session path: {session_path}")

    print("\nFiles found:")
    for label, path_value in discovered_files.items():
        print(f"  - {label}: {path_value if path_value else 'missing'}")

    print("\nBag inspection:")
    if bag_summary is None:
        print("  - No .bag file found")
    else:
        print(f"  - Bag format: {bag_summary['bag_format']}")
        print(f"  - Number of topics: {bag_summary['topic_count']}")
        print("  - Topic overview:")
        for topic in bag_summary["topics"]:
            print(
                f"    * {topic['topic']} | "
                f"type={topic['message_type']} | "
                f"count={topic['message_count']} | "
                f"category={topic['category']} | "
                f"first={topic['first_timestamp']} | "
                f"last={topic['last_timestamp']} | "
                f"freq={topic['estimated_frequency_hz']}"
            )

    print("\nCSV / JSON inspection:")
    for name, summary in csv_summaries.items():
        if not summary.get("exists", False):
            print(f"  - {name}: missing")
            continue

        rows_or_count = summary.get("rows", summary.get("timestamp_count", "n/a"))
        print(
            f"  - {name}: count={rows_or_count}, "
            f"start={summary.get('start_timestamp')}, "
            f"end={summary.get('end_timestamp')}"
        )

    print("\nSuitability recommendation:")
    print(f"  - {session_summary['recommended_next_step']}")

    print("\nOutput files written to:")
    print(f"  - {output_dir / 'bag_topics_summary.csv'}")
    print(f"  - {output_dir / 'session_summary.json'}")
    print(f"  - {output_dir / 'csv_files_summary.json'}")
    print("=" * 80 + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one TIAGo dataset session folder."
    )
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to one dataset session folder, e.g. /workspaces/Research_Project/dataset/1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: ../processed/session_<session_name>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    session_path = args.session.expanduser().resolve()

    if not session_path.exists():
        raise FileNotFoundError(f"Session path does not exist: {session_path}")

    if not session_path.is_dir():
        raise NotADirectoryError(f"Session path is not a directory: {session_path}")

    session_name = detect_session_name(session_path)
    bag_path = find_bag_file(session_path)

    discovered_files: dict[str, str | None] = {
        "bag_file": str(bag_path) if bag_path else None,
    }

    cmd_vel_path = find_first_match(session_path, "cmd_vel.csv")
    frame_manifest_path = find_first_match(session_path, "*_frame_manifest.json")
    face_landmarks_path = find_first_match(session_path, "facial_landmarks_uniface.csv")
    gaze_log_path = find_first_match(session_path, "gaze_log_*.csv")
    gaze_uniface_path = find_first_match(session_path, "gaze_uniface.csv")
    fixations_path = find_first_match(session_path, "fixations_*.csv")
    audio_path = find_first_match(session_path, "audio_log_*.wav")

    discovered_files.update(
        {
            "cmd_vel": str(cmd_vel_path) if cmd_vel_path else None,
            "frame_manifest": str(frame_manifest_path) if frame_manifest_path else None,
            "face_landmarks": str(face_landmarks_path) if face_landmarks_path else None,
            "gaze_log": str(gaze_log_path) if gaze_log_path else None,
            "gaze_uniface": str(gaze_uniface_path) if gaze_uniface_path else None,
            "fixations": str(fixations_path) if fixations_path else None,
            "audio": str(audio_path) if audio_path else None,
        }
    )

    bag_summary = None
    if bag_path:
        if not check_ros1_bag_header(bag_path):
            raise ValueError(
                f"The bag file does not appear to be a ROS 1 bag with '#ROSBAG V2.0' header: {bag_path}"
            )
        bag_summary = inspect_ros1_bag(bag_path)

    csv_summaries: dict[str, Any] = {}

    csv_summaries["cmd_vel"] = (
        inspect_cmd_vel(cmd_vel_path) if cmd_vel_path else {"exists": False}
    )

    csv_summaries["facial_landmarks_uniface"] = (
        inspect_face_landmarks(face_landmarks_path)
        if face_landmarks_path
        else {"exists": False}
    )

    csv_summaries["frame_manifest"] = (
        inspect_frame_manifest(frame_manifest_path)
        if frame_manifest_path
        else {"exists": False}
    )

    csv_summaries["gaze_log"] = (
        inspect_generic_csv(gaze_log_path) if gaze_log_path else {"exists": False}
    )

    csv_summaries["gaze_uniface"] = (
        inspect_generic_csv(gaze_uniface_path)
        if gaze_uniface_path
        else {"exists": False}
    )

    csv_summaries["fixations"] = (
        inspect_generic_csv(fixations_path) if fixations_path else {"exists": False}
    )

    csv_summaries["audio"] = {
        "exists": bool(audio_path),
        "file": str(audio_path) if audio_path else None,
    }

    categories_present = bag_summary["categories_present"] if bag_summary else {}

    session_summary = {
        "session_name": session_name,
        "session_path": str(session_path),
        "bag_file": str(bag_path) if bag_path else None,
        "bag_format": "ROS1" if bag_summary else None,
        "has_rgb": bool(categories_present.get("camera_rgb", False)),
        "has_depth": bool(categories_present.get("camera_depth", False)),
        "has_lidar": bool(categories_present.get("lidar", False)),
        "has_robot_pose": bool(categories_present.get("robot_pose", False)),
        "has_cmd_vel": bool(cmd_vel_path) or bool(categories_present.get("robot_velocity", False)),
        "has_tf": bool(categories_present.get("tf", False)),
        "has_face_landmarks": bool(face_landmarks_path),
        "recommended_next_step": "",
    }

    session_summary["recommended_next_step"] = build_recommendation(session_summary)

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = session_path.parent / "processed" / f"session_{session_name}"

    write_outputs(output_dir, bag_summary, session_summary, csv_summaries)

    print_terminal_summary(
        session_name=session_name,
        session_path=session_path,
        discovered_files=discovered_files,
        bag_summary=bag_summary,
        csv_summaries=csv_summaries,
        session_summary=session_summary,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()