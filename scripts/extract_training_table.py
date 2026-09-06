"""
STEP 1 OF THE DATA PIPELINE - turn one raw recording into a table of numbers.

    cd /workspaces/Research_Project
    python3 scripts/extract_training_table.py --session dataset/1
    python3 scripts/extract_training_table.py --session dataset/3

============================================================================
IF YOU HAVE NEVER SEEN THIS PROJECT BEFORE, READ THIS FIRST
============================================================================
The PLUS-HRI dataset is a set of recordings of a human driving a TIAGo robot
around a room where people are standing and talking. Each recording ("session")
is a folder containing several very different kinds of file:

    1.bag                     robot sensors, in ROS 1 bag format
    1.mp4                     what the robot's camera saw
    cmd_vel.csv               the joystick commands the human gave
    detected_people*.csv      where people are in each video frame
    1_frame_manifest.json     the timestamp of every video frame

A machine-learning model cannot read any of that. It needs one flat table:
one row per moment in time, every column a number. Producing that table is
this script's entire job.

============================================================================
WHAT COMES OUT, AND WHY EACH PIECE MATTERS
============================================================================
One CSV per session with a row roughly every 30th of a second:

    timestamp              when this moment happened
    robot_x, robot_y       where the robot was, in metres
    robot_yaw              which way it was facing, in radians
    linear_x, angular_z    how fast it was driving and turning
    lidar_min_range        distance to the NEAREST obstacle
    lidar_mean_range       average distance all round - a "how open is this
                           space" measure
    num_faces, face_*      what the camera saw

Two of these carry more weight than the rest:

  * robot_x / robot_y / robot_yaw is where the TRAINING TARGET comes from.
    The thing the model learns to predict is "where did the human stop", and
    the only record of that is the robot's own odometry. No bag, no answer key.

  * lidar_min_range is used later as a stand-in for "how far away is the
    group". It is not a true group distance - it is the nearest obstacle in
    any direction - and that approximation is documented as a limitation
    throughout the project, because the video is uncalibrated and no real
    metric distance is recoverable from it.

============================================================================
THE AWKWARD PARTS, AND WHY THE CODE LOOKS THE WAY IT DOES
============================================================================
Three problems make this longer than you would expect:

  1. ROS 1 BAGS ON A ROS 2 MACHINE. The recordings are ROS 1 format; this
     project runs ROS 2 Humble. Rather than installing ROS 1, the `rosbags`
     library reads the old format directly, using the ROS1_NOETIC typestore so
     the message definitions are interpreted correctly.

  2. EVERY SOURCE HAS ITS OWN CLOCK RATE. The bag, the video and the CSVs are
     each sampled at different rates and none of their timestamps line up.
     Rows are therefore merged by NEAREST timestamp within a tolerance, not by
     exact match, which is what merge_asof does below.

  3. TIMESTAMPS COME IN TWO UNITS. Some are seconds, some nanoseconds. They
     are normalised on the way in - mixing them silently produces times
     millions of seconds apart and merges that match nothing.

Orientation arrives as a quaternion (four numbers describing a 3-D rotation)
and is converted to a single yaw angle, since the robot only turns in the
plane.

============================================================================
WHERE THIS SITS IN THE PIPELINE
============================================================================
    extract_training_table.py     <- YOU ARE HERE, run once per session
    check_dataset_readiness.py       merges all sessions into one table
    build_approach_pose_dataset.py   labels which rows are approaches
    split_dataset.py                 splits into train / val / test
    train_approach_pose_model.py     trains the model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


POSE_TOPICS = [
    "/robot_pose",
    "/mobile_base_controller/odom",
    "/dlo_node/odom",
]

LIDAR_TOPIC = "/scan"

POSE_TOLERANCE_SEC = 0.10
# 0.6s, not 0.1s: sessions 1/3 have continuous per-frame ground truth, but
# the other 22 sessions' detected_people.csv is sampled ~once per second
# (see extract_person_detections.py). 0.1s tolerance meant most cmd_vel rows
# fell just outside the nearest detection and got dropped to NaN (~80-88%
# missing, confirmed via check_dataset_readiness.py on 4 Aug 2026). 0.6s
# comfortably covers half the ~1s sampling gap either side, without changing
# which match is picked (merge_asof always picks the nearest one - the
# tolerance only decides whether it's close enough to accept).
FACE_TOLERANCE_SEC = 0.60
LIDAR_TOLERANCE_SEC = 0.15


def find_bag_file(session_path: Path) -> Path | None:
    bag_files = sorted(session_path.glob("*.bag"))
    return bag_files[0] if bag_files else None


def find_first_match(session_path: Path, pattern: str) -> Path | None:
    matches = sorted(session_path.glob(pattern))
    return matches[0] if matches else None


def normalise_timestamp(value: Any) -> float:
    """
    Convert timestamp to seconds if needed.

    If the value looks like nanoseconds, divide by 1e9.
    """
    value = float(value)

    if abs(value) > 1e12:
        return value / 1e9

    return value


def normalise_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.apply(lambda value: normalise_timestamp(value) if pd.notna(value) else np.nan)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Convert quaternion orientation into yaw angle in radians.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def read_cmd_vel_csv(session_path: Path) -> pd.DataFrame:
    csv_path = session_path / "cmd_vel.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Required file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["timestamp", "msg.linear.x", "msg.angular.z"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"cmd_vel.csv is missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    df["timestamp"] = normalise_timestamp_series(df["timestamp"])
    df = df.rename(
        columns={
            "msg.linear.x": "linear_x",
            "msg.angular.z": "angular_z",
        }
    )

    df = df.dropna(subset=["timestamp", "linear_x", "angular_z"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def select_pose_topic(reader: Reader) -> str | None:
    available_topics = {connection.topic for connection in reader.connections}

    for topic_name in POSE_TOPICS:
        if topic_name in available_topics:
            return topic_name

    return None


def read_robot_pose_from_bag(bag_path: Path) -> tuple[pd.DataFrame, str | None]:
    """
    Read robot pose from the first available pose topic.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows: list[dict[str, Any]] = []

    with Reader(bag_path) as reader:
        pose_topic = select_pose_topic(reader)
        if pose_topic is None:
            return pd.DataFrame(columns=["timestamp", "robot_x", "robot_y", "robot_yaw"]), None

        pose_connections = [conn for conn in reader.connections if conn.topic == pose_topic]

        for connection, timestamp, rawdata in reader.messages(connections=pose_connections):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

            if pose_topic == "/robot_pose":
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation
            else:
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation

            rows.append(
                {
                    "timestamp": timestamp / 1e9,
                    "robot_x": float(position.x),
                    "robot_y": float(position.y),
                    "robot_yaw": float(
                        quaternion_to_yaw(
                            float(orientation.x),
                            float(orientation.y),
                            float(orientation.z),
                            float(orientation.w),
                        )
                    ),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "robot_x", "robot_y", "robot_yaw"]), pose_topic

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, pose_topic


def read_lidar_from_bag(bag_path: Path) -> pd.DataFrame:
    """
    Read /scan and summarise each LaserScan into min and mean range.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows: list[dict[str, Any]] = []

    with Reader(bag_path) as reader:
        lidar_connections = [conn for conn in reader.connections if conn.topic == LIDAR_TOPIC]
        if not lidar_connections:
            return pd.DataFrame(columns=["timestamp", "lidar_min_range", "lidar_mean_range"])

        for connection, timestamp, rawdata in reader.messages(connections=lidar_connections):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

            valid_ranges = [
                float(value)
                for value in msg.ranges
                if value is not None and math.isfinite(value) and float(value) > 0.0
            ]

            if valid_ranges:
                lidar_min = float(np.min(valid_ranges))
                lidar_mean = float(np.mean(valid_ranges))
            else:
                lidar_min = np.nan
                lidar_mean = np.nan

            rows.append(
                {
                    "timestamp": timestamp / 1e9,
                    "lidar_min_range": lidar_min,
                    "lidar_mean_range": lidar_mean,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "lidar_min_range", "lidar_mean_range"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def read_frame_manifest(session_path: Path) -> dict[int, float]:
    """
    Build a frame_index -> timestamp(seconds) mapping if possible.
    """
    manifest_path = find_first_match(session_path, "*_frame_manifest.json")
    if manifest_path is None or not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    frame_map: dict[int, float] = {}

    if isinstance(payload, dict):
        if "timestamps" in payload and isinstance(payload["timestamps"], list):
            for index, timestamp in enumerate(payload["timestamps"]):
                try:
                    frame_map[index] = normalise_timestamp(timestamp)
                except Exception:
                    continue

        elif "frame_timestamps" in payload and isinstance(payload["frame_timestamps"], list):
            for index, timestamp in enumerate(payload["frame_timestamps"]):
                try:
                    frame_map[index] = normalise_timestamp(timestamp)
                except Exception:
                    continue

        elif "frames" in payload and isinstance(payload["frames"], list):
            for item in payload["frames"]:
                if not isinstance(item, dict):
                    continue

                frame_index = item.get("frame_index", item.get("index"))
                timestamp = item.get("timestamp", item.get("time"))

                if frame_index is None or timestamp is None:
                    continue

                try:
                    frame_map[int(frame_index)] = normalise_timestamp(timestamp)
                except Exception:
                    continue

    elif isinstance(payload, list):
        for index, timestamp in enumerate(payload):
            try:
                frame_map[index] = normalise_timestamp(timestamp)
            except Exception:
                continue

    return frame_map


def choose_face_xy_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if "x" in df.columns and "y" in df.columns:
        return "x", "y"

    if "position_x" in df.columns and "position_y" in df.columns:
        return "position_x", "position_y"

    return None, None


def summarise_face_group(group: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    """
    Summarise all face landmarks for one timestamp.
    """
    valid_points = group[[x_col, y_col]].dropna()

    if valid_points.empty:
        return {
            "num_faces": 0,
            "face_center_x": np.nan,
            "face_center_y": np.nan,
            "face_bbox_min_x": np.nan,
            "face_bbox_min_y": np.nan,
            "face_bbox_max_x": np.nan,
            "face_bbox_max_y": np.nan,
        }

    if "face_index" in group.columns:
        per_face = (
            group.dropna(subset=[x_col, y_col])
            .groupby("face_index")
            .agg(face_center_x=(x_col, "mean"), face_center_y=(y_col, "mean"))
            .reset_index()
        )
        num_faces = int(per_face["face_index"].nunique()) if not per_face.empty else 0
        face_center_x = float(per_face["face_center_x"].mean()) if not per_face.empty else np.nan
        face_center_y = float(per_face["face_center_y"].mean()) if not per_face.empty else np.nan
    else:
        num_faces = 1
        face_center_x = float(valid_points[x_col].mean())
        face_center_y = float(valid_points[y_col].mean())

    return {
        "num_faces": num_faces,
        "face_center_x": face_center_x,
        "face_center_y": face_center_y,
        "face_bbox_min_x": float(valid_points[x_col].min()),
        "face_bbox_min_y": float(valid_points[y_col].min()),
        "face_bbox_max_x": float(valid_points[x_col].max()),
        "face_bbox_max_y": float(valid_points[y_col].max()),
    }


def read_face_landmarks(session_path: Path) -> tuple[pd.DataFrame, int]:
    """
    Read facial_landmarks_uniface.csv and create one row per timestamp.

    Fallback: if this session has no facial_landmarks_uniface.csv (true for
    every session except 1 and 3), use detected_people.csv instead, produced
    by extract_person_detections.py. It's already written in the exact same
    per-timestamp schema (timestamp, num_faces, face_center_x/y,
    face_bbox_min/max_x/y), so it can be used directly with no reshaping.
    """
    csv_path = session_path / "facial_landmarks_uniface.csv"
    detected_path = session_path / "detected_people.csv"

    if not csv_path.exists():
        if detected_path.exists():
            df = pd.read_csv(detected_path)
            df["timestamp"] = normalise_timestamp_series(df["timestamp"])
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return df, len(df)

        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "num_faces",
                "face_center_x",
                "face_center_y",
                "face_bbox_min_x",
                "face_bbox_min_y",
                "face_bbox_max_x",
                "face_bbox_max_y",
            ]
        )
        return empty, 0

    df = pd.read_csv(csv_path)
    raw_row_count = len(df)

    x_col, y_col = choose_face_xy_columns(df)
    if x_col is None or y_col is None:
        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "num_faces",
                "face_center_x",
                "face_center_y",
                "face_bbox_min_x",
                "face_bbox_min_y",
                "face_bbox_max_x",
                "face_bbox_max_y",
            ]
        )
        return empty, raw_row_count

    if "timestamp" in df.columns:
        df["timestamp"] = normalise_timestamp_series(df["timestamp"])
    else:
        frame_map = read_frame_manifest(session_path)

        if "frame_index" not in df.columns or not frame_map:
            empty = pd.DataFrame(
                columns=[
                    "timestamp",
                    "num_faces",
                    "face_center_x",
                    "face_center_y",
                    "face_bbox_min_x",
                    "face_bbox_min_y",
                    "face_bbox_max_x",
                    "face_bbox_max_y",
                ]
            )
            return empty, raw_row_count

        df["timestamp"] = pd.to_numeric(df["frame_index"], errors="coerce").map(frame_map)

    df = df.dropna(subset=["timestamp"])

    if df.empty:
        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "num_faces",
                "face_center_x",
                "face_center_y",
                "face_bbox_min_x",
                "face_bbox_min_y",
                "face_bbox_max_x",
                "face_bbox_max_y",
            ]
        )
        return empty, raw_row_count

    summary_rows = []

    for timestamp_value, group in df.groupby("timestamp"):
        summary = summarise_face_group(group, x_col, y_col)
        summary["timestamp"] = float(timestamp_value)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("timestamp").reset_index(drop=True)

    return summary_df, raw_row_count


def build_training_table(
    cmd_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    face_df: pd.DataFrame,
    lidar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one synchronised training table using cmd_vel timestamps as the base timeline.
    """
    if cmd_df.empty:
        raise ValueError("cmd_vel data is empty after preprocessing.")

    if pose_df.empty:
        raise ValueError("Robot pose data is missing or empty. Cannot build training table.")

    cmd_df = cmd_df.sort_values("timestamp").reset_index(drop=True)
    pose_df = pose_df.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        cmd_df,
        pose_df,
        on="timestamp",
        direction="nearest",
        tolerance=POSE_TOLERANCE_SEC,
    )

    if not face_df.empty:
        face_df = face_df.sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            merged,
            face_df,
            on="timestamp",
            direction="nearest",
            tolerance=FACE_TOLERANCE_SEC,
        )
    else:
        for column in [
            "num_faces",
            "face_center_x",
            "face_center_y",
            "face_bbox_min_x",
            "face_bbox_min_y",
            "face_bbox_max_x",
            "face_bbox_max_y",
        ]:
            merged[column] = np.nan

    if not lidar_df.empty:
        lidar_df = lidar_df.sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            merged,
            lidar_df,
            on="timestamp",
            direction="nearest",
            tolerance=LIDAR_TOLERANCE_SEC,
        )
    else:
        merged["lidar_min_range"] = np.nan
        merged["lidar_mean_range"] = np.nan

    merged = merged.dropna(subset=["robot_x", "robot_y", "robot_yaw", "linear_x", "angular_z"])

    if "num_faces" in merged.columns:
        merged["num_faces"] = merged["num_faces"].fillna(0).astype(int)
    else:
        merged["num_faces"] = 0

    ordered_columns = [
        "timestamp",
        "robot_x",
        "robot_y",
        "robot_yaw",
        "linear_x",
        "angular_z",
        "num_faces",
        "face_center_x",
        "face_center_y",
        "face_bbox_min_x",
        "face_bbox_min_y",
        "face_bbox_max_x",
        "face_bbox_max_y",
        "lidar_min_range",
        "lidar_mean_range",
    ]

    for column in ordered_columns:
        if column not in merged.columns:
            merged[column] = np.nan

    merged = merged[ordered_columns].sort_values("timestamp").reset_index(drop=True)
    return merged


def write_outputs(training_df: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    training_csv_path = output_dir / "training_table.csv"
    summary_json_path = output_dir / "training_table_summary.json"

    training_df.to_csv(training_csv_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def build_summary(
    session_name: str,
    session_path: Path,
    bag_path: Path,
    pose_topic_used: str | None,
    training_df: pd.DataFrame,
    face_row_count: int,
    lidar_row_count: int,
    has_face_landmarks: bool,
    has_lidar: bool,
) -> dict[str, Any]:
    if training_df.empty:
        time_start = None
        time_end = None
        duration_sec = 0.0
    else:
        time_start = float(training_df["timestamp"].min())
        time_end = float(training_df["timestamp"].max())
        duration_sec = float(time_end - time_start)

    notes_parts = [
        "Base timeline uses cmd_vel timestamps.",
        "Pose, face landmarks, and LiDAR are matched using nearest-neighbour timestamp alignment.",
    ]

    if not has_face_landmarks:
        notes_parts.append("Face landmarks were missing or could not be aligned, so face columns may contain NaN and num_faces defaults to 0.")

    if not has_lidar:
        notes_parts.append("LiDAR was missing, so LiDAR summary columns contain NaN.")

    if pose_topic_used is None:
        notes_parts.append("No pose topic was found in the expected priority list.")

    return {
        "session_name": session_name,
        "session_path": str(session_path.resolve()),
        "bag_file": str(bag_path.resolve()),
        "pose_topic_used": pose_topic_used,
        "row_count": int(len(training_df)),
        "time_start": time_start,
        "time_end": time_end,
        "duration_sec": round(duration_sec, 6),
        "has_robot_pose": pose_topic_used is not None,
        "has_cmd_vel": True,
        "has_face_landmarks": has_face_landmarks,
        "has_lidar": has_lidar,
        "columns": training_df.columns.tolist(),
        "face_rows_found": int(face_row_count),
        "lidar_rows_found": int(lidar_row_count),
        "notes": " ".join(notes_parts),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a training-ready Behavioural Cloning table from one TIAGo dataset session."
    )
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to one dataset session folder, for example dataset/1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    session_path = args.session.expanduser().resolve()

    if not session_path.exists():
        raise FileNotFoundError(f"Session folder does not exist: {session_path}")

    if not session_path.is_dir():
        raise NotADirectoryError(f"Session path is not a directory: {session_path}")

    session_name = session_path.name
    bag_path = find_bag_file(session_path)

    if bag_path is None:
        raise FileNotFoundError(f"No .bag file found in session folder: {session_path}")

    cmd_df = read_cmd_vel_csv(session_path)
    pose_df, pose_topic_used = read_robot_pose_from_bag(bag_path)
    lidar_df = read_lidar_from_bag(bag_path)
    face_df, face_row_count = read_face_landmarks(session_path)

    training_df = build_training_table(cmd_df, pose_df, face_df, lidar_df)

    output_dir = session_path.parent / "processed" / f"session_{session_name}"

    summary = build_summary(
        session_name=session_name,
        session_path=session_path,
        bag_path=bag_path,
        pose_topic_used=pose_topic_used,
        training_df=training_df,
        face_row_count=face_row_count,
        lidar_row_count=len(lidar_df),
        has_face_landmarks=not face_df.empty,
        has_lidar=not lidar_df.empty,
    )

    write_outputs(training_df, summary, output_dir)

    print(f"\nTraining table created for session {session_name}\n")
    print(f"Rows: {len(training_df)}")
    print(f"Pose topic used: {pose_topic_used}")
    print(f"Face rows found: {face_row_count}")
    print(f"LiDAR rows found: {len(lidar_df)}")
    print("\nOutput:")
    print(output_dir / "training_table.csv")
    print(output_dir / "training_table_summary.json")
    print()


if __name__ == "__main__":
    main() 

 