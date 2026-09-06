"""
Run YOLOv8n person detection on sampled frames from a session's video, and
write detected_people.csv in the same format as the facial-landmark summary
(timestamp, num_faces, face_center_x/y, face_bbox_min/max_x/y), so
extract_training_table.py can use it as a drop-in fallback for sessions that
have no facial_landmarks_uniface.csv (i.e. everything except sessions 1 & 3).

Sampling: roughly 1 detection per second of video (using the real frame
timestamps from the session's *_frame_manifest.json), not every single
frame - this keeps runtime reasonable across all 24 sessions. A person's
position doesn't meaningfully change within a fraction of a second, and the
downstream merge_asof step already tolerates small timestamp gaps.

Setup (once):
    pip install --upgrade matplotlib   # if not already done
    pip install ultralytics

Usage - one session:
    python3 scripts/extract_person_detections.py --session dataset/9

Usage - all sessions without real face data (run this loop):
    for s in 5 7 8 9 10 11 12 14 15 26 27 28 30 31 49 51 52 54 55 58 59 60; do
        echo "=== session $s ==="
        python3 scripts/extract_person_detections.py --session dataset/$s
    done

Output:
    dataset/<session>/detected_people.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


def read_frame_manifest(session_path: Path) -> dict[int, float]:
    manifest_path = next(session_path.glob("*_frame_manifest.json"), None)
    if manifest_path is None:
        return {}

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frame_map: dict[int, float] = {}

    def normalise(value: float) -> float:
        return value / 1e9 if abs(value) > 1e12 else value

    if isinstance(payload, list):
        for index, timestamp in enumerate(payload):
            frame_map[index] = normalise(float(timestamp))
    elif isinstance(payload, dict):
        for key in ("timestamps", "frame_timestamps"):
            if key in payload and isinstance(payload[key], list):
                for index, timestamp in enumerate(payload[key]):
                    frame_map[index] = normalise(float(timestamp))
                break

    return frame_map


def find_video(session_path: Path) -> Path | None:
    # Prefer the top-level full-session video (matches the frame manifest)
    candidate = session_path / f"{session_path.name}.mp4"
    if candidate.exists():
        return candidate
    videos = sorted(session_path.glob("*.mp4"))
    return videos[0] if videos else None


def run_detection(session_path: Path, sample_every_sec: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
      - summary_rows: one row per sampled frame, same aggregated schema as
        before (num_faces, averaged face_center_x/y, bbox envelope) - this is
        what extract_training_table.py consumes for the BC feature tables.
      - individual_rows: one row per DETECTED PERSON per sampled frame
        (not averaged together) - needed for Phase C group clustering, since
        you can't tell "two people standing close together" apart from "two
        people on opposite sides of the room" once they've been averaged
        into a single point.
    """
    from ultralytics import YOLO

    frame_map = read_frame_manifest(session_path)
    if not frame_map:
        raise FileNotFoundError(f"No *_frame_manifest.json found in {session_path}")

    video_path = find_video(session_path)
    if video_path is None:
        raise FileNotFoundError(f"No .mp4 found in {session_path}")

    timestamps = sorted(frame_map.values())
    duration = timestamps[-1] - timestamps[0]
    fps_estimate = len(frame_map) / duration if duration > 0 else 15.0
    step = max(1, round(fps_estimate * sample_every_sec))
    print(f"  video: {video_path.name}, ~{fps_estimate:.1f} fps, sampling every {step} frames (~{sample_every_sec}s)")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    summary_rows: list[dict[str, Any]] = []
    individual_rows: list[dict[str, Any]] = []
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0 and frame_idx in frame_map:
            timestamp = frame_map[frame_idx]
            results = model(frame, classes=[0], verbose=False)  # class 0 = person
            boxes = results[0].boxes

            if len(boxes) == 0:
                summary_rows.append({
                    "timestamp": timestamp, "num_faces": 0,
                    "face_center_x": None, "face_center_y": None,
                    "face_bbox_min_x": None, "face_bbox_min_y": None,
                    "face_bbox_max_x": None, "face_bbox_max_y": None,
                })
            else:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
                centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2

                summary_rows.append({
                    "timestamp": timestamp, "num_faces": len(boxes),
                    "face_center_x": float(centers_x.mean()), "face_center_y": float(centers_y.mean()),
                    "face_bbox_min_x": float(xyxy[:, 0].min()), "face_bbox_min_y": float(xyxy[:, 1].min()),
                    "face_bbox_max_x": float(xyxy[:, 2].max()), "face_bbox_max_y": float(xyxy[:, 3].max()),
                })

                for person_idx in range(len(boxes)):
                    individual_rows.append({
                        "timestamp": timestamp,
                        "person_index": person_idx,
                        "center_x": float(centers_x[person_idx]),
                        "center_y": float(centers_y[person_idx]),
                        "bbox_min_x": float(xyxy[person_idx, 0]),
                        "bbox_min_y": float(xyxy[person_idx, 1]),
                        "bbox_max_x": float(xyxy[person_idx, 2]),
                        "bbox_max_y": float(xyxy[person_idx, 3]),
                        "confidence": float(confs[person_idx]),
                    })
            processed += 1

        frame_idx += 1

    cap.release()
    print(f"  processed {processed} sampled frames out of {frame_idx} total")
    return pd.DataFrame(summary_rows), pd.DataFrame(individual_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--sample-every-sec", type=float, default=1.0)
    args = parser.parse_args()

    session_path = args.session.expanduser().resolve()
    print(f"Session: {session_path.name}")

    summary_df, individual_df = run_detection(session_path, args.sample_every_sec)

    summary_path = session_path / "detected_people.csv"
    summary_df.to_csv(summary_path, index=False)

    individual_path = session_path / "detected_people_individual.csv"
    individual_df.to_csv(individual_path, index=False)

    people_found = (summary_df["num_faces"] > 0).sum()
    print(f"  rows with at least 1 person detected: {people_found}/{len(summary_df)}")
    print(f"  individual detections written: {len(individual_df)} rows")
    print(f"  written to: {summary_path}")
    print(f"  written to: {individual_path}\n")


if __name__ == "__main__":
    main()
