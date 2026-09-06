"""
Step 1 of the YOLOv8n vs LocateAnything-3B comparison: export a fixed set of
frames from sessions 1 and 3 (the only two with ground-truth face
annotations), together with the YOLO result and the ground-truth count for
each frame.

This runs on ANY machine - no GPU, no torch, no LocateAnything install. It
just pulls frames out of the videos and writes a manifest. You then copy the
exported folder to your GPU machine and run
run_locateanything_comparison.py there (step 2), which only needs the images
and the manifest, not the 26 GB dataset.

Frames are chosen only from moments where the ground truth says at least one
face was present, because recall is the metric being compared - a frame with
no people in it tells you nothing about whether a detector finds people.

Output: dataset/processed/detector_comparison/
    frame_XXX_sY.jpg          the images to run both detectors on
    comparison_manifest.csv   session, timestamp, ground-truth count,
                              YOLO count (already known), LA-3B count (blank,
                              filled in by step 2)

Usage:
    python3 scripts/export_detector_comparison_frames.py
    python3 scripts/export_detector_comparison_frames.py --num-frames 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT = DATASET_DIR / "processed" / "detector_comparison"
VALIDATION_SESSIONS = ["1", "3"]
MATCH_TOLERANCE_SEC = 1.0


def normalise_timestamp(value: float) -> float:
    return value / 1e9 if abs(value) > 1e12 else value


def read_frame_manifest(session_path: Path) -> dict[int, float]:
    manifest_path = next(session_path.glob("*_frame_manifest.json"), None)
    if manifest_path is None:
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frame_map: dict[int, float] = {}
    if isinstance(payload, list):
        for index, timestamp in enumerate(payload):
            frame_map[index] = normalise_timestamp(float(timestamp))
    elif isinstance(payload, dict):
        for key in ("timestamps", "frame_timestamps"):
            if key in payload and isinstance(payload[key], list):
                for index, timestamp in enumerate(payload[key]):
                    frame_map[index] = normalise_timestamp(float(timestamp))
                break
    return frame_map


def load_ground_truth(session_path: Path) -> pd.DataFrame:
    df = pd.read_csv(session_path / "facial_landmarks_uniface.csv")
    x_col = "x" if "x" in df.columns else "position_x"
    y_col = "y" if "y" in df.columns else "position_y"
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").apply(normalise_timestamp)
    df = df.dropna(subset=["timestamp", x_col, y_col])
    if "face_index" in df.columns:
        summary = df.groupby("timestamp")["face_index"].nunique().reset_index(name="num_faces_gt")
    else:
        summary = df.groupby("timestamp").size().reset_index(name="num_faces_gt")
    return summary.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=30,
                        help="total frames to export across sessions 1 and 3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_session = max(1, args.num_frames // len(VALIDATION_SESSIONS))
    manifest_rows = []
    saved = 0

    for session_id in VALIDATION_SESSIONS:
        session_path = DATASET_DIR / session_id
        video_path = session_path / f"{session_id}.mp4"
        if not video_path.exists():
            print(f"session {session_id}: no video, skipping")
            continue

        gt = load_ground_truth(session_path)
        gt = gt[gt["num_faces_gt"] > 0]

        yolo_path = session_path / "detected_people.csv"
        yolo = pd.read_csv(yolo_path)[["timestamp", "num_faces"]].rename(
            columns={"num_faces": "yolo_count"}).sort_values("timestamp")

        merged = pd.merge_asof(gt, yolo, on="timestamp", direction="nearest",
                               tolerance=MATCH_TOLERANCE_SEC)
        merged["yolo_count"] = merged["yolo_count"].fillna(0).astype(int)

        sample = merged.sample(n=min(per_session, len(merged)), random_state=args.seed)

        frame_map = read_frame_manifest(session_path)
        indices = np.array(sorted(frame_map))
        times = np.array([frame_map[i] for i in indices])

        wanted = {}
        for _, row in sample.iterrows():
            nearest = int(indices[np.abs(times - row["timestamp"]).argmin()])
            wanted[nearest] = row

        # Sequential read - seeking with CAP_PROP_POS_FRAMES decodes corrupt
        # frames on these videos (lands on a non-keyframe).
        cap = cv2.VideoCapture(str(video_path))
        max_wanted = max(wanted)
        frame_idx = 0
        while frame_idx <= max_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx not in wanted:
                frame_idx += 1
                continue
            row = wanted[frame_idx]
            frame_idx += 1

            name = f"frame_{saved:03d}_s{session_id}.jpg"
            cv2.imwrite(str(output_dir / name), frame)
            manifest_rows.append({
                "frame_file": name,
                "session_id": session_id,
                "timestamp": float(row["timestamp"]),
                "num_faces_gt": int(row["num_faces_gt"]),
                "yolo_count": int(row["yolo_count"]),
                "la3b_count": "",  # filled in by step 2 on the GPU machine
            })
            saved += 1
        cap.release()
        print(f"session {session_id}: exported {len(wanted)} frame(s)")

    if not manifest_rows:
        print("Nothing exported.")
        return

    manifest_path = output_dir / "comparison_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"\n{saved} frames exported to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nNext: copy that whole folder to your GPU machine and run")
    print("    python3 run_locateanything_comparison.py --frames-dir <copied folder>")


if __name__ == "__main__":
    main()
