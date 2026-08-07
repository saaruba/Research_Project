#!/usr/bin/env python3
"""
First real test of person detection on your actual video footage, using
YOLOv8n (a small, fast, pretrained object detector - not trained by us,
just downloaded and used as-is, the same way the proposal plans to use
LocateAnything as a pretrained perception model).

Why this script exists: a quick test with OpenCV's older built-in
HOG people-detector gave poor, badly-placed boxes on this dataset's footage
(dim indoor scenes, people facing away from camera). YOLOv8n is a modern
deep-learning detector and should do much better. This script proves that
before you invest more time building the full Phase B pipeline.

Setup (only needed once):
    pip install ultralytics

Usage - detect people in one frame extracted from a session's video:
    python3 scripts/test_person_detector.py --session dataset/9 --time 30

Usage - detect people in an existing image file:
    python3 scripts/test_person_detector.py --image /path/to/frame.jpg

Output:
    Prints how many people were found and their confidence scores.
    Saves an annotated image next to the input, e.g. 9_frame_detected.jpg,
    so you can see the boxes and judge the quality yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_frame(session_dir: Path, time_sec: float) -> Path:
    import cv2

    mp4_files = sorted(session_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 file found in {session_dir}")

    # Prefer the top-level full-session video over segment clips if both exist
    video_path = mp4_files[0]
    for candidate in mp4_files:
        if candidate.stem == session_dir.name:
            video_path = candidate
            break

    output_path = session_dir.parent.parent / "processed" / f"{session_dir.name}_test_frame.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read a frame at {time_sec}s from {video_path}. "
            "Try a smaller --time value (the video may be shorter than that)."
        )

    cv2.imwrite(str(output_path), frame)
    print(f"Extracted frame from {video_path.name} at {time_sec}s -> {output_path}")
    return output_path


def run_detector(image_path: Path) -> None:
    from ultralytics import YOLO
    import cv2

    print("\nLoading YOLOv8n (downloads automatically on first run, ~6MB)...")
    model = YOLO("yolov8n.pt")

    print(f"Running detection on {image_path}...")
    results = model(str(image_path), classes=[0], verbose=False)  # class 0 = person in COCO

    result = results[0]
    boxes = result.boxes

    print(f"\nPeople detected: {len(boxes)}")
    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        print(f"  person {i + 1}: confidence={conf:.2f}, box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    annotated = result.plot()
    output_path = image_path.parent / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\nAnnotated image saved to: {output_path}")
    print("Open that file and look at the boxes yourself before trusting this on the full dataset.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test person detection on one frame.")
    parser.add_argument("--session", type=Path, help="Path to a session folder, e.g. dataset/9")
    parser.add_argument("--time", type=float, default=30.0, help="Time in seconds to grab the frame from")
    parser.add_argument("--image", type=Path, help="Path to an existing image, instead of --session")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.image:
        image_path = args.image.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    elif args.session:
        image_path = extract_frame(args.session.expanduser().resolve(), args.time)
    else:
        raise ValueError("Provide either --session or --image")

    run_detector(image_path)


if __name__ == "__main__":
    main()
