"""
Objective 3 validation, step 1: extract a random sample of frames with the
estimated O-space centre drawn on them, ready for manual hand-labelling.

WHY THE PROPOSAL'S ORIGINAL CRITERION HAD TO CHANGE
----------------------------------------------------
The proposal commits to:
    "...validated against a manually hand-labelled subset of at least 30
     frames, targeting an O-space-centre estimate within 0.3m of the manual
     annotation for at least 70% of labelled frames."

The 30-frame hand-labelled subset is fine and is what this script prepares.
The "within 0.3m" part is NOT measurable in this project: 0.3 m is a metric
tolerance, but the PLUS-HRI video is uncalibrated and has no depth channel,
so there is no honest pixel->metre conversion available. Reporting any
figure in metres here would be fabricated.

RE-SPECIFIED CRITERION (defensible, and preserves the original intent):
    "The estimated O-space centre must fall within 0.5x the mean detected
     person bounding-box width of the hand-labelled centre, for at least
     70% of labelled frames."

Rationale for 0.5x bbox width: an adult's shoulder width is roughly
0.45-0.50 m, and the detector's bounding-box width is a direct pixel
measure of that same quantity in each frame. So 0.5x bbox width corresponds
to roughly 0.22-0.25 m in the real world - i.e. the same order as the
original 0.3 m target, but expressed in a unit this dataset can actually
measure. Because the tolerance scales with apparent person size, it also
self-corrects for perspective: people further from the camera get a
proportionally tighter pixel tolerance, which is the correct behaviour.

State this re-specification explicitly in the dissertation (Methodology
and/or Limitations). It is a legitimate adaptation to a real data
constraint, not a lowering of the bar - but it must be disclosed, and
ideally agreed with your supervisor first.

WHAT THIS SCRIPT PRODUCES
-------------------------
  dataset/processed/ospace_validation/frame_XXX.jpg
      Each sampled frame with the ESTIMATED O-space centre marked (red
      cross) and each detected person's box (thin green). Look at the
      picture and decide where the true conversational-group centre is.
  dataset/processed/ospace_validation/labels_template.csv
      One row per frame, with `label_x` and `label_y` left BLANK for you to
      fill in. Everything else (session, timestamp, estimate, bbox scale)
      is pre-filled.

HOW TO LABEL (about 20-30 minutes for 30 frames)
------------------------------------------------
  1. Open a frame image in any viewer that shows pixel coordinates
     (Windows Photos won't; IrfanView, GIMP, or even MS Paint will - Paint
     shows the cursor position in the bottom-left status bar).
  2. Decide by eye where the centre of the conversational group's shared
     space is - the empty middle of the circle people have formed. If the
     frame shows no real conversational group (e.g. people walking past,
     or only one person), write `none` in label_x and it will be excluded.
  3. Type that pixel x and y into labels_template.csv.
  4. Save it as labels.csv (same folder), then run
     scripts/validate_ospace_estimate.py.

Usage:
    python3 scripts/prepare_ospace_validation.py
    python3 scripts/prepare_ospace_validation.py --num-frames 40 --seed 7
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
OUTPUT_DIR: Path = DATASET_DIR / "processed" / "ospace_validation"

MIN_GROUP_SIZE = 2  # a "group" needs at least 2 people to have an O-space at all


def read_frame_manifest(session_path: Path) -> dict[int, float]:
    manifest_path = next(session_path.glob("*_frame_manifest.json"), None)
    if manifest_path is None:
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def normalise(value: float) -> float:
        return value / 1e9 if abs(value) > 1e12 else value

    frame_map: dict[int, float] = {}
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


def collect_candidates() -> pd.DataFrame:
    """Every (session, timestamp) where a real multi-person group was detected."""
    rows = []
    for groups_path in sorted(DATASET_DIR.glob("*/detected_groups.csv")):
        session_id = groups_path.parent.name
        df = pd.read_csv(groups_path)
        if df.empty:
            continue
        good = df[(df["is_largest_group"]) & (df["num_people"] >= MIN_GROUP_SIZE)].copy()
        good["session_id"] = session_id
        rows.append(good)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=30,
                        help="how many frames to sample for labelling (proposal requires >= 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="where to write frames + labelling template")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()

    candidates = collect_candidates()
    if candidates.empty:
        print("No multi-person groups found - run cluster_groups.py first.")
        return

    print(f"Found {len(candidates)} frames containing a group of {MIN_GROUP_SIZE}+ people "
          f"across {candidates['session_id'].nunique()} sessions.")

    sample = candidates.sample(n=min(args.num_frames, len(candidates)), random_state=args.seed)
    sample = sample.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_rows = []
    saved = 0

    for session_id, session_sample in sample.groupby("session_id"):
        session_path = DATASET_DIR / session_id
        video_path = session_path / f"{session_id}.mp4"
        if not video_path.exists():
            print(f"  session {session_id}: no video, skipping")
            continue

        frame_map = read_frame_manifest(session_path)
        if not frame_map:
            print(f"  session {session_id}: no frame manifest, skipping")
            continue

        # Map each wanted timestamp back to its nearest frame index.
        indices = np.array(sorted(frame_map))
        times = np.array([frame_map[i] for i in indices])

        individual_path = session_path / "detected_people_individual.csv"
        individual = pd.read_csv(individual_path) if individual_path.exists() else pd.DataFrame()

        # NOTE: do NOT use cap.set(CAP_PROP_POS_FRAMES, ...) to jump to a frame.
        # These videos are encoded such that seeking lands on a non-keyframe and
        # decodes to a corrupted (solid green) image. Reading sequentially and
        # picking out the wanted frame indices is slower but actually correct -
        # this is the same approach extract_person_detections.py uses.
        wanted = {}
        for _, row in session_sample.iterrows():
            nearest = int(indices[np.abs(times - row["timestamp"]).argmin()])
            wanted[nearest] = row

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

            annotated = frame.copy()
            if not individual.empty:
                # rtol MUST be 0 here: np.isclose defaults to rtol=1e-5, which on a
                # ~1.76e9 UNIX timestamp is a tolerance of ~17,000 SECONDS - every
                # detection in the session would match and the frame would be buried
                # under thousands of boxes.
                people = individual[np.isclose(individual["timestamp"], row["timestamp"],
                                                rtol=0.0, atol=1e-6)]
                for _, person in people.iterrows():
                    cv2.rectangle(annotated,
                                  (int(person["bbox_min_x"]), int(person["bbox_min_y"])),
                                  (int(person["bbox_max_x"]), int(person["bbox_max_y"])),
                                  (0, 200, 0), 1)

            cx, cy = int(row["group_center_x"]), int(row["group_center_y"])
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(annotated, "estimated O-space centre", (max(cx - 120, 5), max(cy - 18, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            name = f"frame_{saved:03d}_s{session_id}.jpg"
            cv2.imwrite(str(output_dir / name), annotated)

            label_rows.append({
                "frame_file": name,
                "session_id": session_id,
                "timestamp": row["timestamp"],
                "num_people": int(row["num_people"]),
                "estimated_x": float(row["group_center_x"]),
                "estimated_y": float(row["group_center_y"]),
                "avg_bbox_width": float(row["avg_bbox_width"]),
                "label_x": "",   # <- YOU FILL THIS IN
                "label_y": "",   # <- YOU FILL THIS IN
            })
            saved += 1
        cap.release()
        print(f"  session {session_id}: {len(session_sample)} frame(s) exported")

    if not label_rows:
        print("\nNo frames could be exported - check that session videos are present.")
        return

    template_path = output_dir / "labels_template.csv"
    pd.DataFrame(label_rows).to_csv(template_path, index=False)

    print(f"\n{saved} frames written to: {output_dir}")
    print(f"Labelling template:      {template_path}")
    print("\nNext: fill in label_x / label_y for each frame (or 'none' if there's no real")
    print("conversational group), save the file as labels.csv in the same folder, then run:")
    print("    python3 scripts/validate_ospace_estimate.py")


if __name__ == "__main__":
    main()
