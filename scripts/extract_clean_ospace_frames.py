#!/usr/bin/env python3
"""
Objective 3, step 1b: extract the 30 validation frames WITHOUT any annotation.

    python3 scripts/extract_clean_ospace_frames.py
    python3 scripts/extract_clean_ospace_frames.py --sessions 1 3 5

============================================================================
WHY THIS EXISTS
============================================================================
prepare_ospace_validation.py saves each frame with the detector's boxes and a
red cross marking the ESTIMATED O-space centre burned into the pixels. That is
useful for eyeballing what the pipeline did, but it is unusable as a labelling
surface: a human asked to mark the true centre while a red cross is already
sitting on the image will anchor to it, and the resulting "validation" would
measure agreement with the algorithm rather than accuracy against a human
judgement.

This regenerates exactly the same 30 frames - same sessions, same timestamps,
matched by frame_file name so nothing can drift - with no overlay at all.

It is resumable: frames already present in the output directory are skipped, so
it can be run in several passes over a large video set.

Note on seeking: cap.set(CAP_PROP_POS_FRAMES, ...) is NOT used. These videos
decode to a corrupted solid-green image when seeked to a non-keyframe. Reading
sequentially to the wanted index is slower but correct - the same approach
prepare_ospace_validation.py and extract_person_detections.py both take.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "dataset"
LABEL_DIR = DATASET / "processed" / "ospace_labelling"
TEMPLATE = LABEL_DIR / "labels_template.csv"
CLEAN_DIR = LABEL_DIR / "clean"


def read_frame_manifest(session_path: Path) -> dict[int, float]:
    manifest_path = next(session_path.glob("*_frame_manifest.json"), None)
    if manifest_path is None:
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    def normalise(v: float) -> float:
        return v / 1e9 if abs(v) > 1e12 else v

    out: dict[int, float] = {}
    if isinstance(payload, list):
        for i, t in enumerate(payload):
            out[i] = normalise(float(t))
    elif isinstance(payload, dict):
        for key in ("timestamps", "frame_timestamps"):
            if key in payload and isinstance(payload[key], list):
                for i, t in enumerate(payload[key]):
                    out[i] = normalise(float(t))
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", type=int, default=None,
                    help="only these session ids (default: all needed)")
    ap.add_argument("--out", type=Path, default=CLEAN_DIR)
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"missing {TEMPLATE} - run prepare_ospace_validation.py first")

    args.out.mkdir(parents=True, exist_ok=True)
    wanted_all = pd.read_csv(TEMPLATE)

    todo = [r for _, r in wanted_all.iterrows()
            if not (args.out / r["frame_file"]).exists()]
    if args.sessions:
        todo = [r for r in todo if int(r["session_id"]) in args.sessions]

    have = len(wanted_all) - len([r for _, r in wanted_all.iterrows()
                                  if not (args.out / r["frame_file"]).exists()])
    print(f"{have}/{len(wanted_all)} clean frames already present; "
          f"{len(todo)} to extract in this pass")
    if not todo:
        print("Nothing to do.")
        return

    by_session: dict[int, list] = {}
    for r in todo:
        by_session.setdefault(int(r["session_id"]), []).append(r)

    for sid in sorted(by_session):
        rows = by_session[sid]
        session_path = DATASET / str(sid)
        video = session_path / f"{sid}.mp4"
        if not video.exists():
            print(f"  s{sid}: no video at {video} - skipped")
            continue

        frame_map = read_frame_manifest(session_path)
        if not frame_map:
            print(f"  s{sid}: no frame manifest - skipped")
            continue

        indices = np.array(sorted(frame_map))
        times = np.array([frame_map[i] for i in indices])
        wanted = {}
        for r in rows:
            nearest = int(indices[np.abs(times - float(r["timestamp"])).argmin()])
            wanted[nearest] = r

        started = time.perf_counter()
        cap = cv2.VideoCapture(str(video))
        max_wanted = max(wanted)
        idx, saved = 0, 0
        while idx <= max_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                cv2.imwrite(str(args.out / wanted[idx]["frame_file"]), frame)
                saved += 1
            idx += 1
        cap.release()
        print(f"  s{sid}: {saved}/{len(rows)} frame(s) in "
              f"{time.perf_counter() - started:.0f}s "
              f"(scanned {idx} of {max_wanted} frames)")

    n = len(list(args.out.glob('*.jpg')))
    print(f"\nClean frames in {args.out}: {n}/{len(wanted_all)}")
    if n < len(wanted_all):
        print("Run again to continue - it resumes where it stopped.")


if __name__ == "__main__":
    main()
