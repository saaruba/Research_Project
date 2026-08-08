#!/usr/bin/env python3
"""
Step 2 of the YOLOv8n vs LocateAnything-3B comparison. RUN THIS ON YOUR GPU
MACHINE, not in the ROS2 devcontainer.

WHAT THIS IS FOR (be clear about this in the write-up)
-------------------------------------------------------
Objective 2 is ALREADY SATISFIED: YOLOv8n measured 99.7% recall against the
sessions 1/3 ground truth, versus the proposal's >= 80% target. Running
LocateAnything-3B cannot improve that - there is 0.3% of headroom - and it
will not change any downstream result, because every group, O-space,
training table and model in this project is built on detections that already
exist.

The ONLY purpose of this script is evidential: your proposal names
nvidia/LocateAnything-3B as the intended perception model, and the project
substituted YOLOv8n. That substitution is already permitted by the
proposal's own risk table ("keep a simpler person-detection baseline"), but
"we measured both and YOLOv8n was sufficient" is a considerably stronger
justification in a viva than "the install was awkward". This produces that
measurement on 30 frames - about half a day's work, not days.

Note: LocateAnything-3B is used for INFERENCE only. Nothing is trained here.
Neither detector in this project is trained by you; both are pretrained
models being applied.

SETUP (on the GPU machine, in a FRESH virtual environment - do NOT install
this into the devcontainer environment; LA-3B pins numpy==1.25.0 which will
break the numpy 2.x that pandas/scikit-learn need in the data pipeline):

    python3 -m venv la3b_env
    source la3b_env/bin/activate          # Windows: la3b_env\\Scripts\\activate
    pip install opencv-python-headless==4.11.0.86 transformers==4.57.1 \\
        numpy==1.25.0 Pillow==11.1.0 peft torchvision decord==0.6.0 lmdb==1.7.5 pandas
    pip install torch     # match your CUDA version: https://pytorch.org/get-started/locally/

USAGE:
    # 1. On the project machine:
    python3 scripts/export_detector_comparison_frames.py
    # 2. Copy dataset/processed/detector_comparison/ to the GPU machine.
    # 3. There, in the la3b_env venv:
    python3 run_locateanything_comparison.py --frames-dir /path/to/detector_comparison
    # 4. Copy the resulting comparison_manifest.csv back into the project.

The script fills in the la3b_count column and prints a recall comparison
using the same definition as validate_detector_recall.py: of the frames
where ground truth says >= 1 person is present, in what fraction did the
detector also find >= 1 person?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

MODEL_NAME = "nvidia/LocateAnything-3B"


class LocateAnythingWorker:
    """Minimal worker, adapted from the NVIDIA model card."""

    def __init__(self, model_path: str = MODEL_NAME, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16
        print(f"Loading {model_path} (several GB on first run)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=self.dtype, trust_remote_code=True
        ).to(device).eval()
        print("Model loaded.\n")

    def detect(self, image, categories=("person",), max_new_tokens: int = 2048) -> str:
        with self.torch.no_grad():
            cats = "</c>".join(categories)
            prompt = f"Locate all the instances that matches the following description: {cats}."
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
            return response[0] if isinstance(response, tuple) else response

    @staticmethod
    def count_boxes(answer: str) -> int:
        return len(re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", str(answer)))


def report(df: pd.DataFrame) -> None:
    scored = df[pd.to_numeric(df["la3b_count"], errors="coerce").notna()].copy()
    scored["la3b_count"] = scored["la3b_count"].astype(int)
    positives = scored[scored["num_faces_gt"] > 0]

    if positives.empty:
        print("No ground-truth positive frames to score.")
        return

    yolo_recall = (positives["yolo_count"] > 0).mean() * 100
    la3b_recall = (positives["la3b_count"] > 0).mean() * 100

    print("\n" + "=" * 68)
    print("DETECTOR COMPARISON - recall on ground-truth positive frames")
    print("=" * 68)
    print(f"Frames scored: {len(positives)}")
    print(f"  YOLOv8n            recall: {yolo_recall:5.1f}%   "
          f"(mean {positives['yolo_count'].mean():.2f} people/frame)")
    print(f"  LocateAnything-3B  recall: {la3b_recall:5.1f}%   "
          f"(mean {positives['la3b_count'].mean():.2f} people/frame)")
    print(f"\nObjective 2 target: >= 80%")
    print("\nNote: this 30-frame sample is a spot-check for justification purposes.")
    print("The headline 99.7% YOLOv8n figure comes from validate_detector_recall.py,")
    print("which scores every annotated moment in sessions 1 and 3 - cite that one.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-only", action="store_true",
                        help="skip inference, just re-print the comparison from an existing manifest")
    args = parser.parse_args()

    frames_dir = args.frames_dir.expanduser().resolve()
    manifest_path = frames_dir / "comparison_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found - run export_detector_comparison_frames.py first")

    df = pd.read_csv(manifest_path)

    if args.report_only:
        report(df)
        return

    from PIL import Image

    worker = LocateAnythingWorker(device=args.device)

    counts = []
    for position, row in enumerate(df.itertuples(index=False), start=1):
        image_path = frames_dir / row.frame_file
        if not image_path.exists():
            print(f"  [{position}/{len(df)}] {row.frame_file}: MISSING, skipping")
            counts.append(None)
            continue
        image = Image.open(image_path).convert("RGB")
        answer = worker.detect(image)
        count = worker.count_boxes(answer)
        counts.append(count)
        print(f"  [{position}/{len(df)}] {row.frame_file}: "
              f"LA-3B={count}  YOLO={row.yolo_count}  GT={row.num_faces_gt}")

    df["la3b_count"] = counts
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest updated: {manifest_path}")

    report(df)


if __name__ == "__main__":
    main()
