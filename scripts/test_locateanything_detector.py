#!/usr/bin/env python3
"""
Test person detection using the model actually named in your proposal:
nvidia/LocateAnything-3B (Wang et al., NVIDIA, 2026).

IMPORTANT: run this on your GPU machine, in its OWN fresh virtual
environment - do NOT install this into the same Python environment as your
ROS2 devcontainer / data pipeline scripts. LocateAnything-3B pins
numpy==1.25.0, which will conflict with the numpy 2.x that pandas/scikit-learn
need there and could break scipy/matplotlib/sklearn all over again (see the
[[tiago-devcontainer-python-quirks]] lesson from testing YOLOv8n).

Setup (on your GPU machine, in a NEW virtual environment):
    python3 -m venv la3b_env
    source la3b_env/bin/activate
    pip install opencv-python-headless==4.11.0.86 transformers==4.57.1 numpy==1.25.0 Pillow==11.1.0 peft torchvision decord==0.6.0 lmdb==1.7.5
    # install torch separately, matching your CUDA version - see https://pytorch.org/get-started/locally/
    pip install torch

Usage:
    python3 test_locateanything_detector.py --image /path/to/frame.jpg

The image itself can just be copied over from this project's
dataset/processed/ folder (e.g. the same 9_test_frame.jpg used for the
YOLOv8n test), so you're comparing both detectors on the exact same frame.

Output:
    Prints each detected person's box.
    Saves an annotated image next to the input, e.g. frame_la3b_detected.jpg,
    so you can compare it directly against the YOLOv8n result.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


MODEL_NAME = "nvidia/LocateAnything-3B"


class LocateAnythingWorker:
    """Minimal worker, adapted directly from the NVIDIA model card."""

    def __init__(self, model_path: str = MODEL_NAME, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, categories: list[str], max_new_tokens: int = 2048) -> str:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
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
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width, "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width, "y2": y2 / 1000 * image_height,
            })
        return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    print("Loading nvidia/LocateAnything-3B (several GB download on first run)...")
    worker = LocateAnythingWorker()

    print("Running detection...")
    answer = worker.detect(img, ["person"])
    print("\nRaw model output:", answer)

    w, h = img.size
    boxes = worker.parse_boxes(answer, w, h)
    print(f"\nPeople detected: {len(boxes)}")

    cv_img = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
        print(f"  person {i + 1}: box=({x1}, {y1}, {x2}, {y2})")
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cv_img, "person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = args.image.parent / f"{args.image.stem}_la3b_detected.jpg"
    cv2.imwrite(str(output_path), cv_img)
    print(f"\nAnnotated image saved to: {output_path}")
    print("Compare this against the YOLOv8n result on the same frame before deciding which to use for all 24 sessions.")


if __name__ == "__main__":
    main()
