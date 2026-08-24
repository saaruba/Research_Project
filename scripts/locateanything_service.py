#!/usr/bin/env python3
"""
LocateAnything-3B as a local detection SERVICE, so the ROS 2 perception node
can use it despite an unresolvable Python dependency conflict.

WHY A SERVICE AND NOT JUST AN IMPORT
-------------------------------------
LocateAnything-3B pins numpy==1.25.0. ROS 2 Humble's rclpy, cv_bridge and the
rest of this project's pipeline need numpy 2.x. Both cannot live in one Python
process - this project already broke its entire data pipeline once by trying
(numpy was silently downgraded and pandas/sklearn/OpenCV stopped importing).

Two processes, two environments, talking over localhost, solves it cleanly:

    ROS 2 node (system python, numpy 2.x)  --HTTP-->  this service (la3b_env, numpy 1.25)

Only stdlib is used on the wire (http.server + JSON + base64), so no extra
dependencies are introduced on either side.

MEASURED PERFORMANCE - READ BEFORE USING THIS LIVE
---------------------------------------------------
On the project hardware (RTX 3050 Ti Laptop, 4.3 GB VRAM):
    LocateAnything-3B : 25.63 s/frame  (0.04 FPS)
    YOLOv8n           : ~0.005 s/frame (~200 FPS)
LA-3B needs ~7.7 GB for its weights in bf16, which exceeds the available VRAM,
so it runs with CUDA system-memory fallback - technically on the GPU, but
starved and shuttling weights over PCIe.

Consequence: this is NOT viable as a continuous perception source for
navigation, which needs 10-30 Hz. It IS viable for a ONE-SHOT approach
decision, where the robot pauses, looks once, decides, and then drives using
Nav2's own reactive obstacle avoidance. That is how the perception node uses
it when detector:=locateanything.

RUN (in the la3b_env virtualenv, NOT the ROS environment)
----------------------------------------------------------
    source la3b_env/bin/activate
    python3 scripts/locateanything_service.py
    # then, in a separate ROS terminal:
    ros2 run tiago_group_approach group_perception_node --ros-args -p detector:=locateanything

Health check:
    curl http://127.0.0.1:8765/health
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_NAME = "nvidia/LocateAnything-3B"


class Detector:
    """Wraps LocateAnything-3B. Greedy decoding for reproducibility."""

    def __init__(self, model_path: str = MODEL_NAME, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16

        print(f"Loading {model_path} ... (several GB on first run)", flush=True)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=self.dtype, trust_remote_code=True
        ).to(device).eval()
        print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM {total:.1f} GB", flush=True)
            if total < 8.0:
                print("WARNING: under 8 GB VRAM. LocateAnything-3B needs ~7.7 GB for",
                      "weights alone; expect system-memory fallback and ~25 s/frame.",
                      flush=True)
        else:
            print("WARNING: no CUDA device - CPU inference will be extremely slow.",
                  flush=True)

    def detect(self, image, max_new_tokens: int = 2048) -> list[dict]:
        with self.torch.no_grad():
            prompt = ("Locate all the instances that matches the following "
                      "description: person.")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos,
                                    return_tensors="pt").to(self.device)

            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                repetition_penalty=1.1,
                do_sample=False,          # greedy: same image -> same boxes
                verbose=False,
            )
            answer = response[0] if isinstance(response, tuple) else response

        width, height = image.size
        raw = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", str(answer)):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            # LocateAnything emits coordinates normalised to 0-1000.
            raw.append({
                "x1": x1 / 1000.0 * width,  "y1": y1 / 1000.0 * height,
                "x2": x2 / 1000.0 * width,  "y2": y2 / 1000.0 * height,
            })

        return self._sanitise(raw, width, height)

    # ------------------------------------------------------------------ output
    # DEGENERATE OUTPUT GUARD (added Aug 2026)
    #
    # Observed live: a single restaurant frame produced 341 "person" boxes in
    # 9.9 s. The world contains 15 people, and the camera can see at most a
    # handful of them. This is the classic autoregressive failure - the decoder
    # falls into a repetition loop emitting <box>...</box> until max_new_tokens
    # runs out - and greedy decoding with repetition_penalty=1.1 does not
    # prevent it.
    #
    # Passing 341 boxes downstream is worse than passing none: the perception
    # node back-projects each one, and the clustering then invents groups all
    # over the map. So the raw output is cleaned here, at the boundary, where
    # the failure can also be COUNTED - the raw-vs-kept gap is itself a
    # measurement of how unreliable this detector is, and is logged for exactly
    # that reason.
    #
    # Nothing here is model-specific tuning; it is the minimum needed to stop
    # obviously impossible output reaching the pipeline.
    MAX_BOXES = 20              # far more than any single camera frame contains
    MIN_AREA_FRAC = 0.0005      # smaller than this is noise, not a person
    MAX_AREA_FRAC = 0.60        # a "person" filling most of the frame is not one
    DEDUP_TOL_PX = 8.0          # boxes within this are the same detection

    def _sanitise(self, raw: list[dict], width: int, height: int) -> list[dict]:
        frame_area = float(width * height)
        kept: list[dict] = []

        for b in raw:
            x1, y1 = max(0.0, b["x1"]), max(0.0, b["y1"])
            x2, y2 = min(float(width), b["x2"]), min(float(height), b["y2"])
            if x2 <= x1 or y2 <= y1:
                continue                                  # degenerate
            frac = ((x2 - x1) * (y2 - y1)) / frame_area
            if frac < self.MIN_AREA_FRAC or frac > self.MAX_AREA_FRAC:
                continue

            # Drop repeats of a box we already have.
            if any(abs(x1 - k["x1"]) < self.DEDUP_TOL_PX
                   and abs(y1 - k["y1"]) < self.DEDUP_TOL_PX
                   and abs(x2 - k["x2"]) < self.DEDUP_TOL_PX
                   and abs(y2 - k["y2"]) < self.DEDUP_TOL_PX for k in kept):
                continue

            kept.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        if len(raw) > self.MAX_BOXES:
            print(f"  DEGENERATE OUTPUT: model emitted {len(raw)} boxes for one "
                  f"frame; {len(kept)} survived filtering, capping at "
                  f"{self.MAX_BOXES}. This is a decoder repetition loop, not "
                  f"{len(raw)} people.", flush=True)

        return kept[:self.MAX_BOXES]


class Handler(BaseHTTPRequestHandler):
    detector: Detector | None = None

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise; we log our own timing

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": MODEL_NAME,
                             "loaded": Handler.detector is not None})
        else:
            self._json(404, {"error": "use POST /detect or GET /health"})

    def do_POST(self):
        if self.path != "/detect":
            self._json(404, {"error": "unknown endpoint"})
            return
        if Handler.detector is None:
            self._json(503, {"error": "model not loaded"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            raw = base64.b64decode(payload["image"])

            from PIL import Image
            image = Image.open(io.BytesIO(raw)).convert("RGB")

            started = time.perf_counter()
            boxes = Handler.detector.detect(image)
            elapsed = time.perf_counter() - started

            print(f"detect: {len(boxes)} person(s) in {elapsed:.2f}s", flush=True)
            self._json(200, {"boxes": boxes, "inference_seconds": round(elapsed, 3)})
        except Exception as exc:  # noqa: BLE001 - report anything back to the caller
            print(f"ERROR: {exc}", flush=True)
            self._json(500, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    Handler.detector = Detector(device=args.device)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"\nLocateAnything service listening on http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  POST /detect   {\"image\": \"<base64 jpeg>\"}")
    print("\nStart the ROS node with:  -p detector:=locateanything")
    print("Ctrl-C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
