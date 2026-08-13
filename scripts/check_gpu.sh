#!/usr/bin/env bash
# ============================================================================
# GPU availability diagnostic - run INSIDE the devcontainer.
#
#   bash scripts/check_gpu.sh
#
# Works through the chain in order, because the fix depends entirely on which
# link is broken:
#
#   Windows host has NVIDIA GPU + driver
#        -> Docker Desktop uses WSL2 backend with GPU support
#             -> container was started with --gpus all
#                  -> nvidia-smi works inside the container
#                       -> torch.cuda.is_available() is True
#
# Any one of these failing gives you CPU-only inference, which is why
# LocateAnything-3B measured 25.6 s/frame instead of a few seconds.
# ============================================================================

echo "============================================================"
echo "GPU DIAGNOSTIC (inside the container)"
echo "============================================================"

# --- 1. Is the GPU device visible to this container at all? -----------------
echo ""
echo "[1] nvidia-smi inside the container:"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        nvidia-smi | head -12
        echo "    -> GPU IS visible to the container."
    else
        echo "    nvidia-smi exists but FAILED to run."
        echo "    -> the driver is not reachable from inside the container."
        echo "       Container was probably started without --gpus all."
    fi
else
    echo "    nvidia-smi NOT FOUND."
    echo "    -> This container has no NVIDIA runtime. It was started without"
    echo "       GPU passthrough, OR the host has no NVIDIA GPU."
fi

# --- 2. Does a GPU device node exist? ---------------------------------------
# NOTE: on WSL2 (Docker Desktop on Windows) the GPU is exposed through
# /dev/dxg, NOT /dev/nvidia*. Checking only for /dev/nvidia* gives a false
# "no GPU" result on a perfectly working WSL2 setup.
echo ""
echo "[2] GPU device nodes:"
FOUND_NODE=0
if ls /dev/nvidia* >/dev/null 2>&1; then
    ls -1 /dev/nvidia* | sed 's/^/    /'
    echo "    -> native NVIDIA device nodes present (Linux host passthrough)."
    FOUND_NODE=1
fi
if [ -e /dev/dxg ]; then
    echo "    /dev/dxg"
    echo "    -> WSL2 GPU paravirtualisation present. This is how Docker Desktop"
    echo "       on Windows exposes the GPU - /dev/nvidia* is NOT expected here."
    FOUND_NODE=1
fi
if [ "$FOUND_NODE" -eq 0 ]; then
    echo "    none found (neither /dev/nvidia* nor /dev/dxg)."
    echo "    -> the container has no GPU passthrough."
fi

# --- 3. What does PyTorch think? --------------------------------------------
echo ""
echo "[3] PyTorch CUDA status:"

check_torch() {
    python3 - <<'PY' 2>/dev/null || echo "    (torch not installed in this environment)"
try:
    import torch
    print(f"    torch version : {torch.__version__}")
    print(f"    built with CUDA: {torch.version.cuda}")
    print(f"    cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    device: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"    VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
        if total < 8e9:
            print("    !! LocateAnything-3B needs roughly 8 GB VRAM in bf16.")
            print("       With less than that it will fail or fall back to CPU.")
    else:
        print("    -> PyTorch cannot see a GPU. Inference will run on CPU.")
except ImportError:
    raise SystemExit(1)
PY
}

if [ -f "$(dirname "${BASH_SOURCE[0]}")/../la3b_env/bin/activate" ]; then
    echo "  -- inside la3b_env (the environment that actually runs the model):"
    # shellcheck disable=SC1091
    source "$(dirname "${BASH_SOURCE[0]}")/../la3b_env/bin/activate"
    check_torch
    deactivate
else
    check_torch
fi

# --- Verdict ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "WHAT TO DO NEXT"
echo "============================================================"
cat <<'GUIDE'
CASE 1 - no GPU visible at all ([1] and [2] both found nothing)
---------------------------------------------------------------
  Add GPU passthrough to .devcontainer/devcontainer.json as a top-level key:
      "runArgs": ["--gpus", "all"],
  then Ctrl+Shift+P -> "Dev Containers: Rebuild Container".
  Requires Docker Desktop on the WSL2 backend + a recent NVIDIA driver on the
  Windows host (driver-side; you do NOT install a driver in the container).

CASE 2 - GPU IS visible but VRAM is under ~8 GB  <-- MEASURED ON THIS MACHINE
------------------------------------------------------------------------------
  Passthrough is already working. Do NOT rebuild the container - that would
  change nothing. The constraint is memory capacity:

      LocateAnything-3B weights on disk : ~7.7 GB (bf16, 2 safetensors shards)
      RTX 3050 Ti Laptop VRAM           :  4.3 GB total, ~3.4 GB free

  The weights alone need roughly double the available VRAM. It does not fit.

  Why it did not crash with an out-of-memory error: recent NVIDIA drivers on
  Windows/WSL2 enable "CUDA system memory fallback" by default. Instead of
  failing, the allocation silently spills into host RAM and is shuttled over
  PCIe on every access. That is almost certainly the real cause of the
  25.6 s/frame figure - the GPU is technically in use, but is starved.

  Options, in order of how much they cost you:

  (a) ACCEPT AND REPORT IT - recommended with a deadline close.
      Costs nothing, changes no conclusion. Suggested wording:
        "Inference was measured at 25.6 s per frame. LocateAnything-3B
         requires approximately 7.7 GB for its weights in bf16, exceeding the
         4 GB VRAM of the available RTX 3050 Ti Laptop GPU; the model
         therefore ran with system-memory fallback rather than fully resident
         on the GPU. Even allowing one to two orders of magnitude speed-up on
         hardware with sufficient VRAM, this remains far below the rate
         required for reactive navigation."

  (b) 4-BIT QUANTISATION - only if you have time to spare.
        pip install bitsandbytes accelerate
      then load with load_in_4bit=True. This shrinks the weights to roughly
      1.9 GB, which fits comfortably, and should give genuine GPU speed.
      CAVEATS: quantisation slightly changes the model's outputs, so the
      30-frame accuracy comparison must be re-run and the quantisation must be
      disclosed in the write-up. Also, LocateAnything-3B uses custom remote
      code, so bitsandbytes integration is not guaranteed to work first time.

  (c) RUN ON A DIFFERENT MACHINE with >= 8 GB VRAM, if the lab has one.

  Whichever you choose, the conclusion is unchanged: even at 1 s/frame,
  LocateAnything-3B is hundreds of times too slow for a 10-30 Hz reactive
  navigation loop, so YOLOv8n stays in the live system.
GUIDE
