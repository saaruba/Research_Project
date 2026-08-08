#!/usr/bin/env bash
# ============================================================================
# TIAGo Group-Approach Project - one-command environment setup
#
#   ./setup_environment.sh              install the main data pipeline
#   ./setup_environment.sh --venv       install into a fresh ./venv instead
#   ./setup_environment.sh --check      verify an existing install, change nothing
#   ./setup_environment.sh --cpu-torch  install CPU-only PyTorch first (~200 MB
#                                       instead of the ~4 GB CUDA stack that
#                                       ultralytics would otherwise pull in)
#
# This sets up the DATA PIPELINE only (extraction, detection, clustering,
# training, evaluation). It deliberately does NOT install:
#   - ROS 2 packages   -> not on PyPI; come from the ROS 2 Humble system
#                         install, and are built with `colcon build`
#   - LocateAnything-3B -> conflicting numpy pin; needs its own venv, see
#                          requirements-locateanything.txt
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USE_VENV=0
CHECK_ONLY=0
CPU_TORCH=0
for arg in "$@"; do
    case "$arg" in
        --venv)      USE_VENV=1 ;;
        --check)     CHECK_ONLY=1 ;;
        --cpu-torch) CPU_TORCH=1 ;;
        -h|--help)
            sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown option: $arg (try --help)" >&2
            exit 1 ;;
    esac
done

echo "============================================================"
echo "TIAGo Group-Approach Project - environment setup"
echo "============================================================"

# --- Python version check ---------------------------------------------------
# Verified on 3.10; 3.9 lacks some typing syntax used in the scripts.
PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python version: $PY_VERSION"
if [ "$(printf '%s\n3.9\n' "$PY_VERSION" | sort -V | head -1)" != "3.9" ]; then
    echo "WARNING: Python 3.9+ required (3.10 recommended). Found $PY_VERSION." >&2
fi

verify_install() {
    echo ""
    echo "Verifying imports..."
    python3 - <<'PY'
import importlib, sys

# (import name, pip name, what breaks without it)
CHECKS = [
    ("numpy",   "numpy",                    "everything"),
    ("pandas",  "pandas",                   "everything"),
    ("sklearn", "scikit-learn",             "model training / grid search"),
    ("joblib",  "joblib",                   "saving + loading trained models"),
    ("cv2",     "opencv-python-headless",   "video reading / frame export"),
    ("rosbags", "rosbags",                  "reading the ROS 1 dataset bags"),
    ("PIL",     "Pillow",                   "image loading"),
]
OPTIONAL = [
    ("ultralytics", "ultralytics",
     "re-running YOLO person detection (not needed if using cached detections)"),
]

failed = []
for module, pip_name, purpose in CHECKS:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        print(f"  OK       {pip_name:<26} {version}")
    except ImportError:
        print(f"  MISSING  {pip_name:<26} -> breaks: {purpose}")
        failed.append(pip_name)

for module, pip_name, purpose in OPTIONAL:
    try:
        m = importlib.import_module(module)
        print(f"  OK       {pip_name:<26} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  optional {pip_name:<26} not installed - only needed for: {purpose}")

# numpy 1.x here means a LocateAnything install has contaminated this env.
try:
    import numpy
    if numpy.__version__.startswith("1."):
        print("\n  !! WARNING: numpy 1.x detected in this environment.")
        print("     The data pipeline expects numpy 2.x. If you installed")
        print("     requirements-locateanything.txt here by mistake, that is the")
        print("     cause - it must live in its own separate virtual environment.")
except ImportError:
    pass

if failed:
    print("\nMissing required package(s): " + ", ".join(failed))
    sys.exit(1)
print("\nAll required packages present.")
PY
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    verify_install
    exit 0
fi

# --- Optional virtual environment -------------------------------------------
if [ "$USE_VENV" -eq 1 ]; then
    if [ ! -d venv ]; then
        echo ""
        echo "Creating virtual environment at ./venv ..."
        python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "Activated: $(which python3)"
    PIP_FLAGS=""
else
    # Outside a venv, Ubuntu 22.04 / Debian mark the system Python as
    # externally-managed and refuse plain `pip install`.
    PIP_FLAGS="--break-system-packages"
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        PIP_FLAGS=""   # already inside a venv, flag not needed
    fi
fi

# --- Install ----------------------------------------------------------------
echo ""
# shellcheck disable=SC2086
python3 -m pip install --upgrade pip $PIP_FLAGS

if [ "$CPU_TORCH" -eq 1 ]; then
    echo ""
    echo "Installing CPU-only PyTorch first (~200 MB) so that ultralytics does"
    echo "not pull in the ~4 GB CUDA stack..."
    # shellcheck disable=SC2086
    python3 -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu $PIP_FLAGS
fi

echo ""
echo "Installing from requirements.txt ..."
if [ "$CPU_TORCH" -eq 0 ]; then
    echo "NOTE: ultralytics depends on PyTorch, and on Linux pip resolves that to"
    echo "      the CUDA build (~3-4 GB with the NVIDIA libraries). If you do not"
    echo "      need to re-run YOLO detection, Ctrl-C now and either:"
    echo "        - re-run with --cpu-torch, or"
    echo "        - comment out the ultralytics line in requirements.txt"
    echo "      (cached detections mean everything downstream works without it)."
fi
echo ""

# shellcheck disable=SC2086
python3 -m pip install -r requirements.txt $PIP_FLAGS

verify_install

echo ""
echo "============================================================"
echo "Data pipeline ready."
echo ""
echo "NOT installed by this script (by design):"
echo "  * ROS 2 packages (rclpy, nav2_msgs, tf2_ros, ...)"
echo "      -> provided by the ROS 2 Humble system install."
echo "      -> build the project's nodes with:"
echo "           colcon build --packages-select tiago_group_approach"
echo "  * LocateAnything-3B (torch, transformers, ...)"
echo "      -> conflicting numpy pin; needs its own environment:"
echo "           python3 -m venv la3b_env && source la3b_env/bin/activate"
echo "           pip install -r requirements-locateanything.txt"
echo "============================================================"
