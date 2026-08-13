#!/usr/bin/env bash
# ============================================================================
# LocateAnything-3B environment setup - SAFE version
#
#   ./scripts/setup_locateanything.sh           create venv + install
#   ./scripts/setup_locateanything.sh --repair  fix a contaminated main env
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The obvious manual sequence is dangerous on this devcontainer:
#
#     python3 -m venv la3b_env          # <-- FAILS: python3.10-venv missing
#     source la3b_env/bin/activate      # <-- fails, no venv
#     pip install -r requirements-locateanything.txt   # <-- lands in ~/.local!
#
# Because the shell keeps going after each failure, the pip install silently
# targets your USER site-packages and downgrades numpy 2.2.6 -> 1.25.0,
# breaking pandas / scikit-learn / opencv-python for the entire data pipeline.
#
# This script makes that impossible: it verifies the venv exists and is
# active before installing anything, and aborts loudly otherwise.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/la3b_env"

# ---------------------------------------------------------------------------
# --repair : undo a contaminated main environment
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--repair" ]; then
    echo "============================================================"
    echo "REPAIRING the main data-pipeline environment"
    echo "============================================================"
    echo "Restoring numpy 2.x (LocateAnything's numpy 1.25.0 downgrade breaks"
    echo "pandas, scikit-learn and opencv-python)..."
    echo ""
    python3 -m pip install --user --force-reinstall \
        "numpy==2.2.6" "pandas==2.3.3" "scikit-learn==1.7.2"
    echo ""
    echo "Verifying..."
    python3 - <<'PY'
import importlib, sys
ok = True
for mod, expect in [("numpy", "2."), ("pandas", "2."), ("sklearn", "1.")]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        good = v.startswith(expect)
        print(f"  {'OK     ' if good else 'WRONG  '} {mod}=={v}")
        ok &= good
    except Exception as exc:
        print(f"  BROKEN  {mod}: {exc}")
        ok = False
try:
    import cv2, pandas  # the pair that actually breaks on an ABI mismatch
    print(f"  OK      cv2=={cv2.__version__} imports cleanly alongside pandas")
except Exception as exc:
    print(f"  BROKEN  cv2/pandas: {exc}")
    ok = False
sys.exit(0 if ok else 1)
PY
    echo ""
    echo "Main environment repaired. Now run this script with no arguments"
    echo "to set LocateAnything up properly, in its own venv."
    exit 0
fi

echo "============================================================"
echo "LocateAnything-3B setup (isolated environment)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Ensure a venv can actually be created
# ---------------------------------------------------------------------------
# A venv is only usable if bin/activate exists. A FAILED `python3 -m venv`
# still leaves the directory behind, so testing `-d "$VENV_DIR"` alone is not
# enough - that is exactly how this script failed the first time round.
create_venv() {
    local target="$1"
    echo "Creating virtual environment at $target ..."
    if python3 -m venv "$target" 2>/dev/null && [ -f "$target/bin/activate" ]; then
        return 0
    fi
    echo "  python3 -m venv failed (python3.10-venv is not installed)."
    echo "  Falling back to virtualenv, which needs no root..."
    rm -rf "$target" 2>/dev/null || true
    python3 -m pip install --user --quiet virtualenv
    if python3 -m virtualenv "$target" && [ -f "$target/bin/activate" ]; then
        return 0
    fi
    return 1
}

if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Reusing existing virtual environment at $VENV_DIR"
else
    if [ -d "$VENV_DIR" ]; then
        echo "Found a BROKEN/incomplete venv at $VENV_DIR (no bin/activate)."
        echo "This is the leftover of a failed 'python3 -m venv'. Removing it..."
        if ! rm -rf "$VENV_DIR" 2>/dev/null; then
            # Mounted volumes sometimes refuse deletion; just use a new name.
            VENV_DIR="$SCRIPT_DIR/la3b_env2"
            echo "Could not delete it (mounted volume?). Using $VENV_DIR instead."
        fi
    fi

    if ! create_venv "$VENV_DIR"; then
        echo "" >&2
        echo "ERROR: could not create a working virtual environment." >&2
        echo "Install the venv module and re-run:" >&2
        echo "    sudo apt update && sudo apt install -y python3.10-venv" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. Activate and HARD-VERIFY we are inside it before installing anything
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ACTIVE_PYTHON="$(command -v python3)"
echo "Active python3: $ACTIVE_PYTHON"

case "$ACTIVE_PYTHON" in
    "$VENV_DIR"/*) ;;
    *)
        echo "" >&2
        echo "ABORTING: python3 is not the one inside $VENV_DIR." >&2
        echo "Installing now would contaminate your main environment and" >&2
        echo "downgrade numpy, breaking the data pipeline." >&2
        exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 3. Install
# ---------------------------------------------------------------------------
echo ""
echo "Installing LocateAnything requirements into the venv..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements-locateanything.txt

echo ""
echo "Installing PyTorch..."
echo "NOTE: this pulls the default CUDA wheel. If your GPU needs a specific"
echo "      CUDA version, Ctrl-C and install it manually from"
echo "      https://pytorch.org/get-started/locally/"
python3 -m pip install torch

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo ""
echo "Verifying the environment..."
python3 - <<'PY'
import sys
import numpy, transformers
print(f"  python       {sys.executable}")
print(f"  numpy        {numpy.__version__}   (1.25.0 expected HERE - correct inside this venv)")
print(f"  transformers {transformers.__version__}")
try:
    import torch
    print(f"  torch        {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  !! No GPU visible. LocateAnything-3B will be extremely slow on CPU.")
except ImportError:
    print("  torch NOT installed")
import jinja2
print(f"  jinja2       {jinja2.__version__}   (needs >= 3.1.0 for apply_chat_template)")
PY

echo ""
echo "============================================================"
echo "Ready. Run the comparison with this venv ACTIVE:"
echo ""
echo "    source la3b_env/bin/activate"
echo "    python3 scripts/run_locateanything_comparison.py \\"
echo "        --frames-dir dataset/processed/detector_comparison"
echo ""
echo "When finished, leave the venv before using the data pipeline again:"
echo "    deactivate"
echo "============================================================"
