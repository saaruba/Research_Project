#!/usr/bin/env bash
# ============================================================================
# THE WHOLE EXPERIMENT, UNATTENDED.
#
#     bash scripts/run_overnight.sh
#     bash scripts/run_overnight.sh restaurant_testing 10
#
# Args: [world] [trials per policy]
#
# One command, no supervision. It starts the simulation, runs every policy with
# YOLOv8n, then repeats the whole set with LocateAnything-3B, and writes a
# summary for each. Results are kept in SEPARATE folders so the two detectors
# can never be mixed in one table:
#
#     dataset/processed/results_yolo/            + summary.txt
#     dataset/processed/results_locateanything/  + summary.txt
#     /tmp/overnight.log                          full transcript
#
# ----------------------------------------------------------------------------
# DESIGN RULE: THE YOLO RESULTS MUST SURVIVE ANYTHING
# ----------------------------------------------------------------------------
# The YOLO batch is the dissertation's primary result. LocateAnything-3B is an
# additional demonstration that closes a gap in the original proposal. So the
# LA-3B phase is written to fail SOFTLY: every step that could hang - the venv,
# the CUDA check, the model download, the service coming up - is bounded by a
# timeout, and any failure skips straight to the summaries with the YOLO data
# already written and safe.
#
# Nothing here overwrites earlier results. Both folders are new.
# ============================================================================

set -o pipefail

WORLD="${1:-restaurant_testing}"
N="${2:-10}"
POLICIES="${POLICIES:-rule bc mlp}"
PROJECT=/workspaces/Research_Project

YOLO_DIR="${PROJECT}/dataset/processed/results_yolo"
LA_DIR="${PROJECT}/dataset/processed/results_locateanything"
GT="${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.groundtruth.json"

# Conversational groups only - a lone individual has no O-space, so approaching
# one cannot demonstrate group-approach behaviour.
export MIN_GROUP_SIZE="${MIN_GROUP_SIZE:-2}"

mkdir -p "$YOLO_DIR" "$LA_DIR"

source /opt/ros/humble/setup.bash
[ -f "${PROJECT}/install/setup.bash" ] && source "${PROJECT}/install/setup.bash"

say() { echo ""; echo "############################################################"; \
        echo "#  $*"; echo "############################################################"; }

SIM_PID=""
SVC_PID=""
cleanup() {
    echo ""
    echo "Shutting everything down..."
    [ -n "$SVC_PID" ] && kill "$SVC_PID" 2>/dev/null
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null
    sleep 3
    pkill -9 -f gzserver 2>/dev/null
    pkill -9 -f gzclient 2>/dev/null
    pkill -9 -f locateanything_service 2>/dev/null
}
trap cleanup EXIT INT TERM

say "OVERNIGHT EXPERIMENT   world=${WORLD}   ${N} trial(s) x ${POLICIES}"
echo "  min_group_size : ${MIN_GROUP_SIZE} (conversational groups only)"
echo "  YOLO results   : ${YOLO_DIR}"
echo "  LA-3B results  : ${LA_DIR}"
echo "  started        : $(date)"

# --- Preflight ---------------------------------------------------------------
if [ ! -f "$GT" ]; then
    echo "FATAL: no ground truth at $GT" >&2
    exit 1
fi
python3 -c "
import json,sys
d=json.load(open('$GT'))
n=[g for g in d['groups'] if g['num_people']>=${MIN_GROUP_SIZE}]
print(f'  targets        : {len(n)} group(s) with >= ${MIN_GROUP_SIZE} people')
sys.exit(0 if n else 1)" || { echo "FATAL: no groups meet MIN_GROUP_SIZE" >&2; exit 1; }

# --- 1. Simulation -----------------------------------------------------------
say "PHASE 1/5  Starting the simulation"
bash "${PROJECT}/scripts/run_everything.sh" "$WORLD" rule --no-pipeline \
    > /tmp/overnight_sim.log 2>&1 &
SIM_PID=$!

echo "  waiting for the robot (up to 8 minutes)..."
READY=0
for i in $(seq 1 96); do
    if timeout 5 ros2 topic echo /mobile_base_controller/odom --once >/dev/null 2>&1; then
        READY=1; break
    fi
    sleep 5
done
if [ "$READY" -eq 0 ]; then
    echo "FATAL: the simulation never came up. See /tmp/overnight_sim.log" >&2
    tail -30 /tmp/overnight_sim.log >&2
    exit 1
fi
echo "  simulation is up. Settling for 30 s..."
sleep 30

# --- 2. YOLO batch -----------------------------------------------------------
say "PHASE 2/5  YOLOv8n  -  ${N} trial(s) of each policy"
# Wall-clock budget. Every completed trial is written to disk as it finishes,
# so cutting the phase short loses nothing already recorded.
RESULTS_DIR="$YOLO_DIR" DETECTOR=yolo \
    timeout "${YOLO_BUDGET:-5h}" \
    bash "${PROJECT}/scripts/run_trials.sh" "$WORLD" "$N" "$POLICIES" \
    || echo "  (the YOLO batch ended early or reported errors - results kept)"

YOLO_COUNT=$(ls -1 "$YOLO_DIR"/*.json 2>/dev/null | wc -l)
echo ""
echo "  YOLO trials written: ${YOLO_COUNT}"

# --- 3. LocateAnything-3B service -------------------------------------------
# Everything below is optional and bounded. If any step fails we go straight to
# the summaries with the YOLO results intact.
say "PHASE 3/5  LocateAnything-3B  -  starting the service"
LA_OK=0

if [ ! -d "${PROJECT}/la3b_env" ]; then
    echo "  la3b_env not found - skipping LocateAnything."
else
    # shellcheck disable=SC1091
    source "${PROJECT}/la3b_env/bin/activate"
    CUDA=$(python3 -c "import torch;print(torch.cuda.is_available())" 2>/dev/null || echo False)
    echo "  CUDA available in la3b_env: ${CUDA}"

    if [ "$CUDA" != "True" ]; then
        echo "  Without CUDA the model runs at ~25 s/frame, which cannot drive"
        echo "  a 2 Hz control loop. Skipping LocateAnything."
        deactivate 2>/dev/null || true
    else
        python3 "${PROJECT}/scripts/locateanything_service.py" \
            > /tmp/la3b_service.log 2>&1 &
        SVC_PID=$!
        deactivate 2>/dev/null || true

        # Generous: the first start may download several GB of weights.
        echo "  waiting for the service (up to 25 minutes - first run downloads weights)..."
        for i in $(seq 1 150); do
            if curl -s --max-time 5 http://127.0.0.1:8765/health >/dev/null 2>&1; then
                LA_OK=1; break
            fi
            kill -0 "$SVC_PID" 2>/dev/null || { echo "  the service exited early."; break; }
            sleep 10
        done

        if [ "$LA_OK" -eq 1 ]; then
            echo "  service is up."
            grep -iE "s/frame|seconds per frame|device|cuda" /tmp/la3b_service.log \
                | tail -5 | sed 's/^/    /'
        else
            echo "  service did not become healthy. Last lines:" >&2
            tail -20 /tmp/la3b_service.log >&2
        fi
    fi
fi

# --- 4. LocateAnything batch -------------------------------------------------
if [ "$LA_OK" -eq 1 ]; then
    say "PHASE 4/5  LocateAnything-3B  -  ${N} trial(s) of each policy"
    # LocateAnything-3B is a 3-billion-parameter vision-language model doing
    # autoregressive decoding, so a frame costs far more than YOLOv8n's ~5 ms.
    # Its true rate is unknown until it runs, and at several seconds per frame
    # thirty trials could occupy fifteen hours. The budget stops the phase
    # cleanly and summarises whatever completed.
    # ------------------------------------------------------------------------
    # PERIODIC MODE (Aug 2026) - why this phase is no longer one-shot
    # ------------------------------------------------------------------------
    # One-shot mode detects ONCE per node lifetime, and nothing publishes
    # /perception/trigger, so the earlier LA-3B batch gave each trial exactly
    # one look from the start pose: 31 inferences across 30 trials. Every group
    # outside that single camera frustum was never seen, the policies had
    # nothing to act on, and the batch measured open-loop patrol rather than
    # approach behaviour - which is why all three policies scored alike.
    #
    # Periodic mode re-looks every RETRIGGER_PERIOD_S with the inference on a
    # worker thread, so the node keeps publishing instead of freezing for 8.4 s.
    # At ~8.4 s/inference and a ~300 s trial this yields roughly 30 looks per
    # trial instead of 1. It is still not closed-loop perception at 0.12 Hz,
    # and the headline finding is unchanged; it simply lets the comparison be
    # made fairly.
    #
    # Set ONESHOT=true to reproduce the original one-shot batch exactly.
    RESULTS_DIR="$LA_DIR" DETECTOR=locateanything \
        ONESHOT="${ONESHOT:-periodic}" \
        RETRIGGER_PERIOD_S="${RETRIGGER_PERIOD_S:-10.0}" \
        timeout "${LA_BUDGET:-4h}" \
        bash "${PROJECT}/scripts/run_trials.sh" "$WORLD" "$N" "$POLICIES" \
        || echo "  (the LA-3B batch ended early or reported errors - results kept)"
else
    say "PHASE 4/5  SKIPPED  -  LocateAnything-3B unavailable"
    echo "  The YOLO results are complete and unaffected."
    echo "  Diagnose later with: tail -50 /tmp/la3b_service.log"
fi

# --- 5. Summaries ------------------------------------------------------------
say "PHASE 5/5  Summaries"
for pair in "YOLOv8n:${YOLO_DIR}" "LocateAnything-3B:${LA_DIR}"; do
    name="${pair%%:*}"; dir="${pair##*:}"
    count=$(ls -1 "$dir"/*.json 2>/dev/null | wc -l)
    echo ""
    echo "=== ${name}  (${count} trial(s)) ==="
    if [ "$count" -eq 0 ]; then
        echo "  no trials"
        continue
    fi
    # Re-score every trial against ground truth with identical criteria, so the
    # two detectors are judged the same way.
    python3 "${PROJECT}/scripts/rescore_sim_results.py" \
        --results "$dir" --groundtruth "$GT" \
        --min-group-size "$MIN_GROUP_SIZE" --apply \
        > "${dir}/rescore.txt" 2>&1 || true
    python3 "${PROJECT}/scripts/summarise_sim_results.py" --results-dir "$dir" \
        2>&1 | tee "${dir}/summary.txt"
done

say "DONE  $(date)"
echo "  YOLO   : ${YOLO_DIR}/summary.txt"
echo "  LA-3B  : ${LA_DIR}/summary.txt"
echo "  Log    : /tmp/overnight.log"
