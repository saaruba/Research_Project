#!/usr/bin/env bash
# ============================================================================
# Run the whole experiment unattended.
#
#     bash scripts/run_trials.sh                       # 5 each of rule, bc, mlp
#     bash scripts/run_trials.sh restaurant_testing 5 "rule bc mlp"
#     bash scripts/run_trials.sh restaurant_testing 3 "rule bc"
#
# Args:  [world]  [trials per policy]  [policies]
#
# Terminal 1 must ALREADY be running the simulation:
#     bash scripts/run_everything.sh <world> rule --no-pipeline
#
# ----------------------------------------------------------------------------
# WHY BATCH
# ----------------------------------------------------------------------------
# Each trial takes about 5 minutes and ends itself, but it still needed a
# person to notice and press Ctrl-C. Fifteen trials is over an hour of sitting
# and watching. This loops them instead, resetting the robot to the same start
# pose before each one, and stops early if the simulation dies rather than
# grinding out a dozen empty result files.
#
# WHAT COMES OUT
#     dataset/processed/sim_results/<policy>_<timestamp>.json   - the metrics
#     dataset/processed/sim_bags/<policy>_<timestamp>/          - the raw data
#
# Then:
#     python3 scripts/summarise_sim_results.py
# ============================================================================

set -o pipefail

WORLD="${1:-restaurant_testing}"
N="${2:-5}"
POLICIES="${3:-rule bc mlp}"
PROJECT=/workspaces/Research_Project

source /opt/ros/humble/setup.bash
[ -f "${PROJECT}/install/setup.bash" ] && source "${PROJECT}/install/setup.bash"

TOTAL=0
for p in $POLICIES; do TOTAL=$((TOTAL + N)); done

echo "============================================================"
echo "  EXPERIMENT: ${WORLD}"
echo "  ${N} trial(s) each of: ${POLICIES}"
echo "  ${TOTAL} trials, roughly $((TOTAL * 5)) minutes"
echo "  detector: ${DETECTOR:-yolo}"
echo "  results : ${RESULTS_DIR:-dataset/processed/sim_results}"
echo "============================================================"

# Fail fast rather than producing a pile of meaningless files.
if ! timeout 30 ros2 topic echo /mobile_base_controller/odom --once >/dev/null 2>&1; then
    echo "" >&2
    echo "The simulation does not appear to be running." >&2
    echo "Start it in another terminal first:" >&2
    echo "  bash scripts/run_everything.sh ${WORLD} rule --no-pipeline" >&2
    exit 1
fi

DONE=0
FAILED=0
START_ALL=$(date +%s)

for policy in $POLICIES; do
    for i in $(seq 1 "$N"); do
        DONE=$((DONE + 1))
        echo ""
        echo "############################################################"
        echo "#  TRIAL ${DONE}/${TOTAL}   policy=${policy}   run ${i}/${N}"
        echo "############################################################"

        if ! pgrep -f gzserver >/dev/null 2>&1; then
            echo "gzserver is gone - the simulation died. Stopping here." >&2
            echo "Completed ${DONE} of ${TOTAL} trials before the failure." >&2
            break 2
        fi

        bash "${PROJECT}/scripts/run_pipeline.sh" "$WORLD" "$policy" \
            || { echo "  trial reported an error"; FAILED=$((FAILED + 1)); }

        # Let the graph settle so the next trial starts clean.
        sleep 10
    done
done

ELAPSED=$(( $(date +%s) - START_ALL ))
echo ""
echo "============================================================"
echo "  FINISHED: ${DONE} trial(s) in $((ELAPSED / 60)) min"
[ "$FAILED" -gt 0 ] && echo "  ${FAILED} reported an error - check them before using"
echo ""
echo "  Results:"
ls -1t "${RESULTS_DIR:-${PROJECT}/dataset/processed/sim_results}"/*.json 2>/dev/null \
    | head -"$TOTAL" | sed 's|.*/|    |'
echo ""
echo "  Aggregate them:"
echo "    python3 scripts/summarise_sim_results.py"
echo "============================================================"
