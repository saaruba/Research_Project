#!/usr/bin/env bash
# ============================================================================
# GUIDED DEMONSTRATION RUN - for screen recording
#
#   bash scripts/record_demo.sh offline     # ~3 min, terminal only
#   bash scripts/record_demo.sh yolo        # ~18 min, 3 trials
#   bash scripts/record_demo.sh la          # ~18 min, 3 trials
#   bash scripts/record_demo.sh results     # ~2 min, terminal only
#   bash scripts/record_demo.sh all         # everything, ~40 min
#
# ----------------------------------------------------------------------------
# WHAT THIS IS FOR
# ----------------------------------------------------------------------------
# Demonstrating this project means showing several separate things: what the
# data looks like, how the models score offline, and then the robot actually
# driving under three different policies and two different detectors. Typing
# all of that live is slow and easy to fumble on camera.
#
# This script runs the whole demonstration in a fixed order, printing a large
# labelled banner before each step so the recording has clear chapters, and
# pausing between them so you can narrate or move a window before continuing.
#
# It runs ONE trial per policy, not ten. The reported results come from the
# full 60-trial batch already in dataset/processed/results_FINAL_20260824;
# these single runs are to show the behaviour, not to re-measure it. Demo
# output is written to a separate folder and cannot overwrite anything.
#
# ----------------------------------------------------------------------------
# BEFORE YOU RECORD
# ----------------------------------------------------------------------------
# The offline and results parts need nothing running.
#
# The yolo and la parts need the simulation up in another terminal:
#     bash scripts/run_everything.sh restaurant_testing rule --no-pipeline
#
# The la part additionally needs the detector service, in its own terminal:
#     source la3b_env/bin/activate
#     python3 scripts/locateanything_service.py
#
# ----------------------------------------------------------------------------
# OPTIONS
# ----------------------------------------------------------------------------
#   NO_PAUSE=1   do not wait for a keypress between steps (unattended)
#   PAUSE_SECS=8 auto-continue after N seconds instead of waiting for a key
# ============================================================================

set -o pipefail

PART="${1:-all}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT" || exit 1

DEMO_DIR="$PROJECT/dataset/processed/demo_$(date +%Y%m%d)"
GT="$PROJECT/src/tiago_social_worlds/worlds/restaurant_testing.groundtruth.json"
FINAL="$PROJECT/dataset/processed/results_FINAL_20260824"

# ---------------------------------------------------------------- presentation
banner() {
    echo ""
    echo "################################################################################"
    printf "#  %-76s#\n" ""
    printf "#  %-76s#\n" "$1"
    [ -n "$2" ] && printf "#  %-76s#\n" "$2"
    printf "#  %-76s#\n" ""
    echo "################################################################################"
    echo ""
}

step() { echo ""; echo "-------- $1"; echo ""; }

pause() {
    [ -n "$NO_PAUSE" ] && { sleep 2; return; }
    echo ""
    if [ -n "$PAUSE_SECS" ]; then
        echo "    ... continuing in ${PAUSE_SECS}s"
        sleep "$PAUSE_SECS"
    else
        echo "    [ press ENTER to continue ]"
        read -r _
    fi
}

countdown() {
    echo ""
    echo "    Switch to the Gazebo / RViz window now."
    for i in 5 4 3 2 1; do printf "\r    starting in %d " "$i"; sleep 1; done
    printf "\r    running...        \n\n"
}

sim_is_up() {
    timeout 10 ros2 topic echo /mobile_base_controller/odom --once >/dev/null 2>&1
}

# ============================================================================
# PART 1 - OFFLINE: the data, the models, the validation
# ============================================================================
part_offline() {
    banner "PART 1  -  OFFLINE" "The data, the trained models, and how they were validated"
    pause

    step "1.1  What the robot learned from"
    python3 - <<'PY'
import pandas as pd
d = pd.read_csv("dataset/processed/approach_pose_dataset.csv")
TR=[1,3,7,8,11,12,26,27,28,30,31,49,51,52,55,58,60]; VA=[10,14,15,54]; TE=[5,9,59]
print(f"  PLUS-HRI recordings used     : {d.session_id.nunique()} sessions")
print(f"  Moments inside an approach   : {len(d):,} rows")
print()
print("  Split by WHOLE SESSION, never by row, so the test set contains")
print("  rooms and people the model has never seen:")
for name, s in (("train", TR), ("validation", VA), ("test", TE)):
    print(f"     {name:<11} {len(s):>2} sessions   {len(d[d.session_id.isin(s)]):>7,} rows")
print()
print("  The model sees SEVEN NUMBERS per moment - no images:")
for c in ["lidar_min_range","lidar_mean_range","linear_x_prev",
          "angular_z_prev","num_people","group_bearing_rad","group_scale_norm"]:
    print(f"     {c}")
print()
print("  And predicts where to stop, relative to where it is now:")
print("     target_dx, target_dy, target_dyaw")
PY
    pause

    step "1.2  How the models score on sessions they have never seen"
    echo "    Objective 4 thresholds: position < 0.4 m, orientation < 20 deg"
    echo ""
    python3 scripts/evaluate_approach_pose.py 2>&1 | tail -25
    pause

    step "1.3  Objective 3 - does the group detector find the conversation?"
    echo "    Scored against 30 hand-labelled frames from 18 different sessions."
    echo ""
    python3 scripts/validate_ospace_estimate.py 2>&1 | tail -22
    pause

    step "1.4  The unit test for the threaded perception mode"
    python3 tests/test_periodic_perception.py 2>&1 | tail -9
    pause
}

# ============================================================================
# PART 2 / 3 - LIVE TRIALS
# ============================================================================
run_live() {
    local detector="$1" oneshot="$2" label="$3" outdir="$4"

    if ! sim_is_up; then
        banner "SIMULATION IS NOT RUNNING" "Start it in another terminal, then re-run this part"
        echo "    bash scripts/run_everything.sh restaurant_testing rule --no-pipeline"
        echo ""
        return 1
    fi

    if [ "$detector" = "locateanything" ]; then
        if ! curl -s --max-time 5 http://127.0.0.1:8765/health >/dev/null 2>&1; then
            banner "LOCATEANYTHING SERVICE IS NOT RUNNING" "Start it in its own terminal first"
            echo "    source la3b_env/bin/activate"
            echo "    python3 scripts/locateanything_service.py"
            echo ""
            return 1
        fi
    fi

    mkdir -p "$outdir"

    for pol in rule bc_ft mlp_ft; do
        case "$pol" in
          rule)   desc="RULE BASELINE - hand-coded geometry, no learning" ;;
          bc_ft)  desc="BEHAVIOURAL CLONING - Random Forest" ;;
          mlp_ft) desc="BEHAVIOURAL CLONING - Multi-Layer Perceptron" ;;
        esac
        banner "$label  |  $desc" "One full patrol: find groups, approach, dwell 30 s, move on"
        pause
        countdown

        MIN_GROUP_SIZE=2 \
        RESULTS_DIR="$outdir" \
        DETECTOR="$detector" \
        ONESHOT="$oneshot" \
        RETRIGGER_PERIOD_S=2 \
        DWELL_TIME_S=30 \
        MAX_APPROACH_TIME=60 \
        bash scripts/run_pipeline.sh restaurant_testing "$pol"

        step "Trial finished - $pol under $label"
        pause
    done
}

part_yolo() {
    banner "PART 2  -  LIVE, YOLOv8n" "3.2 M parameters, about 5 ms per frame, runs at 2 Hz"
    echo "    Watch for:"
    echo "      - green boxes on people in the RViz camera panel"
    echo "      - the robot driving to a gap in the group, not into the middle"
    echo "      - it stopping and holding position for 30 seconds"
    echo "      - then leaving and continuing its patrol"
    pause
    run_live yolo auto "YOLOv8n" "$DEMO_DIR/yolo"
}

part_la() {
    banner "PART 3  -  LIVE, LocateAnything-3B" "3 B parameters, a vision-language model, runs at 0.5 Hz"
    echo "    Same three policies, same world, only the detector has changed."
    echo ""
    echo "    Watch for:"
    echo "      - 'PERIODIC: look complete in N s' in the terminal"
    echo "      - detection happening far less often than with YOLO"
    echo "      - the rule baseline coping worst with the slower perception"
    pause
    run_live locateanything periodic "LocateAnything-3B" "$DEMO_DIR/locateanything"
}

# ============================================================================
# PART 4 - RESULTS
# ============================================================================
part_results() {
    banner "PART 4  -  RESULTS" "From the full 60-trial experiment, not the demo runs"

    step "4.1  YOLOv8n - 30 trials"
    python3 scripts/summarise_sim_results.py --results-dir "$FINAL/yolo" 2>&1 | tail -30
    pause

    step "4.2  LocateAnything-3B - 30 trials"
    python3 scripts/summarise_sim_results.py --results-dir "$FINAL/locateanything" 2>&1 | tail -30
    pause

    step "4.3  The headline finding"
    cat <<'TXT'
    Position error       Orientation error
    (lower is better)    (lower is better)

    BC - MLP     0.158 m        79.1 deg     best placement, worst facing
    BC - RF      0.236 m        74.7 deg
    Rule         0.250 m        64.6 deg     worst placement, best facing

    The two columns rank in OPPOSITE ORDERS, and the same inversion appears
    under both detectors independently.

    Learning from human demonstrations produces better POSITIONING than a
    hand-coded rule, but worse ORIENTATION - because the rule simply turns to
    face the group by construction and cannot be badly wrong, while the
    learned models have to predict the facing and that was always their
    weakest output.

    That is the argument for a hybrid: learned position, geometric orientation.
TXT
    pause

    step "4.4  The figures"
    ls -1 "$FINAL/figures"/*.png | sed 's|.*/|    |'
    echo ""
    echo "    Open these in the file manager to show them on camera."
    pause
}

# ============================================================================
case "$PART" in
    offline) part_offline ;;
    yolo)    part_yolo ;;
    la)      part_la ;;
    results) part_results ;;
    all)     part_offline; part_yolo; part_la; part_results ;;
    *)
        echo "usage: bash scripts/record_demo.sh [offline|yolo|la|results|all]"
        exit 1 ;;
esac

banner "DEMONSTRATION COMPLETE" "Demo trial output: $DEMO_DIR"
