#!/usr/bin/env bash
# ============================================================================
# THE ROBOT'S BEHAVIOUR, ON ITS OWN.
#
#     bash scripts/run_pipeline.sh                      # restaurant_testing, rule
#     bash scripts/run_pipeline.sh restaurant_testing bc
#
# Run this in a SECOND terminal, after the simulation is up:
#
#     terminal 1:  bash scripts/run_everything.sh restaurant_testing rule --no-pipeline
#     terminal 2:  bash scripts/run_pipeline.sh
#
# ----------------------------------------------------------------------------
# WHY SEPARATE THIS OUT
# ----------------------------------------------------------------------------
# Bring-up (Gazebo, TIAGo, Nav2, the map, localisation) and BEHAVIOUR
# (perception -> group detection -> policy -> Nav2 goal) are different things
# that fail for different reasons. Bundled together, a bring-up problem and a
# behaviour problem look identical: "the robot isn't moving".
#
# Split, the bring-up runs once and stays up, and the behaviour can be
# restarted as often as you like - including switching policy:=rule to
# policy:=bc for the Objective 4 comparison - without paying the 2-3 minute
# Gazebo start each time.
# ============================================================================

set -o pipefail

WORLD="${1:-restaurant_testing}"
POLICY="${2:-rule}"
PROJECT=/workspaces/Research_Project
GT="${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.groundtruth.json"

source /opt/ros/humble/setup.bash
[ -f "${PROJECT}/install/setup.bash" ] && source "${PROJECT}/install/setup.bash"

echo "============================================================"
echo "  GROUP-APPROACH PIPELINE   world=${WORLD}  policy=${POLICY}"
echo "============================================================"

# --- Is the simulation actually up? -----------------------------------------
# Checked explicitly, because running this against a dead simulation is how
# the pipeline ends up sitting on "Waiting for CameraInfo..." forever.
echo ""
echo "[1] Checking the simulation is alive..."
python3 - <<'PY' || { echo "" >&2; echo "    Start the simulation FIRST, in another terminal, and wait for it to" >&2; echo "    print 'Simulation, map and localisation are up':" >&2; echo "      bash scripts/run_everything.sh restaurant_testing rule --no-pipeline" >&2; exit 1; }
import sys, time
import rclpy
from nav_msgs.msg import Odometry

TOPIC = '/mobile_base_controller/odom'

rclpy.init()
n = rclpy.create_node('pipeline_precheck')
got = {'v': False}
n.create_subscription(Odometry, TOPIC, lambda _: got.__setitem__('v', True), 10)

# 60 s, not 30. Discovery is slow on this container, and a precheck that gives
# up early sends you off restarting a simulation that was actually fine.
end = time.time() + 60
last = 0
while time.time() < end and not got['v']:
    rclpy.spin_once(n, timeout_sec=0.5)
    el = int(60 - (end - time.time()))
    if el >= last + 15:
        last = el
        print(f"    {el}s - publishers on {TOPIC}: {n.count_publishers(TOPIC)}",
              flush=True)

if got['v']:
    print("    OK - odometry is flowing")
else:
    pubs = n.count_publishers(TOPIC)
    print(f"    NO ODOMETRY after 60 s (publishers seen: {pubs})")
    if pubs > 0:
        print("    A publisher EXISTS but no message arrived - the simulation is")
        print("    probably still starting up. Wait a moment and try again.")
    else:
        print("    Nothing is publishing odometry at all.")
rclpy.shutdown()
sys.exit(0 if got['v'] else 1)
PY

# --- Ground truth ------------------------------------------------------------
if [ ! -f "$GT" ]; then
    echo "" >&2
    echo "[3] No ground truth at $GT" >&2
    echo "    Generate it:" >&2
    echo "      python3 scripts/extract_world_groundtruth.py \\" >&2
    echo "          --world ${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.world" >&2
    exit 1
fi

# --- Reset the robot to the same start pose ---------------------------------
# EVERY TRIAL MUST START IDENTICALLY.
#
# Terminal 1 stays up across trials, so without this the robot begins each run
# wherever the last one left it. Measured directly: one trial's very first
# trajectory sample was already (-2.46, -2.48) - standing among the people,
# 0.7 m from someone, before the policy had done anything. Metrics from a run
# like that say nothing about the policy.
#
# Teleporting the robot back to the origin makes rule and bc trials comparable,
# which is the whole basis of the Objective 4 comparison.
echo ""
echo "[2] Resetting the robot to the start pose (0, 0)..."
python3 - <<'PY' || echo "    WARNING: could not reset - trials may not be comparable"
import sys
import rclpy
from gazebo_msgs.srv import SetEntityState

rclpy.init()
n = rclpy.create_node('trial_reset')
cli = n.create_client(SetEntityState, '/gazebo/set_entity_state')

if not cli.wait_for_service(timeout_sec=15.0):
    print("    /gazebo/set_entity_state unavailable "
          "(is gazebo_ros_state in the world file?)")
    rclpy.shutdown(); sys.exit(1)

req = SetEntityState.Request()
req.state.name = 'tiago'
req.state.pose.position.x = 0.0
req.state.pose.position.y = 0.0
req.state.pose.position.z = 0.0
req.state.pose.orientation.w = 1.0
req.state.reference_frame = 'world'

fut = cli.call_async(req)
rclpy.spin_until_future_complete(n, fut, timeout_sec=15.0)
res = fut.result()
ok = bool(res and res.success)
print("    robot reset to (0, 0) facing +x" if ok
      else f"    reset rejected: {getattr(res, 'status_message', 'no response')}")
rclpy.shutdown()
sys.exit(0 if ok else 1)
PY
sleep 3

# min_group_size must match the SMALLEST group in the world, not the total
# headcount. A world with two conversations and one person standing alone has
# seven people but a smallest group of one - and with min_group_size=2 the lone
# person is invisible to the pipeline and never approached.
read -r NPEOPLE NGROUPS MINSIZE <<< "$(python3 -c "
import json
d = json.load(open('$GT'))
g = d['groups']
sizes = [x['num_people'] for x in g] or [2]
print(sum(sizes), len(g), max(1, min(sizes)))" 2>/dev/null || echo "2 1 2")"

echo ""
echo "[3] World has ${NPEOPLE} person(s) in ${NGROUPS} group(s)"
echo "    smallest group = ${MINSIZE} -> min_group_size=${MINSIZE}"

# --- Go ----------------------------------------------------------------------
echo ""
echo "[4] Launching perception -> policy -> Nav2 -> metrics"
echo "    (output is live below - Ctrl-C stops the behaviour only,"
echo "     the simulation in the other terminal keeps running)"
echo ""
echo "    Look for:"
echo "      'Group centroid (...) -> approach pose (...)'"
echo "      'Nav2 goal accepted - robot is moving.'"
echo "------------------------------------------------------------"

# --- Optional: record a rosbag of the trial ---------------------------------
# BAG=0 to skip. Recording costs a little disk and nothing else, and it makes
# each trial reproducible after the fact: you can replay exactly what the robot
# saw and did without re-running the simulation.
#
# NOTE ON RE-TRAINING FROM THESE BAGS - see docs. They are EVIDENCE, not
# training data. Behavioural Cloning learns to copy a demonstrator; a bag of
# the robot copying itself contains no correction signal, so training on it
# reinforces current behaviour, mistakes included. Improving from experience
# needs either human corrections (DAgger) or a reward signal (RL).
if [ "${BAG:-1}" = "1" ]; then
    BAGDIR="${PROJECT}/dataset/processed/sim_bags/${POLICY}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$(dirname "$BAGDIR")"
    echo ""
    echo "[5] Recording a bag to ${BAGDIR}"
    ros2 bag record -o "$BAGDIR" \
        /scan_raw \
        /mobile_base_controller/odom \
        /mobile_base_controller/cmd_vel_unstamped \
        /group_centroid \
        /detected_people \
        /tf /tf_static \
        /head_front_camera/rgb/camera_info \
        > /tmp/rosbag.log 2>&1 &
    BAG_PID=$!
    trap 'echo ""; echo "Stopping the bag recorder..."; kill -INT '"$BAG_PID"' 2>/dev/null; sleep 3' EXIT INT TERM
else
    echo ""
    echo "[5] Bag recording disabled (BAG=0)"
fi

echo "------------------------------------------------------------"

# --- Which model does this policy use? ---------------------------------------
MODELS="${PROJECT}/dataset/processed/models"
case "$POLICY" in
    mlp) MODEL_ARG="model_path:=${MODELS}/approach_pose_mlp_tuned.joblib" ;;
    bc)  MODEL_ARG="model_path:=${MODELS}/approach_pose_random_forest_tuned.joblib" ;;
    *)   MODEL_ARG="" ;;
esac
[ -n "$MODEL_ARG" ] && echo "    model: ${MODEL_ARG#model_path:=}"

RESULTS="${PROJECT}/dataset/processed/sim_results"
BEFORE=$(ls -1 "$RESULTS" 2>/dev/null | wc -l)

# --- Stall watchdog ----------------------------------------------------------
# Ends a trial that has gone dead, so an unattended overnight batch cannot lose
# hours to one wedged robot. If the base has not moved at all for STALL_TIMEOUT
# seconds, it publishes /metrics/finish - the same signal the mission sends on
# completion - so the recorder still writes a proper results file and the trial
# closes down cleanly rather than being killed mid-write.
#
# 60 s is comfortably longer than any legitimate pause: the dwell at a group is
# 6 s, a blocked waypoint waits 7 s, and the unwedge reflex fires at 18 s.
STALL_TIMEOUT="${STALL_TIMEOUT:-60}"
python3 - "$STALL_TIMEOUT" > /tmp/stall_watchdog.log 2>&1 <<'PY' &
import sys, time, math
import rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

limit = float(sys.argv[1])
rclpy.init()
n = rclpy.create_node('stall_watchdog')
pub = n.create_publisher(Empty, '/metrics/finish', 10)
state = {'xy': None, 'moved': time.time()}

def cb(msg):
    p = msg.pose.pose.position
    now = time.time()
    if state['xy'] is None or math.dist((p.x, p.y), state['xy']) > 0.05:
        state['xy'] = (p.x, p.y)
        state['moved'] = now

n.create_subscription(Odometry, '/mobile_base_controller/odom', cb, 10)

while rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.5)
    if state['xy'] is not None and time.time() - state['moved'] > limit:
        print(f"STALLED: no motion for {limit:.0f}s - ending the trial", flush=True)
        for _ in range(3):
            pub.publish(Empty())
            time.sleep(0.5)
        break
rclpy.shutdown()
PY
STALL_PID=$!

ros2 launch tiago_group_approach group_approach.launch.py \
    policy:="$POLICY" \
    min_group_size:="$MINSIZE" \
    groundtruth:="$GT" \
    $MODEL_ARG &
LAUNCH_PID=$!

# --- Stop when the trial has actually finished -------------------------------
# The mission ends itself and the recorder writes a results file, but ros2
# launch keeps the nodes alive afterwards, so every trial needed a manual
# Ctrl-C. Watching for a NEW file in sim_results/ is a reliable completion
# signal and makes unattended batches possible - see scripts/run_trials.sh.
# AUTO_EXIT=0 restores the old behaviour if you want to watch a run.
if [ "${AUTO_EXIT:-1}" = "1" ]; then
    LIMIT="${TRIAL_TIMEOUT:-1800}"
    START=$(date +%s)
    while kill -0 "$LAUNCH_PID" 2>/dev/null; do
        sleep 5
        NOW=$(ls -1 "$RESULTS" 2>/dev/null | wc -l)
        if [ "$NOW" -gt "$BEFORE" ]; then
            echo ""
            echo "    Trial complete - results written. Shutting the trial down."
            sleep 8            # let the recorder flush and the bag close
            kill -INT "$LAUNCH_PID" 2>/dev/null

            # Make sure it ACTUALLY dies. A SIGINT to ros2 launch does not
            # always bring its children down: one trial's policy node outlived
            # its own trial by six minutes, still issuing recovery commands to
            # a robot that had finished. Leftover nodes would then interfere
            # with the next trial in a batch.
            for i in $(seq 1 15); do
                kill -0 "$LAUNCH_PID" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$LAUNCH_PID" 2>/dev/null; then
                echo "    (launch ignored SIGINT - forcing it down)"
                kill -9 "$LAUNCH_PID" 2>/dev/null
            fi
            pkill -9 -f group_perception_node 2>/dev/null
            pkill -9 -f group_approach_baseline_node 2>/dev/null
            pkill -9 -f bc_policy_node 2>/dev/null
            pkill -9 -f mission_node 2>/dev/null
            pkill -9 -f metrics_recorder_node 2>/dev/null
            kill "$STALL_PID" 2>/dev/null
            sleep 3
            break
        fi
        if [ $(( $(date +%s) - START )) -gt "$LIMIT" ]; then
            echo "" >&2
            echo "    TIMEOUT after ${LIMIT}s with no results file - stopping." >&2
            kill -INT "$LAUNCH_PID" 2>/dev/null
            break
        fi
    done
fi

# ESCALATE if SIGINT is ignored.
#
# `ros2 launch` does not always pass SIGINT on cleanly - the perception node in
# particular has needed SIGKILL before. Without escalation the script waits
# forever on a launch that will never exit, which stalls an entire unattended
# batch after its first trial.
for i in $(seq 1 20); do
    kill -0 "$LAUNCH_PID" 2>/dev/null || break
    sleep 1
done
if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "    Launch ignored SIGINT after 20s - terminating it."
    kill -TERM "$LAUNCH_PID" 2>/dev/null
    sleep 5
    kill -9 "$LAUNCH_PID" 2>/dev/null
fi

# Anything the launch left behind would collide with the next trial's nodes.
pkill -9 -f group_perception_node 2>/dev/null
pkill -9 -f group_approach_baseline_node 2>/dev/null
pkill -9 -f bc_policy_node 2>/dev/null
pkill -9 -f mission_node 2>/dev/null
pkill -9 -f metrics_recorder_node 2>/dev/null
sleep 3

wait "$LAUNCH_PID" 2>/dev/null
echo ""
echo "Latest result:"
ls -t "$RESULTS"/*.json 2>/dev/null | head -1
