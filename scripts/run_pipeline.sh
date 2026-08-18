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
    echo "[2] No ground truth at $GT" >&2
    echo "    Generate it:" >&2
    echo "      python3 scripts/extract_world_groundtruth.py \\" >&2
    echo "          --world ${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.world" >&2
    exit 1
fi

NPEOPLE=$(python3 -c "
import json; d=json.load(open('$GT'))
print(sum(g['num_people'] for g in d['groups']))" 2>/dev/null || echo 2)
MINSIZE=2
[ "${NPEOPLE:-2}" -lt 2 ] && MINSIZE=1

echo ""
echo "[2] World has ${NPEOPLE} person(s) -> min_group_size=${MINSIZE}"

# --- Go ----------------------------------------------------------------------
echo ""
echo "[3] Launching perception -> policy -> Nav2 -> metrics"
echo "    (output is live below - Ctrl-C stops the behaviour only,"
echo "     the simulation in the other terminal keeps running)"
echo ""
echo "    Look for:"
echo "      'Group centroid (...) -> approach pose (...)'"
echo "      'Nav2 goal accepted - robot is moving.'"
echo "------------------------------------------------------------"

exec ros2 launch tiago_group_approach group_approach.launch.py \
    policy:="$POLICY" \
    min_group_size:="$MINSIZE" \
    groundtruth:="$GT"
