#!/usr/bin/env bash
# ============================================================================
# Serve the prebuilt map AND localise the robot on it, in one go.
#
#   bash scripts/start_localisation.sh
#   bash scripts/start_localisation.sh restaurant_final     # different world
#
# Run this AFTER the simulation is up and /mobile_base_controller/odom works.
#
# ----------------------------------------------------------------------------
# THE TWO PROBLEMS THIS SOLVES
# ----------------------------------------------------------------------------
# 1. map_server is a LIFECYCLE node. Starting it is not enough - it sits in
#    "Waiting on external lifecycle transitions to activate" and publishes
#    nothing until something configures and activates it. Running
#        ros2 run nav2_map_server map_server ... && ros2 lifecycle set ...
#    does not work, because map_server never exits, so the lifecycle commands
#    on the same line never run. It must go to the BACKGROUND first.
#
# 2. With a prebuilt map (slam:=False), AMCL has no idea where the robot is
#    until it is given an initial pose. Until then there is no map -> odom
#    transform, so Nav2 reports:
#        Invalid frame ID "map" passed to canTransform
#        Message Filter dropping message: frame 'odom' ... queue is full
#    and RViz shows Global Status: Error with an empty grid.
#
# Rather than asking you to click "2D Pose Estimate" and guess, this script
# reads the robot's TRUE pose from Gazebo (via /gazebo/model_states, published
# by the gazebo_ros_state plugin in the world) and publishes it to /initialpose.
# AMCL then starts perfectly localised, with no manual clicking and no
# estimation error introduced by a shaky mouse.
# ============================================================================

set -o pipefail   # not -u: ROS setup.bash reads undefined variables

WORLD="${1:-restaurant_testing}"
PROJECT=/workspaces/Research_Project
MAP_YAML="${PROJECT}/src/tiago_social_worlds/maps/${WORLD}.yaml"

echo "============================================================"
echo "LOCALISATION BRINGUP  -  world: ${WORLD}"
echo "============================================================"

if [ ! -f "$MAP_YAML" ]; then
    echo "ERROR: map not found: $MAP_YAML" >&2
    echo "Generate it with:" >&2
    echo "  python3 scripts/world_to_map.py \\" >&2
    echo "      --world ${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.world \\" >&2
    echo "      --output ${PROJECT}/src/tiago_social_worlds/maps/${WORLD}" >&2
    exit 1
fi
echo "Map: $MAP_YAML"

# --- 1. Is a map already being published? -----------------------------------
echo ""
echo "[1] Checking for an existing map publisher..."
if timeout 5 ros2 topic echo /map --once >/dev/null 2>&1; then
    echo "    A map is ALREADY being published."
    echo "    Skipping map_server - going straight to localisation."
    SKIP_MAP=1
else
    echo "    Nothing on /map. Starting map_server."
    SKIP_MAP=0
fi

# --- 2. Start and ACTIVATE map_server ----------------------------------------
if [ "$SKIP_MAP" -eq 0 ]; then
    echo ""
    echo "[2] Starting map_server in the background..."

    # Reuse an already-running map_server rather than starting a second one.
    if ros2 node list 2>/dev/null | grep -q "^/map_server$"; then
        echo "    map_server is ALREADY running - reusing it."
    else
        ros2 run nav2_map_server map_server --ros-args \
            -p yaml_filename:="$MAP_YAML" \
            -p use_sim_time:=true \
            > /tmp/map_server.log 2>&1 &
        echo "    pid $!  (log: /tmp/map_server.log)"
    fi

    # ------------------------------------------------------------------
    # Drive the lifecycle by POLLING THE STATE, never by sleeping a fixed
    # amount. This container runs at ~0.26x real time and 2 FPS, and loading
    # the 420x320 map took about 10 seconds - far longer than any sensible
    # fixed sleep. An earlier version of this script slept 4s, so `configure`
    # timed out ("Transitioning failed") and the follow-up `activate` was then
    # rejected with "Unknown transition requested" because the node was still
    # unconfigured. Waiting for the actual state avoids guessing entirely.
    # ------------------------------------------------------------------
    wait_for_state() {
        local want="$1" limit="${2:-90}" i=0 state=""
        while [ $i -lt "$limit" ]; do
            state=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
            if [ "$state" = "$want" ]; then
                return 0
            fi
            sleep 1
            i=$((i + 1))
        done
        echo "    (still '$state' after ${limit}s, wanted '$want')"
        return 1
    }

    echo "    waiting for map_server to appear..."
    for i in $(seq 1 60); do
        ros2 node list 2>/dev/null | grep -q "^/map_server$" && break
        sleep 1
    done

    STATE=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
    echo "    current state: ${STATE:-unknown}"

    if [ "$STATE" = "unconfigured" ]; then
        echo "    configuring (this loads the map - can take 10-20s here)..."
        ros2 lifecycle set /map_server configure >/dev/null 2>&1 || true
        wait_for_state inactive 90 && echo "    configured"
    fi

    STATE=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
    if [ "$STATE" = "inactive" ]; then
        echo "    activating..."
        ros2 lifecycle set /map_server activate >/dev/null 2>&1 || true
        wait_for_state active 60 && echo "    activated"
    fi

    STATE=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
    echo "    final state: ${STATE:-unknown}"

    if timeout 20 ros2 topic echo /map --once >/dev/null 2>&1; then
        echo "    OK - /map is publishing"
    else
        echo "    FAILED - nothing on /map. Last lines of the log:" >&2
        tail -20 /tmp/map_server.log >&2
        echo "" >&2
        echo "    Try manually:  ros2 lifecycle set /map_server activate" >&2
        exit 1
    fi
fi

# --- 3. Read the robot's TRUE pose from Gazebo -------------------------------
echo ""
echo "[3] Reading the robot's true pose from Gazebo..."
POSE=$(python3 - <<'PY'
import math, sys
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates

class Grab(Node):
    def __init__(self):
        super().__init__('pose_grabber')
        self.result = None
        self.create_subscription(ModelStates, '/gazebo/model_states', self.cb, 10)
    def cb(self, msg):
        if self.result is not None:
            return
        # PAL spawns the robot as "tiago"; fall back to any name containing it.
        for want in ('tiago',):
            for i, name in enumerate(msg.name):
                if want in name.lower():
                    p = msg.pose[i]
                    q = p.orientation
                    yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                                     1 - 2*(q.y*q.y + q.z*q.z))
                    self.result = (p.position.x, p.position.y, yaw)
                    return

rclpy.init()
n = Grab()
for _ in range(60):                      # up to ~6 s
    rclpy.spin_once(n, timeout_sec=0.1)
    if n.result:
        break
rclpy.shutdown()
if n.result:
    print("%.4f %.4f %.4f" % n.result)
else:
    sys.exit(1)
PY
) || {
    echo "    Could not read the pose from /gazebo/model_states." >&2
    echo "    Falling back to odometry (accurate only if the robot started at the origin)." >&2
    POSE=$(timeout 5 ros2 topic echo /mobile_base_controller/odom --once 2>/dev/null | \
        python3 -c "
import sys, math, re
t = sys.stdin.read()
def grab(block, key):
    m = re.search(block + r'.*?' + key + r':\s*([-\d.e]+)', t, re.S)
    return float(m.group(1)) if m else 0.0
x = grab('position', 'x'); y = grab('position', 'y')
z = grab('orientation', 'z'); w = grab('orientation', 'w')
print('%.4f %.4f %.4f' % (x, y, math.atan2(2*w*z, 1-2*z*z)))
")
}

read -r RX RY RYAW <<< "$POSE"
echo "    robot is at x=${RX}  y=${RY}  yaw=${RYAW} rad"

# --- 4. Hand that pose to AMCL ------------------------------------------------
echo ""
echo "[4] Publishing the initial pose to /initialpose..."
QZ=$(python3 -c "import math; print(math.sin($RYAW/2))")
QW=$(python3 -c "import math; print(math.cos($RYAW/2))")

# Published a few times: AMCL occasionally misses the first message while its
# subscription is still coming up.
for i in 1 2 3; do
    ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
"{header: {frame_id: 'map'},
  pose: {pose: {position: {x: ${RX}, y: ${RY}, z: 0.0},
                orientation: {z: ${QZ}, w: ${QW}}},
         covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0,
                      0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]}}" \
        >/dev/null 2>&1
    sleep 1
done
echo "    sent"

# --- 4b. Is AMCL even running? ------------------------------------------------
# Without AMCL nothing publishes map -> odom, so no amount of /initialpose
# messages will produce a map frame. This is the difference between "Nav2 is
# running" and "Nav2 can localise".
echo ""
echo "[4b] Checking AMCL..."
if ros2 node list 2>/dev/null | grep -q amcl; then
    echo "    AMCL is running"
else
    echo "    !! AMCL is NOT running." >&2
    echo "    Nav2 was almost certainly launched with slam:=True, which uses" >&2
    echo "    slam_toolbox instead. Relaunch the simulation with slam:=False:" >&2
    echo "      ros2 launch tiago_gazebo tiago_gazebo.launch.py \\" >&2
    echo "          is_public_sim:=True world_name:=${WORLD} \\" >&2
    echo "          navigation:=True slam:=False moveit:=False" >&2
fi

# --- 5. Verify the transform chain -------------------------------------------
echo ""
echo "[5] Verifying map -> base_footprint (allowing time on this slow machine)..."
sleep 8
if timeout 10 ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | head -8 | grep -q Translation; then
    echo ""
    echo "============================================================"
    echo "  LOCALISED. The robot now knows where it is on the map."
    echo ""
    echo "  In RViz the laser scan should sit on top of the walls."
    echo "  You can now send a Nav2 Goal, and start the pipeline:"
    echo "    ros2 launch tiago_group_approach group_approach.launch.py \\"
    echo "        policy:=rule \\"
    echo "        groundtruth:=${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.groundtruth.json"
    echo "============================================================"
else
    echo "    No map -> base_footprint transform yet." >&2
    echo "    Check AMCL is running:  ros2 node list | grep amcl" >&2
    echo "    If AMCL is absent, Nav2 was launched with slam:=True - relaunch with slam:=False." >&2
fi
