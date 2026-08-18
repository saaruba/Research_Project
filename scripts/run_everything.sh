#!/usr/bin/env bash
# ============================================================================
# ONE COMMAND: simulation + localisation + group-approach pipeline.
#
#     bash scripts/run_everything.sh
#     bash scripts/run_everything.sh restaurant_final bc
#
# Args:  [world]  [policy: rule|bc]
#
# Everything runs from THIS terminal. Ctrl-C stops all of it cleanly.
#
# ----------------------------------------------------------------------------
# WHY A SINGLE SCRIPT
# ----------------------------------------------------------------------------
# Running these by hand across four terminals kept failing for one reason or
# another that was invisible until much later:
#   - the pipeline was started while Gazebo had already died, so perception sat
#     on "Waiting for CameraInfo..." forever against a world that wasn't there;
#   - the transform publisher was started before the robot existed;
#   - the map was served but nothing published map->odom.
# Every stage below WAITS for hard evidence that the previous one is alive
# before continuing, and says exactly what it is waiting for. If a stage fails,
# it stops there instead of letting later stages fail confusingly.
# ============================================================================

set -o pipefail

WORLD="${1:-restaurant_testing}"
POLICY="${2:-rule}"
PROJECT=/workspaces/Research_Project
GT="${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.groundtruth.json"

# ----------------------------------------------------------------------------
# EVERYTHING this project starts, by process-name pattern.
#
# An earlier version only killed gzserver/gzclient. That was not enough: the
# Nav2 nodes, controllers and robot_state_publisher from a previous attempt
# stayed alive, kept their node names and topics, and the next launch silently
# collided with them - the robot never spawned, while the OLD planner_server
# carried on printing 'Invalid frame ID "map"' into the terminal. If you ever
# see log lines appear that this script did not redirect to a file, that is
# what is happening.
# ----------------------------------------------------------------------------
STALE_PATTERNS=(
    gzserver gzclient 'ros2 launch' robot_state_publisher
    static_transform_publisher map_server amcl rviz2
    planner_server controller_server bt_navigator behavior_server
    smoother_server waypoint_follower velocity_smoother lifecycle_manager
    component_container spawn_entity
    group_perception_node group_approach_baseline_node bc_policy_node
    metrics_recorder_node
)

kill_stale() {
    for pat in "${STALE_PATTERNS[@]}"; do
        pkill -f "$pat" 2>/dev/null
    done
    sleep 3
    # Anything that ignored SIGTERM gets SIGKILL - a half-dead Nav2 node still
    # owns its name on the ROS graph and will break the next launch.
    for pat in "${STALE_PATTERNS[@]}"; do
        pkill -9 -f "$pat" 2>/dev/null
    done

    # ------------------------------------------------------------------
    # CATCH-ALL. Killing by node name was not enough: a full PAL launch
    # (amcl, map_server, twist_mux, play_motion2, arm_tucker, the lifecycle
    # managers) survived the list above, because those processes' command
    # lines do not contain the strings we were matching on. Every ROS node
    # runs out of one of these two trees, so killing by install path catches
    # all of them without needing to know their names in advance.
    #
    # This is safe: nothing else in this container runs from there, and this
    # script itself is plain bash, so it cannot kill itself.
    # ------------------------------------------------------------------
    pkill -9 -f '/opt/ros/humble/lib/' 2>/dev/null
    pkill -9 -f "${PROJECT}/install/" 2>/dev/null
    sleep 2

    # ------------------------------------------------------------------
    # The ros2 daemon CACHES the node graph. `ros2 node list` keeps reporting
    # nodes that are already dead until that cache refreshes, which is why the
    # previous run listed 24 "still alive" nodes while several of them had in
    # fact just been killed. Stopping the daemon forces the next query to
    # rediscover from scratch, so the check below reflects reality.
    # ------------------------------------------------------------------
    ros2 daemon stop >/dev/null 2>&1
    sleep 3
}

PIDS=()
CLEANED=0
cleanup() {
    # Guard against running twice. The trap fires on INT and again on EXIT, so
    # "Shutting everything down... Done." was printed two and three times.
    [ "$CLEANED" -eq 1 ] && return
    CLEANED=1
    echo ""
    echo "Shutting everything down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    sleep 2
    kill_stale
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  world : $WORLD"
echo "  policy: $POLICY"
echo "============================================================"

# --- 0. Clean slate ----------------------------------------------------------
echo ""
echo "[0/5] Clearing any leftover processes..."
kill_stale

source /opt/ros/humble/setup.bash
[ -f "${PROJECT}/install/setup.bash" ] && source "${PROJECT}/install/setup.bash"
[ -f "${PROJECT}/scripts/setup_gazebo_env.sh" ] && source "${PROJECT}/scripts/setup_gazebo_env.sh" >/dev/null

# ----------------------------------------------------------------------------
# Confirm the graph is actually empty before launching anything onto it.
#
# CRITICAL: an empty `ros2 node list` does NOT mean the graph is clear. Right
# after `ros2 daemon stop`, the first query always comes back empty because a
# fresh daemon has not finished discovery yet. A previous version of this check
# accepted that first empty read and cheerfully launched on top of a live
# simulation - which is exactly how a Gazebo from an earlier session survived
# three "clean" runs. So: require several CONSECUTIVE empty reads, with a
# settle delay first, and cross-check against the process table, which cannot
# lie about what is running.
# ----------------------------------------------------------------------------
echo -n "      letting discovery settle "
sleep 8
echo -n "      verifying the ROS graph is clear "
LEFTOVER=""
for attempt in 1 2 3 4 5 6; do
    CLEAN_STREAK=0
    for probe in 1 2 3; do
        LEFTOVER=$(timeout 15 ros2 node list 2>/dev/null \
                     | grep -v '^$' | grep -v '_ros2cli_' | grep -v 'transform_listener_impl' || true)
        # The process table is the ground truth, not the ROS graph.
        REALPROCS=$(pgrep -f '/opt/ros/humble/lib/|gzserver|gzclient' 2>/dev/null | head -5 || true)
        if [ -z "$LEFTOVER" ] && [ -z "$REALPROCS" ]; then
            CLEAN_STREAK=$((CLEAN_STREAK + 1))
        else
            CLEAN_STREAK=0
            [ -z "$LEFTOVER" ] && LEFTOVER="(processes alive but not yet on the graph)"
            break
        fi
        sleep 3
    done
    [ "$CLEAN_STREAK" -ge 3 ] && { LEFTOVER=""; break; }
    echo -n "."
    kill_stale
done
echo ""

if [ -n "$LEFTOVER" ]; then
    echo "      STILL not clear. These nodes refuse to die:" >&2
    echo "$LEFTOVER" | sed 's/^/        /' >&2
    echo "" >&2
    echo "      Are they real processes? (if this is empty, they are daemon ghosts" >&2
    echo "      and you can ignore this and re-run):" >&2
    ps -eo pid,cmd | grep -E '/opt/ros/humble/lib/|ros2 launch' | grep -v grep \
        | head -20 | sed 's/^/        /' >&2
    echo "" >&2
    echo "      If real processes ARE listed, the surest fix is to rebuild the" >&2
    echo "      devcontainer (VS Code: Dev Containers: Rebuild Container), which" >&2
    echo "      guarantees a clean ROS graph, then run this script again." >&2
    exit 1
fi
echo "      OK - ROS graph is clear"

# --- 1. Simulation -----------------------------------------------------------
echo ""
# ----------------------------------------------------------------------------
# SYNC THE WORLD FILE INTO pal_gazebo_worlds.
#
# `world_name:=restaurant_testing` is a NAME, not a path. PAL resolves it
# inside /opt/ros/humble/share/pal_gazebo_worlds/worlds/, which holds a COPY
# made by install_sim_stack.sh. Editing the world in this project therefore has
# NO effect on what Gazebo loads until the copy is refreshed - so the one-person
# edit could sit in the repo while Gazebo kept loading an older world entirely.
# Copying it every run makes that class of confusion impossible.
# ----------------------------------------------------------------------------
echo ""
echo "[1a] Syncing the world file into pal_gazebo_worlds..."
SRC_WORLD="${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.world"
PAL_WORLDS="$(ros2 pkg prefix pal_gazebo_worlds 2>/dev/null)/share/pal_gazebo_worlds/worlds"

if [ ! -f "$SRC_WORLD" ]; then
    echo "      ERROR: no world file at $SRC_WORLD" >&2
    exit 1
fi

if [ -d "$PAL_WORLDS" ]; then
    if [ -f "${PAL_WORLDS}/${WORLD}.world" ] \
       && diff -q "$SRC_WORLD" "${PAL_WORLDS}/${WORLD}.world" >/dev/null 2>&1; then
        echo "      already up to date"
    else
        echo "      >>> the installed copy was STALE or missing - refreshing it <<<"
        sudo cp "$SRC_WORLD" "${PAL_WORLDS}/${WORLD}.world" \
            && echo "      copied $(basename "$SRC_WORLD") -> pal_gazebo_worlds" \
            || { echo "      ERROR: copy failed (sudo?)" >&2; exit 1; }
    fi
else
    echo "      WARNING: pal_gazebo_worlds worlds/ not found - cannot sync" >&2
fi

echo ""
echo "[1/5] Starting Gazebo + TIAGo + Nav2 (this takes 1-3 minutes here)..."

# A UNIQUE log file per run. Do not reuse /tmp/sim.log: processes from an
# earlier launch keep their file handle open, so truncating it with '>' does
# not stop them writing into it. That produced a log that looked like this
# run's output but was actually a previous simulation's - the sim-time stamps
# carried on from where the old one had got to, which is how we found it.
SIMLOG="/tmp/sim_$(date +%s).log"

# ARM_TYPE=no-arm removes the arm from the robot model entirely - the cleanest
# option for a navigation-only study, but only if that URDF variant is
# installed. Left unset by default so a missing variant cannot break the spawn.
ARM_ARG=""
[ -n "${ARM_TYPE:-}" ] && ARM_ARG="arm_type:=${ARM_TYPE}"

ros2 launch tiago_gazebo tiago_gazebo.launch.py \
    is_public_sim:=True \
    world_name:="$WORLD" \
    navigation:=True \
    slam:=False \
    moveit:=False \
    $ARM_ARG \
    > "$SIMLOG" 2>&1 &
PIDS+=($!)
echo "      log: $SIMLOG"

# ----------------------------------------------------------------------------
# Wait for odometry - the only reliable proof the robot actually spawned.
#
# Done with ONE long-lived subscriber rather than 90 short `ros2 topic echo
# --once` calls. Each of those pays the full discovery cost from scratch, and
# on this container (0.26x real time) discovery alone can exceed the 3-second
# timeout, so the check could fail forever against a perfectly healthy robot.
# ----------------------------------------------------------------------------
echo "      waiting for the robot to spawn (up to 4 minutes)..."
python3 - <<'PY'
import sys
import rclpy
from nav_msgs.msg import Odometry

rclpy.init()
node = rclpy.create_node('spawn_waiter')
got = {'msg': False}

def cb(_):
    got['msg'] = True

node.create_subscription(Odometry, '/mobile_base_controller/odom', cb, 10)

import time
deadline = time.time() + 240
last_report = 0
while time.time() < deadline and not got['msg']:
    rclpy.spin_once(node, timeout_sec=0.5)
    elapsed = int(time.time() - (deadline - 240))
    if elapsed >= last_report + 20:
        last_report = elapsed
        pubs = node.count_publishers('/mobile_base_controller/odom')
        print(f"        {elapsed}s - publishers on /mobile_base_controller/odom: {pubs}",
              flush=True)

rclpy.shutdown()
sys.exit(0 if got['msg'] else 1)
PY
ROBOT_UP=$?
[ "$ROBOT_UP" -eq 0 ] && ROBOT_UP=1 || ROBOT_UP=0
if [ "$ROBOT_UP" -eq 0 ]; then
    echo "      FAILED - no odometry after 3 minutes. The robot did not spawn." >&2
    echo "      Check /tmp/sim.log, especially for:" >&2
    echo "        'Service /spawn_entity unavailable'  -> gazebo_ros_factory not loaded" >&2
    echo "        'Unable to find uri[model://...]'    -> a model is missing" >&2
    echo "" >&2
    echo "      ---- matching lines from $SIMLOG ----" >&2
    grep -iE "spawn|error|fail|unable to find" "$SIMLOG" 2>/dev/null | tail -20 >&2
    echo "" >&2
    echo "      ---- last 30 lines of $SIMLOG (regardless) ----" >&2
    # Printed unconditionally: an EMPTY log is itself the diagnosis - it means
    # 'ros2 launch' died instantly and never produced any output at all.
    tail -30 "$SIMLOG" 2>/dev/null || echo "      (no log file at all)" >&2
    echo "      ------------------------------------------" >&2
    exit 1
fi
echo "      OK - robot is spawned and publishing odometry"

# --- 1b. Tuck the arm --------------------------------------------------------
# TIAGo spawns with the arm extended. For a navigation study that is pure
# nuisance: it widens the robot's footprint in the costmap, it can clip tables,
# and it is visually confusing. PAL ships a 'home' motion that folds it away.
#
# Non-fatal by design - if play_motion2 is unavailable the run continues with
# the arm out, which is untidy but harmless.
#
# To remove the arm entirely instead, launch with ARM_TYPE=no-arm:
#     ARM_TYPE=no-arm bash scripts/run_everything.sh
echo ""
echo "[1b] Tucking the arm out of the way..."
ARM_TUCKED=0
timeout 60 ros2 action send_goal /play_motion2 \
    play_motion2_msgs/action/PlayMotion2 \
    "{motion_name: 'home', skip_planning: false}" >/dev/null 2>&1 \
    && ARM_TUCKED=1

# play_motion2 needs its motion definitions loaded, which does not always
# happen in the public sim. Fall back to commanding the arm controller
# directly with TIAGo's standard folded ("travel") joint configuration - no
# motion library, no MoveIt, just a joint trajectory.
if [ "$ARM_TUCKED" -eq 0 ]; then
    timeout 40 ros2 topic pub --once /arm_controller/joint_trajectory \
        trajectory_msgs/msg/JointTrajectory \
        "{joint_names: ['arm_1_joint','arm_2_joint','arm_3_joint','arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint'],
          points: [{positions: [0.20, -1.34, -0.20, 1.94, -1.57, 1.37, 0.0],
                    time_from_start: {sec: 4, nanosec: 0}}]}" >/dev/null 2>&1 \
        && ARM_TUCKED=1
    sleep 5
fi

if [ "$ARM_TUCKED" -eq 1 ]; then
    echo "      arm tucked"
else
    echo "      could not tuck the arm (harmless - continuing)"
    echo "      to remove it entirely:  ARM_TYPE=no-arm bash scripts/run_everything.sh"
fi

# --- 2. Camera ---------------------------------------------------------------
echo ""
echo "[2/5] Waiting for the camera (up to 3 minutes)..."
# Same persistent-subscriber approach as stage 1. The old version used 45 x
# `timeout 3 ros2 topic echo --once`, each paying full ROS discovery cost from
# scratch - on this container that alone can exceed 3 seconds, so the check
# could report "no camera" against a camera that was streaming perfectly.
# On failure it now lists every CameraInfo topic that DOES exist, because the
# most likely remaining cause is simply a different topic name.
python3 - <<'PY'
import sys, time
import rclpy
from sensor_msgs.msg import CameraInfo

WANT = '/head_front_camera/rgb/camera_info'

rclpy.init()
node = rclpy.create_node('camera_waiter')
got = {'msg': False}
node.create_subscription(CameraInfo, WANT, lambda _: got.__setitem__('msg', True), 10)

deadline = time.time() + 180
start = time.time()
last = 0
while time.time() < deadline and not got['msg']:
    rclpy.spin_once(node, timeout_sec=0.5)
    el = int(time.time() - start)
    if el >= last + 20:
        last = el
        print(f"        {el}s - publishers on {WANT}: "
              f"{node.count_publishers(WANT)}", flush=True)

if not got['msg']:
    print("\n        No message on that topic. CameraInfo topics that DO exist:",
          flush=True)
    found = [n for n, t in node.get_topic_names_and_types()
             if 'sensor_msgs/msg/CameraInfo' in t]
    for n in found:
        print(f"          {n}   (publishers: {node.count_publishers(n)})", flush=True)
    if not found:
        print("          (none at all - the camera plugin did not load)", flush=True)
    print("\n        All camera-ish topics:", flush=True)
    for n, _t in node.get_topic_names_and_types():
        if 'camera' in n.lower() or 'image' in n.lower():
            print(f"          {n}", flush=True)

rclpy.shutdown()
sys.exit(0 if got['msg'] else 1)
PY
if [ $? -ne 0 ]; then
    echo "      FAILED - no camera_info. Perception cannot run." >&2
    echo "      If a DIFFERENT CameraInfo topic is listed above, that is the fix:" >&2
    echo "      tell me the name and I will point the perception node at it." >&2
    exit 1
fi
echo "      OK - camera is streaming"

# --- 3. map -> odom ----------------------------------------------------------
# Nav2 needs a 'map' frame. AMCL is supposed to publish map->odom once it has
# localised, but on this container it never converges, so Nav2 spins forever on
#   Invalid frame ID "map" passed to canTransform
# The robot spawns at the world origin and the odom frame is created there, so
# map and odom coincide and the identity transform is exactly correct.
echo ""
echo "[3/5] Localisation: activating the map, then publishing map -> odom..."

# --- 3a. map_server MUST be active -------------------------------------------
# RViz showed "Localization: inactive": PAL brings map_server and AMCL up as
# LIFECYCLE nodes, and with slam:=False they sit unconfigured until something
# activates them. While map_server is inactive nothing publishes /map, the
# global costmap has no static layer, and Nav2 cannot plan even once the
# transform exists.
MAP_YAML="${PROJECT}/src/tiago_social_worlds/maps/${WORLD}.yaml"
if [ ! -f "$MAP_YAML" ]; then
    echo "      ERROR: no map at $MAP_YAML" >&2
    echo "      Generate it:  python3 scripts/world_to_map.py \\" >&2
    echo "          --world ${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.world \\" >&2
    echo "          --output ${PROJECT}/src/tiago_social_worlds/maps/${WORLD}" >&2
    exit 1
fi

# THE yaml_filename PARAMETER IS THE WHOLE PROBLEM.
# `ros2 lifecycle set /map_server configure` failed and left the node
# 'unconfigured'. configure() is what LOADS the map, so with no yaml_filename
# set (PAL does not set one for a custom world) it has nothing to load and the
# transition fails. Setting the parameter first makes configure succeed.
echo "      pointing map_server at: $MAP_YAML"
ros2 param set /map_server yaml_filename "$MAP_YAML" >/dev/null 2>&1 \
    && echo "      parameter set" \
    || echo "      WARNING: could not set yaml_filename (is /map_server running?)"
ros2 param set /map_server use_sim_time true >/dev/null 2>&1 || true

echo "      activating map_server..."
for target in configure activate; do
    want=$([ "$target" = configure ] && echo inactive || echo active)
    ros2 lifecycle set /map_server "$target" >/dev/null 2>&1 || true
    for i in $(seq 1 40); do
        [ "$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')" = "$want" ] && break
        sleep 1
    done
done
MS_STATE=$(ros2 lifecycle get /map_server 2>/dev/null || echo unknown)
echo "      map_server: $MS_STATE"

# Fallback: ask the lifecycle MANAGER to bring the localisation stack up. Doing
# it through the manager keeps its bond bookkeeping consistent, so it will not
# later decide the node died and shut it down again.
case "$MS_STATE" in
    active*) : ;;
    *)
        echo "      still not active - asking lifecycle_manager_localization to start it..."
        timeout 45 ros2 service call /lifecycle_manager_localization/manage_nodes \
            nav2_msgs/srv/ManageLifecycleNodes "{command: 0}" >/dev/null 2>&1 || true
        sleep 5
        echo "      map_server: $(ros2 lifecycle get /map_server 2>/dev/null || echo unknown)"
        ;;
esac

# Prove it: /map must actually publish, or the global costmap has no static layer.
# /map is published TRANSIENT_LOCAL (latched): map_server sends it once and
# holds it for late subscribers. `ros2 topic echo --once` uses VOLATILE QoS by
# default, so it never matches and hangs until the timeout - reporting "nothing
# on /map" while the map is being served perfectly well. Subscribe with the
# matching durability instead.
echo "      waiting for /map to publish..."
python3 - <<'PY'
import sys, time
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid

rclpy.init()
n = rclpy.create_node('map_check')
got = {}
qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
n.create_subscription(OccupancyGrid, '/map',
                      lambda m: got.setdefault('m', m), qos)
end = time.time() + 40
while time.time() < end and 'm' not in got:
    rclpy.spin_once(n, timeout_sec=0.2)
if 'm' in got:
    m = got['m']
    print(f"        map: {m.info.width}x{m.info.height} @ {m.info.resolution} m/px, "
          f"origin ({m.info.origin.position.x:.1f}, {m.info.origin.position.y:.1f})")
rclpy.shutdown()
sys.exit(0 if 'm' in got else 1)
PY
if [ $? -eq 0 ]; then
    echo "      OK - /map is publishing"
else
    echo "      WARNING - nothing on /map. Nav2 will not be able to plan." >&2
    tail -15 /tmp/map_server.log 2>/dev/null >&2 || true
fi

# --- 3b. AMCL must NOT fight us ----------------------------------------------
# We supply map->odom directly from simulator ground truth. If AMCL later
# converges it would publish a second, competing map->odom.
if ros2 node list 2>/dev/null | grep -q "^/amcl$"; then
    ros2 lifecycle set /amcl deactivate >/dev/null 2>&1 || true
    echo "      amcl deactivated (we provide map->odom ourselves)"
fi

# --- 3c. The transform -------------------------------------------------------
# use_sim_time matters here. Everything else in the stack timestamps with
# Gazebo's clock (sim time ~360 s), while a node on wall-clock time stamps with
# ~1.79e9. A TF buffer mixing the two cannot interpolate, which is why the old
# verification failed even when the transform was being published correctly.
# ----------------------------------------------------------------------------
# COMPUTE map->odom, DO NOT ASSUME IT.
#
# This used to publish identity, on the assumption that TIAGo spawns at the
# world origin with yaw 0. It does not - and the symptom was visible: the robot
# faced one direction in Gazebo and a different one in RViz. Nav2 was planning
# from a false pose, so goals were ACCEPTED and then ABORTED.
#
# The world loads the gazebo_ros_state plugin, so the true pose is published on
# /gazebo/model_states. The exact correction is:
#
#     T_map_odom = T_map_base * inverse(T_odom_base)
#
# For the dissertation, state this in the Methodology: localisation was taken
# from simulator ground truth so that navigation error could not confound the
# comparison between the rule-based and BC policies. The laser, costmaps and
# planners all still behave normally; only the global correction is exact.
# ----------------------------------------------------------------------------
echo "      computing map -> odom from simulator ground truth..."
TF=$(python3 - <<'PY'
import math, sys
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry

def yaw_of(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

class Collect(Node):
    def __init__(self):
        super().__init__('gt_tf_calc')
        self.truth = None
        self.odom = None
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_models, 10)
        self.create_subscription(Odometry, '/mobile_base_controller/odom', self.on_odom, 10)

    def on_models(self, msg):
        if self.truth is not None:
            return
        for i, name in enumerate(msg.name):
            if 'tiago' in name.lower():
                p = msg.pose[i]
                self.truth = (p.position.x, p.position.y, yaw_of(p.orientation))
                return

    def on_odom(self, msg):
        if self.odom is None:
            p = msg.pose.pose
            self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

rclpy.init()
n = Collect()
for _ in range(300):                       # up to ~30 s
    rclpy.spin_once(n, timeout_sec=0.1)
    if n.truth and n.odom:
        break
rclpy.shutdown()

if not n.truth or not n.odom:
    sys.exit(1)

mbx, mby, mbyaw = n.truth
obx, oby, obyaw = n.odom
yaw = mbyaw - obyaw
c, s = math.cos(yaw), math.sin(yaw)
x = mbx - (obx*c - oby*s)
y = mby - (obx*s + oby*c)
print(f"{x:.6f} {y:.6f} {yaw:.6f} {mbx:.3f} {mby:.3f} {math.degrees(mbyaw):.1f}")
PY
) && GT_OK=1 || GT_OK=0

if [ "$GT_OK" -eq 1 ]; then
    read -r TX TY TYAW MBX MBY MBDEG <<< "$TF"
    echo "      robot TRUE pose in Gazebo : x=${MBX} y=${MBY} yaw=${MBDEG} deg"
    echo "      => map->odom              : x=${TX} y=${TY} yaw=${TYAW} rad"
else
    echo "      Ground truth unavailable - falling back to identity." >&2
    echo "      (If the robot's heading in RViz disagrees with Gazebo, this is why.)" >&2
    TX=0; TY=0; TYAW=0
fi

# A STATIC transform is only correct for one instant. Odometry drifts - and it
# drifts violently when the wheels slip against an obstacle - so a correction
# computed once at startup becomes progressively wrong. That is why the robot
# appeared in one place in Gazebo and another in RViz, why the costmap was
# offset from the real furniture, and very likely why detections landed at the
# wrong map coordinates. gt_localisation_node recomputes and republishes the
# correction at 30 Hz instead.
ros2 run tiago_group_approach gt_localisation_node \
    --ros-args -p use_sim_time:=true \
    > /tmp/map_odom_tf.log 2>&1 &
PIDS+=($!)
sleep 8

# Verified with a real TF listener, not `tf2_echo`. tf2_echo kept reporting
# failure while the transform was demonstrably fine - perception was placing
# the person within 10 cm of ground truth THROUGH THIS VERY CHAIN. A check that
# cries wolf is worse than no check: it sent us hunting a localisation bug that
# did not exist.
echo "      verifying map -> base_footprint..."
python3 - <<'PY'
import sys, time
import rclpy, tf2_ros
from rclpy.duration import Duration

rclpy.init()
n = rclpy.create_node('tf_check')
n.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                  rclpy.parameter.Parameter.Type.BOOL, True)])
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf, n)

ok = False
end = time.time() + 40
while time.time() < end and not ok:
    rclpy.spin_once(n, timeout_sec=0.2)
    try:
        t = buf.lookup_transform('map', 'base_footprint',
                                 rclpy.time.Time(), timeout=Duration(seconds=0.5))
        p = t.transform.translation
        print(f"        robot on the map: x={p.x:.3f} y={p.y:.3f}")
        ok = True
    except Exception:
        pass
rclpy.shutdown()
sys.exit(0 if ok else 1)
PY
TF_OK=$?
[ "$TF_OK" -eq 0 ] && TF_OK=1 || TF_OK=0
if [ "$TF_OK" -eq 0 ]; then
    echo "      WARNING - transform still not resolving. Nav2 may not plan." >&2
    echo "      (continuing anyway - perception will still run)" >&2
else
    echo "      OK - the robot is localised on the map"
fi

# --- 4. Pipeline -------------------------------------------------------------
echo ""
# --- 3d. Swap in this project's RViz configuration ---------------------------
# PAL launches nav2_default_view.rviz, which is built for a different robot:
# its "Realsense" display points at camera topics TIAGo does not publish here,
# so it sits on "No Image", and there is no display at all for the detector
# output. Every run then needs the same manual Add -> Image -> pick topic
# dance before you can see anything.
#
# This project's config has the annotated detection view enabled by default,
# plus the map, laser, TF and group markers - and nothing pointing at hardware
# that is not there.
RVIZ_CFG="${PROJECT}/src/tiago_group_approach/rviz/group_approach.rviz"
if [ -f "$RVIZ_CFG" ] && [ "${RVIZ:-1}" = "1" ]; then
    echo ""
    echo "[3d] Swapping in this project's RViz layout..."

    # ----------------------------------------------------------------------
    # DO NOT KILL PAL's RViz.
    #
    # An earlier version of this step did, and it was killing the entire
    # simulation. PAL's RViz is a process managed by `ros2 launch`; when a
    # managed process exits unexpectedly, launch tears the WHOLE launch down
    # with it. The log said so plainly:
    #
    #     [gzserver-1] signal_handler(signum=15)
    #     [gzserver-1] Shutdown request received....
    #     [INFO] [gzclient-2]: process has finished cleanly
    #
    # signum 15 is SIGTERM, and "finished cleanly" is an orderly shutdown, not
    # a crash. The giveaway was which nodes survived: /rviz and
    # /gt_localisation_node, both started by THIS script, outside PAL's launch.
    #
    # So we simply add our own RViz alongside PAL's. Two windows, but a stable
    # simulation - and stability is worth far more than tidiness here.
    # ----------------------------------------------------------------------
    FREE_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}')
    echo "      available memory: ${FREE_MB:-unknown} MB"
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 700 ]; then
        echo "      LOW MEMORY - skipping our RViz to protect the simulation." >&2
    else
        ros2 run rviz2 rviz2 -d "$RVIZ_CFG" \
            --ros-args -p use_sim_time:=true \
            > /tmp/rviz.log 2>&1 &
        PIDS+=($!)
        echo "      opened a SECOND RViz window with the detection view."
        echo "      PAL's original RViz is left alone on purpose - closing it"
        echo "      shuts down the whole simulation. Just minimise it."
    fi
elif [ "${RVIZ:-1}" != "1" ]; then
    echo ""
    echo "[3d] RViz swap skipped (RVIZ=0)."
fi

if [ "${3:-}" = "--no-pipeline" ] || [ "${NO_PIPELINE:-0}" = "1" ]; then
    echo "[4/5] SKIPPED - bring-up only (--no-pipeline)."
    echo ""
    echo "============================================================"
    echo "  Simulation, map and localisation are up. Leave this running."
    echo ""
    echo "  Prove the base can move at all (new terminal):"
    echo "      bash scripts/drive_test.sh"
    echo ""
    echo "  Then run the robot's behaviour separately (new terminal):"
    echo "      bash scripts/run_pipeline.sh ${WORLD} ${POLICY}"
    echo ""
    echo "  Do NOT close ANY of the windows PAL opened (Gazebo, its RViz)."
    echo "  They are managed by ros2 launch: if one exits, launch shuts the"
    echo "  WHOLE simulation down. Minimise them instead. Ctrl-C here to stop."
    echo ""
    echo "  Lighter run, if things keep dying:"
    echo "      RVIZ=0 bash scripts/run_everything.sh ${WORLD} ${POLICY} --no-pipeline"
    echo "============================================================"

    # ----------------------------------------------------------------------
    # HOLD HERE - and report, rather than exiting quietly.
    #
    # This used to be a bare `wait`. The problem: `wait` returns as soon as it
    # is interrupted or a child is reaped, and the `exit 0` after it then fired
    # the cleanup trap, which killed Gazebo, RViz and Nav2. From the outside
    # that looks like "everything closed on its own for no reason".
    #
    # Closing the Gazebo GUI window can do it too: gzclient exits, PAL's launch
    # treats that as a shutdown, and the whole stack comes down.
    #
    # So instead of waiting blindly, poll the components we care about and SAY
    # which one disappeared first, with the tail of its log. Nothing is killed
    # automatically - Ctrl-C is the only thing that shuts this down.
    # ----------------------------------------------------------------------
    echo ""
    echo "      (monitoring - this terminal stays open until you press Ctrl-C)"
    # Require THREE consecutive failures before declaring death. A single
    # pgrep miss is not evidence: on a loaded container a process can fail to
    # match transiently, and the first version of this monitor announced
    # gzserver dead while printing an EMPTY log - two claims that cannot both
    # be true. Better to be slow and right.
    MISSES=0
    while true; do
        sleep 15
        if pgrep -f "gzserver" >/dev/null 2>&1; then
            MISSES=0
            continue
        fi
        MISSES=$((MISSES + 1))
        echo "      (gzserver not found - check ${MISSES}/3)"
        [ "$MISSES" -lt 3 ] && continue

        echo ""
        echo "  ============================================================" >&2
        echo "  *** gzserver GONE - confirmed 3x at $(date +%H:%M:%S) ***" >&2
        echo "  ============================================================" >&2

        echo "  Gazebo-ish processes still alive:" >&2
        ps -eo pid,etime,cmd 2>/dev/null | grep -iE "gzserver|gzclient|gazebo" \
            | grep -v grep | sed 's/^/    /' >&2 || echo "    (none)" >&2

        echo "" >&2
        echo "  Simulation log: $SIMLOG" >&2
        if [ -s "$SIMLOG" ]; then
            echo "  size: $(wc -c < "$SIMLOG") bytes. Last 30 lines:" >&2
            tail -30 "$SIMLOG" >&2
        elif [ -f "$SIMLOG" ]; then
            # An EMPTY log is itself a finding: ros2 launch produced no output
            # at all, which does not happen during a normal shutdown.
            echo "  THE LOG IS EMPTY (0 bytes)." >&2
            ls -la /tmp/sim_*.log 2>/dev/null | tail -5 | sed 's/^/    /' >&2
        else
            echo "  THE LOG FILE DOES NOT EXIST." >&2
        fi

        echo "" >&2
        echo "  ROS nodes still up:" >&2
        timeout 15 ros2 node list 2>/dev/null | head -12 | sed 's/^/    /' >&2

        echo "" >&2
        free -m 2>/dev/null | sed 's/^/    /' >&2
        echo "" >&2
        echo "  Nothing has been shut down - inspect freely. Ctrl-C when done." >&2
        echo "  ============================================================" >&2
        while true; do sleep 3600; done
    done
fi

echo "[4/5] Starting the group-approach pipeline..."
if [ ! -f "$GT" ]; then
    echo "      No ground truth at $GT" >&2
    echo "      Generate it: python3 scripts/extract_world_groundtruth.py --world ..." >&2
    exit 1
fi

# One person in the world is still a valid approach target for this test.
NPEOPLE=$(python3 -c "
import json; d=json.load(open('$GT'))
print(sum(g['num_people'] for g in d['groups']))" 2>/dev/null || echo 2)
MINSIZE=2
[ "${NPEOPLE:-2}" -lt 2 ] && MINSIZE=1
echo "      world has ${NPEOPLE} person(s) -> min_group_size=${MINSIZE}"

ros2 launch tiago_group_approach group_approach.launch.py \
    policy:="$POLICY" \
    min_group_size:="$MINSIZE" \
    groundtruth:="$GT" \
    > /tmp/pipeline.log 2>&1 &
PIDS+=($!)
sleep 20
echo "      log: /tmp/pipeline.log"

# --- 5. Ready ----------------------------------------------------------------
echo ""
echo "[5/5] Everything is up."

# --- 5a. Turn the robot to face the person -----------------------------------
# Perception is passive: it can only cluster people the camera can actually
# see. TIAGo spawns at the origin facing +x, and the person in this world is at
# (-3, 0) - directly BEHIND the robot. Without this the pipeline would sit
# waiting for a detection that can never arrive, which looks identical to
# "the robot is broken".
if [ "$TF_OK" -eq 1 ]; then
    echo ""
    echo "      Sending an orientation goal so the camera faces the person..."
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
        "{header: {frame_id: 'map'}, pose: {position: {x: -1.5, y: 0.0, z: 0.0},
          orientation: {x: 0.0, y: 0.0, z: 1.0, w: 0.0}}}" >/dev/null 2>&1 \
        && echo "      goal sent - the robot should start turning" \
        || echo "      could not send the goal (is Nav2 up?)"
else
    echo ""
    echo "      SKIPPING the orientation goal - there is no map transform, so"
    echo "      Nav2 would reject it. Fix stage 3 first."
fi

echo ""
echo "============================================================"
echo "  WATCH THE PIPELINE:      tail -f /tmp/pipeline.log"
echo "  WATCH THE SIMULATION:    tail -f $SIMLOG"
echo ""
echo "  If it does not move, send the goal again by hand:"
echo ""
echo "    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \\"
echo "      \"{header: {frame_id: 'map'}, pose: {position: {x: -1.5, y: 0.0}, \\"
echo "        orientation: {z: 1.0, w: 0.0}}}\""
echo ""
echo "  Then look for this in the pipeline log:"
echo "    'Group centroid (...) -> approach pose (...)'"
echo "    'Nav2 goal accepted - robot is moving.'"
echo ""
echo "  Ctrl-C here stops everything."
echo "============================================================"

wait
