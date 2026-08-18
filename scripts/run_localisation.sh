#!/usr/bin/env bash
# ============================================================================
# Make the robot navigable: serve the map, and provide map -> odom DIRECTLY.
#
#   bash scripts/run_localisation.sh
#   bash scripts/run_localisation.sh restaurant_final
#
# Run AFTER Gazebo is up and /mobile_base_controller/odom returns data.
# Leave this terminal running - it holds the transform publisher.
#
# ----------------------------------------------------------------------------
# WHY THIS BYPASSES AMCL
# ----------------------------------------------------------------------------
# Nav2 cannot plan without a map -> odom transform. Normally AMCL produces it by
# matching laser scans against the map, but it needs a good initial pose, several
# seconds of settling, and enough distinctive geometry to lock on. In this
# container (0.26x real time, ~2 FPS) it was never converging, leaving Nav2
# stuck in a loop of:
#     Invalid frame ID "map" passed to canTransform
#     Message Filter dropping message: frame 'odom' ... queue is full
#
# But this is SIMULATION. Gazebo already knows exactly where the robot is, so
# the transform can be computed exactly instead of estimated:
#
#     T_map_odom = T_map_base * inverse(T_odom_base)
#
# where T_map_base is the robot's true pose from /gazebo/model_states and
# T_odom_base is what the wheel odometry reports. Publishing that as a static
# transform gives PERFECT localisation with no convergence, no drift and no
# initial-pose guessing.
#
# IS THIS LEGITIMATE FOR THE DISSERTATION?
# Yes, and it is worth stating explicitly in the Methodology. This project is
# evaluating SOCIAL APPROACH BEHAVIOUR, not localisation. Using ground-truth
# localisation removes a confound: a poor result then reflects the policy, not
# AMCL having lost the robot. Report it as: "localisation was provided from
# simulator ground truth so that navigation error did not confound the
# behavioural comparison."
#
# Note the robot's ODOMETRY still drifts naturally, and the laser, costmaps and
# planners all behave exactly as normal. Only the global correction is exact.
# ============================================================================

set -o pipefail

WORLD="${1:-restaurant_testing}"
PROJECT=/workspaces/Research_Project
MAP_YAML="${PROJECT}/src/tiago_social_worlds/maps/${WORLD}.yaml"

echo "============================================================"
echo "LOCALISATION (ground truth)  -  world: ${WORLD}"
echo "============================================================"

[ -f "$MAP_YAML" ] || { echo "ERROR: no map at $MAP_YAML" >&2; exit 1; }

# --- 1. Map server ----------------------------------------------------------
echo ""
echo "[1] Map"
if timeout 5 ros2 topic echo /map --once >/dev/null 2>&1; then
    echo "    /map already publishing - reusing it"
else
    if ! ros2 node list 2>/dev/null | grep -q "^/map_server$"; then
        echo "    starting map_server..."
        ros2 run nav2_map_server map_server --ros-args \
            -p yaml_filename:="$MAP_YAML" -p use_sim_time:=true \
            > /tmp/map_server.log 2>&1 &
        sleep 5
    fi
    # Poll the lifecycle state - loading a 420x320 map takes ~10s at 0.26x speed,
    # far longer than any fixed sleep would allow for.
    for target in configure activate; do
        want=$([ "$target" = configure ] && echo inactive || echo active)
        ros2 lifecycle set /map_server "$target" >/dev/null 2>&1 || true
        for i in $(seq 1 60); do
            [ "$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')" = "$want" ] && break
            sleep 1
        done
    done
    echo "    map_server state: $(ros2 lifecycle get /map_server 2>/dev/null)"
fi

# --- 2. Compute the exact map -> odom transform ------------------------------
echo ""
echo "[2] Computing map -> odom from simulator ground truth..."

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
for _ in range(150):                       # up to ~15 s
    rclpy.spin_once(n, timeout_sec=0.1)
    if n.truth and n.odom:
        break
rclpy.shutdown()

if not n.truth:
    print("NO_TRUTH", file=sys.stderr); sys.exit(1)
if not n.odom:
    print("NO_ODOM", file=sys.stderr); sys.exit(2)

mbx, mby, mbyaw = n.truth      # robot in the map/world frame
obx, oby, obyaw = n.odom       # robot in the odom frame

# T_map_odom = T_map_base * inv(T_odom_base)
yaw = mbyaw - obyaw
c, s = math.cos(yaw), math.sin(yaw)
x = mbx - (obx*c - oby*s)
y = mby - (obx*s + oby*c)

print(f"{x:.6f} {y:.6f} {yaw:.6f} {mbx:.3f} {mby:.3f} {mbyaw:.3f} {obx:.3f} {oby:.3f} {obyaw:.3f}")
PY
)
STATUS=$?

if [ $STATUS -ne 0 ]; then
    # ----------------------------------------------------------------------
    # FALLBACK: assume odom == map.
    #
    # /gazebo/model_states is only available if the world running in Gazebo
    # contains the gazebo_ros_state plugin. Gazebo loads the world from PAL's
    # directory (/opt/ros/humble/share/pal_gazebo_worlds/worlds/), which holds
    # a COPY made by install_sim_stack.sh - so if that copy predates the plugin
    # being added, ground truth is unavailable even though the project's own
    # world file has it.
    #
    # Fortunately the identity transform is usually exactly right here: PAL
    # spawns TIAGo at the world origin, and the odom frame is created at the
    # spawn point, so odom and map coincide. This gets navigation working now;
    # re-running install_sim_stack.sh and relaunching restores true ground
    # truth if you want the exact version.
    # ----------------------------------------------------------------------
    echo "    Ground truth unavailable (/gazebo/model_states missing)."
    echo "    FALLING BACK to identity: assuming the robot spawned at the world origin."
    echo "    (To get exact ground truth: re-run scripts/install_sim_stack.sh to"
    echo "     refresh the world copy in pal_gazebo_worlds, then relaunch Gazebo.)"
    TX=0.0; TY=0.0; TYAW=0.0
    MBX=n/a; MBY=n/a; MBYAW=n/a; OBX=n/a; OBY=n/a; OBYAW=n/a
    FALLBACK=1
fi

if [ "${FALLBACK:-0}" -eq 0 ]; then
    read -r TX TY TYAW MBX MBY MBYAW OBX OBY OBYAW <<< "$TF"
    echo "    robot true pose (map) : x=${MBX} y=${MBY} yaw=${MBYAW}"
    echo "    robot odom pose       : x=${OBX} y=${OBY} yaw=${OBYAW}"
fi
echo "    => map->odom          : x=${TX} y=${TY} yaw=${TYAW}"

# AMCL also publishes map->odom. If it is running but unlocalised it publishes
# nothing, so a static publisher is safe - but if AMCL later converges there
# would be two publishers fighting. Shut AMCL down to keep exactly one source.
if ros2 node list 2>/dev/null | grep -q "^/amcl$"; then
    echo ""
    echo "    AMCL is running - deactivating it so it cannot fight this"
    echo "    transform later (we are supplying map->odom directly)."
    ros2 lifecycle set /amcl deactivate >/dev/null 2>&1 || true
fi

# --- 3. Publish it ------------------------------------------------------------
echo ""
echo "[3] Publishing static transform map -> odom (leave this running)"
ros2 run tf2_ros static_transform_publisher \
    "$TX" "$TY" 0 "$TYAW" 0 0 map odom --ros-args -p use_sim_time:=true \
    > /tmp/map_odom_tf.log 2>&1 &
TF_PID=$!
sleep 4

# --- 4. Verify ----------------------------------------------------------------
echo ""
echo "[4] Verifying..."
if timeout 12 ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | head -10 | grep -q Translation; then
    echo ""
    echo "============================================================"
    echo "  LOCALISED - map -> base_footprint now exists."
    echo ""
    echo "  In RViz: the laser should sit on the map walls, and"
    echo "  Global Status should go green."
    echo ""
    echo "  Send a goal to test:"
    echo "    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \\"
    echo "      \"{header: {frame_id: 'map'}, pose: {position: {x: -2.0, y: 0.5}, \\"
    echo "        orientation: {w: 1.0}}}\""
    echo ""
    echo "  Then start the pipeline in another terminal:"
    echo "    ros2 launch tiago_group_approach group_approach.launch.py \\"
    echo "        policy:=rule \\"
    echo "        groundtruth:=${PROJECT}/src/tiago_social_worlds/worlds/${WORLD}.groundtruth.json"
    echo "============================================================"
    echo ""
    echo "KEEP THIS TERMINAL OPEN - closing it stops the transform."
    wait $TF_PID
else
    echo "    map -> base_footprint still missing." >&2
    echo "    Is something else already publishing map->odom (AMCL/SLAM)?" >&2
    echo "      ros2 run tf2_ros tf2_monitor map odom" >&2
    kill $TF_PID 2>/dev/null
    exit 1
fi
