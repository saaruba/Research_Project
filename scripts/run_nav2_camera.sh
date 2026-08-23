#!/usr/bin/env bash
# ============================================================================
# Nav2 with CAMERA-BASED obstacle avoidance.
#
#     bash scripts/run_nav2_camera.sh
#
# Started automatically by run_everything.sh when CAMERA_OBSTACLES=1. It brings
# up two things PAL's own navigation does not:
#
#   1. pointcloud_to_laserscan - projects the head camera's depth cloud into a
#      LaserScan on /scan_depth, covering 0.25-1.60 m above the floor.
#   2. Nav2, using config/nav2_camera_obstacles.yaml, which registers
#      /scan_depth as a second observation source alongside the base laser.
#
# ----------------------------------------------------------------------------
# WHY THIS IS NEEDED
# ----------------------------------------------------------------------------
# TIAGo's base laser scans at ~0.2 m. A dining table is thin legs at that
# height and a wide top at ~0.75 m, so the tabletop never enters the costmap
# and the robot drives its upper body into furniture that Nav2 believes is not
# there. The same blind spot hides anything else mounted above knee height.
#
# The depth camera sees all of it. Feeding it in as a second scan is the
# smallest change that fixes the actual cause.
#
# ----------------------------------------------------------------------------
# THIS IS OPT-IN, AND THE OLD PATH IS UNTOUCHED
# ----------------------------------------------------------------------------
# Without CAMERA_OBSTACLES=1, run_everything.sh still launches PAL's navigation
# exactly as before and none of this runs. The 60 trials already recorded used
# that path and remain reproducible.
# ============================================================================

set -o pipefail

PROJECT=/workspaces/Research_Project
PARAMS="${PROJECT}/config/nav2_camera_obstacles.yaml"
CLOUD_TOPIC="${CLOUD_TOPIC:-/head_front_camera/depth/rgb/points}"

source /opt/ros/humble/setup.bash
[ -f "${PROJECT}/install/setup.bash" ] && source "${PROJECT}/install/setup.bash"

echo "============================================================"
echo "  NAV2 WITH CAMERA OBSTACLES"
echo "============================================================"

if [ ! -f "$PARAMS" ]; then
    echo "ERROR: $PARAMS not found." >&2
    echo "Generate it first:" >&2
    echo "  python3 scripts/make_nav2_camera_params.py" >&2
    exit 1
fi

if ! ros2 pkg list 2>/dev/null | grep -q pointcloud_to_laserscan; then
    echo "ERROR: pointcloud_to_laserscan is not installed." >&2
    echo "  sudo apt update && sudo apt install -y ros-humble-pointcloud-to-laserscan" >&2
    exit 1
fi

# --- 1. Depth cloud -> LaserScan ---------------------------------------------
# target_frame is the base, so the height band is measured from the floor
# rather than from the tilted camera.
echo ""
echo "[1] Projecting ${CLOUD_TOPIC} into /scan_depth ..."
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node \
    --ros-args \
    -r cloud_in:="${CLOUD_TOPIC}" \
    -r scan:=/scan_depth \
    -p target_frame:=base_footprint \
    -p transform_tolerance:=0.05 \
    -p min_height:=0.25 \
    -p max_height:=1.60 \
    -p angle_min:=-1.0 \
    -p angle_max:=1.0 \
    -p angle_increment:=0.0087 \
    -p scan_time:=0.3333 \
    -p range_min:=0.30 \
    -p range_max:=4.0 \
    -p use_inf:=true \
    -p use_sim_time:=true \
    > /tmp/pc2scan.log 2>&1 &
PC_PID=$!
sleep 5

# --- 2. Nav2 with the patched parameters --------------------------------------
echo "[2] Starting Nav2 with ${PARAMS##*/} ..."
ros2 launch nav2_bringup navigation_launch.py \
    params_file:="$PARAMS" \
    use_sim_time:=true \
    > /tmp/nav2_camera.log 2>&1 &
NAV_PID=$!

cleanup() {
    echo ""
    echo "Stopping camera-obstacle Nav2..."
    kill "$PC_PID" "$NAV_PID" 2>/dev/null
    sleep 2
    kill -9 "$PC_PID" "$NAV_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# --- 3. Confirm the depth scan is actually flowing ---------------------------
echo "[3] Waiting for /scan_depth ..."
python3 - <<'PY'
import sys, time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

rclpy.init()
n = rclpy.create_node('depth_scan_check')
got = {}
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
n.create_subscription(LaserScan, '/scan_depth',
                      lambda m: got.setdefault('m', m), qos)
end = time.time() + 60
while time.time() < end and 'm' not in got:
    rclpy.spin_once(n, timeout_sec=0.5)
if 'm' in got:
    m = got['m']
    finite = [r for r in m.ranges if r == r and r < m.range_max]
    print(f"    OK - /scan_depth: {len(m.ranges)} beams, "
          f"{len(finite)} returns, closest "
          f"{min(finite) if finite else float('nan'):.2f} m")
else:
    print("    NO /scan_depth. Check /tmp/pc2scan.log - is the point cloud "
          "topic correct?")
rclpy.shutdown()
sys.exit(0 if 'm' in got else 1)
PY

echo ""
echo "============================================================"
echo "  Camera obstacles are live. Leave this terminal running."
echo "  Logs: /tmp/pc2scan.log  /tmp/nav2_camera.log"
echo "============================================================"
wait
