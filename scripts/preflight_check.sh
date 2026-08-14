#!/usr/bin/env bash
# ============================================================================
# "Why isn't the robot moving?" - checks every link in the chain, in order.
#
#   bash scripts/preflight_check.sh
#
# Run it with the simulation ALREADY RUNNING. Each check corresponds to one
# thing that must be true for TIAGo to drive to a group. The first FAIL is
# your problem - later checks often fail simply as a consequence of it, so
# fix them top-down and re-run.
# ============================================================================

set -o pipefail   # NOT 'set -u': ROS setup.bash reads undefined variables

PASS=0
FAIL=0

ok()   { printf "  \033[0;32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[0;31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL+1)); }
info() { printf "        %s\n" "$1"; }

echo "============================================================"
echo "PREFLIGHT - is everything the robot needs actually running?"
echo "============================================================"

# --- 0. environment ---------------------------------------------------------
echo ""
echo "[0] Environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    bad "inside virtualenv $VIRTUAL_ENV - run 'deactivate' first"
else
    ok "not in a virtualenv"
fi
if [ -z "${ROS_DISTRO:-}" ]; then
    bad "ROS not sourced - run: source /opt/ros/humble/setup.bash"
else
    ok "ROS $ROS_DISTRO sourced"
fi
if python3 -c "import ultralytics" 2>/dev/null; then
    ok "ultralytics importable by $(command -v python3)"
else
    bad "ultralytics NOT importable - perception node will die on startup"
    info "fix: python3 -m pip install --user ultralytics"
fi

# --- 1. simulator ------------------------------------------------------------
echo ""
echo "[1] Simulator and sensors"
TOPICS=$(ros2 topic list 2>/dev/null)
check_topic() {
    if echo "$TOPICS" | grep -q "^$1$"; then ok "$1"; else bad "$1 missing"; fi
}
check_topic /scan_raw
check_topic /head_front_camera/rgb/image_raw
check_topic /head_front_camera/depth/image_raw
check_topic /head_front_camera/rgb/camera_info

# Is the camera actually PUBLISHING, or just advertised? A topic can exist
# with nobody sending anything, which looks fine in `topic list` and produces
# a perception node that silently never runs.
echo ""
echo "    camera actually streaming?"
if timeout 8 ros2 topic hz /head_front_camera/rgb/image_raw --window 5 2>/dev/null | head -2 | grep -q "average rate"; then
    ok "RGB camera is publishing"
else
    bad "no frames on the RGB camera within 8s"
    info "Gazebo may be paused, or the head camera is disabled"
fi

# --- 2. navigation -----------------------------------------------------------
echo ""
echo "[2] Nav2 - THE most common reason the robot never moves"
ACTIONS=$(ros2 action list 2>/dev/null)
if echo "$ACTIONS" | grep -q "navigate_to_pose"; then
    ok "navigate_to_pose action server exists"
else
    bad "navigate_to_pose MISSING - Nav2 is not running"
    info "You launched the sim without navigation. Relaunch with:"
    info "  ros2 launch tiago_gazebo tiago_gazebo.launch.py \\"
    info "      is_public_sim:=True world_name:=restaurant_testing \\"
    info "      navigation:=True slam:=True moveit:=False arm_type:=no-arm"
fi

# Nav2 cannot plan without a map->base_footprint transform chain.
echo ""
echo "    localisation (map -> base_footprint TF)?"
if timeout 8 ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | head -5 | grep -q "Translation"; then
    ok "robot is localised in the map frame"
else
    bad "no map -> base_footprint transform"
    info "SLAM/AMCL has not localised yet. With slam:=True this usually"
    info "appears a few seconds after launch; with AMCL you must first set"
    info "an initial pose using RViz's '2D Pose Estimate' tool."
fi

# --- 3. this project's pipeline ---------------------------------------------
echo ""
echo "[3] Group-approach pipeline"
NODES=$(ros2 node list 2>/dev/null)
for n in group_perception_node metrics_recorder_node; do
    if echo "$NODES" | grep -q "$n"; then ok "$n running"; else bad "$n not running"; fi
done
if echo "$NODES" | grep -qE "group_approach_baseline_node|bc_policy_node"; then
    ok "a policy node is running"
else
    bad "no policy node running"
fi

echo ""
echo "    is perception finding anyone? (waiting up to 15s for /group_centroid)"
CENTROID=$(timeout 15 ros2 topic echo /group_centroid --once 2>/dev/null)
if [ -n "$CENTROID" ]; then
    ok "/group_centroid is publishing"
    echo "$CENTROID" | grep -E "^\s+(x|y):" | head -2 | sed 's/^/        /'
else
    bad "nothing on /group_centroid within 15s"
    info "The policy has no target, so the robot will never move."
    info "Causes, in order of likelihood:"
    info "  - no people in the camera's view: turn the robot to face a group"
    info "  - detected people are further apart than group_distance_m (1.5 m)"
    info "  - perception node died: check its terminal for a traceback"
fi

# --- summary -----------------------------------------------------------------
echo ""
echo "============================================================"
printf "  %d passed, %d failed\n" "$PASS" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "  Everything is up. The robot should drive once it sees a group."
else
    echo "  Fix the FIRST failure above, then re-run this script."
fi
echo "============================================================"
