#!/usr/bin/env bash
# ============================================================================
# Install the simulation stack this project needs: Nav2 + the TIAGo robot.
#
#   bash scripts/install_sim_stack.sh          install what is missing
#   bash scripts/install_sim_stack.sh --check  report only, install nothing
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The devcontainer was recreated during this project and everything installed
# outside /workspaces was lost - including the TIAGo simulation that used to
# live at ~/tiago_ws. A diagnostic on 13 Aug 2026 confirmed the container has
# ROS 2 Humble, Gazebo 11.10.2, RViz2 and gazebo_ros, but NO Nav2 and NO TIAGo
# robot packages at all.
#
# This script is idempotent: it checks first and only installs what is
# genuinely absent, so it is safe to re-run and safe to call from
# .devcontainer/post-create.sh on every container start.
#
# IMPORTANT: apt installs into /opt/ros/humble, which is INSIDE the container
# and therefore ephemeral. That is why this must be wired into post-create.sh
# rather than run once by hand - otherwise the next rebuild silently loses the
# simulator again and you are back to "Nav2 was never verified".
# ============================================================================

# Deliberately NOT -e: one failed package should not abort the whole run, we
# report per-package status instead.
#
# NOTE: 'set -u' (nounset) is NOT used either. ROS 2's own setup.bash reads
# AMENT_TRACE_SETUP_FILES without defining it first, so sourcing it under
# 'set -u' dies with "AMENT_TRACE_SETUP_FILES: unbound variable". Any script in
# this repo that sources a ROS setup file must avoid nounset, or wrap the
# source in 'set +u' / 'set -u'.
set -o pipefail

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "============================================================"
echo "SIMULATION STACK INSTALLER"
echo "============================================================"

if [ -z "${ROS_DISTRO:-}" ]; then
    if [ -f /opt/ros/humble/setup.bash ]; then
        echo "ROS 2 is not sourced. Sourcing Humble..."
        # shellcheck disable=SC1091
        source /opt/ros/humble/setup.bash
    else
        echo "WARNING: /opt/ros/humble/setup.bash not found - is this the right container?"
    fi
fi
echo "ROS_DISTRO: ${ROS_DISTRO:-unknown}"

# Packages the project actually needs, with why - so future-you knows what is
# safe to drop if disk space or install time becomes a problem.
PACKAGES=(
    "ros-humble-navigation2:Nav2 core - planners, controllers, behaviour tree"
    "ros-humble-nav2-bringup:Nav2 launch files + default params (needed to bring Nav2 up at all)"
    "ros-humble-tiago-description:TIAGo URDF/meshes - without this there is no robot to spawn"
    "ros-humble-tiago-simulation:TIAGo Gazebo integration, controllers, sensor plugins"
    "ros-humble-gazebo-ros-pkgs:Gazebo<->ROS bridge, spawn_entity service"
    "ros-humble-teleop-twist-keyboard:manual driving, useful for sanity-checking the robot moves"
)

echo ""
echo "--- Current status ---"
MISSING=()
for entry in "${PACKAGES[@]}"; do
    pkg="${entry%%:*}"
    why="${entry#*:}"
    if dpkg -s "$pkg" >/dev/null 2>&1; then
        printf "  INSTALLED  %-38s\n" "$pkg"
    else
        printf "  MISSING    %-38s  (%s)\n" "$pkg" "$why"
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo ""
    echo "Everything is already installed."
    CHECK_ONLY=1
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo ""
    echo "(check-only mode - nothing installed)"
else
    echo ""
    echo "--- Installing ${#MISSING[@]} package(s) ---"
    echo "This downloads several hundred MB and takes a few minutes."
    echo ""

    sudo apt-get update -qq

    FAILED=()
    for pkg in "${MISSING[@]}"; do
        echo ">>> $pkg"
        if sudo apt-get install -y --no-install-recommends "$pkg"; then
            echo "    OK"
        else
            echo "    FAILED"
            FAILED+=("$pkg")
        fi
    done

    if [ ${#FAILED[@]} -gt 0 ]; then
        echo ""
        echo "!! These packages failed to install:"
        printf '     %s\n' "${FAILED[@]}"
        echo ""
        echo "   If it was ros-humble-tiago-simulation or ros-humble-tiago-description,"
        echo "   the apt version may not exist for Humble in your sources. Fallback is a"
        echo "   source build INTO THE PERSISTENT MOUNT so it survives rebuilds:"
        echo ""
        echo "     mkdir -p /workspaces/tiago_ws/src && cd /workspaces/tiago_ws"
        echo "     vcs import --input \\"
        echo "       https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/humble-devel/tiago_public.repos src"
        echo "     rosdep install --from-paths src --ignore-src -y"
        echo "     colcon build --symlink-install"
        echo ""
        echo "   Then add 'source /workspaces/tiago_ws/install/setup.bash' to ~/.bashrc."
    fi
fi

# --- Install this project's custom worlds into PAL's search path ------------
# tiago_gazebo.launch.py takes `world_name` (a NAME, "will be converted to full
# path"), not an arbitrary path - it resolves the name inside pal_gazebo_worlds.
# The least invasive way to use our own scenes is therefore to drop them into
# that package's worlds/ directory, after which
#     world_name:=restaurant_humans
# just works alongside PAL's own worlds.
#
# That directory is under /opt/ros/humble, so it is EPHEMERAL - which is
# exactly why this is part of this script rather than a one-off manual copy.
# post-create.sh re-runs it on every container start.
install_custom_worlds() {
    local pal_worlds
    pal_worlds="$(ros2 pkg prefix pal_gazebo_worlds 2>/dev/null)/share/pal_gazebo_worlds/worlds"
    local src="/workspaces/Research_Project/src/tiago_social_worlds/worlds"

    if [ ! -d "$pal_worlds" ]; then
        echo "  pal_gazebo_worlds worlds/ not found - skipping custom world install."
        echo "  (looked in: $pal_worlds)"
        return
    fi
    if [ ! -d "$src" ]; then
        echo "  project worlds/ not found at $src - skipping."
        return
    fi

    local copied=0
    for world in "$src"/*.world; do
        [ -e "$world" ] || continue
        if sudo cp "$world" "$pal_worlds/" 2>/dev/null; then
            echo "  installed $(basename "$world")"
            copied=$((copied + 1))
        else
            echo "  FAILED to copy $(basename "$world") (permissions?)"
        fi
    done
    echo "  $copied world(s) available to tiago_gazebo via world_name:=<name>"
}

echo ""
echo "--- Installing custom worlds into pal_gazebo_worlds ---"
install_custom_worlds

# --- Verify -----------------------------------------------------------------
echo ""
echo "--- Verification (fresh ROS environment) ---"
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

verify() {
    local pattern="$1" label="$2"
    local n
    n=$(ros2 pkg list 2>/dev/null | grep -cE "$pattern" || true)
    if [ "${n:-0}" -gt 0 ]; then
        printf "  OK       %-24s (%s package(s))\n" "$label" "$n"
    else
        printf "  MISSING  %-24s\n" "$label"
    fi
}

verify '^nav2_'                 "Nav2"
verify '^tiago_description$'    "tiago_description"
verify '^tiago_gazebo$|^tiago_simulation$' "tiago gazebo/sim"
verify 'gazebo_ros'             "gazebo_ros"

echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
cat <<'GUIDE'
1. Open a NEW terminal (or re-source) so the new packages are on the path:
       source /opt/ros/humble/setup.bash
       source /workspaces/Research_Project/install/setup.bash

2. Sanity-check the robot exists before anything else:
       ros2 pkg list | grep tiago

3. MAKE THIS SURVIVE THE NEXT REBUILD - this is the step that matters.
   Add to .devcontainer/post-create.sh, near the top:

       bash /workspaces/Research_Project/scripts/install_sim_stack.sh || true

   Without it, the next container rebuild silently removes Nav2 and TIAGo
   again, exactly as happened before.
GUIDE
