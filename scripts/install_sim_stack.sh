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
    # This container's base image ships Python WITHOUT pip or venv. That is why
    # `python3 -m venv` fails with "ensurepip is not available" and why
    # `python3 -m pip` reports "No module named pip". Both are needed: pip to
    # install ultralytics for the ROS perception node, venv for the isolated
    # LocateAnything environment.
    "python3-pip:pip for the SYSTEM python - needed to install ultralytics"
    "python3-venv:venv module - needed for the LocateAnything virtualenv"
    "ros-humble-navigation2:Nav2 core - planners, controllers, behaviour tree"
    "ros-humble-nav2-bringup:Nav2 launch files + default params (needed to bring Nav2 up at all)"
    "ros-humble-tiago-description:TIAGo URDF/meshes - without this there is no robot to spawn"
    "ros-humble-tiago-simulation:TIAGo Gazebo integration, controllers, sensor plugins"
    "ros-humble-gazebo-ros-pkgs:Gazebo<->ROS bridge, spawn_entity service"
    "ros-humble-teleop-twist-keyboard:manual driving, useful for sanity-checking the robot moves"
    "ros-humble-slam-toolbox:SLAM - lets Nav2 localise in a custom world with no prebuilt map"
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

# --- Python packages the ROS nodes import -----------------------------------
# group_perception_node imports ultralytics (YOLOv8). It runs under
# /usr/bin/python3 - NOT the la3b_env venv - so ultralytics must be installed
# for the system python. Installing it into the venv instead is a very easy
# mistake to make and produces a confusing ModuleNotFoundError at launch.
echo ""
echo "--- Python packages for the ROS nodes ---"
if python3 -c "import ultralytics" 2>/dev/null; then
    echo "  ultralytics already importable by /usr/bin/python3"
elif python3 -m pip --version >/dev/null 2>&1; then
    echo "  installing ultralytics for the system python (this pulls PyTorch - several minutes)..."
    python3 -m pip install --user ultralytics || \
        echo "  FAILED - install manually: python3 -m pip install --user ultralytics"
    # ------------------------------------------------------------------
    # REPAIR THE BUILD TOOLCHAIN AFTER INSTALLING ultralytics.
    #
    # ultralytics drags in a modern setuptools, which then calls
    #   packaging.utils.canonicalize_version(..., strip_trailing_zero=True)
    # That keyword only exists in packaging >= 24.0, but Ubuntu 22.04 ships
    # packaging 21.3 - so every subsequent `colcon build` dies with:
    #   TypeError: canonicalize_version() got an unexpected keyword argument
    # Separately, setuptools >= 80 drops the setup.py entry points colcon
    # relies on for ament_python packages.
    # Pinning both keeps colcon working after a Python install.
    # ------------------------------------------------------------------
    echo "  repairing build toolchain (packaging / setuptools) for colcon..."
    python3 -m pip install --user --upgrade "packaging>=24.0" "setuptools<80" || \
        echo "  FAILED - run manually: python3 -m pip install --user --upgrade 'packaging>=24.0' 'setuptools<80'"
else
    echo "  pip is STILL unavailable for /usr/bin/python3."
    echo "  Install it first, then re-run this script:"
    echo "      sudo apt-get update && sudo apt-get install -y python3-pip"
fi

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
