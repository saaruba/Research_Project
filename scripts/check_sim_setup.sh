#!/usr/bin/env bash
# ============================================================================
# Simulation stack diagnostic - what is ACTUALLY installed in this container?
#
#   deactivate            # leave the la3b_env venv first!
#   bash scripts/check_sim_setup.sh
#
# Context: the devcontainer was recreated at least once during this project
# (the hostname changed). Anything installed inside the container but OUTSIDE
# /workspaces does not survive that - including a TIAGo workspace at
# ~/tiago_ws. This script establishes ground truth rather than trusting the
# setup notes in docs/.
# ============================================================================

echo "============================================================"
echo "SIMULATION STACK DIAGNOSTIC"
echo "============================================================"
echo "hostname : $(hostname)"
echo "user     : $(whoami)"
echo "date     : $(date)"

# --- venv contamination check -----------------------------------------------
echo ""
echo "[0] Python environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "    !! You are inside a virtualenv: $VIRTUAL_ENV"
    echo "       Run 'deactivate' before doing ROS work - the la3b_env has"
    echo "       numpy 1.25 and its own python, which can shadow ROS packages."
else
    echo "    OK - not inside a virtualenv"
fi
echo "    python3: $(command -v python3)"

# --- ROS 2 ------------------------------------------------------------------
echo ""
echo "[1] ROS 2"
if [ -z "${ROS_DISTRO:-}" ]; then
    echo "    ROS_DISTRO not set - ROS 2 is not sourced in this shell."
    echo "    Try: source /opt/ros/humble/setup.bash"
else
    echo "    ROS_DISTRO   : $ROS_DISTRO"
    echo "    AMENT_PREFIX_PATH entries:"
    echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | sed 's/^/      /'
fi
if command -v ros2 >/dev/null 2>&1; then
    TOTAL=$(ros2 pkg list 2>/dev/null | wc -l)
    echo "    total ROS packages visible: $TOTAL"
else
    echo "    'ros2' command NOT FOUND"
fi

# --- The things the project actually needs ----------------------------------
echo ""
echo "[2] Required components"

check_pkg() {
    local pattern="$1" label="$2"
    local hits
    hits=$(ros2 pkg list 2>/dev/null | grep -icE "$pattern" || true)
    if [ "${hits:-0}" -gt 0 ]; then
        printf "    %-22s PRESENT (%s package(s))\n" "$label" "$hits"
    else
        printf "    %-22s MISSING\n" "$label"
    fi
}

check_pkg '^nav2_|^nav2$'        "Nav2"
check_pkg 'gazebo_ros'           "gazebo_ros"
check_pkg '^rviz2$|^rviz_'       "RViz2"
check_pkg 'tiago'                "TIAGo (any)"
check_pkg 'tiago_description'    "  tiago_description"
check_pkg 'tiago_gazebo'         "  tiago_gazebo"
check_pkg 'tiago_2dnav|tiago_nav' "  tiago navigation"
check_pkg 'play_motion|pal_'     "PAL support pkgs"
check_pkg 'slam_toolbox'         "slam_toolbox"

# --- Binaries ---------------------------------------------------------------
echo ""
echo "[3] Simulator binaries"
for bin in gazebo gzserver gzclient rviz2; do
    if command -v "$bin" >/dev/null 2>&1; then
        printf "    %-10s %s\n" "$bin" "$(command -v $bin)"
    else
        printf "    %-10s MISSING\n" "$bin"
    fi
done
if command -v gazebo >/dev/null 2>&1; then
    echo "    gazebo version: $(gazebo --version 2>/dev/null | head -1)"
fi

# --- Did an old TIAGo workspace survive? ------------------------------------
echo ""
echo "[4] TIAGo workspace on disk"
for d in "$HOME/tiago_ws" /opt/tiago_ws /workspaces/tiago_ws; do
    if [ -d "$d" ]; then
        echo "    FOUND: $d"
        ls "$d" 2>/dev/null | sed 's/^/        /'
    else
        echo "    absent: $d"
    fi
done
echo ""
echo "    NOTE: only /workspaces/... survives a container rebuild. Anything"
echo "    installed to \$HOME is ephemeral. If TIAGo was installed to"
echo "    ~/tiago_ws previously, that is why it is gone."

# --- apt availability -------------------------------------------------------
echo ""
echo "[5] Are the packages installable from apt?"
if command -v apt-cache >/dev/null 2>&1; then
    for p in ros-humble-nav2-bringup ros-humble-navigation2 \
             ros-humble-tiago-description ros-humble-tiago-simulation \
             ros-humble-gazebo-ros-pkgs; do
        if apt-cache show "$p" >/dev/null 2>&1; then
            printf "    %-36s available\n" "$p"
        else
            printf "    %-36s NOT in apt sources\n" "$p"
        fi
    done
else
    echo "    apt-cache unavailable"
fi

echo ""
echo "============================================================"
echo "Paste this whole output back so the install plan targets what"
echo "is actually missing, rather than guessing."
echo "============================================================"
