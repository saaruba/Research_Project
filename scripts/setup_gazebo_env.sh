# ============================================================================
# Gazebo environment for this project.  SOURCE it, do not execute it:
#
#     source scripts/setup_gazebo_env.sh
#
# Then launch the simulation in the SAME terminal.
#
# ----------------------------------------------------------------------------
# WHY THIS IS NEEDED - the bug it fixes
# ----------------------------------------------------------------------------
# restaurant_testing.world references
#     model://free-wooden-round-dining-table-3d-model
# That model IS present, at models/restaurant_furniture/..., but Gazebo only
# searches GAZEBO_MODEL_PATH. Not finding it, Gazebo falls back to the ONLINE
# model database:
#
#     [Wrn] Getting models from[http://models.gazebosim.org/].
#           This may take a few seconds.
#     [Err] Unable to find uri[model://free-wooden-round-dining-table-3d-model]
#
# In a container with slow or blocked outbound network that lookup stalls
# gzserver for a long time - long enough that spawn_entity's 30 second wait for
# /spawn_entity expires and the robot is never inserted:
#
#     [spawn_entity] Service /spawn_entity unavailable. Exiting.
#
# and then everything downstream fails: no controller_manager, no /odom, Nav2
# stuck on "Invalid frame ID 'odom'", no camera, no perception.
#
# Pointing GAZEBO_MODEL_PATH at the right directory AND disabling the online
# database removes both the stall and the error.
# ============================================================================

PROJECT_ROOT=/workspaces/Research_Project

# --- Where Gazebo looks for model:// references ------------------------------
# NOTE the restaurant_furniture entry: Gazebo expects
#   <entry>/free-wooden-round-dining-table-3d-model/model.config
# so the parent of the model folder must be on the path, not models/ itself.
export GAZEBO_MODEL_PATH="\
${PROJECT_ROOT}/models/restaurant_furniture:\
${PROJECT_ROOT}/models:\
${PROJECT_ROOT}/src/tiago_social_worlds/models:\
${HOME}/.gazebo/models:\
/usr/share/gazebo-11/models:\
/opt/ros/humble/share/pal_gazebo_worlds/models:\
${GAZEBO_MODEL_PATH}"

# --- Stop Gazebo reaching out to the internet --------------------------------
# An empty database URI makes a missing model fail instantly instead of hanging
# on an HTTP request. Everything this project needs is available locally.
export GAZEBO_MODEL_DATABASE_URI=""

# --- Shaders and media -------------------------------------------------------
# Fixes: [Err] Unable to find shader lib. Shader generating will fail.
#        Your GAZEBO_RESOURCE_PATH is probably improperly set.
# which is also why gzclient crashed with "Failed to initialize scene".
export GAZEBO_RESOURCE_PATH="/usr/share/gazebo-11:/usr/share/gazebo:${GAZEBO_RESOURCE_PATH}"

# --- Actor meshes (LIRS-HMLG) referenced by absolute file:// paths -----------
export GAZEBO_PLUGIN_PATH="/opt/ros/humble/lib:${GAZEBO_PLUGIN_PATH}"

# --- Containerised display ---------------------------------------------------
export DISPLAY=${DISPLAY:-:1}
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-lcas}
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null

echo "Gazebo environment configured:"
echo "  GAZEBO_MODEL_PATH  (first entry): ${PROJECT_ROOT}/models/restaurant_furniture"
echo "  GAZEBO_MODEL_DATABASE_URI       : (disabled - no online lookups)"
echo "  GAZEBO_RESOURCE_PATH            : /usr/share/gazebo-11"
echo "  DISPLAY                         : $DISPLAY"
echo ""
if [ -d "${PROJECT_ROOT}/models/restaurant_furniture/free-wooden-round-dining-table-3d-model" ]; then
    echo "  OK   table model found locally - Gazebo will not go looking online"
else
    echo "  WARN table model NOT found - Gazebo may still report a missing uri"
fi
