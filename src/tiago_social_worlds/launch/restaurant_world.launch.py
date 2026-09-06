
"""
LAUNCH THE EMPTY RESTAURANT IN GAZEBO - no people, no robot.

    ros2 launch tiago_social_worlds restaurant_world.launch.py

============================================================================
WHAT THIS DOES
============================================================================
Opens the restaurant room by itself: walls, floor, tables and chairs, and
nothing else. No humans, no TIAGo.

That makes it the right thing to run when you are working on the ROOM - moving
furniture, checking the map matches the world, confirming Gazebo renders at
all - because it starts in seconds and nothing else can be blamed for what you
see.

For the full scene with people, use restaurant_humans.launch.py. For an actual
experiment, use scripts/run_everything.sh, which brings up the world, the
robot, the map, localisation and Nav2 together.

============================================================================
THE ENVIRONMENT VARIABLES BELOW
============================================================================
This project runs inside a Docker container with a virtual display (VNC), not
on a normal desktop. Gazebo Classic assumes a real GPU and real display, and
without these variables it either renders a black window or cannot find the
model files at all. They are set here rather than in your shell so that
launching always works the same way regardless of which terminal you use.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """Create the restaurant Gazebo launch description."""

    gazebo_ros_share = get_package_share_directory("gazebo_ros")
    restaurant_share = get_package_share_directory("tiago_social_worlds")

    world_file = os.path.join(
        restaurant_share,
        "worlds",
        "restaurant_shell.world",
    )

    gazebo_launch_file = os.path.join(
        gazebo_ros_share,
        "launch",
        "gazebo.launch.py",
    )

    custom_models = os.path.join(
        restaurant_share,
        "models",
    )

    home_models = os.path.expanduser("~/.gazebo/models")

    gazebo_model_path = ":".join([
        custom_models,
        home_models,
        "/usr/share/gazebo-11/models",
        "/opt/ros/humble/share/pal_gazebo_worlds/models",
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            "world": world_file,
            "verbose": "true",
            "gui": "true",
        }.items(),
    )

    return LaunchDescription([
        # VNC display used by the development container.
        SetEnvironmentVariable(
            name="DISPLAY",
            value=":1",
        ),

        # Software rendering is more reliable through browser-based VNC.
        SetEnvironmentVariable(
            name="LIBGL_ALWAYS_SOFTWARE",
            value="1",
        ),

        # Prevent Qt shared-memory failures inside Docker.
        SetEnvironmentVariable(
            name="QT_X11_NO_MITSHM",
            value="1",
        ),

        # Gazebo Classic shaders, materials and media files.
        SetEnvironmentVariable(
            name="GAZEBO_RESOURCE_PATH",
            value="/usr/share/gazebo-11",
        ),

        # Prevent Gazebo from treating every ROS package as a model.
        SetEnvironmentVariable(
            name="GAZEBO_MODEL_PATH",
            value=gazebo_model_path,
        ),

        gazebo,
    ])
