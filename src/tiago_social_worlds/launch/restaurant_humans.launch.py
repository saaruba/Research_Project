
"""
LAUNCH THE RESTAURANT WITH PEOPLE IN IT - the scene, without the robot.

    ros2 launch tiago_social_worlds restaurant_humans.launch.py

============================================================================
WHAT THIS DOES
============================================================================
Opens the restaurant room WITH its human figures placed: the conversational
groups the robot is meant to approach, plus lone individuals acting as
distractors.

Use it to check the scene itself - are the groups standing in sensible
formations, are people textured rather than grey, does the camera see them
clearly. It does not spawn TIAGo, so nothing moves and nothing can go wrong
with navigation while you are looking.

    restaurant_world.launch.py    room only, fastest to start
    restaurant_humans.launch.py   room + people  <- THIS FILE
    scripts/run_everything.sh     room + people + robot + map + Nav2

============================================================================
ONE THING WORTH KNOWING ABOUT THE PEOPLE
============================================================================
Gazebo <actor> elements are VISUAL ONLY - they have no collision geometry, so
the robot's LiDAR passes straight through them and Nav2 will happily plan a
path through a person. This project works around it by adding separate
collision cylinders (see scripts/add_person_collisions.py). If you edit the
world by hand and people stop being obstacles, that is why.

============================================================================
THE ENVIRONMENT VARIABLES BELOW
============================================================================
This project runs inside a Docker container with a virtual display (VNC).
Gazebo Classic assumes a real GPU and display; without these settings it
renders a black window or cannot locate the model files.
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
        "restaurant_humans.world",
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
