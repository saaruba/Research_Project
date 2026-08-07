#!/usr/bin/env python3

"""
Launch the custom restaurant shell in Gazebo Classic.

The environment variables below prevent Gazebo Classic rendering and
model-path problems when running inside the project's Docker/VNC setup.
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
