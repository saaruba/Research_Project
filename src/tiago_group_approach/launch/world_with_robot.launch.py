#!/usr/bin/env python3
"""
Start the custom world AND put TIAGo into it - the step that fails with
PAL's own launch when a custom world is used.

WHY THIS EXISTS
---------------
A Gazebo .world file contains the room, the furniture and the people. It does
NOT contain the robot. TIAGo is described separately (tiago_description) and
must be INSERTED into the running world by the /spawn_entity service.

That service is provided by libgazebo_ros_factory.so, which is a Gazebo SYSTEM
plugin - it has to be passed on the gzserver command line (-s ...), and cannot
be declared inside the <world> element. When tiago_gazebo.launch.py is given a
custom world_name it reports

    Private gazebo world package not found.

and starts gzserver without that plugin, so the world loads beautifully and the
robot is never inserted:

    [spawn_entity] Service /spawn_entity unavailable.
                   Was Gazebo started with GazeboRosFactory?

Everything downstream then fails for the same single reason - no robot means no
controller_manager, no /odom, no camera, and Nav2 waits forever on a transform
that will never arrive.

This launch avoids the problem by starting Gazebo through gazebo_ros's OWN
gazebo.launch.py, which always loads the init and factory plugins, and only
then running PAL's robot_spawn.launch.py to insert TIAGo and its controllers.

USAGE
-----
    ros2 launch tiago_group_approach world_with_robot.launch.py

    # a different world (name only - it is resolved in this project's worlds/)
    ros2 launch tiago_group_approach world_with_robot.launch.py world:=restaurant_final

    # headless (much faster on a software-rendered container)
    ros2 launch tiago_group_approach world_with_robot.launch.py gui:=false

Navigation is NOT started here - bring it up separately once you have confirmed
the robot exists and /mobile_base_controller/odom is publishing. Debugging one
layer at a time is the whole point of splitting them.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

PROJECT_WORLDS = "/workspaces/Research_Project/src/tiago_social_worlds/worlds"


def generate_launch_description():
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")

    gazebo_ros_share = get_package_share_directory("gazebo_ros")
    gazebo_launch = os.path.join(gazebo_ros_share, "launch", "gazebo.launch.py")

    # Full path to the world file, built from the short name.
    world_path = PythonExpression(
        ["'", PROJECT_WORLDS, "/' + '", world, "' + '.world'"])

    # Gazebo must find the LIRS actor meshes and any custom models.
    model_path = ":".join([
        "/workspaces/Research_Project/models",
        "/workspaces/Research_Project/src/tiago_social_worlds/models",
        os.path.expanduser("~/.gazebo/models"),
        "/usr/share/gazebo-11/models",
        "/opt/ros/humble/share/pal_gazebo_worlds/models",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "world", default_value="restaurant_testing",
            description="World name (without .world), resolved in this project's worlds/"),
        DeclareLaunchArgument(
            "gui", default_value="true",
            description="Show the Gazebo GUI. false is much faster here."),

        # Environment for the containerised/VNC setup.
        ExecuteProcess(cmd=["true"], output="log"),  # placeholder to keep ordering readable

        LogInfo(msg=["Starting Gazebo with world: ", world_path]),

        # gazebo_ros's own launch - this one DOES load libgazebo_ros_init.so and
        # libgazebo_ros_factory.so, which is what makes /spawn_entity exist.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                "world": world_path,
                "verbose": "true",
                "gui": gui,
                # Explicit, so the factory plugin is loaded even if a future
                # gazebo_ros version changes its defaults.
                "init": "true",
                "factory": "true",
            }.items(),
        ),

        # Give gzserver time to load the world (actor meshes are large and this
        # container renders in software). Spawning too early is exactly how the
        # 30-second /spawn_entity timeout happens.
        TimerAction(
            period=15.0,
            actions=[
                LogInfo(msg="Gazebo should be up - spawning TIAGo now..."),
                # robot_spawn.launch.py declares every hardware-variant argument
                # as REQUIRED (no defaults), so all of them must be supplied or
                # the launch aborts with
                #   "Included launch description missing required argument ..."
                # These values mirror tiago_gazebo.launch.py's own defaults, so
                # the robot spawns in its standard configuration.
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(
                        get_package_share_directory("tiago_gazebo"),
                        "launch", "robot_spawn.launch.py")),
                    launch_arguments={
                        "robot_name": "tiago",
                        "base_type": "pmb2",
                        "has_screen": "False",
                        "arm_type": "tiago-arm",
                        "arm_motor_model": "parker",
                        "end_effector": "pal-gripper",
                        "ft_sensor": "schunk-ft",
                        "wrist_model": "wrist-2017",
                        "camera_model": "orbbec-astra",
                        "laser_model": "sick-571",
                        "namespace": "",
                        "is_public_sim": "True",
                        "use_sim_time": "True",
                        # Spawn pose: a few metres from both groups (which sit at
                        # about (-3.3, -0.3) and (-0.3, 4.7)) and clear of walls.
                        "x": "0.0",
                        "y": "0.0",
                        "z": "0.0",
                        "yaw": "0.0",
                    }.items(),
                ),
            ],
        ),

        TimerAction(
            period=35.0,
            actions=[LogInfo(msg=(
                "\n"
                "============================================================\n"
                "CHECK THE ROBOT ACTUALLY EXISTS before starting anything else:\n"
                "  ros2 topic echo /mobile_base_controller/odom --once\n"
                "If that returns data, the robot is in the world and you can\n"
                "start navigation and the group-approach pipeline.\n"
                "============================================================"))],
        ),
    ])
