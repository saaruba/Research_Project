#!/usr/bin/env python3
"""
Phase G: the full group-approach pipeline, in one launch.

    camera+depth -> perception -> /group_centroid -> policy -> Nav2 -> metrics

Run this AFTER the simulation and Nav2 are already up, e.g.:

    # terminal 1 - simulator + Nav2 (PAL's own launch)
    ros2 launch tiago_gazebo tiago_gazebo.launch.py \\
        is_public_sim:=True world_name:=restaurant_humans \\
        navigation:=True moveit:=False arm_type:=no-arm

    # terminal 2 - this pipeline, rule-based policy
    ros2 launch tiago_group_approach group_approach.launch.py policy:=rule

    # ... or the learned policy, for the Objective 4 comparison
    ros2 launch tiago_group_approach group_approach.launch.py policy:=bc

THE EXPERIMENT THIS ENABLES
----------------------------
Objective 4 requires comparing Behavioural Cloning against a rule-based
baseline. Both policy nodes consume the same /group_centroid and both emit
NavigateToPose goals, so the ONLY thing that changes between runs is
`policy:=`. Same world, same start pose, same groups, one variable. That is
the experiment; everything else is bookkeeping.

Set `metrics:=false` while debugging so you do not litter the results
directory with half-finished runs.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PROJECT_ROOT = '/workspaces/Research_Project'
DEFAULT_MODEL = os.path.join(
    PROJECT_ROOT, 'dataset', 'processed', 'models',
    'approach_pose_random_forest_tuned.joblib')
DEFAULT_GROUNDTRUTH = os.path.join(
    PROJECT_ROOT, 'src', 'tiago_social_worlds', 'worlds',
    'restaurant_humans.groundtruth.json')
DEFAULT_RESULTS = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'sim_results')


def generate_launch_description():
    policy = LaunchConfiguration('policy')
    metrics = LaunchConfiguration('metrics')
    groundtruth = LaunchConfiguration('groundtruth')
    model_path = LaunchConfiguration('model_path')
    output_dir = LaunchConfiguration('output_dir')
    group_distance = LaunchConfiguration('group_distance_m')
    standoff = LaunchConfiguration('standoff_distance')

    return LaunchDescription([
        DeclareLaunchArgument(
            'policy', default_value='rule',
            description="Which policy to run: 'rule' (geometric baseline) or "
                        "'bc' (trained Random Forest) or 'mlp' (trained MLP). "
                        "'bc' and 'mlp' run the same node with a different "
                        ".joblib - pass model_path to choose it."),
        DeclareLaunchArgument(
            'metrics', default_value='true',
            description='Record evaluation metrics for this run'),
        DeclareLaunchArgument(
            'groundtruth', default_value=DEFAULT_GROUNDTRUTH,
            description='World ground-truth JSON, for the social metrics. '
                        'Without it, O-space/distance/cut-through cannot be scored.'),
        DeclareLaunchArgument('model_path', default_value=DEFAULT_MODEL),
        DeclareLaunchArgument('output_dir', default_value=DEFAULT_RESULTS),
        DeclareLaunchArgument(
            'detector', default_value='yolo',
            description="Perception backend: 'yolo' (YOLOv8n, ~200 FPS, default) "
                        "or 'locateanything' (nvidia/LocateAnything-3B via the local "
                        "service, ~25 s/frame - start scripts/locateanything_service.py "
                        "in the la3b_env venv first; auto-enables one-shot mode)"),
        DeclareLaunchArgument(
            'group_distance_m', default_value='1.5',
            description='Max separation for two people to count as one group'),
        DeclareLaunchArgument(
            'min_group_size', default_value='2',
            description="How many people make a 'group'. Default 2 - a lone "
                        "person is not a conversational group, so with a "
                        "one-person world nothing is ever published on "
                        "/group_centroid and the policy waits forever. Set to 1 "
                        "to treat a single person as an approach target."),
        DeclareLaunchArgument(
            'mission', default_value='true',
            description='Run the scripted patrol tour: (-5,5) -> (3,6) -> '
                        '(8,1) -> (8,-6) -> (-8,-4) -> back to (-5,5), then '
                        'stop and report. Repeatable, so rule vs BC differ '
                        'only in behaviour.'),
        DeclareLaunchArgument(
            'obstacle_wait', default_value='15.0',
            description='Seconds to wait when a waypoint is blocked, in case '
                        'the obstacle is a person walking past'),
        DeclareLaunchArgument(
            'explore', default_value='false',
            description='Patrol the room when no group is visible. Without it '
                        'the robot only moves once it can already see someone, '
                        'and it spawns facing away from them.'),
        DeclareLaunchArgument(
            'idle_timeout', default_value='12.0',
            description='Seconds without a detection before patrolling resumes'),
        DeclareLaunchArgument(
            'confidence', default_value='0.45',
            description="YOLO person-detection threshold. The old default of "
                        "0.4 is permissive: in a synthetic Gazebo scene it "
                        "accepts furniture and wall edges as people, and with "
                        "min_group_size=1 a single false positive is enough to "
                        "send the robot off to approach nothing."),
        DeclareLaunchArgument(
            'standoff_distance', default_value='1.2',
            description="Rule-based policy's standoff from the group centre "
                        "(Hall's proxemics social-space boundary)"),

        LogInfo(msg=['Group-approach pipeline starting with policy: ', policy]),

        # --- Perception (always) ---------------------------------------------
        Node(
            package='tiago_group_approach',
            executable='group_perception_node',
            name='group_perception_node',
            output='screen',
            parameters=[{
                # Same string->type cast issue as min_group_size below.
                'group_distance_m': ParameterValue(group_distance, value_type=float),
                # A LaunchConfiguration is always a STRING. The node declares
                # min_group_size as an int, and ROS 2 rejects a type mismatch at
                # startup, so it must be cast explicitly.
                'min_group_size': ParameterValue(
                    LaunchConfiguration('min_group_size'), value_type=int),
                'detector': LaunchConfiguration('detector'),
                'confidence': ParameterValue(
                    LaunchConfiguration('confidence'), value_type=float),
                'rgb_topic': '/head_front_camera/rgb/image_raw',
                'depth_topic': '/head_front_camera/depth/image_raw',
                'camera_info_topic': '/head_front_camera/rgb/camera_info',
            }],
        ),

        # --- Policy: rule-based ----------------------------------------------
        Node(
            package='tiago_group_approach',
            executable='group_approach_baseline_node',
            name='group_approach_baseline_node',
            output='screen',
            condition=IfCondition(PythonExpression(["'", policy, "' == 'rule'"])),
            # float cast: the node declares standoff_distance as 1.2 (float),
            # but a LaunchConfiguration is a string.
            parameters=[{'standoff_distance': ParameterValue(standoff, value_type=float)}],
        ),

        # --- Policy: learned BC ----------------------------------------------
        Node(
            package='tiago_group_approach',
            executable='bc_policy_node',
            name='bc_policy_node',
            output='screen',
            # Both learned policies use this same node - only the .joblib
            # differs. Running the MLP as well as the Random Forest matches the
            # offline evaluation, which compared rule / RF / MLP / naive; live
            # results for all three make the two analyses directly comparable
            # instead of the live study testing a subset.
            condition=IfCondition(PythonExpression(
                ["'", policy, "' in ('bc', 'mlp')"])),
            parameters=[{'model_path': model_path}],
        ),

        # --- Exploration -------------------------------------------------------
        # Without this the robot never moves unless it can already see someone,
        # and it spawns facing away from them. Set explore:=false to disable.
        Node(
            package='tiago_group_approach',
            executable='explore_node',
            name='explore_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('explore')),
            parameters=[{
                'idle_timeout': ParameterValue(
                    LaunchConfiguration('idle_timeout'), value_type=float),
            }],
        ),

        # --- Scripted mission ---------------------------------------------------
        Node(
            package='tiago_group_approach',
            executable='mission_node',
            name='mission_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('mission')),
            parameters=[{
                'obstacle_wait': ParameterValue(
                    LaunchConfiguration('obstacle_wait'), value_type=float),
            }],
        ),

        # --- Metrics ----------------------------------------------------------
        Node(
            package='tiago_group_approach',
            executable='metrics_recorder_node',
            name='metrics_recorder_node',
            output='screen',
            condition=IfCondition(metrics),
            parameters=[{
                'groundtruth': groundtruth,
                'policy_name': policy,
                'output_dir': output_dir,
            }],
        ),
    ])
