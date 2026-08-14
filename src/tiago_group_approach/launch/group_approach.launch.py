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
                        "'bc' (trained Behavioural Cloning model)"),
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
                'group_distance_m': group_distance,
                'detector': LaunchConfiguration('detector'),
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
            parameters=[{'standoff_distance': standoff}],
        ),

        # --- Policy: learned BC ----------------------------------------------
        Node(
            package='tiago_group_approach',
            executable='bc_policy_node',
            name='bc_policy_node',
            output='screen',
            condition=IfCondition(PythonExpression(["'", policy, "' == 'bc'"])),
            parameters=[{'model_path': model_path}],
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
