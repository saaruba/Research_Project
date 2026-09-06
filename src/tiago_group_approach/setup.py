"""
PACKAGE INSTALL SCRIPT for tiago_group_approach - required by ROS 2.

============================================================================
WHAT THIS IS, IN PLAIN TERMS
============================================================================
ROS 2 does not run Python files straight from the folder you wrote them in.
Each package is BUILT and INSTALLED first, into an install/ directory, and it
is that installed copy the robot actually runs. This file is the recipe for
that install step: it tells the build system which Python modules exist, which
extra files (launch files, RViz layouts) to copy across, and which commands to
create so `ros2 run` can find them.

You almost never run this file yourself. It is invoked by:

    colcon build --packages-select tiago_group_approach
    source install/setup.bash

============================================================================
THE TRAP THAT CATCHES EVERYONE
============================================================================
Editing a node's .py file changes the SOURCE, not the installed copy. Until
you re-run colcon build, ROS keeps running the old version - so your change
appears to do nothing. If a fix seems to have been ignored, that is almost
always why.

The data_files section below is the other common trip hazard: launch files and
RViz configs must be listed explicitly or `ros2 launch` cannot find them, and
the error message does not make the cause obvious.
"""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tiago_group_approach'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files must be installed explicitly or `ros2 launch` cannot
        # find them - a very common and confusing ROS 2 packaging omission.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lcas',
    maintainer_email='saarunathan@gmail.com',
    description='Rule-based group-approach baseline for TIAGo (Phase E), sends Nav2 goals outside a detected group\'s O-space.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Perception: camera + depth -> /group_centroid in the map frame
            'group_perception_node = tiago_group_approach.group_perception_node:main',
            # Policies (drop-in swappable, both consume /group_centroid)
            'group_approach_baseline_node = tiago_group_approach.group_approach_baseline_node:main',
            'bc_policy_node = tiago_group_approach.bc_policy_node:main',
            # Localisation: continuous map->odom from Gazebo ground truth
            'gt_localisation_node = tiago_group_approach.gt_localisation_node:main',
            # Exploration: patrols the room so the camera can find people at all
            'explore_node = tiago_group_approach.explore_node:main',
            # Scripted patrol tour that ends the run and reports what it saw
            'mission_node = tiago_group_approach.mission_node:main',
            # Evaluation
            'metrics_recorder_node = tiago_group_approach.metrics_recorder_node:main',
        ],
    },
)
