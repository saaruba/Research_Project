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
