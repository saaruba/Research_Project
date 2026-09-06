"""
PACKAGE INSTALL SCRIPT for tiago_head_control - required by ROS 2.

This is the build recipe for the small head-control package, in the same form
as src/tiago_group_approach/setup.py (see that file for a fuller explanation
of what a ROS 2 setup.py does and why editing a node is not enough on its own).

The package contains one node, head_scan_node, which aims TIAGo's head camera.
It is a setup and debugging aid and is not part of the group-approach
experiment.

    colcon build --packages-select tiago_head_control
    source install/setup.bash
    ros2 run tiago_head_control head_scan_node
"""

from setuptools import find_packages, setup

package_name = 'tiago_head_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lcas',
    maintainer_email='saarunathan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'head_scan_node = tiago_head_control.head_scan_node:main',
        ],
    },
)
