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
            'group_approach_baseline_node = tiago_group_approach.group_approach_baseline_node:main',
        ],
    },
)
