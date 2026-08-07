cmake_minimum_required(VERSION 3.8)

project(tiago_social_worlds)

find_package(ament_cmake REQUIRED)

# Install only the ROS package resources.
#
# Large external Gazebo models are intentionally stored outside this
# ROS package at:
#
# /workspaces/Research_Project/models
#
# Gazebo accesses those models through GAZEBO_MODEL_PATH, so they do
# not need to be copied into the colcon install directory.
install(
  DIRECTORY
    launch
    worlds
    config
    maps
    rviz
    scripts
    docs
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

ament_package()