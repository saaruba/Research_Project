#!/usr/bin/env python3
"""
Phase E: rule-based group-approach baseline.

Rule (matches the O-space decision already made in Phase C, re-applied here
to REAL-WORLD map-frame coordinates instead of image pixels):
    1. Get the robot's current position from TF (map -> base_link).
    2. Get a detected group's centroid (subscribed from /group_centroid,
       geometry_msgs/PointStamped, in the map frame).
    3. Draw a line from the robot's position through the group centroid.
    4. Place the goal pose `standoff_distance` metres short of the centroid
       along that line (i.e. outside the group's O-space), facing back
       toward the group centre.
    5. Send that pose to Nav2 (`NavigateToPose` action) so TIAGo drives
       there.

This is deliberately the simplest thing that could work - it is the
COMPARISON POINT for the trained BC model (Phase F), not the final
approach. Per the checklist: build this first, prove the pipeline shape
end-to-end, then see if the learned model beats it.

Standoff distance default (1.2 m) is chosen per Hall's proxemics as the
boundary between "personal" and "social" space for approaching a group of
strangers - a defensible, citable default, and it's exposed as a ROS 2
parameter so it can be tuned/justified in the dissertation without a code
change.

HOW TO TEST (in the TIAGo devcontainer, with the sim + Nav2 already
running):
    # Terminal 1: launch the restaurant world (existing package)
    ros2 launch tiago_social_worlds restaurant_world.launch.py

    # Terminal 2: bring up Nav2 for TIAGo (adjust to however your
    # tiago_simulation workspace normally launches navigation - this
    # package does not replace that step, it only sends goals to it)

    # Terminal 3: run this baseline node
    ros2 run tiago_group_approach group_approach_baseline_node

    # Terminal 4: simulate a detected group centroid at map (2.0, 1.0)
    ros2 topic pub --once /group_centroid geometry_msgs/msg/PointStamped \
        "{header: {frame_id: 'map'}, point: {x: 2.0, y: 1.0, z: 0.0}}"

    # TIAGo should navigate to a point ~1.2m short of (2.0, 1.0), facing it.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose

import tf2_ros
from tf2_ros import TransformException


class GroupApproachBaselineNode(Node):
    def __init__(self):
        super().__init__('group_approach_baseline_node')

        self.declare_parameter('standoff_distance', 1.2)  # metres, Hall's social-space boundary
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.subscription = self.create_subscription(
            PointStamped,
            '/group_centroid',
            self.group_centroid_callback,
            10,
        )

        self.get_logger().info(
            'Group-approach baseline node ready. '
            f'standoff_distance={self.get_parameter("standoff_distance").value} m. '
            'Waiting for a group centroid on /group_centroid...'
        )

    def get_robot_position(self):
        map_frame = self.get_parameter('map_frame').value
        robot_frame = self.get_parameter('robot_frame').value
        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame, robot_frame, rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
            return transform.transform.translation.x, transform.transform.translation.y
        except TransformException as ex:
            self.get_logger().error(f'Could not get robot position from TF: {ex}')
            return None

    def compute_approach_pose(self, group_x: float, group_y: float, robot_x: float, robot_y: float):
        standoff = self.get_parameter('standoff_distance').value

        direction_x = group_x - robot_x
        direction_y = group_y - robot_y
        distance = math.hypot(direction_x, direction_y)

        if distance < 1e-6:
            self.get_logger().warn('Robot is already at the group centroid - cannot compute a direction.')
            return None

        unit_x = direction_x / distance
        unit_y = direction_y / distance

        # Stand `standoff` metres short of the centroid, outside its O-space.
        approach_x = group_x - unit_x * standoff
        approach_y = group_y - unit_y * standoff

        # Face back toward the group centre.
        facing_yaw = math.atan2(group_y - approach_y, group_x - approach_x)

        if distance < standoff:
            self.get_logger().warn(
                f'Robot is only {distance:.2f}m from the group centroid, closer than the '
                f'{standoff:.2f}m standoff distance - the computed approach point may sit '
                'behind the robot. Check the group detection input.'
            )

        return approach_x, approach_y, facing_yaw

    def group_centroid_callback(self, msg: PointStamped):
        robot_position = self.get_robot_position()
        if robot_position is None:
            return
        robot_x, robot_y = robot_position

        result = self.compute_approach_pose(msg.point.x, msg.point.y, robot_x, robot_y)
        if result is None:
            return
        approach_x, approach_y, facing_yaw = result

        self.get_logger().info(
            f'Group centroid ({msg.point.x:.2f}, {msg.point.y:.2f}) -> '
            f'approach pose ({approach_x:.2f}, {approach_y:.2f}), facing {math.degrees(facing_yaw):.1f} deg'
        )

        self.send_nav_goal(approach_x, approach_y, facing_yaw, frame_id=msg.header.frame_id or self.get_parameter('map_frame').value)

    def send_nav_goal(self, x: float, y: float, yaw: float, frame_id: str):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 "navigate_to_pose" action server not available - is Nav2 running?')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'Sending Nav2 goal: ({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f} deg')
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 goal was rejected.')
            return
        self.get_logger().info('Nav2 goal accepted - robot is moving.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Nav2 goal finished. Result: {result}')


def main(args=None):
    rclpy.init(args=args)
    node = GroupApproachBaselineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
