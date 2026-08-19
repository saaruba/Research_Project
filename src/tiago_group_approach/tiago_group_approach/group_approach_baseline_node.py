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

from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose

import tf2_ros
from tf2_ros import TransformException


class GroupApproachBaselineNode(Node):
    def __init__(self):
        super().__init__('group_approach_baseline_node')

        self.declare_parameter('standoff_distance', 1.2)  # metres, Hall's social-space boundary
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')

        # --- Goal throttling ------------------------------------------------
        # Perception publishes a centroid at camera rate. Without throttling,
        # every message sent a fresh NavigateToPose goal, each one PREEMPTING
        # the last, so Nav2 restarted planning continuously and the robot
        # barely moved - which matches the recorded runs where final_pose was
        # still essentially the spawn point. A new goal is now only sent if the
        # target has actually moved, or the previous goal has finished.
        self.declare_parameter('goal_update_threshold_m', 0.40)
        self.declare_parameter('min_goal_interval_s', 3.0)

        # --- Personal-space clearance ----------------------------------------
        # The standoff alone is not enough. It is measured from the group
        # CENTROID, and the centroid is computed from whoever the camera can
        # currently see. With a group of four and only two visible, the
        # centroid sits off to one side, and a point 1.2 m from it can land
        # *inside* the real group. Measured in a live run: O-space intrusion
        # true, closest approach 0.43 m - inside intimate distance.
        #
        # So the chosen pose is now also checked against EVERY detected person
        # individually, and pushed back along the approach line until it clears
        # them all.
        self.declare_parameter('min_person_clearance', 1.0)
        self.declare_parameter('max_standoff', 3.0)

        self._people: list[tuple[float, float]] = []
        self._goal_in_flight = False
        self._last_goal_xy = None
        self._last_goal_time = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Individual people, so the goal can be checked against each of them
        # rather than only against the group's average position.
        self.create_subscription(
            PoseArray, '/detected_people', self.people_callback, 10)

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

    def people_callback(self, msg: PoseArray) -> None:
        """Latest individual person positions, in the map frame."""
        self._people = [(p.position.x, p.position.y) for p in msg.poses]

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

        # Stand `standoff` metres short of the centroid, outside its O-space...
        approach_x = group_x - unit_x * standoff
        approach_y = group_y - unit_y * standoff

        # ...then back off further, if that spot crowds anybody.
        #
        # Walk outward along the same approach line in 10 cm steps until the
        # pose clears every detected person by min_person_clearance. This keeps
        # the rule's character - approach along the line of sight, stop short,
        # face the group - while making it respect the people who define the
        # O-space rather than only their average position.
        clearance = self.get_parameter('min_person_clearance').value
        max_standoff = self.get_parameter('max_standoff').value
        if self._people:
            extra = 0.0
            while standoff + extra <= max_standoff:
                cx = group_x - unit_x * (standoff + extra)
                cy = group_y - unit_y * (standoff + extra)
                nearest = min(math.dist((cx, cy), p) for p in self._people)
                if nearest >= clearance:
                    break
                extra += 0.1
            if extra > 0.0:
                approach_x = group_x - unit_x * (standoff + extra)
                approach_y = group_y - unit_y * (standoff + extra)
                self.get_logger().info(
                    f'Backed off an extra {extra:.1f} m: the {standoff:.1f} m '
                    f'pose was within {clearance:.1f} m of a person '
                    f'({len(self._people)} detected).')

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

        # --- Should this become a new Nav2 goal? -----------------------------
        now = self.get_clock().now().nanoseconds / 1e9
        threshold = self.get_parameter('goal_update_threshold_m').value
        interval = self.get_parameter('min_goal_interval_s').value

        if now - self._last_goal_time < interval:
            return

        if self._goal_in_flight and self._last_goal_xy is not None:
            moved = math.hypot(approach_x - self._last_goal_xy[0],
                               approach_y - self._last_goal_xy[1])
            if moved < threshold:
                # Same target, goal already running - let the robot drive
                # instead of restarting the plan.
                return
            self.get_logger().info(
                f'Target moved {moved:.2f} m (> {threshold:.2f} m) - re-planning.')

        self._last_goal_xy = (approach_x, approach_y)
        self._last_goal_time = now

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
        self._goal_in_flight = True
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 goal was rejected.')
            self._goal_in_flight = False
            return
        self.get_logger().info('Nav2 goal accepted - robot is moving.')
        self._goal_in_flight = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Nav2 goal finished. Result: {result}')
        # Clearing this re-arms the node: the next centroid starts a fresh
        # approach. The node therefore keeps approaching for as long as the
        # simulation runs, rather than stopping after one successful approach.
        self._goal_in_flight = False


def main(args=None):
    rclpy.init(args=args)
    node = GroupApproachBaselineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
