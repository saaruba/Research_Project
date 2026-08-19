#!/usr/bin/env python3
"""
Scripted patrol mission: drive a fixed tour, observe, then stop and report.

THE MISSION
    start   (-5, +5)     <- drives here first, then the tour begins
            (+3, +6)
            (+8, +1)
            (+8, -6)
            (-8, -4)
    finish  (-5, +5)     <- back to start, then the run ENDS automatically

On completion the node publishes to /metrics/finish, which makes the metrics
recorder write its results file, and prints a summary of what was observed:
how many people were seen, how many groups, and where.

WHY A SCRIPTED TOUR RATHER THAN FREE EXPLORATION
    A fixed route is repeatable. Every run covers the same ground in the same
    order, so a difference between the rule-based and BC policies is a
    difference in BEHAVIOUR, not in which corner the robot happened to wander
    into. That is what makes the comparison in Objective 4 meaningful.

BLOCKED GOALS AND MOVING OBSTACLES
    If Nav2 aborts a waypoint, the node does NOT immediately give up. A person
    walking across the corridor blocks the path for a few seconds and then
    clears it. So on failure it waits `obstacle_wait` seconds (default 15) to
    let a moving obstacle pass, then retries the same waypoint. Only after
    `max_retries` consecutive failures does it treat the waypoint as genuinely
    unreachable and move on - otherwise a single parked table would stall the
    entire mission forever.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Empty as EmptyMsg

import tf2_ros
from tf2_ros import TransformException


DEFAULT_MISSION = [
    -5.0,  5.0,
     3.0,  6.0,
     8.0,  1.0,
     8.0, -6.0,
    -8.0, -4.0,
    -5.0,  5.0,
]


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')

        self.declare_parameter('waypoints', DEFAULT_MISSION)
        self.declare_parameter('obstacle_wait', 15.0)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('stop_on_complete', True)
        self.declare_parameter('pause_for_approach', True)
        # Hard ceiling on how long the mission will wait for one approach.
        # Without it a robot parked in front of a group re-arms the pause
        # forever and the tour never continues.
        self.declare_parameter('max_approach_time', 90.0)

        flat = list(self.get_parameter('waypoints').value)
        self.waypoints = [(flat[i], flat[i + 1])
                          for i in range(0, len(flat) - 1, 2)]
        self.index = 0
        self.retries = 0
        self.active = False
        self.finished = False
        self.waiting_until = 0.0

        # --- Observation log ---------------------------------------------
        self.people_seen = 0
        self.max_people_at_once = 0
        self.group_sightings = 0
        self.group_positions: list[tuple[float, float]] = []
        self.approach_paused_until = 0.0
        self.visited: list[tuple[float, float]] = []   # groups already approached
        self.approach_started: float | None = None
        self.approach_target: tuple[float, float] | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.finish_pub = self.create_publisher(EmptyMsg, '/metrics/finish', 10)
        self.create_subscription(PointStamped, '/group_centroid',
                                 self.on_group, 10)
        self.create_subscription(PoseArray, '/detected_people',
                                 self.on_people, 10)

        self.create_timer(2.0, self.tick)
        self.get_logger().info(
            'Mission ready: ' +
            ' -> '.join(f'({x:.0f},{y:.0f})' for x, y in self.waypoints) +
            '\n  Waiting for Nav2...')

    # ------------------------------------------------------------------ time
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ---------------------------------------------------------- observations
    def on_people(self, msg: PoseArray) -> None:
        n = len(msg.poses)
        if n > self.max_people_at_once:
            self.max_people_at_once = n
        self.people_seen += n

    def on_group(self, msg: PointStamped) -> None:
        self.group_sightings += 1
        pos = (msg.point.x, msg.point.y)
        self.group_positions.append(pos)

        if not self.get_parameter('pause_for_approach').value:
            return

        # ------------------------------------------------------------------
        # Approach ONCE per group, then move on.
        #
        # This used to just push the pause 20 s into the future on every
        # detection. Since the robot ends its approach parked in front of the
        # group, it kept seeing them, kept re-arming the pause, and the tour
        # never resumed - it simply stood there indefinitely. Observed
        # directly: the robot reached the group and stayed put for the rest of
        # the run.
        #
        # Two changes fix it. A group already visited is ignored, so it cannot
        # re-trigger a pause; and any single approach is capped in wall time,
        # so even an approach that never converges cannot stall the mission.
        # ------------------------------------------------------------------
        if self.already_visited(pos):
            return

        now = self.now_s()
        if self.approach_started is None:
            self.approach_started = now
            self.approach_target = pos
            self.get_logger().info(
                f'New group at ({pos[0]:.2f}, {pos[1]:.2f}) - pausing the '
                f'mission to approach it.')

        limit = self.get_parameter('max_approach_time').value
        if now - self.approach_started > limit:
            self.get_logger().info(
                f'Approach time limit ({limit:.0f}s) reached - marking this '
                'group as visited and resuming the tour.')
            self.visited.append(self.approach_target or pos)
            self.approach_started = None
            self.approach_target = None
            self.approach_paused_until = 0.0
            return

        self.approach_paused_until = now + 5.0

    def already_visited(self, pos) -> bool:
        return any(math.hypot(pos[0] - v[0], pos[1] - v[1]) < 2.0
                   for v in self.visited)

    # -------------------------------------------------------------- main loop
    def tick(self) -> None:
        if self.finished or not self.nav.server_is_ready():
            return
        if self.now_s() < self.waiting_until:
            return
        if self.now_s() < self.approach_paused_until:
            return
        if self.active:
            return
        self.send_waypoint()

    def send_waypoint(self) -> None:
        if self.index >= len(self.waypoints):
            self.complete()
            return

        x, y = self.waypoints[self.index]
        here = self.robot_xy()
        yaw = math.atan2(y - here[1], x - here[0]) if here else 0.0

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.get_parameter('map_frame').value
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        label = 'START' if self.index == 0 else (
            'RETURN TO START' if self.index == len(self.waypoints) - 1
            else f'waypoint {self.index + 1}')
        self.get_logger().info(
            f'MISSION -> {label}: ({x:.1f}, {y:.1f})'
            + (f'   [retry {self.retries}]' if self.retries else ''))

        self.active = True
        self.nav.send_goal_async(goal).add_done_callback(self.on_response)

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.get_parameter('map_frame').value,
                self.get_parameter('robot_frame').value,
                rclpy.time.Time(), timeout=Duration(seconds=0.5))
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None

    def on_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Goal send failed: {exc}')
            self.active = False
            return
        if not handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal.')
            self.active = False
            self.retry_or_skip()
            return
        handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, future) -> None:
        self.active = False
        try:
            status = future.result().status
        except Exception:
            status = 0

        # 4 == SUCCEEDED in action_msgs/GoalStatus
        if status == 4 and self.arrived():
            self.get_logger().info(
                f'Reached waypoint {self.index + 1}/{len(self.waypoints)}.')
            self.index += 1
            self.retries = 0
            if self.index >= len(self.waypoints):
                self.complete()
            return

        self.retry_or_skip()

    def arrived(self) -> bool:
        here = self.robot_xy()
        if here is None:
            return True          # cannot check - trust Nav2
        tx, ty = self.waypoints[self.index]
        return math.hypot(here[0] - tx, here[1] - ty) < 1.2

    def retry_or_skip(self) -> None:
        wait = self.get_parameter('obstacle_wait').value
        self.retries += 1
        if self.retries <= self.get_parameter('max_retries').value:
            self.get_logger().warn(
                f'Waypoint {self.index + 1} blocked. Waiting {wait:.0f}s in '
                'case it is a person walking past, then retrying.')
            self.waiting_until = self.now_s() + wait
        else:
            self.get_logger().warn(
                f'Waypoint {self.index + 1} unreachable after '
                f'{self.retries - 1} retries - skipping it.')
            self.index += 1
            self.retries = 0
            if self.index >= len(self.waypoints):
                self.complete()

    # ---------------------------------------------------------------- report
    def complete(self) -> None:
        if self.finished:
            return
        self.finished = True

        unique = self.cluster_sightings()
        log = self.get_logger()
        log.info('')
        log.info('=' * 60)
        log.info('MISSION COMPLETE - returned to the start point')
        log.info('=' * 60)
        log.info(f'  waypoints visited     : {self.index}/{len(self.waypoints)}')
        log.info(f'  group sightings       : {self.group_sightings}')
        log.info(f'  groups approached     : {len(self.visited)}')
        log.info(f'  distinct group locations: {len(unique)}')
        for i, (x, y, n) in enumerate(unique, 1):
            log.info(f'      group {i}: ({x:.2f}, {y:.2f})  seen {n} times')
        log.info(f'  most people at once   : {self.max_people_at_once}')
        log.info('=' * 60)
        log.info('Telling the metrics recorder to write its results file...')

        self.finish_pub.publish(EmptyMsg())

    def cluster_sightings(self) -> list[tuple[float, float, int]]:
        """Merge repeated sightings of the same group into one entry."""
        out: list[list[float]] = []          # [sum_x, sum_y, count]
        for x, y in self.group_positions:
            for e in out:
                if math.hypot(x - e[0] / e[2], y - e[1] / e[2]) < 1.5:
                    e[0] += x
                    e[1] += y
                    e[2] += 1
                    break
            else:
                out.append([x, y, 1])
        return [(e[0] / e[2], e[1] / e[2], int(e[2])) for e in out]


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
