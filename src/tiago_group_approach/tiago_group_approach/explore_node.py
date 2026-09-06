"""
Autonomous exploration: patrol the room until people are found.

WHY THIS NODE EXISTS
--------------------
Perception is PASSIVE. It can only cluster people the camera can actually see,
and the policy nodes only act once /group_centroid is published. TIAGo spawns
at the world origin facing +x, while the person in restaurant_testing.world
stands at (-3, 0) - directly behind it. The robot therefore waited forever for
a detection that could never arrive, which is indistinguishable from "the robot
is broken".

This node closes that loop. It drives a patrol route around the room so the
camera sweeps the whole space. The moment a group is detected it stands down
and lets the approach policy take over; if the group is lost for a while, it
resumes patrolling.

    no detections  ->  PATROL   (this node drives)
    group detected ->  YIELD    (the policy node drives)

Both this node and the policy node send NavigateToPose goals to the same Nav2
action server, and Nav2 runs one goal at a time - a new goal preempts the old
one. That is exactly the desired behaviour: an approach goal interrupts the
patrol immediately.

PARAMETERS
    patrol_points   flat [x1,y1,x2,y2,...] list, map frame
    idle_timeout    seconds without a detection before patrolling resumes
    enabled         set false to disable exploration entirely
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


# Default route for restaurant_testing.world.
# Room is 20 x 15 m: walls at x = +/-9.9 and y = +/-7.4. Avoided:
#   - the stage       at (7.45, 5.20), 4.5 x 3.6
#   - the kitchen     behind the partition at x < -6.2, y > 2.1
#   - the dining tables scattered through the middle
# Points are kept well clear of walls so the local costmap always has room.
DEFAULT_PATROL = [
     4.0, -4.0,
     4.0,  2.0,
     0.0,  4.0,
    -4.0,  4.0,
    -4.0, -1.0,     # looks toward the person at (-3, 0)
     0.0, -4.0,
]


class ExploreNode(Node):
    def __init__(self):
        super().__init__('explore_node')

        self.declare_parameter('patrol_points', DEFAULT_PATROL)
        self.declare_parameter('idle_timeout', 12.0)
        self.declare_parameter('enabled', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('arrival_tolerance', 0.8)

        flat = list(self.get_parameter('patrol_points').value)
        self.points = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
        self.index = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.last_detection = None      # sim time, seconds
        self.goal_handle = None
        self.patrolling = False

        self.create_subscription(PointStamped, '/group_centroid',
                                 self.on_group, 10)
        self.create_timer(2.0, self.tick)

        if not self.get_parameter('enabled').value:
            self.get_logger().info('Exploration DISABLED by parameter.')
        else:
            self.get_logger().info(
                f'Exploration ready: {len(self.points)} patrol points, '
                f'idle_timeout={self.get_parameter("idle_timeout").value}s. '
                'Waiting for Nav2...')

    # ------------------------------------------------------------------ time
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ------------------------------------------------------------ detections
    def on_group(self, _msg: PointStamped) -> None:
        first = self.last_detection is None
        self.last_detection = self.now_s()
        if self.patrolling:
            self.get_logger().info(
                'Group detected - standing down, the approach policy takes over.')
            self.cancel_patrol()
        elif first:
            self.get_logger().info('Group detected before patrol started.')

    def cancel_patrol(self) -> None:
        self.patrolling = False
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.goal_handle = None

    # ----------------------------------------------------------- main loop
    def tick(self) -> None:
        if not self.get_parameter('enabled').value:
            return
        if not self.nav.server_is_ready():
            return

        idle = self.get_parameter('idle_timeout').value
        seen_recently = (self.last_detection is not None
                         and self.now_s() - self.last_detection < idle)

        if seen_recently:
            return                     # the policy node is driving

        if self.patrolling:
            if self.reached_current():
                self.get_logger().info('Patrol point reached.')
                self.index = (self.index + 1) % len(self.points)
                self.send_patrol_goal()
            return

        # Nothing seen for a while - explore.
        if self.last_detection is not None:
            self.get_logger().info(
                f'No group for {idle:.0f}s - resuming patrol.')
        self.send_patrol_goal()

    def reached_current(self) -> bool:
        pose = self.robot_xy()
        if pose is None:
            return False
        tx, ty = self.points[self.index]
        tol = self.get_parameter('arrival_tolerance').value
        return math.hypot(pose[0] - tx, pose[1] - ty) < tol

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.get_parameter('map_frame').value,
                self.get_parameter('robot_frame').value,
                rclpy.time.Time(), timeout=Duration(seconds=0.5))
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None

    def send_patrol_goal(self) -> None:
        x, y = self.points[self.index]

        # Face the direction of travel, so the camera looks where the robot is
        # going rather than at whatever happens to be behind it.
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

        self.get_logger().info(
            f'PATROL -> point {self.index + 1}/{len(self.points)} '
            f'({x:.1f}, {y:.1f})')
        self.patrolling = True
        future = self.nav.send_goal_async(goal)
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Patrol goal failed to send: {exc}')
            self.patrolling = False
            return
        if not handle.accepted:
            self.get_logger().warn('Patrol goal rejected by Nav2.')
            self.patrolling = False
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, _future) -> None:
        # Finished, aborted or preempted - all handled the same way: move on.
        # An abort usually means that patrol point is unreachable (a table has
        # been placed on it), so skipping to the next one keeps exploring
        # rather than retrying the same blocked goal forever.
        if self.patrolling:
            self.index = (self.index + 1) % len(self.points)
        self.goal_handle = None
        self.patrolling = False


def main(args=None):
    rclpy.init(args=args)
    node = ExploreNode()
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
