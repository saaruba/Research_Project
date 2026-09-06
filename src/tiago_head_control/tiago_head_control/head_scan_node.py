"""
HEAD SCAN NODE - point TIAGo's head camera where you want it.

============================================================================
IF YOU HAVE NEVER SEEN THIS PROJECT BEFORE, READ THIS FIRST
============================================================================
TIAGo's head is motorised: it can pan (turn left and right) and tilt (look up
and down). The camera this project relies on for finding people is mounted in
that head, so where the head points decides what the robot can see.

By default TIAGo's head looks straight ahead and level. That is a poor angle
for spotting people standing a couple of metres away, because it puts their
heads at the very top of the frame or out of it entirely. This node moves the
head to a chosen pan/tilt so the camera is aimed usefully.

It is a SETUP AND DEBUGGING TOOL, not part of the main experiment. The
group-approach pipeline does not call it. Use it when you want to check what
the camera sees, or to park the head at a sensible angle before a run.

============================================================================
HOW IT WORKS
============================================================================
Moving a robot joint in ROS 2 is not a matter of setting a variable. You send
a GOAL to an "action server" - a long-running task the robot accepts, works
on, and reports back about - and the controller moves the joint smoothly to
get there.

  1. Connect to the /head_controller/follow_joint_trajectory action server.
  2. Wait until it is available. If TIAGo is not running, this waits forever
     rather than failing, which is the usual reason it appears to hang.
  3. Build a JointTrajectory: the two joint names, the target angles in
     RADIANS (not degrees), and how long the movement should take.
  4. Send it and wait for the controller to report completion.

Angles are in radians throughout, because that is what ROS uses. To convert:
radians = degrees * pi / 180. So -0.6 rad is roughly 34 degrees downward.

============================================================================
RUN IT
============================================================================
    # TIAGo must already be running (Gazebo or the real robot)
    ros2 run tiago_head_control head_scan_node

If it prints "Waiting for head controller action server..." and never moves
on, the robot is not running or the controller has not started yet.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class HeadScanNode(Node):
    def __init__(self):
        super().__init__('head_scan_node')

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/head_controller/follow_joint_trajectory'
        )

        self.get_logger().info('Waiting for head controller action server...')
        self.action_client.wait_for_server()
        self.get_logger().info('Head controller action server is ready.')

        self.scan_positions = [
            [0.5, 0.0],    # look left
            [0.0, 0.0],    # look centre
            [-0.5, 0.0],   # look right
            [0.0, 0.0],    # look centre again
        ]

        self.current_index = 0
        self.send_next_goal()

    def send_next_goal(self):
        if self.current_index >= len(self.scan_positions):
            self.get_logger().info('Head scan completed.')
            rclpy.shutdown()
            return

        pan, tilt = self.scan_positions[self.current_index]

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'head_1_joint',
            'head_2_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0

        goal_msg.trajectory.points.append(point)

        self.get_logger().info(
            f'Sending head goal: pan={pan}, tilt={tilt}'
        )

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Head goal was rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('Head goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result

        self.get_logger().info(
            f'Head goal finished with error_code: {result.error_code}'
        )

        self.current_index += 1
        self.send_next_goal()


def main(args=None):
    rclpy.init(args=args)
    node = HeadScanNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()