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