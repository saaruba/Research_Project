"""
Ground-truth localisation: publish map -> odom CONTINUOUSLY.

THE BUG THIS FIXES
------------------
The previous approach computed map->odom once at startup and published it as a
STATIC transform. That is correct for exactly one instant and wrong from then
on, because it cannot track odometry drift.

Wheel odometry drifts constantly, and it drifts violently whenever the wheels
slip - which is precisely what happens when the robot bumps an obstacle. So the
longer a run went on, and the more the robot collided with things, the further
the map estimate diverged from reality. That single fault produced every one of
these symptoms at once:

  * the robot appeared in one place in Gazebo and a different place in RViz,
    with the gap growing over the run;
  * the costmap was offset from the real world, so the robot swerved around
    obstacles that were not there and drove straight into ones that were;
  * detections were placed at the wrong world coordinates. A person correctly
    detected 4 m in front of the camera was transformed into the map through a
    stale map->odom and landed metres away. What looked like a YOLO false
    positive at (4.4, 2.8) was very likely a CORRECT detection of the real
    person at (-3, 0), ruined by a stale transform.

Publishing the correction continuously at 30 Hz means the map estimate tracks
the true pose exactly, no matter how much the wheels slip.

    T_map_odom = T_map_base * inverse(T_odom_base)

recomputed on every odometry message, where T_map_base comes from Gazebo's
own state plugin and T_odom_base is what the wheel odometry reports.

FOR THE DISSERTATION
    "Localisation was provided from simulator ground truth rather than AMCL, so
     that navigation error could not confound the comparison between the
     rule-based and Behavioural Cloning approach policies. All other navigation
     components - laser scanning, costmaps, global and local planning - operated
     normally."
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GroundTruthLocalisation(Node):
    def __init__(self):
        super().__init__('gt_localisation_node')

        self.declare_parameter('model_name', 'tiago')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('publish_rate_hz', 30.0)

        self.truth = None       # (x, y, yaw) of base in the map/world frame
        self.odom = None        # (x, y, yaw) of base in the odom frame

        # Gazebo publishes model_states best-effort and very fast.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self.on_models, qos)
        self.create_subscription(Odometry, '/mobile_base_controller/odom',
                                 self.on_odom, 10)

        self.br = TransformBroadcaster(self)
        rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / max(1.0, rate), self.publish_tf)

        self.warned = False
        self.reported = False
        self.get_logger().info(
            f'Ground-truth localisation started - publishing '
            f'{self.get_parameter("map_frame").value} -> '
            f'{self.get_parameter("odom_frame").value} at {rate:.0f} Hz.')

    # ------------------------------------------------------------- callbacks
    def on_models(self, msg: ModelStates) -> None:
        want = self.get_parameter('model_name').value.lower()
        for i, name in enumerate(msg.name):
            if want in name.lower():
                p = msg.pose[i]
                self.truth = (p.position.x, p.position.y, yaw_of(p.orientation))
                return

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    # ----------------------------------------------------------------- output
    def publish_tf(self) -> None:
        if self.truth is None or self.odom is None:
            if not self.warned:
                missing = []
                if self.truth is None:
                    missing.append('/gazebo/model_states (is gazebo_ros_state '
                                   'in the world file?)')
                if self.odom is None:
                    missing.append('/mobile_base_controller/odom')
                self.get_logger().warn('Waiting for: ' + ', '.join(missing),
                                       throttle_duration_sec=10.0)
            return

        mbx, mby, mbyaw = self.truth
        obx, oby, obyaw = self.odom

        # T_map_odom = T_map_base * inv(T_odom_base)
        yaw = mbyaw - obyaw
        c, s = math.cos(yaw), math.sin(yaw)
        x = mbx - (obx * c - oby * s)
        y = mby - (obx * s + oby * c)

        if not self.reported:
            self.reported = True
            self.get_logger().info(
                f'Locked on. Robot true pose ({mbx:.2f}, {mby:.2f}, '
                f'{math.degrees(mbyaw):.1f} deg); initial correction '
                f'({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f} deg).')

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').value
        t.child_frame_id = self.get_parameter('odom_frame').value
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthLocalisation()
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
