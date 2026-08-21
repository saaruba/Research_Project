#!/usr/bin/env python3
"""
Phase F/G: the LEARNED policy node. Loads the trained Behavioural Cloning
model and drives TIAGo to the approach pose it predicts.

This is the counterpart to group_approach_baseline_node.py (the hand-coded
geometric rule). Both consume /group_centroid and both send a NavigateToPose
goal, so they are drop-in swappable - which is exactly what Objective 4's
"compare Behavioural Cloning against a rule-based baseline" requires. Running
the same world, same start pose and same groups through each policy in turn
is the experiment.

WHICH MODEL IS LOADED, AND WHY
-------------------------------
Default is approach_pose_random_forest_tuned.joblib. On the held-out test
sessions the tuned Random Forest reached 0.365 m mean position error and
25.8 deg orientation error - better than the untuned Random Forest (0.401 m /
29.3 deg) and clearly better than both MLP variants (tuned MLP: 0.395 m /
31.0 deg). It is also the only learned policy that beat the rule-based
baseline on ANY metric (orientation: 25.8 deg vs 29.1 deg).

Be honest about the rest when writing this up: the rule-based baseline still
wins on position (0.305 m vs 0.365 m), and NO policy met the proposal's
<20 deg orientation threshold. Swap models with the `model_path` parameter to
reproduce the comparison.

FEATURE VECTOR - MUST MATCH TRAINING EXACTLY
---------------------------------------------
The model was trained on 7 features, in this order:
    lidar_min_range, lidar_mean_range, linear_x_prev, angular_z_prev,
    num_people, group_bearing_rad, group_scale_norm
Any mismatch in order or meaning silently produces garbage predictions rather
than an error, so the assembly below is deliberately explicit.

The model predicts a RELATIVE pose (target_dx, target_dy, target_dyaw) in the
robot's own frame - not absolute map coordinates. That was a deliberate
training decision: absolute (x, y) does not transfer between rooms, and was
measured to hurt generalisation. So the prediction is converted to a map-frame
goal here, using the robot's current pose from TF.

CAVEAT WORTH STATING IN THE DISSERTATION
-----------------------------------------
`group_scale_norm` was defined during training as (mean person bbox width /
image width) - an apparent-size proxy for distance, because the recorded
dataset had no depth. In simulation we have true metric distance, so this node
reconstructs an equivalent value from the real distance to the group. That is
a domain shift between training and deployment and is a legitimate limitation
to report: the model was trained on a proxy and is being fed a reconstruction
of that proxy.

RUN
---
    ros2 run tiago_group_approach bc_policy_node
    ros2 run tiago_group_approach bc_policy_node --ros-args \\
        -p model_path:=/workspaces/Research_Project/dataset/processed/models/approach_pose_mlp_tuned.joblib
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan

import tf2_ros
from tf2_ros import TransformException

DEFAULT_MODEL = ("/workspaces/Research_Project/dataset/processed/models/"
                 "approach_pose_random_forest_tuned.joblib")

# Order matters - see the module docstring.
FEATURE_ORDER = [
    "lidar_min_range", "lidar_mean_range", "linear_x_prev", "angular_z_prev",
    "num_people", "group_bearing_rad", "group_scale_norm",
]

# Feature parity between training and inference is not optional. Every value in
# FEATURE_ORDER must be produced live in the same units, on the same scale, and
# with the same meaning as the column of the same name in the training table
# (build_approach_pose_dataset.py). A single feature that silently differs
# degrades predictions in a way that looks exactly like a bad model.

IMAGE_WIDTH = 640.0          # matches the training-time normalisation
TYPICAL_PERSON_WIDTH_M = 0.5  # adult shoulder width, for the scale proxy
ASSUMED_FOCAL_PX = 525.0      # nominal RGB focal length, for the scale proxy


class BCPolicyNode(Node):
    def __init__(self):
        super().__init__('bc_policy_node')

        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('min_standoff_m', 0.6)
        self.declare_parameter('max_standoff_m', 3.0)

        model_path = Path(self.get_parameter('model_path').value)
        if not model_path.exists():
            self.get_logger().fatal(
                f"Model not found: {model_path}\n"
                f"  Train it first:  python3 scripts/grid_search_approach_pose.py"
            )
            raise SystemExit(1)

        import joblib
        self.model = joblib.load(model_path)
        self.get_logger().info(f"Loaded BC model: {model_path.name}")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.latest_scan: LaserScan | None = None
        # The model uses the PREVIOUS action as a feature. At the moment of an
        # approach decision the robot is effectively stationary, so zeros are
        # the honest value here rather than a fabricated history.
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0

        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value,
                                 self.scan_callback, 10)
        # Live headcount, so the num_people feature matches training.
        self._num_people = 2
        self.create_subscription(
            PoseArray, '/detected_people',
            lambda m: setattr(self, '_num_people', max(1, len(m.poses))), 10)

        self.create_subscription(PointStamped, '/group_centroid',
                                 self.group_callback, 10)

        self.get_logger().info(
            "BC policy node ready. Waiting for /group_centroid ...")

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def lidar_features(self) -> tuple[float, float]:
        """min and mean range, matching how extract_training_table.py computed them."""
        if self.latest_scan is None:
            return 3.0, 3.0   # neutral fallback; logged by the caller
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        valid = ranges[np.isfinite(ranges) & (ranges > 0.0)]
        if valid.size == 0:
            return 3.0, 3.0
        return float(valid.min()), float(valid.mean())

    def robot_pose(self) -> tuple[float, float, float] | None:
        map_frame = self.get_parameter('map_frame').value
        robot_frame = self.get_parameter('robot_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(
                map_frame, robot_frame, rclpy.time.Time(), timeout=Duration(seconds=1.0))
        except TransformException as exc:
            self.get_logger().error(f"TF {map_frame}->{robot_frame} failed: {exc}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def group_callback(self, msg: PointStamped) -> None:
        pose = self.robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        gx, gy = msg.point.x, msg.point.y
        dx_world, dy_world = gx - rx, gy - ry
        distance = math.hypot(dx_world, dy_world)
        if distance < 1e-3:
            self.get_logger().warn("Robot is already at the group centroid.")
            return

        # Bearing to the group, in the robot's own frame (+ = left).
        bearing = math.atan2(dy_world, dx_world) - ryaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))

        # Reconstruct the training-time apparent-size proxy from true distance.
        # See the docstring caveat: training used pixels, deployment has metres.
        apparent_px = (TYPICAL_PERSON_WIDTH_M * ASSUMED_FOCAL_PX) / max(distance, 0.3)
        group_scale_norm = min(apparent_px / IMAGE_WIDTH, 1.0)

        lidar_min, lidar_mean = self.lidar_features()
        if self.latest_scan is None:
            self.get_logger().warn(
                f"No LaserScan on {self.get_parameter('scan_topic').value} yet - "
                "using fallback LiDAR features; prediction will be unreliable.",
                throttle_duration_sec=10.0)

        features = {
            "lidar_min_range": lidar_min,
            "lidar_mean_range": lidar_mean,
            "linear_x_prev": self.prev_linear_x,
            "angular_z_prev": self.prev_angular_z,
            # The ACTUAL number of people perception is reporting.
            #
            # This was hard-coded to 3.0 with a comment promising it would be
            # "refined below if perception reports it" - and it never was. One
            # of the seven features the models were trained on was therefore a
            # constant at inference time, while training saw it vary between 1
            # and 6. Both learned policies have been predicting from a
            # corrupted input on every call, which is a far more likely cause
            # of poor live behaviour than anything in the models themselves.
            "num_people": float(self._num_people),
            "group_bearing_rad": bearing,
            "group_scale_norm": group_scale_norm,
        }
        x = np.array([[features[k] for k in FEATURE_ORDER]], dtype=np.float64)

        pred = self.model.predict(x)[0]
        pred_dx, pred_dy, pred_dyaw = float(pred[0]), float(pred[1]), float(pred[2])

        # Sanity-clamp: a regression model can extrapolate to nonsense on
        # out-of-distribution input. Refusing an absurd goal is better than
        # sending Nav2 somewhere meaningless and calling it a result.
        predicted_travel = math.hypot(pred_dx, pred_dy)
        min_standoff = self.get_parameter('min_standoff_m').value
        max_standoff = self.get_parameter('max_standoff_m').value

        # CLAMP, do not reject.
        #
        # Rejecting an implausible prediction meant the robot did nothing at
        # all: no goal, no motion, and from the outside it looked as though the
        # learned policy simply refused to approach anybody. Since the group's
        # position is known independently of the model, an over-long prediction
        # can be scaled back onto the line towards the group instead of thrown
        # away. The model still chooses the direction and the standoff; only
        # physically impossible magnitudes are corrected.
        limit = distance + max_standoff
        if predicted_travel > limit and predicted_travel > 1e-6:
            scale = limit / predicted_travel
            self.get_logger().warn(
                f"Prediction of {predicted_travel:.2f} m exceeds the plausible "
                f"{limit:.2f} m (group is {distance:.2f} m away) - scaling by "
                f"{scale:.2f} rather than discarding it.")
            pred_dx *= scale
            pred_dy *= scale

        # Relative (robot-frame) prediction -> absolute map goal.
        cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
        goal_x = rx + pred_dx * cos_y - pred_dy * sin_y
        goal_y = ry + pred_dx * sin_y + pred_dy * cos_y
        goal_yaw = ryaw + pred_dyaw

        # Never end up inside the group: enforce a minimum standoff.
        to_group = math.hypot(gx - goal_x, gy - goal_y)
        if to_group < min_standoff:
            ux, uy = (goal_x - gx) / max(to_group, 1e-6), (goal_y - gy) / max(to_group, 1e-6)
            goal_x, goal_y = gx + ux * min_standoff, gy + uy * min_standoff
            self.get_logger().warn(
                f"Predicted pose was {to_group:.2f} m from the group centre; "
                f"pushed out to the {min_standoff:.2f} m minimum standoff.")

        self.get_logger().info(
            f"BC prediction: d=({pred_dx:+.2f}, {pred_dy:+.2f}) m, "
            f"dyaw={math.degrees(pred_dyaw):+.1f} deg  ->  "
            f"goal ({goal_x:.2f}, {goal_y:.2f}) @ {math.degrees(goal_yaw):.1f} deg")

        self.send_goal(goal_x, goal_y, goal_yaw)

    def send_goal(self, x: float, y: float, yaw: float) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "Nav2 'navigate_to_pose' unavailable - is Nav2 running? "
                "(launch with navigation:=True)")
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.get_parameter('map_frame').value
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Nav2 rejected the goal.")
            return
        self.get_logger().info("Nav2 accepted the goal - robot is moving.")
        handle.get_result_async().add_done_callback(
            lambda f: self.get_logger().info("Nav2 goal finished."))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BCPolicyNode()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
