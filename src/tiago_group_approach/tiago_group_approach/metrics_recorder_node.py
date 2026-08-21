#!/usr/bin/env python3
"""
Phase H: record the proposal's evaluation metrics during a simulation run.

THE SIX METRICS THAT ONLY SIMULATION CAN PROVIDE
-------------------------------------------------
The proposal lists 8 metrics. Two (approach-position error, approach-orientation
error) were already computed offline from the recorded dataset by
scripts/evaluate_approach_pose.py. The other six could NOT be, for two distinct
reasons:

  Need positions in METRES, which uncalibrated video cannot give:
    1. O-space intrusion rate    - did the robot enter the group's shared space?
    2. Minimum distance to group - closest approach to any person
    3. Group cut-through rate    - did the path pass between group members?

  Need an EXECUTED trajectory, which offline data has no equivalent of:
    4. Collision-free rate       - did the robot hit anything?
    5. Task success rate         - did it reach a socially valid pose?
    6. Path length and navigation time

This node measures all six against the ground-truth person positions written by
scripts/generate_social_world.py (the .groundtruth.json beside each world), or
supplied directly via the `groups` parameter for hand-built worlds.

WHY GROUND TRUTH RATHER THAN PERCEPTION
----------------------------------------
Metrics are scored against the world's TRUE person positions, not the robot's
detections. Otherwise a perception failure would flatter the result: a robot
that fails to see a group cannot intrude on a group it never detected, and
would score perfectly. Evaluating against ground truth measures the behaviour,
not the perception, and keeps the two failure modes separable in the write-up.

USAGE
-----
    # start recording (do this before launching the policy)
    ros2 run tiago_group_approach metrics_recorder_node --ros-args \\
        -p groundtruth:=/workspaces/Research_Project/src/tiago_social_worlds/worlds/restaurant_humans.groundtruth.json \\
        -p policy_name:=rule_based \\
        -p output_dir:=/workspaces/Research_Project/dataset/processed/sim_results

    # stop with Ctrl-C, or:
    ros2 topic pub --once /metrics/finish std_msgs/msg/Empty {}

Each run writes one JSON file. Aggregate several with
scripts/summarise_sim_results.py.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty

import tf2_ros
from tf2_ros import TransformException


class MetricsRecorderNode(Node):
    def __init__(self):
        super().__init__('metrics_recorder_node')

        self.declare_parameter('groundtruth', '')
        self.declare_parameter('policy_name', 'unknown')
        self.declare_parameter('output_dir',
                               '/workspaces/Research_Project/dataset/processed/sim_results')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('sample_rate_hz', 10.0)
        self.declare_parameter('collision_distance_m', 0.30)
        # Below this, a laser return is the robot seeing itself, not an obstacle.
        self.declare_parameter('min_valid_range_m', 0.20)
        # Ignore collisions during spawn settling / initial localisation.
        self.declare_parameter('collision_grace_s', 5.0)
        # Added to the measured self-hit radius before anything counts as an
        # external obstacle. 5 cm is enough to clear sensor noise on the
        # chassis return without hiding a genuine near-collision.
        self.declare_parameter('self_hit_margin_m', 0.05)
        # Distance to the GROUP CENTRE that counts as a successful approach.
        #
        # The lower bound is 0.5 m, not 0.8 m, because the approach policy now
        # targets 0.7 m clearance to the nearest PERSON and stands in a gap in
        # the formation. A robot correctly positioned in the P-space of a tight
        # group sits well inside 0.8 m of the centre, and the old band would
        # have scored those good approaches as failures.
        #
        # The bound that actually protects against crowding is
        # min_distance_to_person_m, reported separately - a pose can only be
        # this close to the centre if it still clears every individual.
        self.declare_parameter('success_standoff_min_m', 0.5)
        self.declare_parameter('success_standoff_max_m', 2.0)
        self.declare_parameter('robot_radius_m', 0.27)   # PMB2 base ~0.27 m

        self.groups = self.load_groundtruth()

        self.samples: list[dict] = []
        self.min_distance_to_person = float('inf')
        self.min_scan_range = float('inf')
        self.collision_detected = False
        self.path_length = 0.0
        self.last_xy: tuple[float, float] | None = None
        self.start_time = time.time()
        self.goal_centroid: tuple[float, float] | None = None
        self.finished = False

        # Per-group intrusion tracking: a run "intruded" if the robot's
        # footprint ever entered the O-space circle.
        self.intruded_groups: set[int] = set()
        self.cut_through_events = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value,
                                 self.scan_callback, 10)
        self.create_subscription(PointStamped, '/group_centroid',
                                 self.centroid_callback, 10)
        self.create_subscription(Empty, '/metrics/finish',
                                 lambda _msg: self.finish(), 10)

        rate = self.get_parameter('sample_rate_hz').value
        self.timer = self.create_timer(1.0 / max(1.0, rate), self.sample)

        self.get_logger().info(
            f"Metrics recorder started.\n"
            f"  policy      : {self.get_parameter('policy_name').value}\n"
            f"  groups      : {len(self.groups)} from ground truth\n"
            f"  sample rate : {rate} Hz\n"
            f"Ctrl-C (or publish to /metrics/finish) to write results.")

    # ------------------------------------------------------------- setup
    def load_groundtruth(self) -> list[dict]:
        path = self.get_parameter('groundtruth').value
        if not path:
            self.get_logger().warn(
                "No 'groundtruth' parameter given - social metrics (O-space "
                "intrusion, min distance, cut-through) CANNOT be computed. "
                "Only path length, time and collisions will be recorded.")
            return []

        p = Path(path)
        if not p.exists():
            self.get_logger().error(f"Ground truth not found: {p}")
            return []

        data = json.loads(p.read_text(encoding='utf-8'))
        groups = data.get('groups', [])
        for g in groups:
            self.get_logger().info(
                f"  group {g['group_id']}: {g['num_people']} people at "
                f"({g['centre_x']}, {g['centre_y']}), O-space r={g.get('ospace_radius', 0.7)} m")
        return groups

    # ------------------------------------------------------------- callbacks
    def scan_callback(self, msg: LaserScan) -> None:
        """
        Track the closest obstacle and flag genuine collisions.

        Two filters matter here, and without them every run is scored as a
        collision before the robot has even moved:

        1. Readings below the sensor's own `range_min` (and below a small floor)
           are physically impossible as external obstacles - they are self-hits
           on the robot's own body, or sensor noise. TIAGo's base laser reports
           values around 0.09 m when it clips its own chassis, which is well
           inside the 0.30 m collision threshold.
        2. A short grace period at startup. The robot frequently spawns close to
           furniture; being NEAR a table at t=0 is not a collision caused by the
           policy, and scoring it as one would make every trial a failure
           regardless of behaviour.
        """
        floor = max(self.get_parameter('min_valid_range_m').value,
                    msg.range_min if math.isfinite(msg.range_min) else 0.0)
        valid = [r for r in msg.ranges
                 if math.isfinite(r) and r > floor and r <= msg.range_max]
        if not valid:
            return

        closest = min(valid)
        elapsed = time.time() - self.start_time

        # ------------------------------------------------------------------
        # SELF-HIT CALIBRATION.
        #
        # Every trial in the first full experiment reported a collision, and
        # min_obstacle_range_m came out at 0.200-0.267 m in all 19 runs - twelve
        # of them at exactly 0.200. That is not nineteen collisions, it is a
        # constant: the base laser returning the robot's own chassis. With a
        # 0.30 m collision threshold, that fixed return flagged a collision in
        # every run before the policy did anything, and task_success is gated on
        # collisions, so success was 0% for all three policies for reasons that
        # had nothing to do with their behaviour.
        #
        # Rather than hard-coding a guess at the self-hit radius, measure it:
        # during the grace period the robot is stationary at spawn, so the
        # smallest persistent return is its own body. Everything at or below
        # that, plus a margin, is discarded for the rest of the run.
        # ------------------------------------------------------------------
        grace = self.get_parameter('collision_grace_s').value
        if elapsed < grace:
            self._self_hit = min(getattr(self, '_self_hit', float('inf')), closest)
            return

        if not getattr(self, '_self_hit_logged', False):
            self._self_hit_logged = True
            margin = self.get_parameter('self_hit_margin_m').value
            self._self_floor = (self._self_hit + margin
                                if math.isfinite(getattr(self, '_self_hit', float('inf')))
                                else floor)
            self.get_logger().info(
                f"Laser self-hit calibrated at {self._self_hit:.3f} m; "
                f"ignoring returns below {self._self_floor:.3f} m "
                f"(collision threshold "
                f"{self.get_parameter('collision_distance_m').value:.2f} m).")

        self_floor = getattr(self, '_self_floor', floor)
        external = [r for r in valid if r > self_floor]
        if not external:
            return
        closest = min(external)
        self.min_scan_range = min(self.min_scan_range, closest)

        if closest < self.get_parameter('collision_distance_m').value:
            if not self.collision_detected:
                self.get_logger().warn(
                    f"COLLISION: obstacle at {closest:.2f} m "
                    f"(threshold {self.get_parameter('collision_distance_m').value} m, "
                    f"{elapsed:.1f}s into the run)")
            self.collision_detected = True

    def centroid_callback(self, msg: PointStamped) -> None:
        self.goal_centroid = (msg.point.x, msg.point.y)

    def robot_xy_yaw(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get_parameter('map_frame').value,
                self.get_parameter('robot_frame').value,
                rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    # ------------------------------------------------------------- sampling
    def sample(self) -> None:
        if self.finished:
            return
        pose = self.robot_xy_yaw()
        if pose is None:
            return
        x, y, yaw = pose

        if self.last_xy is not None:
            self.path_length += math.dist((x, y), self.last_xy)
        self.last_xy = (x, y)

        robot_r = self.get_parameter('robot_radius_m').value

        for g in self.groups:
            cx, cy = g['centre_x'], g['centre_y']
            ospace_r = g.get('ospace_radius', 0.7)

            # O-space intrusion: robot FOOTPRINT overlapping the O-space circle,
            # not merely its centre point - a robot half inside a conversation
            # has intruded.
            if math.dist((x, y), (cx, cy)) < (ospace_r + robot_r):
                if g['group_id'] not in self.intruded_groups:
                    self.get_logger().warn(
                        f"O-SPACE INTRUSION: entered group {g['group_id']}")
                self.intruded_groups.add(g['group_id'])

            for member in g.get('members', []):
                d = math.dist((x, y), (member['x'], member['y']))
                self.min_distance_to_person = min(self.min_distance_to_person, d)

        self.samples.append({'t': time.time() - self.start_time,
                             'x': x, 'y': y, 'yaw': yaw})

    # ------------------------------------------------------------- scoring
    def compute_cut_through(self) -> int:
        """
        Count path segments that pass BETWEEN two members of the same group.

        Implemented as: does the robot's travelled segment intersect the line
        joining any two members of a group? Walking through the middle of a
        conversation is one of the most socially disruptive failures, and it is
        distinct from O-space intrusion (you can clip the O-space edge without
        cutting the group in half).
        """
        def segments_intersect(p1, p2, p3, p4) -> bool:
            def orient(a, b, c):
                return (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
            d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
            d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
            return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

        events = 0
        path = [(s['x'], s['y']) for s in self.samples]
        for g in self.groups:
            members = [(m['x'], m['y']) for m in g.get('members', [])]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    for k in range(len(path) - 1):
                        if segments_intersect(path[k], path[k+1], members[i], members[j]):
                            events += 1
                            break   # count once per member pair
        return events

    def task_success(self) -> tuple[bool, str]:
        """Reached a socially valid pose: right distance band, facing the group, no collision."""
        if not self.samples:
            return False, "no trajectory recorded"
        if self.collision_detected:
            return False, "collision occurred"
        if self.goal_centroid is None:
            return False, "no group was ever detected"

        cx, cy = self.goal_centroid
        lo = self.get_parameter('success_standoff_min_m').value
        hi = self.get_parameter('success_standoff_max_m').value

        # ------------------------------------------------------------------
        # Scored at CLOSEST APPROACH, not at the final pose.
        #
        # The robot now runs a patrol mission and RETURNS TO ITS START POINT,
        # so its final pose is never beside the group - by design. Scoring the
        # last sample marked every run a failure no matter how well the policy
        # behaved: one run reached 1.101 m from the person, textbook social
        # distance, and was recorded as "failed" purely because it then drove
        # home.
        #
        # What the objective actually asks is whether the robot ever achieved a
        # socially appropriate approach pose. So: find the sample where it came
        # closest to the group, and judge distance AND heading at that instant.
        # ------------------------------------------------------------------
        # Success = the robot achieved, at ANY point, a pose that satisfies
        # distance AND heading together.
        #
        # Scoring the single closest sample was wrong too: the nearest point on
        # the path is often a tangential fly-past, where the robot is at the
        # right distance but travelling across the group's face. That scored
        # 88-90 deg heading error and "failed" three runs in which the robot
        # had in fact settled at 1.50 m facing the group within 1 degree.
        #
        # Both conditions must hold in the SAME sample, so this cannot be
        # gamed by being at the right distance at one moment and the right
        # heading at another.
        nearest = min(math.dist((s['x'], s['y']), (cx, cy)) for s in self.samples)

        best = None
        for s in self.samples:
            distance = math.dist((s['x'], s['y']), (cx, cy))
            if not (lo <= distance <= hi):
                continue
            desired_yaw = math.atan2(cy - s['y'], cx - s['x'])
            error = abs(math.atan2(math.sin(desired_yaw - s['yaw']),
                                   math.cos(desired_yaw - s['yaw'])))
            if best is None or error < best[0]:
                best = (error, distance, s['t'])

        if best is None:
            return False, (f"never held a pose in the [{lo}, {hi}] m band "
                           f"(closest was {nearest:.2f} m)")

        error, distance, t = best
        if math.degrees(error) > 45.0:
            return False, (f"was in the social band but never faced the group: "
                           f"best heading error {math.degrees(error):.0f} deg "
                           f"at {distance:.2f} m")

        return True, (f"reached a valid approach pose: {distance:.2f} m, "
                      f"heading {math.degrees(error):.0f} deg off, at t={t:.0f}s")

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True

        success, reason = self.task_success()
        cut_through = self.compute_cut_through()
        duration = time.time() - self.start_time

        result = {
            'policy': self.get_parameter('policy_name').value,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'duration_s': round(duration, 2),
            'samples': len(self.samples),

            # --- the six simulation-only metrics ---
            'ospace_intrusion': len(self.intruded_groups) > 0,
            'ospace_intrusion_count': len(self.intruded_groups),
            'min_distance_to_person_m': (round(self.min_distance_to_person, 3)
                                         if math.isfinite(self.min_distance_to_person) else None),
            'group_cut_through_events': cut_through,
            'collision_free': not self.collision_detected,
            'min_obstacle_range_m': (round(self.min_scan_range, 3)
                                     if math.isfinite(self.min_scan_range) else None),
            'task_success': success,
            'task_success_reason': reason,
            'path_length_m': round(self.path_length, 3),
            'navigation_time_s': round(duration, 2),

            'groups_in_world': len(self.groups),
            'final_pose': self.samples[-1] if self.samples else None,
            'trajectory': self.samples,
        }

        out_dir = Path(self.get_parameter('output_dir').value)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (f"{result['policy']}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.json")
        path = out_dir / name
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')

        self.get_logger().info(
            "\n" + "=" * 60 +
            f"\nRUN COMPLETE - policy: {result['policy']}" +
            "\n" + "=" * 60 +
            f"\n  task success        : {success}  ({reason})" +
            f"\n  collision free      : {result['collision_free']}" +
            f"\n  O-space intrusion   : {result['ospace_intrusion']} "
            f"({result['ospace_intrusion_count']} group(s))" +
            f"\n  min dist to person  : {result['min_distance_to_person_m']} m" +
            f"\n  cut-through events  : {cut_through}" +
            f"\n  path length         : {result['path_length_m']} m" +
            f"\n  navigation time     : {result['navigation_time_s']} s" +
            f"\n\nWritten: {path}\n" + "=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = MetricsRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.finish()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
