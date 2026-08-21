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

from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, Twist
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
        self.declare_parameter('min_goal_interval_s', 2.0)

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
        # 0.7 m to the NEAREST PERSON, not to the group centre.
        #
        # The distinction matters. This group's O-space radius is about 0.71 m,
        # so a 0.7 m standoff measured from the centroid would put the robot in
        # the middle of the conversation. Measured to the nearest person, 0.7 m
        # sits inside Hall's personal zone (0.45-1.2 m) - close enough to
        # address someone, which is what HRI approach studies typically find
        # comfortable for a service robot, and not so close as to be intimate.
        self.declare_parameter('min_person_clearance', 0.8)
        # A gap must be at least this wide to count as "somewhere to stand".
        self.declare_parameter('min_gap_deg', 60.0)
        # BODY RADII. min_person_clearance is the free space we want between
        # the robot's SHELL and a person's body - not between their centres.
        #
        # Treating it as centre-to-centre was a real failure: TIAGo's footprint
        # radius is ~0.30 m and a standing person ~0.25 m, so a 0.7 m
        # centre-to-centre goal leaves 0.15 m of actual gap. The robot wedged
        # itself into a group, came within 0.062 m of somebody, and then spent
        # 84% of a ten-minute run stationary because Nav2 could not find a way
        # out. Adding the radii means 0.7 m requested is 0.7 m delivered.
        self.declare_parameter('robot_radius', 0.30)
        self.declare_parameter('person_radius', 0.25)
        self.declare_parameter('max_standoff', 3.0)

        # --- Stuck recovery ---------------------------------------------------
        self.declare_parameter('stuck_timeout_s', 15.0)
        self.declare_parameter('stuck_move_threshold_m', 0.10)
        self._last_pose_xy = None
        self._last_moved_time = 0.0

        # --- Do not approach people who are walking ---------------------------
        # A person crossing the room is not holding a conversation; there is no
        # F-formation to join and no O-space to respect. Chasing them is both
        # socially wrong and unachievable - the target moves as fast as the
        # robot. They still matter as obstacles, which is Nav2's job, not the
        # approach policy's.
        #
        # Movement is judged from the group centroid over a short window: any
        # centroid drifting faster than this is treated as a passer-by.
        self.declare_parameter('moving_speed_threshold', 0.15)   # m/s
        self.declare_parameter('motion_window_s', 3.0)
        self._centroid_history: list[tuple[float, float, float]] = []  # t, x, y

        # --- Telling the mission we are done ----------------------------------
        # The mission used to wait out a fixed timer before resuming its tour.
        # Publishing the moment the approach actually succeeds lets it carry on
        # immediately, which is what "approach, then leave and continue" means.
        self.approach_done_pub = self.create_publisher(
            PointStamped, '/approach/complete', 10)
        # Claim Nav2 BEFORE driving.
        #
        # The mission and this node both send NavigateToPose goals to the same
        # server, and the last goal wins. Logs showed approach goals accepted
        # and "finished" six MILLISECONDS later - instantly preempted by the
        # mission's next waypoint. The robot never drove to a single approach
        # pose. Announcing the approach first lets the mission stand down.
        self.approach_start_pub = self.create_publisher(
            PointStamped, '/approach/start', 10)
        self._current_target = None

        # --- Memory of what we have already tried -----------------------------
        # Without this the policy retries the same group indefinitely. Observed:
        # a 30-minute run, 208 m driven, 65% of it stationary, endlessly
        # re-approaching one group because no attempt ever satisfied the
        # completion check, so nothing was ever recorded as done.
        #
        # Each group location is remembered along with how many times it has
        # been attempted. After max_attempts it is abandoned for the rest of
        # the run and the mission is told to move on - a group the robot cannot
        # reach is not going to become reachable by trying an eighth time.
        self.declare_parameter('max_attempts_per_group', 3)
        self.declare_parameter('group_memory_radius', 1.5)

        # --- Dwell -------------------------------------------------------------
        # Arriving and immediately leaving is not an approach; it is passing by.
        # The previous behaviour published "complete" the instant Nav2 reported
        # arrival, the mission resumed, and the robot rolled straight on to the
        # next waypoint - exactly what it looked like from the outside.
        #
        # Holding position facing the group for a few seconds is what makes it
        # an approach: it is the moment a person would use to say hello, and it
        # gives the metrics recorder a stable pose to score rather than a
        # fly-past.
        self.declare_parameter('dwell_time_s', 6.0)
        self._dwell_until = 0.0
        self._dwell_group = None
        self.create_timer(1.0, self.dwell_tick)

        # --- Unwedge reflex ----------------------------------------------------
        # A last-resort recovery that does NOT go through Nav2.
        #
        # The robot ended one run motionless for 81+ seconds at (5.72, -3.46),
        # 5 cm from a person, jammed between a group and a table. Nav2's own
        # recovery behaviours assume room to rotate and reverse into; a robot
        # wedged in a crowd has neither, and its recovery loop just spins.
        # Earlier stuck-handling was worse than useless because it only ran
        # while an approach goal was active - once the group had been retired,
        # nothing checked at all.
        #
        # So: a periodic check, independent of any goal, that reverses the base
        # directly on the velocity topic. Crude, and deliberately so - the point
        # is to get free when the navigation stack cannot.
        self.declare_parameter('unwedge_after_s', 18.0)
        self.declare_parameter('unwedge_speed', -0.15)
        self.declare_parameter('unwedge_duration_s', 3.0)
        self._unwedge_until = 0.0
        self.cmd_pub = self.create_publisher(
            Twist, '/mobile_base_controller/cmd_vel_unstamped', 10)
        self.create_timer(2.0, self.unwedge_tick)
        self._attempts: dict = {}          # (rx, ry) -> count
        self._finished_groups: list = []   # done or abandoned
        # 'gap'  - stand in the widest opening of the formation (P-space)
        # 'line' - the original: straight along the robot's line of sight.
        #          This is the variant the OFFLINE evaluation used, so keep it
        #          available for a like-for-like comparison in the write-up.
        self.declare_parameter('approach_mode', 'gap')

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

        # Laser, used only to decide whether reversing is safe.
        from sensor_msgs.msg import LaserScan
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        self.create_subscription(
            LaserScan, '/scan_raw', self.scan_callback,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1))

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

    def group_key(self, x: float, y: float):
        """Snap a group position to a memory cell, so small perception jitter
        does not look like a brand-new group every frame."""
        r = self.get_parameter('group_memory_radius').value
        return (round(x / r), round(y / r))

    def group_finished(self, x: float, y: float) -> bool:
        r = self.get_parameter('group_memory_radius').value
        return any(math.dist((x, y), g) < r for g in self._finished_groups)

    def retire_group(self, x: float, y: float, reason: str) -> None:
        """Stop considering this group, and tell the mission to carry on."""
        if self.group_finished(x, y):
            return
        self._finished_groups.append((x, y))
        self.get_logger().info(
            f'Group at ({x:.2f}, {y:.2f}) retired ({reason}). '
            f'{len(self._finished_groups)} group(s) done this run.')
        done = PointStamped()
        done.header.frame_id = self.get_parameter('map_frame').value
        done.header.stamp = self.get_clock().now().to_msg()
        done.point.x, done.point.y = float(x), float(y)
        self.approach_done_pub.publish(done)

    def group_is_moving(self, x: float, y: float) -> bool:
        """Is this group drifting across the room rather than standing still?"""
        now = self.get_clock().now().nanoseconds / 1e9
        window = self.get_parameter('motion_window_s').value

        self._centroid_history.append((now, x, y))
        self._centroid_history = [h for h in self._centroid_history
                                  if now - h[0] <= window]
        if len(self._centroid_history) < 3:
            return False

        t0, x0, y0 = self._centroid_history[0]
        dt = now - t0
        if dt < 1.0:
            return False
        speed = math.dist((x, y), (x0, y0)) / dt
        return speed > self.get_parameter('moving_speed_threshold').value

    def effective_clearance(self) -> float:
        """Centre-to-centre distance that yields the requested free gap."""
        return (self.get_parameter('min_person_clearance').value
                + self.get_parameter('robot_radius').value
                + self.get_parameter('person_radius').value)

    def check_stuck(self) -> bool:
        """True if the robot has a goal but has not moved for a while.

        Without this the robot could sit pressed against people indefinitely -
        one run was 84% stationary. Nav2's own recovery behaviours assume space
        to rotate and reverse into, which a robot surrounded by a crowd does
        not have, so the approach has to give up on its own.
        """
        pose = self.get_robot_position()
        if pose is None:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        thresh = self.get_parameter('stuck_move_threshold_m').value

        if self._last_pose_xy is None:
            self._last_pose_xy = pose
            self._last_moved_time = now
            return False

        if math.dist(pose, self._last_pose_xy) > thresh:
            self._last_pose_xy = pose
            self._last_moved_time = now
            return False

        return (now - self._last_moved_time
                > self.get_parameter('stuck_timeout_s').value)

    def gap_approach_pose(self, gx: float, gy: float, rx: float, ry: float):
        """
        Stand in the widest opening of the formation, facing in.

        Returns (x, y) or None if no opening clears the people.
        """
        clearance = self.effective_clearance()
        max_r = self.get_parameter('max_standoff').value

        bearings = sorted(math.atan2(p[1] - gy, p[0] - gx) for p in self._people)
        n = len(bearings)

        # Angular midpoints of every gap between neighbouring people, including
        # the wrap-around gap between the last and the first.
        gaps = []
        for i in range(n):
            a = bearings[i]
            b = bearings[(i + 1) % n] + (2 * math.pi if i == n - 1 else 0.0)
            width = b - a
            gaps.append((width, a + width / 2.0))

        robot_bearing = math.atan2(ry - gy, rx - gx)

        def angdiff(a, b):
            return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

        # REACHABILITY BEFORE WIDTH.
        #
        # Sorting by width first sent the robot to the far side of the group:
        # logs showed "177 deg gap ... 179 deg off the robot's current side"
        # over and over, which means walking all the way around - straight
        # through the people. That produced the collisions and the cut-through
        # events.
        #
        # Any gap wide enough to stand in is socially acceptable, so among the
        # adequate ones take the one nearest the robot's current bearing. The
        # nearest adequate opening is the one a person would use.
        min_width = math.radians(self.get_parameter('min_gap_deg').value)
        usable = [g for g in gaps if g[0] >= min_width]
        if usable:
            usable.sort(key=lambda g: angdiff(g[1], robot_bearing))
            gaps = usable
        else:
            gaps.sort(key=lambda g: -g[0])     # nothing ideal - take the widest

        for width, mid in gaps:
            if width < math.radians(30):
                continue                       # too tight to stand in at all
            r = clearance
            while r <= max_r:
                cx = gx + math.cos(mid) * r
                cy = gy + math.sin(mid) * r
                if min(math.dist((cx, cy), p) for p in self._people) >= clearance:
                    self.get_logger().info(
                        f'Approaching through a {math.degrees(width):.0f} deg gap '
                        f'at {r:.2f} m from the group centre '
                        f'({math.degrees(angdiff(mid, robot_bearing)):.0f} deg '
                        'off the robot\'s current side).')
                    return cx, cy
                r += 0.1
        return None

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
        clearance = self.effective_clearance()
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

        # --- Prefer a GAP in the formation ------------------------------------
        # Everything above approaches along the robot's line of sight, which
        # can point straight at somebody's back. People standing in a circle
        # leave openings, and those openings are where a newcomer is expected
        # to arrive - Kendon's P-space, the ring just outside the O-space where
        # participants stand.
        #
        # So: take each person's bearing from the group centre, find the widest
        # angular gap between neighbours, and aim for the middle of it, at a
        # radius that clears the nearest person by min_person_clearance. Among
        # candidate gaps, prefer the one nearest the robot's current bearing,
        # so it does not walk all the way around a group to use a marginally
        # wider opening.
        if self.get_parameter('approach_mode').value == 'gap' and len(self._people) >= 2:
            gap = self.gap_approach_pose(group_x, group_y, robot_x, robot_y)
            if gap is not None:
                approach_x, approach_y = gap

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

        # --- Standing with a group right now? stay put --------------------------
        # No new goals while dwelling, or the robot would drive off mid-greeting.
        if self._dwell_group is not None:
            return

        # --- Already dealt with? -----------------------------------------------
        if self.group_finished(msg.point.x, msg.point.y):
            return

        # --- Walking past? do not chase ---------------------------------------
        if self.group_is_moving(msg.point.x, msg.point.y):
            self.get_logger().info(
                f'Group at ({msg.point.x:.2f}, {msg.point.y:.2f}) is moving - '
                'not a standing conversation, so not approaching. Nav2 will '
                'still avoid them.', throttle_duration_sec=10.0)
            return

        # --- Wedged in? back out before doing anything else -------------------
        if self._goal_in_flight and self.check_stuck():
            self.get_logger().warn(
                'Not moving with a goal active - assuming the robot is wedged. '
                'Retreating away from the group.')
            rx, ry = robot_x, robot_y
            dx, dy = rx - msg.point.x, ry - msg.point.y
            d = math.hypot(dx, dy)
            if d > 1e-6:
                # 1.5 m directly away from the group centre, still facing it.
                back_x = rx + (dx / d) * 1.5
                back_y = ry + (dy / d) * 1.5
                self._goal_in_flight = False
                self._last_pose_xy = None
                self.send_nav_goal(back_x, back_y,
                                   math.atan2(msg.point.y - back_y,
                                              msg.point.x - back_x),
                                   frame_id=msg.header.frame_id or 'map')
            return

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

        # Count this attempt, and give up on the group after too many.
        key = self.group_key(msg.point.x, msg.point.y)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        attempts = self._attempts[key]
        max_attempts = self.get_parameter('max_attempts_per_group').value
        if attempts > max_attempts:
            self.retire_group(msg.point.x, msg.point.y,
                              f'unreachable after {max_attempts} attempts')
            return

        self._last_goal_xy = (approach_x, approach_y)
        self._last_goal_time = now
        self._current_target = (msg.point.x, msg.point.y)
        self.get_logger().info(f'Approach attempt {attempts}/{max_attempts}.')

        # Tell the mission to stand down BEFORE sending the goal, so it stops
        # issuing waypoints that would preempt this approach.
        claim = PointStamped()
        claim.header.frame_id = msg.header.frame_id or self.get_parameter('map_frame').value
        claim.header.stamp = self.get_clock().now().to_msg()
        claim.point.x, claim.point.y = msg.point.x, msg.point.y
        self.approach_start_pub.publish(claim)

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
        self.get_logger().info('Nav2 goal finished.')
        self._goal_in_flight = False

        # Did we actually END UP at the approach pose? Nav2 reports "finished"
        # for aborted and cancelled goals too, so position is checked directly
        # rather than trusted from the status code.
        pose = self.get_robot_position()
        if pose is None or self._last_goal_xy is None or self._current_target is None:
            return

        error = math.dist(pose, self._last_goal_xy)
        if error > 0.75:
            self.get_logger().info(
                f'Ended {error:.2f} m from the intended approach pose - '
                'not counting this as a completed approach.')
            return

        gx, gy = self._current_target
        dwell = self.get_parameter('dwell_time_s').value
        self.get_logger().info(
            f'ARRIVED: standing {math.dist(pose, (gx, gy)):.2f} m from the '
            f'group centre, {error:.2f} m from the intended pose. '
            f'Holding position for {dwell:.0f}s.')
        self._dwell_until = self.get_clock().now().nanoseconds / 1e9 + dwell
        self._dwell_group = (gx, gy)
        self._current_target = None

    def scan_callback(self, msg) -> None:
        self._last_scan = msg

    def rear_is_clear(self) -> bool:
        """Is there space behind the robot? Unknown counts as 'no'."""
        scan = getattr(self, '_last_scan', None)
        if scan is None:
            return False
        n = len(scan.ranges)
        if n == 0:
            return False
        # Beams pointing backwards sit at both ends of the array for a
        # front-facing scanner; take a slice from each end.
        edge = max(1, n // 12)
        rear = list(scan.ranges[:edge]) + list(scan.ranges[-edge:])
        rear = [r for r in rear if math.isfinite(r) and 0.3 < r < 10.0]
        if not rear:
            return False
        return min(rear) > 0.8

    def unwedge_tick(self) -> None:
        """Reverse the base directly if the robot has stopped moving at all."""
        now = self.get_clock().now().nanoseconds / 1e9

        # Mid-recovery: keep the burst going until it is done.
        #
        # Reversing blind was itself a source of collisions - the robot backed
        # into tables it could not see behind itself, because TIAGo's laser
        # covers roughly the front semicircle only. If the rear is not known to
        # be clear, rotate on the spot instead: turning cannot translate the
        # robot into an obstacle, and it usually reveals a free direction.
        if now < self._unwedge_until:
            cmd = Twist()
            if self.rear_is_clear():
                cmd.linear.x = self.get_parameter('unwedge_speed').value
                cmd.angular.z = 0.3
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.6      # rotate in place - always safe
            self.cmd_pub.publish(cmd)
            return

        # Standing with a group on purpose is not being stuck.
        if self._dwell_group is not None:
            self._last_pose_xy = None
            return

        # Neither is having nothing to do.
        #
        # Once the mission has finished, the robot legitimately sits still. The
        # first version of this reflex did not check that, so it reversed the
        # base every 34 seconds for the rest of the session, logging "Nav2
        # could not recover on its own" while there was nothing to recover
        # from. Only intervene if we have actually been trying to drive
        # somewhere recently.
        if now - self._last_goal_time > 120.0:
            self._last_pose_xy = None
            return

        # Neither is standing still because there is nothing left to do.
        #
        # Once the mission has finished, the robot parks at its end waypoint
        # and stops - correctly. The first version of this reflex read that as
        # "wedged" and reversed the robot every 30 seconds for as long as the
        # process lived, filling the log and shunting it around an empty room
        # after the trial had already been recorded.
        #
        # Only intervene while the robot is actually trying to get somewhere:
        # a goal in flight, or one issued in the last minute.
        recent_goal = (now - self._last_goal_time) < 60.0
        if not self._goal_in_flight and not recent_goal:
            self._last_pose_xy = None
            return

        pose = self.get_robot_position()
        if pose is None:
            return

        if self._last_pose_xy is None:
            self._last_pose_xy = pose
            self._last_moved_time = now
            return

        if math.dist(pose, self._last_pose_xy) > 0.05:
            self._last_pose_xy = pose
            self._last_moved_time = now
            return

        if now - self._last_moved_time > self.get_parameter('unwedge_after_s').value:
            self.get_logger().warn(
                f'Motionless for '
                f'{self.get_parameter("unwedge_after_s").value:.0f}s at '
                f'({pose[0]:.2f}, {pose[1]:.2f}) - reversing to get free. '
                'Nav2 could not recover on its own.')
            self._unwedge_until = now + self.get_parameter('unwedge_duration_s').value
            self._last_moved_time = now
            self._last_pose_xy = None
            self._goal_in_flight = False

    def dwell_tick(self) -> None:
        """End the dwell, then release the group and the mission."""
        if self._dwell_group is None:
            return
        if self.get_clock().now().nanoseconds / 1e9 < self._dwell_until:
            return
        gx, gy = self._dwell_group
        self._dwell_group = None
        self.get_logger().info(
            f'APPROACH COMPLETE at ({gx:.2f}, {gy:.2f}) - leaving the group '
            'and resuming the tour.')
        self.retire_group(gx, gy, 'approached successfully')


def main(args=None):
    rclpy.init(args=args)
    node = GroupApproachBaselineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
