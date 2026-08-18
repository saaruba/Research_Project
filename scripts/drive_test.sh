#!/usr/bin/env bash
# ============================================================================
# CAN THE ROBOT MOVE AT ALL?
#
#     bash scripts/drive_test.sh
#
# Run this in a SECOND terminal while the simulation is up.
#
# ----------------------------------------------------------------------------
# WHY THIS EXISTS
# ----------------------------------------------------------------------------
# "The robot is not moving" has at least four completely different causes:
#
#   1. the wheels/controller are not working at all
#   2. the wheels work, but Nav2 never sends a velocity (bad localisation,
#      aborted plan, no map)
#   3. Nav2 sends velocities but something upstream (twist_mux) blocks them
#   4. everything works but nothing ever gives the robot a goal
#
# This script separates (1) and (3) from (2) and (4) by bypassing Nav2
# completely: it publishes a velocity command straight to the base and then
# checks whether the ODOMETRY actually changed. If the robot moves here, the
# hardware side is fine and the problem is entirely in navigation. If it does
# not move here, nothing in Nav2 could ever have worked.
#
# It tries each of TIAGo's plausible command topics in turn, because which one
# is live depends on how twist_mux was configured.
# ============================================================================

set -o pipefail
source /opt/ros/humble/setup.bash

echo "============================================================"
echo "  DRIVE TEST - bypasses Nav2 entirely"
echo "============================================================"

python3 - <<'PY'
import math, time, sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

CANDIDATES = [
    '/mobile_base_controller/cmd_vel_unstamped',
    '/cmd_vel',
    '/key_vel',
    '/nav_vel',
]

def yaw_of(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

class Driver(Node):
    def __init__(self):
        super().__init__('drive_test')
        self.pose = None
        self.create_subscription(Odometry, '/mobile_base_controller/odom',
                                 self.on_odom, 10)
        self.pubs = {t: self.create_publisher(Twist, t, 10) for t in CANDIDATES}

    def on_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def wait_pose(self, limit=30.0):
        end = time.time() + limit
        while time.time() < end and self.pose is None:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self.pose

rclpy.init()
n = Driver()

print("Waiting for odometry...", flush=True)
if n.wait_pose() is None:
    print("  NO ODOMETRY. The simulation is not running, or the robot never "
          "spawned.\n  Start it first:  bash scripts/run_everything.sh")
    rclpy.shutdown()
    sys.exit(1)

print(f"  start pose: x={n.pose[0]:.3f} y={n.pose[1]:.3f} "
      f"yaw={math.degrees(n.pose[2]):.1f} deg\n", flush=True)

moved_on = []
for topic in CANDIDATES:
    subs = n.pubs[topic].get_subscription_count()
    print(f"--- {topic}  (subscribers: {subs})", flush=True)
    if subs == 0:
        print("    nobody is listening - skipping", flush=True)
        continue

    start = n.pose
    msg = Twist()
    msg.angular.z = 0.5          # spin in place: safe, needs no free space
    end = time.time() + 6.0
    while time.time() < end:
        n.pubs[topic].publish(msg)
        rclpy.spin_once(n, timeout_sec=0.1)

    n.pubs[topic].publish(Twist())   # stop
    for _ in range(10):
        rclpy.spin_once(n, timeout_sec=0.1)

    turned = abs(math.degrees(n.pose[2] - start[2]))
    turned = min(turned, 360 - turned)
    print(f"    turned {turned:.1f} deg", flush=True)
    if turned > 3.0:
        print("    *** THE ROBOT MOVED ***", flush=True)
        moved_on.append(topic)

print("\n============================================================")
if moved_on:
    print("RESULT: the base works. Commands take effect on:")
    for t in moved_on:
        print(f"          {t}")
    print("\nSo the wheels, controller and twist_mux are all fine.")
    print("The problem is in NAVIGATION - localisation, the map, or no goal.")
else:
    print("RESULT: the robot did NOT move on any topic.")
    print("\nNothing in Nav2 could have worked. Check that the controller")
    print("is actually loaded:")
    print("    ros2 control list_controllers")
print("============================================================")

rclpy.shutdown()
PY
