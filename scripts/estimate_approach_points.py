#!/usr/bin/env python3
"""
Phase C, final step: given each detected group (detected_groups.csv), compute
a candidate APPROACH POINT just outside its O-space, facing the group centre.

IMPORTANT SCOPE NOTE (read before reusing this for Phase E):
This operates entirely in 2D IMAGE-SPACE pixel coordinates from the recorded
video, normalised by frame size and group scale - there is no camera
calibration or depth in this dataset, so these are NOT real-world (x, y)
metres and cannot be sent to Nav2 as-is. Their purpose here is (a) closing
out Phase C for the recorded dataset (giving a concrete "candidate output"
per group, useful for feature engineering / write-up), and (b) proving out
the geometric rule itself. The LIVE Phase E baseline in the Gazebo
simulation will need the same rule re-applied to real-world (x, y) group
positions coming from the simulation's own perception at runtime - the
formula transfers, the coordinate system does not.

Rule (mutual-facing assumption, consistent with the O-space decision in
detected_groups.csv): approximate the robot's viewpoint as the horizontal
centre of the camera frame ("robot_proxy"). Draw a line from robot_proxy
through the group centroid. The candidate approach point sits on that line,
just outside the group's bounding extent plus a standoff buffer (in units of
the group's average bbox width, consistent with the distance normalisation
already used in cluster_groups.py), facing back toward the group centre.

Output: adds three columns to each session's detected_groups.csv:
    approach_x, approach_y   - candidate standoff point (pixel coords)
    approach_facing_deg      - angle (degrees) the robot should face,
                                 pointing from approach_x/y back to the
                                 group centroid (0 = facing right/+x,
                                 90 = facing down/+y, image convention)

Usage - one session:
    python3 scripts/estimate_approach_points.py --session dataset/9

Usage - all sessions:
    for s in 1 3 5 7 8 9 10 11 12 14 15 26 27 28 30 31 49 51 52 54 55 58 59 60; do
        python3 scripts/estimate_approach_points.py --session dataset/$s
    done
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0
STANDOFF_BBOX_MULTIPLES = 1.5  # standoff distance, in units of the group's avg bbox width


def compute_approach_point(row: pd.Series) -> tuple[float, float, float]:
    robot_proxy = np.array([IMAGE_WIDTH / 2.0, IMAGE_HEIGHT])  # bottom-centre of frame
    group_centre = np.array([row["group_center_x"], row["group_center_y"]])

    direction = group_centre - robot_proxy
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        direction_unit = np.array([0.0, -1.0])
    else:
        direction_unit = direction / dist

    group_half_extent = max(
        row["group_bbox_max_x"] - row["group_center_x"],
        row["group_center_x"] - row["group_bbox_min_x"],
        row["group_bbox_max_y"] - row["group_center_y"],
        row["group_center_y"] - row["group_bbox_min_y"],
    )
    standoff = STANDOFF_BBOX_MULTIPLES * row["avg_bbox_width"]
    stand_distance = group_half_extent + standoff

    approach_point = group_centre - direction_unit * stand_distance
    approach_point[0] = np.clip(approach_point[0], 0, IMAGE_WIDTH)
    approach_point[1] = np.clip(approach_point[1], 0, IMAGE_HEIGHT)

    facing_vector = group_centre - approach_point
    facing_deg = math.degrees(math.atan2(facing_vector[1], facing_vector[0]))

    return float(approach_point[0]), float(approach_point[1]), float(facing_deg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()

    session_path = args.session.expanduser().resolve()
    groups_path = session_path / "detected_groups.csv"
    if not groups_path.exists():
        raise FileNotFoundError(f"{groups_path} not found - run cluster_groups.py first")

    df = pd.read_csv(groups_path)
    if df.empty:
        print(f"Session {session_path.name}: no groups to process")
        return

    results = df.apply(compute_approach_point, axis=1, result_type="expand")
    df["approach_x"], df["approach_y"], df["approach_facing_deg"] = results[0], results[1], results[2]

    df.to_csv(groups_path, index=False)
    print(f"Session {session_path.name}: added approach_x/y/facing_deg to {len(df)} group rows -> {groups_path}")


if __name__ == "__main__":
    main()
