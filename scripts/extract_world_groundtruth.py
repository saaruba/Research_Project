#!/usr/bin/env python3
"""
Extract ground-truth person and group positions from a HAND-BUILT Gazebo world.

WHY THIS IS NEEDED
------------------
metrics_recorder_node scores the social metrics (O-space intrusion, minimum
distance to a person, group cut-through) against where people ACTUALLY are, not
where the robot thinks they are. Worlds produced by generate_social_world.py
ship that ground truth automatically; hand-authored worlds do not. This script
recovers it by parsing the world file.

THE TRICKY BIT: ANIMATED ACTORS DO NOT STORE THEIR POSITION WHERE YOU EXPECT
-----------------------------------------------------------------------------
A Gazebo <actor> can carry its position in two different places:

  1. <actor><pose>            - used by static, non-animated actors
  2. <script><trajectory><waypoint><pose>  - used by ANIMATED actors, whose
                                             actor-level <pose> is often left
                                             at "0 0 0 0 0 0"

Reading only the actor-level pose therefore silently places every animated
person at the world origin - which would make every social metric wrong while
looking perfectly plausible. This script prefers a non-zero actor pose and
falls back to the first trajectory waypoint.

GROUPING
--------
Actors are grouped by the leading `group_NN_` component of their name, which is
the convention used in this project's worlds (group_01_person_01_talking, ...).
Anything that does not match is reported as ungrouped so it is never silently
dropped.

The O-space radius is estimated as the mean distance from the group centroid to
its members - i.e. the radius of the circle the people have formed, which is
exactly the shared space a robot should not enter. That matches the
mutual-facing assumption used throughout Phase C.

Usage:
    python3 scripts/extract_world_groundtruth.py \\
        --world src/tiago_social_worlds/worlds/restaurant_testing.world
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

GROUP_RE = re.compile(r'(group[_-]?\d+)', re.IGNORECASE)


def parse_pose(text: str | None) -> tuple[float, float, float] | None:
    """Return (x, y, yaw) from an SDF pose string, or None if unusable."""
    if not text:
        return None
    parts = text.split()
    if len(parts) < 6:
        return None
    try:
        values = [float(v) for v in parts]
    except ValueError:
        return None
    return values[0], values[1], values[5]


def actor_position(actor: ET.Element) -> tuple[float, float, float] | None:
    """
    Position of one actor, checking BOTH places Gazebo may store it.

    An animated actor typically leaves <actor><pose> at the origin and puts the
    real position in its first trajectory waypoint. Treating a 0,0 actor pose as
    authoritative would drop every animated person onto the world origin.
    """
    direct = parse_pose(actor.findtext('./pose'))
    if direct is not None and (abs(direct[0]) > 1e-6 or abs(direct[1]) > 1e-6):
        return direct

    waypoints = actor.findall('./script/trajectory/waypoint')
    for wp in waypoints:
        pose = parse_pose(wp.findtext('./pose'))
        if pose is not None and (abs(pose[0]) > 1e-6 or abs(pose[1]) > 1e-6):
            return pose

    return direct  # genuinely at the origin, or unparseable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', required=True, type=Path)
    parser.add_argument('--output', type=Path,
                        help='defaults to <world>.groundtruth.json')
    args = parser.parse_args()

    world_path = args.world.expanduser().resolve()
    root = ET.parse(world_path).getroot()

    people: list[dict] = []
    ungrouped: list[str] = []

    for actor in root.findall('.//actor'):
        name = actor.get('name', '?')
        pos = actor_position(actor)
        if pos is None:
            print(f"  WARNING: could not read a position for '{name}' - skipped")
            continue
        x, y, yaw = pos

        match = GROUP_RE.search(name)
        if not match:
            ungrouped.append(name)
            continue

        people.append({
            'name': name,
            'group_key': match.group(1).lower().replace('-', '_'),
            'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 4),
        })

    if not people:
        print("No grouped actors found. Check the actor naming convention "
              "(expected something like group_01_person_01_...).")
        return

    by_group: dict[str, list[dict]] = defaultdict(list)
    for p in people:
        by_group[p['group_key']].append(p)

    groups = []
    for gid, (key, members) in enumerate(sorted(by_group.items())):
        cx = sum(m['x'] for m in members) / len(members)
        cy = sum(m['y'] for m in members) / len(members)
        # O-space radius = mean distance from centroid to members: the radius of
        # the circle the group has formed, which is the space to stay out of.
        radius = sum(math.dist((m['x'], m['y']), (cx, cy)) for m in members) / len(members)
        groups.append({
            'group_id': gid,
            'source_name': key,
            'centre_x': round(cx, 3),
            'centre_y': round(cy, 3),
            'radius': round(radius, 3),
            'ospace_radius': round(max(radius, 0.4), 3),  # floor for very tight pairs
            'num_people': len(members),
            'members': [{'name': m['name'], 'x': m['x'], 'y': m['y'], 'yaw': m['yaw']}
                        for m in members],
        })

    out_path = args.output or world_path.with_suffix('.groundtruth.json')
    out_path.write_text(json.dumps({
        'scenario': world_path.stem,
        'world_file': world_path.name,
        'groups': groups,
    }, indent=2), encoding='utf-8')

    print(f"World: {world_path.name}")
    print(f"People found: {len(people)}   Groups: {len(groups)}")
    for g in groups:
        print(f"\n  group {g['group_id']} ({g['source_name']}): {g['num_people']} people")
        print(f"    centre ({g['centre_x']}, {g['centre_y']})  "
              f"O-space radius {g['ospace_radius']} m")
        for m in g['members']:
            print(f"      {m['name']:34} ({m['x']:6.2f}, {m['y']:6.2f})  "
                  f"yaw {math.degrees(m['yaw']):7.1f} deg")

    if ungrouped:
        print(f"\n  NOT grouped (no group_NN_ in the name): {ungrouped}")

    print(f"\nWritten: {out_path}")


if __name__ == '__main__':
    main()
