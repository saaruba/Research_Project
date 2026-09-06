"""
Give the people in a Gazebo world a body the LiDAR can see.

    python3 scripts/add_person_collisions.py \
        --world src/tiago_social_worlds/worlds/restaurant_testing.world

    # preview without writing
    python3 scripts/add_person_collisions.py --world ... --dry-run

WHY THIS IS NEEDED
------------------
A Gazebo <actor> is a VISUAL ONLY entity. It has a skin and an animation, but
no <collision> geometry, so laser beams pass straight through it. The
consequence in this project was that the robot walked into people:

  * the LiDAR never returned anything for a human, so Nav2's costmap treated
    occupied floor as free and planned paths through standing groups;
  * the camera did detect them, but /detected_people feeds the approach policy,
    not the navigation stack - nothing connected perception to avoidance;
  * a person walking across the robot's path could not be avoided at all,
    because as far as Nav2 was concerned nobody was there.

This script inserts a thin static cylinder at each actor's position, sized like
a standing human. The cylinder has collision but NO visual, so the scene looks
unchanged while the laser finally sees a body.

WHY THE CYLINDERS ARE KEPT OUT OF THE STATIC MAP
------------------------------------------------
world_to_map.py skips models named `person_collision_*` (see PERSON_PREFIX).
That is deliberate. If people were baked into the static map, Nav2 would route
around them from the outset and O-space intrusion would be impossible by
construction - the metric would read zero for reasons that have nothing to do
with the policy. Keeping them out means the space between people stays free in
the map, the robot CAN intrude, and whether it does is a genuine measurement of
the approach behaviour. The laser still sees them live, so real collisions are
avoided.

LIMITATION, WORTH STATING IN THE WRITE-UP
-----------------------------------------
The cylinders are static. An actor that walks away from its start point leaves
its collision body behind, so moving people remain partially invisible to the
navigation stack. Standing groups - the ones that matter for F-formation
approach - are handled correctly.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

PERSON_PREFIX = 'person_collision_'
RADIUS = 0.25          # a standing adult, roughly
HEIGHT = 1.70

TEMPLATE = """
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="body">
        <collision name="body_collision">
          <geometry>
            <cylinder>
              <radius>{radius}</radius>
              <length>{height}</length>
            </cylinder>
          </geometry>
        </collision>
      </link>
    </model>
"""


def actor_xy(actor: ET.Element):
    """An actor's position: its own pose, or its first trajectory waypoint.

    Animated actors leave <actor><pose> at the origin and carry their real
    position in the trajectory, so reading only the actor pose would stack
    every cylinder on top of the world origin.
    """
    pose = actor.findtext('./pose')
    if pose:
        parts = pose.split()
        if len(parts) >= 2:
            x, y = float(parts[0]), float(parts[1])
            if abs(x) > 1e-6 or abs(y) > 1e-6:
                return x, y

    wp = actor.find('./script/trajectory/waypoint/pose')
    if wp is not None and wp.text:
        parts = wp.text.split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True, type=Path)
    ap.add_argument('--radius', type=float, default=RADIUS)
    ap.add_argument('--height', type=float, default=HEIGHT)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    text = args.world.read_text()
    root = ET.fromstring(text)

    actors = root.findall('.//actor')
    if not actors:
        print('No <actor> elements found - nothing to do.')
        return

    if PERSON_PREFIX in text:
        print(f'This world already contains {PERSON_PREFIX}* models.')
        print('Remove them first if you want to regenerate.')
        return

    blocks = []
    for i, actor in enumerate(actors, 1):
        name = actor.get('name', f'actor_{i}')
        pos = actor_xy(actor)
        if pos is None:
            print(f'  no position for {name} - skipped')
            continue
        x, y = pos
        blocks.append(TEMPLATE.format(
            name=f'{PERSON_PREFIX}{i:02d}',
            x=f'{x:.3f}', y=f'{y:.3f}', z=f'{args.height / 2.0:.3f}',
            radius=args.radius, height=args.height))
        print(f'  {name:34s} -> collision cylinder at ({x:6.2f}, {y:6.2f})')

    if not blocks:
        print('Nothing to insert.')
        return

    print(f'\n{len(blocks)} cylinder(s), r={args.radius} m, h={args.height} m')

    if args.dry_run:
        print('(dry run - nothing written)')
        return

    # Insert just before </world>, preserving the rest of the file byte for
    # byte. Rewriting via ElementTree would reformat the whole world and lose
    # the comments that document the layout.
    marker = '</world>'
    idx = text.rfind(marker)
    if idx == -1:
        print('ERROR: no </world> tag found.')
        return

    backup = args.world.with_suffix('.world.pre_collisions')
    if not backup.exists():
        backup.write_text(text)
        print(f'Backup: {backup}')

    args.world.write_text(text[:idx] + ''.join(blocks) + '\n  ' + text[idx:])
    print(f'Updated: {args.world}')
    print('\nNow regenerate the map (the cylinders are excluded from it '
          'on purpose):')
    print(f'  python3 scripts/world_to_map.py --world {args.world} \\')
    print(f'      --output src/tiago_social_worlds/maps/{args.world.stem}')


if __name__ == '__main__':
    main()
