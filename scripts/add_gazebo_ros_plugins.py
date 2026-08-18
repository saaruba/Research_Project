#!/usr/bin/env python3
"""
Add the gazebo_ros system plugins to a world file so TIAGo can actually be
spawned into it.

THE FAILURE THIS FIXES
----------------------
    [spawn_entity.py] ERROR: Service /spawn_entity unavailable.
                             Was Gazebo started with GazeboRosFactory?
    [spawn_entity.py] ERROR: Spawn service failed. Exiting.

`/spawn_entity` is provided by libgazebo_ros_factory.so. PAL's
tiago_gazebo.launch.py loads that plugin when it launches one of its OWN
worlds, but with a custom `world_name` it logs

    Private gazebo world package not found.

and starts gzserver without it. The world then loads perfectly - you can see
the room and the people in Gazebo - but the robot is never inserted, and
every downstream failure follows from that single missing service:

    no robot  ->  no controller_manager
              ->  no /odom frame
              ->  Nav2 stuck on "Invalid frame ID 'odom'"
              ->  no camera topics, so perception never starts

Declaring the plugins inside the world itself makes them load however gzserver
is invoked, so the world works with PAL's launch, with gazebo_ros's own launch,
or with a bare `gazebo` command.

Idempotent: running it twice does not duplicate the plugins.

Usage:
    python3 scripts/add_gazebo_ros_plugins.py \\
        --world src/tiago_social_worlds/worlds/restaurant_testing.world
    # all worlds at once:
    python3 scripts/add_gazebo_ros_plugins.py --all
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORLDS_DIR = PROJECT_ROOT / "src" / "tiago_social_worlds" / "worlds"

# gazebo_ros_init  : ROS time / the /clock topic
# gazebo_ros_factory: /spawn_entity and /delete_entity  <- the missing one
# gazebo_ros_state : /gazebo/get_model_state etc. Useful for evaluation, since
#                    it exposes ground-truth model poses at runtime.
PLUGINS = """
    <!-- Added by scripts/add_gazebo_ros_plugins.py -->
    <!-- gazebo_ros_factory provides /spawn_entity. Without it, TIAGo cannot be
         inserted into this world and every downstream system fails. -->
    <plugin name="gazebo_ros_init" filename="libgazebo_ros_init.so"/>
    <plugin name="gazebo_ros_factory" filename="libgazebo_ros_factory.so"/>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros>
        <namespace>/gazebo</namespace>
      </ros>
      <update_rate>10.0</update_rate>
    </plugin>
"""

MARKER = "libgazebo_ros_factory.so"


def patch(world_path: Path) -> str:
    text = world_path.read_text(encoding="utf-8")

    if MARKER in text:
        return "already patched"

    # Insert immediately after the opening <world ...> tag.
    match = re.search(r"(<world\b[^>]*>)", text)
    if not match:
        return "SKIPPED - no <world> element found"

    backup = world_path.with_suffix(world_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(world_path, backup)

    patched = text[:match.end()] + PLUGINS + text[match.end():]
    world_path.write_text(patched, encoding="utf-8")

    # Verify it is still well-formed, and roll back if not.
    import xml.etree.ElementTree as ET
    try:
        ET.parse(world_path)
    except ET.ParseError as exc:
        shutil.copy2(backup, world_path)
        return f"FAILED - XML broke ({exc}); original restored"

    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path)
    parser.add_argument("--all", action="store_true",
                        help="patch every .world in the project's worlds/ directory")
    args = parser.parse_args()

    if args.all:
        targets = sorted(WORLDS_DIR.glob("*.world"))
    elif args.world:
        targets = [args.world.expanduser().resolve()]
    else:
        parser.error("pass --world <file> or --all")

    for world in targets:
        if not world.exists():
            print(f"  MISSING  {world}")
            continue
        print(f"  {patch(world):<24} {world.name}")

    print("\nNow reinstall the worlds so tiago_gazebo picks up the patched copies:")
    print("    bash scripts/install_sim_stack.sh")


if __name__ == "__main__":
    main()
