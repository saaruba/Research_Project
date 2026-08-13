#!/usr/bin/env python3
"""
Generate a Gazebo Classic world containing conversational groups of people
standing in F-formations, for the TIAGo group-approach experiments.

WHY THIS IS A GENERATOR AND NOT A HAND-WRITTEN .world FILE
-----------------------------------------------------------
Phase H of the project requires evaluating on "at least one unseen group
configuration". Hand-editing SDF XML for every scenario is slow and
error-prone, so this script produces worlds parametrically: change the group
list, get a new scenario. It also writes a companion JSON file recording the
exact ground-truth position of every person and every group centre - which is
what makes the O-space intrusion / min-distance / cut-through metrics
measurable in metres, something the recorded PLUS-HRI video could never
support (no depth, no calibration).

PEOPLE ARE BUILT FROM PRIMITIVES ON PURPOSE
--------------------------------------------
Each person is a capsule-ish torso + sphere head + two legs, not an imported
mesh. Reasons:
  - zero external dependencies: no model:// downloads, no Gazebo online model
    database (which frequently fails inside containers), no missing-mesh
    errors on a machine that has never run it before;
  - what the experiments actually measure is GEOMETRY - where people stand,
    how far the robot stops, whether it enters the O-space. Visual fidelity
    changes none of those numbers.
If you later want photo-realistic people for screenshots, swap the <visual>
blocks for a mesh; the <collision> geometry and the ground-truth JSON stay
valid.

All people are static (`<static>true</static>`) so they stand still rather
than toppling over on contact - appropriate for a standing conversational
group, and it keeps the physics stable.

Usage:
    # default scenario (used for development/testing)
    python3 scripts/generate_social_world.py

    # a different, unseen configuration for final evaluation
    python3 scripts/generate_social_world.py --scenario unseen \\
        --output src/tiago_social_worlds/worlds/restaurant_humans_unseen.world
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "tiago_social_worlds" / "worlds" / "restaurant_humans.world"

# Scenario definitions: each group is (centre_x, centre_y, num_people, radius).
# Radius ~0.7 m matches a typical small standing conversational group, where
# the O-space (the shared empty middle) is roughly 1.2-1.5 m across.
SCENARIOS: dict[str, list[tuple[float, float, int, float]]] = {
    # Development scenario: one 3-person group and one 2-person pair, well
    # separated so clustering should find exactly two groups.
    "default": [
        (2.5, 0.0, 3, 0.70),
        (-1.5, 2.0, 2, 0.60),
    ],
    # Unseen evaluation scenario (Phase H): different group sizes, positions
    # and spacing from the development scenario, so the policy is genuinely
    # tested on a configuration it was not tuned against.
    "unseen": [
        (3.0, 2.5, 4, 0.85),
        (0.0, -2.5, 3, 0.70),
        (-2.5, 1.0, 2, 0.55),
    ],
    # Stress case: two groups close together - tests whether clustering
    # correctly separates them rather than merging into one blob.
    "adjacent": [
        (1.5, 0.0, 3, 0.70),
        (3.6, 0.0, 3, 0.70),
    ],
}


def person_sdf(name: str, x: float, y: float, yaw: float) -> str:
    """One standing person, built from primitives. Roughly 1.7 m tall."""
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose>

      <link name="body">
        <!-- Torso -->
        <collision name="torso_collision">
          <pose>0 0 1.05 0 0 0</pose>
          <geometry><cylinder><radius>0.20</radius><length>0.75</length></cylinder></geometry>
        </collision>
        <visual name="torso_visual">
          <pose>0 0 1.05 0 0 0</pose>
          <geometry><cylinder><radius>0.20</radius><length>0.75</length></cylinder></geometry>
          <material><ambient>0.2 0.3 0.6 1</ambient><diffuse>0.25 0.4 0.75 1</diffuse></material>
        </visual>

        <!-- Head -->
        <collision name="head_collision">
          <pose>0 0 1.60 0 0 0</pose>
          <geometry><sphere><radius>0.115</radius></sphere></geometry>
        </collision>
        <visual name="head_visual">
          <pose>0 0 1.60 0 0 0</pose>
          <geometry><sphere><radius>0.115</radius></sphere></geometry>
          <material><ambient>0.7 0.55 0.45 1</ambient><diffuse>0.85 0.68 0.55 1</diffuse></material>
        </visual>

        <!-- Nose marker: makes facing direction visible in the GUI, which
             matters because F-formation is all about who faces whom. -->
        <visual name="facing_marker">
          <pose>0.13 0 1.60 0 1.5708 0</pose>
          <geometry><cylinder><radius>0.03</radius><length>0.06</length></cylinder></geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient><diffuse>0.9 0.25 0.25 1</diffuse></material>
        </visual>

        <!-- Legs -->
        <collision name="legs_collision">
          <pose>0 0 0.35 0 0 0</pose>
          <geometry><cylinder><radius>0.17</radius><length>0.70</length></cylinder></geometry>
        </collision>
        <visual name="leg_left_visual">
          <pose>0.09 0 0.35 0 0 0</pose>
          <geometry><cylinder><radius>0.075</radius><length>0.70</length></cylinder></geometry>
          <material><ambient>0.15 0.15 0.2 1</ambient><diffuse>0.2 0.2 0.25 1</diffuse></material>
        </visual>
        <visual name="leg_right_visual">
          <pose>-0.09 0 0.35 0 0 0</pose>
          <geometry><cylinder><radius>0.075</radius><length>0.70</length></cylinder></geometry>
          <material><ambient>0.15 0.15 0.2 1</ambient><diffuse>0.2 0.2 0.25 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def build_groups(spec: list[tuple[float, float, int, float]]) -> tuple[str, list[dict]]:
    """Place people evenly on a circle around each group centre, facing inward."""
    models: list[str] = []
    ground_truth: list[dict] = []

    for group_id, (cx, cy, count, radius) in enumerate(spec):
        members = []
        for person_id in range(count):
            theta = 2.0 * math.pi * person_id / count
            px = cx + radius * math.cos(theta)
            py = cy + radius * math.sin(theta)
            # Face the group centre: stand at angle theta, look back along it.
            yaw = theta + math.pi
            name = f"person_g{group_id}_p{person_id}"
            models.append(person_sdf(name, px, py, yaw))
            members.append({
                "name": name,
                "x": round(px, 3),
                "y": round(py, 3),
                "yaw": round(math.atan2(math.sin(yaw), math.cos(yaw)), 4),
            })

        ground_truth.append({
            "group_id": group_id,
            "centre_x": cx,
            "centre_y": cy,
            "radius": radius,
            "num_people": count,
            # The O-space is the shared empty middle. Approximating its radius
            # as the group circle radius is the same mutual-facing assumption
            # used throughout Phase C - stated here so the metric code and the
            # dissertation agree on one definition.
            "ospace_radius": radius,
            "members": members,
        })

    return "\n".join(models), ground_truth


def build_world(spec: list[tuple[float, float, int, float]]) -> tuple[str, list[dict]]:
    people_sdf, ground_truth = build_groups(spec)

    world = f"""<?xml version="1.0" ?>
<!-- GENERATED by scripts/generate_social_world.py - do not edit by hand.
     Regenerate with a different scenario argument to create new
     configurations. NOTE: XML comments may not contain a double hyphen,
     so option flags are written without their leading dashes here. -->
<sdf version="1.6">
  <world name="restaurant_humans">

    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <shadows>false</shadows>
    </scene>

    <!-- Bounding walls: give the LiDAR something to localise against.
         Without walls, AMCL has no features to match and Nav2 localisation
         will drift badly. -->
    <model name="walls">
      <static>true</static>
      <link name="wall_link">
        <collision name="north"><pose>0 6 1.25 0 0 0</pose>
          <geometry><box><size>16 0.15 2.5</size></box></geometry></collision>
        <visual name="north_v"><pose>0 6 1.25 0 0 0</pose>
          <geometry><box><size>16 0.15 2.5</size></box></geometry>
          <material><ambient>0.8 0.8 0.78 1</ambient></material></visual>

        <collision name="south"><pose>0 -6 1.25 0 0 0</pose>
          <geometry><box><size>16 0.15 2.5</size></box></geometry></collision>
        <visual name="south_v"><pose>0 -6 1.25 0 0 0</pose>
          <geometry><box><size>16 0.15 2.5</size></box></geometry>
          <material><ambient>0.8 0.8 0.78 1</ambient></material></visual>

        <collision name="east"><pose>8 0 1.25 0 0 0</pose>
          <geometry><box><size>0.15 12 2.5</size></box></geometry></collision>
        <visual name="east_v"><pose>8 0 1.25 0 0 0</pose>
          <geometry><box><size>0.15 12 2.5</size></box></geometry>
          <material><ambient>0.8 0.8 0.78 1</ambient></material></visual>

        <collision name="west"><pose>-8 0 1.25 0 0 0</pose>
          <geometry><box><size>0.15 12 2.5</size></box></geometry></collision>
        <visual name="west_v"><pose>-8 0 1.25 0 0 0</pose>
          <geometry><box><size>0.15 12 2.5</size></box></geometry>
          <material><ambient>0.8 0.8 0.78 1</ambient></material></visual>
      </link>
    </model>
{people_sdf}

  </world>
</sdf>
"""
    return world, ground_truth


# Room geometry - MUST match the <walls> model in build_world().
ROOM_HALF_X = 8.0
ROOM_HALF_Y = 6.0
WALL_THICKNESS = 0.15
MAP_RESOLUTION = 0.05  # metres per pixel


def write_nav2_map(map_stem: Path) -> tuple[Path, Path]:
    """
    Write an occupancy grid matching the generated world, so Nav2 can localise
    without you having to drive the robot around running SLAM first.

    Because this world is a simple rectangular room with known wall positions,
    the map can be computed exactly rather than mapped - which removes a whole
    error-prone step (SLAM drift, partial coverage, re-running after every
    world change).

    PGM convention used by nav2_map_server with the default thresholds:
        0   = occupied (black)
        254 = free (white)
    People are deliberately NOT drawn into the map: they are dynamic obstacles
    that the local costmap should react to from live sensor data. Baking them
    into the static map would make the planner permanently avoid places where
    a person merely happened to stand once.
    """
    width = int(round(2 * ROOM_HALF_X / MAP_RESOLUTION))
    height = int(round(2 * ROOM_HALF_Y / MAP_RESOLUTION))

    # Start all free, then stamp the walls as occupied.
    grid = bytearray([254] * (width * height))
    wall_px = max(1, int(round(WALL_THICKNESS / MAP_RESOLUTION)))

    def fill(x0: int, y0: int, x1: int, y1: int) -> None:
        for y in range(max(0, y0), min(height, y1)):
            row = y * width
            for x in range(max(0, x0), min(width, x1)):
                grid[row + x] = 0

    fill(0, 0, width, wall_px)                      # south
    fill(0, height - wall_px, width, height)        # north
    fill(0, 0, wall_px, height)                     # west
    fill(width - wall_px, 0, width, height)         # east

    pgm_path = map_stem.with_suffix(".pgm")
    with pgm_path.open("wb") as handle:
        handle.write(b"P5\n")
        handle.write(b"# generated by scripts/generate_social_world.py\n")
        handle.write(f"{width} {height}\n255\n".encode())
        handle.write(bytes(grid))

    yaml_path = map_stem.with_suffix(".yaml")
    yaml_path.write_text(
        f"image: {pgm_path.name}\n"
        f"resolution: {MAP_RESOLUTION}\n"
        # origin = world coordinates of the map's bottom-left pixel
        f"origin: [{-ROOM_HALF_X}, {-ROOM_HALF_Y}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n",
        encoding="utf-8",
    )
    return pgm_path, yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-map", action="store_true",
                        help="skip writing the Nav2 occupancy grid")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the output world even if it was not generated "
                             "by this script (refuses by default, to protect hand-built worlds)")
    args = parser.parse_args()

    spec = SCENARIOS[args.scenario]
    world, ground_truth = build_world(spec)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # REFUSE to clobber an existing world unless explicitly told to.
    # This guard exists because an earlier run of this script silently
    # overwrote a hand-built world containing 15 animated LIRS-HMLG actors
    # with generated primitives. It was recoverable from git, but only by
    # luck. Hand-authored scenes are worth far more than generated ones -
    # never destroy them by default.
    if output.exists() and not args.force:
        existing = output.read_text(encoding="utf-8", errors="ignore")
        generated_by_us = "GENERATED by scripts/generate_social_world.py" in existing
        actor_count = existing.count("<actor")
        if not generated_by_us:
            print(f"REFUSING to overwrite {output}")
            print(f"  That file was NOT generated by this script"
                  f"{f' and contains {actor_count} <actor> element(s)' if actor_count else ''}.")
            print("  Write somewhere else with --output, or pass --force if you")
            print("  really mean to replace it.")
            raise SystemExit(1)

    output.write_text(world, encoding="utf-8")

    gt_path = output.with_suffix(".groundtruth.json")
    gt_path.write_text(json.dumps({
        "scenario": args.scenario,
        "world_file": output.name,
        "groups": ground_truth,
    }, indent=2), encoding="utf-8")

    total_people = sum(g["num_people"] for g in ground_truth)
    print(f"Scenario '{args.scenario}': {len(ground_truth)} group(s), {total_people} people")
    for g in ground_truth:
        print(f"  group {g['group_id']}: {g['num_people']} people around "
              f"({g['centre_x']}, {g['centre_y']}), O-space radius {g['ospace_radius']} m")
    print(f"\nWorld:        {output}")
    print(f"Ground truth: {gt_path}")

    if not args.no_map:
        maps_dir = PROJECT_ROOT / "src" / "tiago_social_worlds" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        pgm_path, yaml_path = write_nav2_map(maps_dir / "restaurant")
        print(f"Nav2 map:     {yaml_path}")
        print(f"              {pgm_path}")
    print("\nThe ground-truth JSON gives every person's exact position in METRES -")
    print("this is what makes the O-space / distance / cut-through metrics")
    print("measurable in simulation, unlike the uncalibrated recorded video.")


if __name__ == "__main__":
    main()
