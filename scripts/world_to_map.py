#!/usr/bin/env python3
"""
Build a Nav2 occupancy grid directly FROM a Gazebo .world file.

WHY THIS EXISTS
---------------
Nav2 needs a static map to localise against. The usual route is to drive the
robot around running SLAM, which is slow, drifts, and has to be redone every
time the world changes. But a hand-built world already states exactly where
every wall is - so the map can simply be computed from it, exactly and
instantly.

This replaces the hardcoded map that an earlier version of
generate_social_world.py produced. That one assumed a 16x12 m room, which did
NOT match restaurant_humans.world (20x15 m, with a doorway gap in the south
wall, kitchen partitions and a stage). A map that disagrees with the world is
worse than no map: AMCL will appear to work and then localise the robot into
a wall.

WHAT COUNTS AS AN OBSTACLE
--------------------------
Only static box models whose vertical extent crosses the LiDAR's height are
rasterised. The reasoning:
  - the floor (top at z=0) is below the laser and must not be drawn, or the
    entire map would be occupied;
  - the stage (0 to 0.30 m) DOES cross a laser mounted at ~0.20 m, so it is a
    real obstacle for navigation and must be included;
  - walls and partitions obviously included.
Actors (people) are deliberately excluded: they are dynamic obstacles for the
local costmap to handle from live sensor data. Baking them into the static map
would make the planner permanently avoid wherever someone happened to stand.

Usage:
    python3 scripts/world_to_map.py \\
        --world src/tiago_social_worlds/worlds/restaurant_humans.world \\
        --output src/tiago_social_worlds/maps/restaurant_humans

    # TIAGo's sick-571 base laser sits at roughly 0.2 m; override if needed
    python3 scripts/world_to_map.py --world ... --laser-height 0.25
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_RESOLUTION = 0.05   # metres per pixel
DEFAULT_LASER_HEIGHT = 0.20  # metres above ground (PMB2 base laser, approx)
PADDING_M = 0.5             # free-space margin around the world extents


def parse_pose(text: str | None) -> tuple[float, float, float, float]:
    """Return (x, y, z, yaw) from an SDF pose string."""
    if not text:
        return 0.0, 0.0, 0.0, 0.0
    parts = [float(v) for v in text.split()]
    while len(parts) < 6:
        parts.append(0.0)
    return parts[0], parts[1], parts[2], parts[5]


# Half-width used for <include> furniture. The round dining table is about
# 1.1 m across, so 0.6 m covers it with a little to spare.
INCLUDE_RADIUS = 0.60


def collect_obstacles(world_path: Path, laser_height: float) -> list[dict]:
    """Every static box that the laser would actually see."""
    root = ET.parse(world_path).getroot()
    obstacles: list[dict] = []

    for model in root.findall(".//model"):
        name = model.get("name", "?")
        mx, my, mz, myaw = parse_pose(model.findtext("./pose"))

        for link in model.findall("./link"):
            lx, ly, lz, lyaw = parse_pose(link.findtext("./pose"))

            for collision in link.findall("./collision"):
                cx, cy, cz, cyaw = parse_pose(collision.findtext("./pose"))
                box = collision.find("./geometry/box/size")
                if box is None:
                    continue
                sx, sy, sz = [float(v) for v in box.text.split()]

                # Compose the (translation-only + yaw) transforms.
                yaw = myaw + lyaw + cyaw
                # rotate the link/collision offset into the model frame
                ox = lx + cx
                oy = ly + cy
                wx = mx + ox * math.cos(myaw) - oy * math.sin(myaw)
                wy = my + ox * math.sin(myaw) + oy * math.cos(myaw)
                wz = mz + lz + cz

                top = wz + sz / 2.0
                bottom = wz - sz / 2.0
                if not (bottom <= laser_height <= top):
                    continue  # laser passes over or under this box

                obstacles.append({
                    "model": name, "x": wx, "y": wy,
                    "sx": sx, "sy": sy, "yaw": yaw,
                })

    # ------------------------------------------------------------------
    # <include> models (furniture referenced as meshes).
    #
    # These were being MISSED ENTIRELY. Only <model> elements with box
    # collision geometry were rasterised, so the five dining tables - each an
    # <include> of a mesh - never appeared in the map. Nav2's global planner
    # therefore routed straight through them, and the only thing that ever
    # noticed a table was the laser clipping a thin leg at 0.2 m, far too late
    # to plan around. The robot ended up wedged against tabletops that, as far
    # as it knew, did not exist.
    #
    # A mesh has no size we can read cheaply, so each include is treated as a
    # square of side 2 * include_radius centred on its pose. Square rather than
    # circle is deliberate: it over-covers at the corners, which is the safe
    # direction to err.
    # ------------------------------------------------------------------
    for inc in root.findall(".//include"):
        uri = (inc.findtext("./uri") or "").strip()
        name = (inc.findtext("./name") or uri.rsplit("/", 1)[-1] or "include")
        ix, iy, _iz, iyaw = parse_pose(inc.findtext("./pose"))
        side = 2.0 * INCLUDE_RADIUS
        obstacles.append({
            "model": f"{name} (include)", "x": ix, "y": iy,
            "sx": side, "sy": side, "yaw": iyaw,
        })

    return obstacles


def rasterise(obstacles: list[dict], resolution: float) -> tuple[bytearray, int, int, float, float]:
    if not obstacles:
        raise SystemExit("No obstacles found at laser height - check --laser-height.")

    # World extents from the obstacle footprints (corners, so rotation is handled).
    xs, ys = [], []
    for o in obstacles:
        hx, hy = o["sx"] / 2.0, o["sy"] / 2.0
        for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
            xs.append(o["x"] + dx * c - dy * s)
            ys.append(o["y"] + dx * s + dy * c)

    min_x, max_x = min(xs) - PADDING_M, max(xs) + PADDING_M
    min_y, max_y = min(ys) - PADDING_M, max(ys) + PADDING_M

    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))
    grid = bytearray([254] * (width * height))

    for o in obstacles:
        hx, hy = o["sx"] / 2.0, o["sy"] / 2.0
        c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
        # Bounding box of this (possibly rotated) rectangle, then point-test.
        corners = []
        for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            corners.append((o["x"] + dx * c - dy * s, o["y"] + dx * s + dy * c))
        bx0 = min(p[0] for p in corners); bx1 = max(p[0] for p in corners)
        by0 = min(p[1] for p in corners); by1 = max(p[1] for p in corners)

        px0 = max(0, int((bx0 - min_x) / resolution))
        px1 = min(width, int(math.ceil((bx1 - min_x) / resolution)))
        py0 = max(0, int((by0 - min_y) / resolution))
        py1 = min(height, int(math.ceil((by1 - min_y) / resolution)))

        for py in range(py0, py1):
            wy = min_y + (py + 0.5) * resolution
            for px in range(px0, px1):
                wx = min_x + (px + 0.5) * resolution
                # transform the world point into the box's own frame
                rx = wx - o["x"]
                ry = wy - o["y"]
                lx = rx * c + ry * s
                ly = -rx * s + ry * c
                if abs(lx) <= hx and abs(ly) <= hy:
                    grid[py * width + px] = 0

    return grid, width, height, min_x, min_y


def write_map(grid: bytearray, width: int, height: int,
              origin_x: float, origin_y: float, resolution: float,
              stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)

    # PGM rows are top-to-bottom; the map origin is the BOTTOM-left corner,
    # so the grid must be flipped vertically on write.
    pgm_path = stem.with_suffix(".pgm")
    with pgm_path.open("wb") as handle:
        handle.write(b"P5\n# generated by scripts/world_to_map.py\n")
        handle.write(f"{width} {height}\n255\n".encode())
        for row in range(height - 1, -1, -1):
            handle.write(bytes(grid[row * width:(row + 1) * width]))

    yaml_path = stem.with_suffix(".yaml")
    yaml_path.write_text(
        f"image: {pgm_path.name}\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin_x:.3f}, {origin_y:.3f}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n",
        encoding="utf-8",
    )
    return pgm_path, yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="output path WITHOUT extension, e.g. maps/restaurant_humans")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--laser-height", type=float, default=DEFAULT_LASER_HEIGHT)
    args = parser.parse_args()

    world_path = args.world.expanduser().resolve()
    obstacles = collect_obstacles(world_path, args.laser_height)

    print(f"World: {world_path.name}")
    print(f"Laser height: {args.laser_height} m")
    print(f"Obstacles visible to the laser: {len(obstacles)}")
    for o in obstacles:
        print(f"  {o['model']:34} at ({o['x']:6.2f}, {o['y']:6.2f})  {o['sx']:.2f} x {o['sy']:.2f} m")

    grid, width, height, min_x, min_y = rasterise(obstacles, args.resolution)
    pgm, yaml_path = write_map(grid, width, height, min_x, min_y, args.resolution,
                               args.output.expanduser().resolve())

    occupied = sum(1 for b in grid if b == 0)
    print(f"\nMap: {width} x {height} px @ {args.resolution} m/px "
          f"= {width*args.resolution:.1f} x {height*args.resolution:.1f} m")
    print(f"Origin: [{min_x:.2f}, {min_y:.2f}]   occupied: {occupied/(width*height)*100:.1f}%")
    print(f"Written: {yaml_path}")
    print(f"         {pgm}")


if __name__ == "__main__":
    main()
