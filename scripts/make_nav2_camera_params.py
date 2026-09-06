"""
Build a Nav2 parameter file that lets the robot see obstacles with its CAMERA.

    python3 scripts/make_nav2_camera_params.py
    python3 scripts/make_nav2_camera_params.py --base /path/to/nav2_params.yaml

THE PROBLEM THIS SOLVES
-----------------------
TIAGo's base laser scans at roughly 0.2 m above the floor. A dining table has
thin legs down there and a wide top at ~0.75 m, so the scanner sees almost
nothing of it: the costmap stays empty where the tabletop actually is, and the
local controller drives the robot's upper body straight into it. Observed
repeatedly - the robot clipping tables while Nav2 reported clear space.

The depth camera looks forward and slightly down and does see the tabletop. The
fix is therefore not smarter planning but a second observation source: convert
the depth point cloud into a laser scan at a height that includes table
surfaces and torsos, and register that scan with both costmaps alongside the
real laser.

WHY A GENERATED FILE RATHER THAN A HAND-WRITTEN ONE
---------------------------------------------------
Nav2's parameter file is long, version-specific, and PAL layers its own values
on top. Writing one from scratch invites subtle mismatches with the installed
version. This script instead PATCHES whatever base file is present, changing
only the observation-source entries and leaving every other tuning value alone.

WHAT IT CHANGES
    local_costmap  obstacle_layer.observation_sources : scan  ->  scan depth
    global_costmap obstacle_layer.observation_sources : scan  ->  scan depth
    (+ a `depth` block for each, with a height band that captures tabletops)

Nothing else is touched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is required:  pip install pyyaml --break-system-packages")

DEFAULT_BASE = Path('/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml')
OUT = Path('config/nav2_camera_obstacles.yaml')

# Height band, in the robot base frame, that the depth-derived scan represents.
# 0.25 m clears the floor; 1.60 m includes tabletops (~0.75 m) and standing
# torsos without picking up the ceiling.
MIN_HEIGHT = 0.25
MAX_HEIGHT = 1.60

DEPTH_SOURCE = {
    'topic': '/scan_depth',
    'data_type': 'LaserScan',
    'max_obstacle_height': MAX_HEIGHT,
    'min_obstacle_height': MIN_HEIGHT,
    'clearing': True,
    'marking': True,
    'obstacle_max_range': 4.0,      # depth is noisy far out
    'obstacle_min_range': 0.3,
    'raytrace_max_range': 4.5,
    'raytrace_min_range': 0.0,
    'inf_is_valid': False,
}


def patch_costmap(node: dict, name: str) -> bool:
    """Add the depth source to one costmap's obstacle layer."""
    params = node.get('ros__parameters')
    if not isinstance(params, dict):
        return False
    layer = params.get('obstacle_layer')
    if not isinstance(layer, dict):
        print(f"    {name}: no obstacle_layer - skipped")
        return False

    sources = str(layer.get('observation_sources', 'scan')).split()
    if 'depth' in sources:
        print(f"    {name}: already has the depth source")
        return False
    sources.append('depth')
    layer['observation_sources'] = ' '.join(sources)
    layer['depth'] = dict(DEPTH_SOURCE)
    print(f"    {name}: observation_sources -> '{layer['observation_sources']}'")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()

    if not args.base.exists():
        raise SystemExit(
            f"Base parameter file not found: {args.base}\n"
            "Point --base at your installed nav2_params.yaml, e.g.\n"
            "  find /opt/ros/humble -name 'nav2_params.yaml'")

    print(f"Base : {args.base}")
    data = yaml.safe_load(args.base.read_text())

    changed = 0
    for key in ('local_costmap', 'global_costmap'):
        node = data.get(key)
        if node is None:
            print(f"    {key}: not present in the base file")
            continue
        # Nav2 nests these as local_costmap: { local_costmap: {ros__parameters}}
        inner = node.get(key, node)
        if patch_costmap(inner, key):
            changed += 1

    if not changed:
        print("\nNothing changed - the base file may already be patched.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        shutil.copy(args.out, args.out.with_suffix('.yaml.bak'))
    args.out.write_text(yaml.safe_dump(data, sort_keys=False, width=120))
    print(f"\nWritten: {args.out}")
    print("\nThe robot will now treat anything the depth camera sees between")
    print(f"{MIN_HEIGHT} m and {MAX_HEIGHT} m as an obstacle - tabletops included.")


if __name__ == '__main__':
    main()
