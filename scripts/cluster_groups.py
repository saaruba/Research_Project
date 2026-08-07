#!/usr/bin/env python3
"""
Phase C, step 1: cluster individually-detected people (from
detected_people_individual.csv) into candidate GROUPS, per frame.

Why a distance threshold isn't enough on its own: we only have 2D pixel
positions from the camera image, not real-world (x, y) coordinates - there's
no depth/calibration step in this project. Two people standing far apart in
the room could still land close together in pixels if they're both far from
the camera; two people standing close together could land far apart in
pixels if one is much closer to the camera than the other. To partly correct
for this, distance between two people is measured in units of their own
apparent size (average bounding-box width), not raw pixels - since a
person's bbox width shrinks the further they are from the camera, this
roughly cancels out perspective distortion. This is a documented
simplification, not true 3D localisation - it should be written up as a
limitation, but it is more defensible than raw pixel distance alone.

Clustering rule: two people are linked into the same group if their pixel
distance is less than GROUP_DISTANCE_THRESHOLD times their average bbox
width. This produces a similarity graph per frame; connected components of
that graph become groups. Groups of size 1 are lone individuals (not a
"group" in the F-formation sense, but kept in the output so downstream code
can still see "1 person here, no group").

Output per session: dataset/<session>/detected_groups.csv, one row per
GROUP per sampled frame, with columns:
    timestamp, group_id, num_people, member_indices,
    group_center_x, group_center_y,      (centroid of member centers)
    group_bbox_min_x/y, group_bbox_max_x/y,  (spatial extent of the group)
    avg_bbox_width,                       (scale proxy - bigger = closer)
    is_largest_group                      (True for the biggest group in
                                            that frame - a first-pass guess
                                            at "the group the robot should
                                            approach", used until a proper
                                            F-formation/O-space step exists)

Usage - one session:
    python3 scripts/cluster_groups.py --session dataset/9

Usage - all sessions:
    for s in 1 3 5 7 8 9 10 11 12 14 15 26 27 28 30 31 49 51 52 54 55 58 59 60; do
        echo "=== session $s ==="
        python3 scripts/cluster_groups.py --session dataset/$s
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_DISTANCE_THRESHOLD = 2.5  # in units of average bbox width


def find_connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """Simple connected-components search over a boolean adjacency matrix."""
    n = adjacency.shape[0]
    visited = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = np.where(adjacency[node])[0]
            for neighbour in neighbours:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        components.append(sorted(component))

    return components


def cluster_frame(frame_df: pd.DataFrame, distance_threshold: float = GROUP_DISTANCE_THRESHOLD) -> list[dict]:
    """Cluster the people detected in a single frame into groups."""
    n = len(frame_df)
    centers = frame_df[["center_x", "center_y"]].to_numpy()
    widths = (frame_df["bbox_max_x"] - frame_df["bbox_min_x"]).to_numpy()

    if n == 1:
        components = [[0]]
    else:
        adjacency = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                pixel_dist = np.linalg.norm(centers[i] - centers[j])
                avg_width = (widths[i] + widths[j]) / 2.0
                normalised_dist = pixel_dist / avg_width if avg_width > 0 else np.inf
                if normalised_dist < distance_threshold:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
        components = find_connected_components(adjacency)

    group_rows = []
    group_sizes = [len(c) for c in components]
    largest_size = max(group_sizes) if group_sizes else 0
    largest_seen = False  # only flag the first largest group if there's a tie

    for group_id, member_idx in enumerate(components):
        member_rows = frame_df.iloc[member_idx]
        member_centers = member_rows[["center_x", "center_y"]].to_numpy()
        member_widths = (member_rows["bbox_max_x"] - member_rows["bbox_min_x"]).to_numpy()

        is_largest = len(member_idx) == largest_size and not largest_seen
        if is_largest:
            largest_seen = True

        group_rows.append({
            "timestamp": frame_df["timestamp"].iloc[0],
            "group_id": group_id,
            "num_people": len(member_idx),
            "member_indices": ";".join(str(frame_df.iloc[i]["person_index"]) for i in member_idx),
            "group_center_x": float(member_centers[:, 0].mean()),
            "group_center_y": float(member_centers[:, 1].mean()),
            "group_bbox_min_x": float(member_rows["bbox_min_x"].min()),
            "group_bbox_min_y": float(member_rows["bbox_min_y"].min()),
            "group_bbox_max_x": float(member_rows["bbox_max_x"].max()),
            "group_bbox_max_y": float(member_rows["bbox_max_y"].max()),
            "avg_bbox_width": float(member_widths.mean()),
            "is_largest_group": bool(is_largest),
        })

    return group_rows


def cluster_session(session_path: Path, distance_threshold: float = GROUP_DISTANCE_THRESHOLD) -> pd.DataFrame:
    individual_path = session_path / "detected_people_individual.csv"
    if not individual_path.exists():
        raise FileNotFoundError(f"{individual_path} not found - run extract_person_detections.py first")

    df = pd.read_csv(individual_path)
    if df.empty:
        return pd.DataFrame()

    all_group_rows: list[dict] = []
    for _, frame_df in df.groupby("timestamp", sort=True):
        all_group_rows.extend(cluster_frame(frame_df.reset_index(drop=True), distance_threshold))

    return pd.DataFrame(all_group_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=GROUP_DISTANCE_THRESHOLD,
                         help="distance threshold in units of average bbox width (default: 2.5)")
    args = parser.parse_args()

    session_path = args.session.expanduser().resolve()
    print(f"Session: {session_path.name}")

    groups_df = cluster_session(session_path, args.threshold)
    if groups_df.empty:
        print("  no person detections found - nothing to cluster")
        return

    output_path = session_path / "detected_groups.csv"
    groups_df.to_csv(output_path, index=False)

    n_frames = groups_df["timestamp"].nunique()
    size_counts = groups_df["num_people"].value_counts().sort_index()
    multi_person_groups = (groups_df["num_people"] > 1).sum()

    print(f"  frames with detections: {n_frames}")
    print(f"  total groups (incl. lone individuals): {len(groups_df)}")
    print(f"  group size breakdown (num_people -> count of groups): {size_counts.to_dict()}")
    print(f"  groups with 2+ people: {multi_person_groups}")
    print(f"  written to: {output_path}\n")


if __name__ == "__main__":
    main()
