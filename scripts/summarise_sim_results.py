#!/usr/bin/env python3
"""
Aggregate simulation runs into the Objective 4 comparison table.

Each run of metrics_recorder_node writes one JSON file. This collects them,
groups by policy, and produces the rule-based vs Behavioural Cloning table
your dissertation's results chapter needs - plus a CSV for plotting.

WHAT IT REPORTS, AND WHY EACH ONE
----------------------------------
  task success rate        headline: did the robot achieve a socially valid
                           approach pose at all?
  collision-free rate      basic safety; a policy that succeeds by driving
                           through people is not a success
  O-space intrusion rate   the core SOCIAL metric - entering a group's shared
                           conversational space is the thing this whole project
                           is trying to avoid
  min distance to person   proxemic comfort; reported as a mean over runs
  cut-through events       walking between two people mid-conversation, the
                           most disruptive failure mode
  path length / time       efficiency. Included because a policy can trivially
                           avoid all social violations by taking absurd detours,
                           and that trade-off must be visible.

Rates are reported with run counts (e.g. "8/10") rather than bare percentages,
because with a handful of runs "80%" hides how little evidence sits behind it.

Usage:
    python3 scripts/summarise_sim_results.py
    python3 scripts/summarise_sim_results.py --results-dir path/to/sim_results
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "dataset" / "processed" / "sim_results"


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_file"] = path.name
            runs.append(data)
        except json.JSONDecodeError:
            print(f"  (skipping unreadable file: {path.name})")
    return runs


def mean_or_none(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def fmt(value, unit: str = "", places: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}{unit}"


def summarise(policy: str, runs: list[dict]) -> dict:
    n = len(runs)
    successes = sum(1 for r in runs if r.get("task_success"))
    collision_free = sum(1 for r in runs if r.get("collision_free"))
    intrusions = sum(1 for r in runs if r.get("ospace_intrusion"))
    cut_through_runs = sum(1 for r in runs if (r.get("group_cut_through_events") or 0) > 0)

    return {
        "policy": policy,
        "runs": n,
        "task_success": successes,
        "task_success_rate": successes / n if n else 0.0,
        "collision_free": collision_free,
        "collision_free_rate": collision_free / n if n else 0.0,
        "ospace_intrusion": intrusions,
        "ospace_intrusion_rate": intrusions / n if n else 0.0,
        "cut_through_runs": cut_through_runs,
        "cut_through_rate": cut_through_runs / n if n else 0.0,
        "mean_cut_through_events": mean_or_none([r.get("group_cut_through_events") for r in runs]),
        "mean_min_distance_m": mean_or_none([r.get("min_distance_to_person_m") for r in runs]),
        "mean_path_length_m": mean_or_none([r.get("path_length_m") for r in runs]),
        "mean_navigation_time_s": mean_or_none([r.get("navigation_time_s") for r in runs]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        print(f"No results directory at {results_dir}")
        print("Run some trials first:")
        print("  ros2 launch tiago_group_approach group_approach.launch.py policy:=rule")
        return

    runs = load_runs(results_dir)
    if not runs:
        print(f"No run files found in {results_dir}")
        return

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_policy[r.get("policy", "unknown")].append(r)

    summaries = [summarise(p, rs) for p, rs in sorted(by_policy.items())]

    print("=" * 78)
    print("OBJECTIVE 4 - SIMULATION COMPARISON")
    print("=" * 78)
    print(f"{len(runs)} run(s) across {len(summaries)} policy/policies\n")

    for s in summaries:
        n = s["runs"]
        print(f"--- {s['policy']}  ({n} run(s)) " + "-" * max(0, 48 - len(s['policy'])))
        print(f"  task success       : {s['task_success']}/{n}   "
              f"({s['task_success_rate']*100:.0f}%)")
        print(f"  collision free     : {s['collision_free']}/{n}   "
              f"({s['collision_free_rate']*100:.0f}%)")
        print(f"  O-space intrusion  : {s['ospace_intrusion']}/{n}   "
              f"({s['ospace_intrusion_rate']*100:.0f}%)   [lower is better]")
        print(f"  cut-through runs   : {s['cut_through_runs']}/{n}   "
              f"({s['cut_through_rate']*100:.0f}%)   [lower is better]")
        print(f"  min dist to person : {fmt(s['mean_min_distance_m'], ' m')}   [mean over runs]")
        print(f"  path length        : {fmt(s['mean_path_length_m'], ' m')}")
        print(f"  navigation time    : {fmt(s['mean_navigation_time_s'], ' s')}")
        print()

    if len(summaries) >= 2:
        print("=" * 78)
        print("HEAD-TO-HEAD")
        print("=" * 78)
        header = f"{'metric':<26}" + "".join(f"{s['policy']:>16}" for s in summaries)
        print(header)
        print("-" * len(header))
        rows = [
            ("task success rate", "task_success_rate", "%", True),
            ("collision-free rate", "collision_free_rate", "%", True),
            ("O-space intrusion rate", "ospace_intrusion_rate", "%", False),
            ("cut-through rate", "cut_through_rate", "%", False),
            ("min dist to person (m)", "mean_min_distance_m", "", True),
            ("path length (m)", "mean_path_length_m", "", False),
            ("navigation time (s)", "mean_navigation_time_s", "", False),
        ]
        for label, key, unit, higher_better in rows:
            cells = ""
            for s in summaries:
                v = s[key]
                if v is None:
                    cells += f"{'n/a':>16}"
                elif unit == "%":
                    cells += f"{v*100:>15.0f}%"
                else:
                    cells += f"{v:>16.2f}"
            arrow = "^" if higher_better else "v"
            print(f"{label:<26}{cells}   ({arrow} better)")

        print("\nNote: with a small number of runs these rates carry wide")
        print("uncertainty. Report the run counts alongside them, and avoid")
        print("claiming a difference that rests on one or two trials.")

    csv_path = results_dir / "summary.csv"
    if summaries:
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nWritten: {csv_path}")


if __name__ == "__main__":
    main()
