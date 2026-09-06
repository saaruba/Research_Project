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


# ---------------------------------------------------------------------------
# APPROACH-POINT ACCURACY  (added Aug 2026)
#
# The binary metrics above saturated in the final batch - every policy scored
# 100% task success under YOLOv8n - so they cannot separate the policies. This
# section adds the two numbers the project is really about: how close, in
# metres, did the robot get to a socially correct place to stand, and how far,
# in degrees, was it from facing the group when it got there.
#
# The slot geometry and scoring live in approach_accuracy.py so there is one
# implementation, not two that can drift apart.
# ---------------------------------------------------------------------------
def _load_approach_module():
    import importlib.util
    path = Path(__file__).resolve().parent / "approach_accuracy.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("approach_accuracy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def approach_accuracy_section(runs: list[dict], args) -> list[dict]:
    """Print position and orientation accuracy per policy. Returns the rows."""
    gt = Path(args.groundtruth).expanduser()
    if not gt.exists():
        print(f"\n(no ground truth at {gt} - skipping approach accuracy)")
        print("  pass --groundtruth <world>.groundtruth.json to enable it")
        return []

    aa = _load_approach_module()
    if aa is None:
        print("\n(scripts/approach_accuracy.py not found - skipping approach accuracy)")
        return []

    groups = aa.build_slots(str(gt), args.approach_offset, args.min_gap_deg)
    if not groups:
        print("\n(no conversational groups in the ground truth - "
              "approach accuracy needs groups of 2+)")
        return []

    rows = []
    for r in runs:
        rows.extend(aa.score_trial(r, groups, args.tolerance,
                                   args.heading_tol, args.stationary_speed))
    if not rows:
        return []

    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    n_slots = sum(len(e["slots"]) for e in groups.values())
    print()
    print("=" * 78)
    print("APPROACH-POINT ACCURACY - position and orientation")
    print("=" * 78)
    print(f"{len(groups)} conversational group(s), {n_slots} ideal standing slot(s)")
    print(f"slots sit {args.approach_offset:g} m outside the O-space, on the bisector")
    print(f"of every formation gap wider than {args.min_gap_deg:g} deg")
    print()
    print(f"{'policy':<10}{'POSITION err (m)':>22}{'ORIENTATION err (deg)':>26}")
    print(f"{'':<10}{'mean':>11}{'median':>11}{'mean':>13}{'median':>13}")
    print("-" * 78)

    for policy in sorted(by_policy):
        pr = by_policy[policy]
        pos = sorted(r["nearest_slot_error_m"] for r in pr)
        head = sorted(r["heading_error_deg"] for r in pr if r["within_distance"])
        pm = statistics.fmean(pos)
        pmed = pos[len(pos) // 2]
        if head:
            hm, hmed = statistics.fmean(head), head[len(head) // 2]
            print(f"{policy:<10}{pm:>11.3f}{pmed:>11.3f}{hm:>13.1f}{hmed:>13.1f}")
        else:
            print(f"{policy:<10}{pm:>11.3f}{pmed:>11.3f}{'n/a':>13}{'n/a':>13}")

    print()
    print("  (lower is better in every column)")
    print(f"  {len(rows)} group-visit(s) scored, "
          f"{len(rows) // max(1, len(by_policy))} per policy")

    print()
    print("Per group, mean position error (m):")
    gids = sorted({r["group_id"] for r in rows})
    header = "  " + f"{'policy':<10}" + "".join(
        f"{'grp ' + str(g):>12}" for g in gids)
    print(header)
    for policy in sorted(by_policy):
        line = f"  {policy:<10}"
        for g in gids:
            vals = [r["nearest_slot_error_m"] for r in by_policy[policy]
                    if r["group_id"] == g]
            line += f"{statistics.fmean(vals):>12.3f}" if vals else f"{'n/a':>12}"
        print(line)

    sizes = {g: groups[g]["group"].get("num_people") for g in gids}
    print("  " + " " * 10 + "".join(f"{'(' + str(sizes[g]) + 'p)':>12}" for g in gids))
    return rows


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
    # --- approach-point accuracy (added Aug 2026) ---------------------------
    parser.add_argument(
        "--groundtruth", default=str(
            PROJECT_ROOT / "src" / "tiago_social_worlds" / "worlds"
            / "restaurant_testing.groundtruth.json"),
        help="world ground truth; supplies the person positions the ideal "
             "approach slots are derived from")
    parser.add_argument("--approach-offset", type=float, default=0.6,
                        help="metres outside the O-space to place a slot")
    parser.add_argument("--min-gap-deg", type=float, default=45.0,
                        help="narrowest formation gap that counts as an opening")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="metres from a slot that counts as being there")
    parser.add_argument("--heading-tol", type=float, default=45.0)
    parser.add_argument("--stationary-speed", type=float, default=0.10,
                        help="m/s below which the robot counts as stopped")
    parser.add_argument("--no-approach-accuracy", action="store_true",
                        help="skip the position/orientation accuracy section")
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

    # --- position and orientation accuracy ----------------------------------
    approach_rows = []
    if not args.no_approach_accuracy:
        approach_rows = approach_accuracy_section(runs, args)
    if approach_rows:
        import csv as _csv
        aa_path = results_dir / "approach_accuracy.csv"
        with aa_path.open("w", newline="", encoding="utf-8") as handle:
            w = _csv.DictWriter(handle, fieldnames=list(approach_rows[0]))
            w.writeheader()
            w.writerows(approach_rows)
        print(f"\nWritten: {aa_path}")

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
