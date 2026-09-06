"""
Re-score recorded trials against ground truth, without re-running them.

    python3 scripts/rescore_sim_results.py \
        --results dataset/processed/sim_results \
        --groundtruth src/tiago_social_worlds/worlds/restaurant_testing.groundtruth.json

    # write the corrected verdicts back into the JSON files
    python3 scripts/rescore_sim_results.py ... --apply

WHY THIS EXISTS
---------------
task_success was being decided against `goal_centroid`, which the metrics
recorder overwrites on every /group_centroid message. At the end of a run it
therefore held whatever perception saw LAST - often a false positive against a
wall, or a group on the far side of the room that the robot never approached.
Runs were being judged against the wrong reference point.

The symptom was unmistakable once the trials were laid side by side: one MLP
run came within 0.51 m of a person AND registered an O-space intrusion, while
being scored "never held a pose in the [0.5, 2.0] m band". Both cannot be true
of the same group; the robot had approached a real group and was then measured
against a phantom.

Because every trial stores its full trajectory at 10 Hz, the correct verdict
can be recomputed offline. Nothing needs to be re-run: 30 trials of simulation
time are preserved, and only the arithmetic changes.

WHAT "SUCCESS" MEANS HERE
-------------------------
The robot achieved, at some instant, a pose that is simultaneously
  * between `--min` and `--max` metres from a REAL group centre, and
  * oriented within `--heading` degrees of that group,
for any one of the ground-truth groups. Both conditions must hold in the SAME
sample, so it cannot be satisfied by being at the right distance at one moment
and the right heading at another.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def score(traj, groups, lo, hi, heading_deg):
    if not traj:
        return False, "no trajectory recorded", None
    best = None
    nearest = float('inf')
    for gx, gy in groups:
        for s in traj:
            d = math.hypot(s['x'] - gx, s['y'] - gy)
            nearest = min(nearest, d)
            if not (lo <= d <= hi):
                continue
            want = math.atan2(gy - s['y'], gx - s['x'])
            err = abs(math.atan2(math.sin(want - s['yaw']),
                                 math.cos(want - s['yaw'])))
            if best is None or err < best[0]:
                best = (err, d, s['t'], gx, gy)
    if best is None:
        return False, (f"never held a pose in the [{lo}, {hi}] m band around any "
                       f"real group (closest {nearest:.2f} m)"), None
    err, d, t, gx, gy = best
    if math.degrees(err) > heading_deg:
        return False, (f"in the band at the group near ({gx:.1f}, {gy:.1f}) but "
                       f"never faced it: best heading error "
                       f"{math.degrees(err):.0f} deg"), (d, math.degrees(err), t)
    return True, (f"valid approach pose at the group near ({gx:.1f}, {gy:.1f}): "
                  f"{d:.2f} m, heading {math.degrees(err):.0f} deg off, "
                  f"t={t:.0f}s"), (d, math.degrees(err), t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path,
                    default=Path('dataset/processed/sim_results'))
    ap.add_argument('--groundtruth', type=Path, required=True)
    ap.add_argument('--min', type=float, default=0.5)
    ap.add_argument('--max', type=float, default=2.0)
    ap.add_argument('--heading', type=float, default=45.0)
    ap.add_argument('--min-group-size', type=int, default=1,
                    help="Only count groups of at least this many people as "
                         "valid targets. Use 2 to score CONVERSATIONAL GROUPS "
                         "only: a lone individual has no F-formation and no "
                         "O-space, so approaching one does not test the "
                         "group-approach claim this project is about.")
    ap.add_argument('--apply', action='store_true',
                    help='write corrected verdicts back into the JSON files')
    args = ap.parse_args()

    gt = json.loads(args.groundtruth.read_text())
    all_groups = gt['groups']
    kept = [g for g in all_groups
            if g.get('num_people', 1) >= args.min_group_size]
    groups = [(g['centre_x'], g['centre_y']) for g in kept]

    print(f"Ground truth: {args.groundtruth.name}")
    print(f"  {len(all_groups)} target(s); scoring against the "
          f"{len(groups)} with >= {args.min_group_size} people")
    for g in all_groups:
        mark = ' ' if g.get('num_people', 1) >= args.min_group_size else ' (excluded)'
        print(f"   ({g['centre_x']:6.2f}, {g['centre_y']:6.2f})  "
              f"{g.get('num_people', 1)} people{mark}")
    if not groups:
        raise SystemExit("No groups meet --min-group-size; nothing to score.")

    files = sorted(p for p in args.results.glob('*.json')
                   if p.name != 'summary.csv')
    print(f"\n{len(files)} trial file(s)\n")
    print(f"{'run':30s} {'old':>5s} {'new':>5s}  reason")
    print("-" * 100)

    changed = 0
    tally: dict[str, list[bool]] = {}
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if 'trajectory' not in d:
            continue
        ok, reason, _ = score(d['trajectory'], groups,
                              args.min, args.max, args.heading)
        old = bool(d.get('task_success'))
        tally.setdefault(d.get('policy', '?'), []).append(ok)
        flag = '' if ok == old else '   <-- CHANGED'
        if ok != old:
            changed += 1
        print(f"{f.stem[:30]:30s} {str(old):>5s} {str(ok):>5s}  {reason[:60]}{flag}")

        if args.apply:
            d['task_success'] = ok
            d['task_success_reason'] = reason
            d['rescored_against_groundtruth'] = True
            f.write_text(json.dumps(d, indent=2))

    print("\n" + "=" * 60)
    print(f"{changed} verdict(s) changed")
    for p, vals in sorted(tally.items()):
        print(f"  {p:6s} success {sum(vals)}/{len(vals)}  ({100*sum(vals)/len(vals):.0f}%)")
    if args.apply:
        print("\nFiles updated. Re-run scripts/summarise_sim_results.py.")
    else:
        print("\n(dry run - pass --apply to write the corrected verdicts)")


if __name__ == '__main__':
    main()
