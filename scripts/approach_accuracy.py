"""
APPROACH-POINT ACCURACY  -  did the robot stand where it should have stood?

    # see the ideal approach points for a world, and sanity-check them
    python3 scripts/approach_accuracy.py --show-slots \
        --groundtruth src/tiago_social_worlds/worlds/restaurant_testing.groundtruth.json

    # score a batch of trials
    python3 scripts/approach_accuracy.py \
        --results dataset/processed/results_FINAL_20260824/yolo \
        --groundtruth src/tiago_social_worlds/worlds/restaurant_testing.groundtruth.json

============================================================================
WHY THIS EXISTS
============================================================================
The existing task_success metric is a single per-trial boolean: did the robot
ever hold a valid pose near ANY group. In the final batch it saturated - every
policy scored 100% under YOLOv8n - so it cannot separate the policies. A metric
at its ceiling measures nothing.

This asks the sharper question the project is actually about: for each group in
the room, how CLOSE did the robot get to a socially correct place to stand?

That turns one boolean per trial into one distance per (trial, group), which is
continuous, has no ceiling, and gives 3 groups x 10 trials = 30 observations
per policy instead of 10.

============================================================================
HOW THE IDEAL APPROACH POINTS ARE DERIVED
============================================================================
They are computed from F-formation geometry, not hand-placed, so the same rule
applies to every group and nothing is tuned to flatter a result.

For each group of two or more people:

  1. Take the bearing of every member as seen from the group centre.
  2. Sort them and measure the angular GAP between neighbouring members.
     These gaps are the openings in the formation - the P-space slots a person
     would step into to join the conversation.
  3. Keep every gap wider than --min-gap-deg. A narrow gap is two people
     standing shoulder to shoulder, not a way in.
  4. Place a slot on the bisector of each kept gap, at
         radius = ospace_radius + --approach-offset
     which is just outside the group's shared space.
  5. The slot's heading faces the group centre.

A four or five person group therefore yields three to five slots, matching the
intuition that there are a handful of right places to stand and many wrong
ones. Standing at ANY slot counts as correct - the robot is not required to
guess a particular one.

Sanity check on restaurant_testing group 3 (five people, centre 5.6,-1.8):
the widest formation gap bisects at about +13 deg, putting a slot near
(7.0, -2.0) - which is where a human reading the map would say to stand.

============================================================================
WHAT IS REPORTED
============================================================================
For each (trial, group) pair the whole 10 Hz trajectory is searched for a
sample that satisfies BOTH conditions at the same instant - not for the single
closest sample, which is usually a tangential fly-past across the group's face.

    nearest_slot_error_m  closest the robot ever came to a slot
    within_distance       it was inside --tolerance of a slot at some point
    heading_error_deg     best heading error among the in-tolerance samples
    slot_error_m          distance at that best-heading sample
    reached               within_distance AND heading within --heading-tol

Reporting within_distance separately from reached matters: "got to the right
place but never turned to face the group" and "never got to the right place"
are different failures and should not be collapsed into one number.

Lone individuals are skipped: a single person has no F-formation, no O-space
opening, and therefore no approach slot.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def slots_for_group(group: dict, offset: float, min_gap_deg: float) -> list[dict]:
    """Ideal standing positions in the openings of one F-formation."""
    members = group.get("members", [])
    if len(members) < 2:
        return []

    cx, cy = group["centre_x"], group["centre_y"]
    radius = group.get("ospace_radius", 0.7) + offset

    bearings = sorted(math.atan2(m["y"] - cy, m["x"] - cx) for m in members)

    slots = []
    n = len(bearings)
    for i in range(n):
        a = bearings[i]
        b = bearings[(i + 1) % n]
        gap = (b - a) % (2 * math.pi)          # always the positive sweep a->b
        if math.degrees(gap) < min_gap_deg:
            continue
        mid = a + gap / 2.0
        slots.append({
            "x": cx + radius * math.cos(mid),
            "y": cy + radius * math.sin(mid),
            "bearing_deg": round(math.degrees(wrap(mid)), 1),
            "gap_deg": round(math.degrees(gap), 1),
        })
    return slots


def build_slots(gt_path: str, offset: float, min_gap_deg: float) -> dict:
    gt = json.load(open(gt_path))
    out = {}
    for g in gt["groups"]:
        if g.get("num_people", 1) < 2:
            continue
        s = slots_for_group(g, offset, min_gap_deg)
        if s:
            out[g["group_id"]] = {"group": g, "slots": s}
    return out


def annotate_speed(traj: list[dict]) -> None:
    """Attach a speed estimate to every trajectory sample, in place."""
    for i, s in enumerate(traj):
        if i == 0:
            s["_v"] = 0.0
            continue
        p = traj[i - 1]
        dt = s.get("t", 0.0) - p.get("t", 0.0)
        s["_v"] = math.hypot(s["x"] - p["x"], s["y"] - p["y"]) / dt if dt > 1e-6 else 0.0


def score_trial(result: dict, groups: dict, tol: float, heading_tol: float,
                stationary_speed: float = 0.10) -> list[dict]:
    traj = result.get("trajectory") or []
    annotate_speed(traj)
    rows = []
    for gid, entry in groups.items():
        g, slots = entry["group"], entry["slots"]
        cx, cy = g["centre_x"], g["centre_y"]

        # --------------------------------------------------------------------
        # SEARCH FOR A SAMPLE THAT SATISFIES BOTH, NOT THE CLOSEST SAMPLE
        #
        # The first version of this took the single sample nearest to a slot
        # and then tested its heading. That is the same mistake that was
        # already fixed once in metrics_recorder_node.task_success: the nearest
        # point on a path is very often a tangential fly-past, where the robot
        # is at the right distance while travelling ACROSS the group's face.
        # It scored 0/30 "reached" on a batch whose worst distance error was
        # 0.446 m - i.e. every trial was inside the 0.5 m tolerance and every
        # one was rejected on heading measured at the wrong instant.
        #
        # The question is whether the robot EVER stood in a slot facing the
        # group. So: consider every sample already within tolerance, and take
        # the best heading among them.
        # --------------------------------------------------------------------
        # ONLY SAMPLES WHERE THE ROBOT HAS STOPPED COUNT  (added Aug 2026)
        #
        # The previous version searched every sample. That produced a striking
        # and misleading pattern: 100% of group-visits came within 0.5 m of a
        # slot, but the best heading while there was 48-75 deg off, identically
        # across all six policy/detector conditions. A result that uniform is
        # not behaviour, it is a measurement artefact.
        #
        # The cause is that the slots (ospace_radius + 0.6 m from the centre)
        # and the policy's own target (1.35 m from the nearest PERSON) are not
        # the same place. The robot therefore clips the slot region while
        # DRIVING PAST toward its own goal, and gets scored mid-transit, facing
        # its direction of travel rather than the group.
        #
        # The question is where the robot stood when it stopped to engage the
        # group, so only near-stationary samples are eligible. Movement between
        # samples gives the speed; anything above --stationary-speed is the
        # robot in transit and is excluded from the heading judgement.
        best_valid = None       # (heading_err, dist, slot_idx, t), stationary & in tol
        nearest = None          # (dist, heading_err, slot_idx, t) over ALL samples
        n_stationary_in_band = 0

        for s in traj:
            desired = math.atan2(cy - s["y"], cx - s["x"])
            herr = math.degrees(abs(wrap(desired - s["yaw"])))
            parked = s.get("_v", 0.0) <= stationary_speed
            for k, slot in enumerate(slots):
                d = math.hypot(s["x"] - slot["x"], s["y"] - slot["y"])
                if nearest is None or d < nearest[0]:
                    nearest = (d, herr, k, s.get("t", 0.0))
                if d <= tol and parked:
                    n_stationary_in_band += 1
                    if best_valid is None or herr < best_valid[0]:
                        best_valid = (herr, d, k, s.get("t", 0.0))

        if nearest is None:
            continue

        if best_valid is not None:
            herr, d, k, t = best_valid
            in_band = True
        else:
            d, herr, k, t = nearest
            in_band = False

        rows.append({
            "policy": result.get("policy", "?"),
            "group_id": gid,
            "num_people": g.get("num_people"),
            "slot_error_m": round(d, 3),
            "nearest_slot_error_m": round(nearest[0], 3),
            "heading_error_deg": round(herr, 1),
            "slot_index": k,
            "t_s": round(t, 1),
            "within_distance": bool(in_band),
            "stationary_samples_in_band": n_stationary_in_band,
            "dwell_seconds_in_band": round(n_stationary_in_band * 0.1, 1),
            "reached": bool(in_band and herr <= heading_tol),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", help="directory of trial *.json files")
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--approach-offset", type=float, default=0.6,
                   help="metres beyond the O-space edge (default 0.6)")
    p.add_argument("--min-gap-deg", type=float, default=45.0)
    p.add_argument("--tolerance", type=float, default=0.5,
                   help="metres from a slot that counts as reaching it")
    p.add_argument("--heading-tol", type=float, default=45.0)
    p.add_argument("--stationary-speed", type=float, default=0.10,
                   help="m/s below which the robot counts as stopped. Only "
                        "stationary samples are judged on heading, because a "
                        "robot driving past a slot is not standing in it.")
    p.add_argument("--show-slots", action="store_true",
                   help="print the computed approach points and exit")
    p.add_argument("--csv", help="write per-(trial,group) rows here")
    args = p.parse_args()

    groups = build_slots(args.groundtruth, args.approach_offset, args.min_gap_deg)

    if args.show_slots or not args.results:
        print("=" * 72)
        print(f"  IDEAL APPROACH POINTS   offset={args.approach_offset} m "
              f"beyond O-space, min gap {args.min_gap_deg:.0f} deg")
        print("=" * 72)
        for gid, e in groups.items():
            g = e["group"]
            print(f"\n  Group {gid}: {g['num_people']} people at "
                  f"({g['centre_x']:.2f}, {g['centre_y']:.2f}), "
                  f"O-space r={g.get('ospace_radius'):.3f} m")
            for i, s in enumerate(e["slots"]):
                print(f"      slot {i}: ({s['x']:6.2f}, {s['y']:6.2f})   "
                      f"bearing {s['bearing_deg']:>6.1f} deg   "
                      f"gap {s['gap_deg']:>5.1f} deg")
        print("\n" + "=" * 72)
        if args.show_slots:
            return

    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    if not files:
        raise SystemExit(f"no trial JSONs in {args.results}")

    all_rows = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {os.path.basename(f)}: {exc}")
            continue
        all_rows.extend(score_trial(r, groups, args.tolerance, args.heading_tol,
                                    args.stationary_speed))

    by_policy = defaultdict(list)
    for r in all_rows:
        by_policy[r["policy"]].append(r)

    print("\n" + "=" * 78)
    print(f"  APPROACH-POINT ACCURACY   {args.results}")
    print(f"  {len(files)} trial(s), {len(groups)} conversational group(s), "
          f"tolerance {args.tolerance} m / {args.heading_tol:.0f} deg")
    print("=" * 78)

    tol_note = f"{args.tolerance:g} m"
    for policy in sorted(by_policy):
        rows = by_policy[policy]
        n = len(rows)
        reached = sum(r["reached"] for r in rows)
        in_band = sum(r["within_distance"] for r in rows)
        errs = sorted(r["nearest_slot_error_m"] for r in rows)
        head = sorted(r["heading_error_deg"] for r in rows if r["within_distance"])
        mean = sum(errs) / n
        median = errs[n // 2]
        print(f"\n--- {policy}  ({n} group-visit(s)) "
              + "-" * max(0, 40 - len(policy)))
        dwelled = sum(1 for r in rows if r["stationary_samples_in_band"] > 0)
        print(f"  came within {tol_note} of a slot : {sum(r['nearest_slot_error_m'] <= args.tolerance for r in rows)}/{n}")
        print(f"  STOPPED within {tol_note}        : {dwelled}/{n}  ({100 * dwelled / n:.0f}%)")
        print(f"  ...AND facing the group : {reached}/{n}  ({100 * reached / n:.0f}%)")
        if head:
            print(f"  best heading in band    : mean {sum(head) / len(head):.1f} deg, "
                  f"median {head[len(head) // 2]:.1f} deg")
        print(f"  slot error, mean        : {mean:.3f} m")
        print(f"  slot error, median      : {median:.3f} m")
        print(f"  slot error, best        : {errs[0]:.3f} m")
        print(f"  slot error, worst       : {errs[-1]:.3f} m")
        per_group = defaultdict(list)
        for r in rows:
            per_group[r["group_id"]].append(r)
        for gid in sorted(per_group):
            g = per_group[gid]
            gr = sum(x["reached"] for x in g)
            gm = sum(x["nearest_slot_error_m"] for x in g) / len(g)
            npeople = g[0]["num_people"]
            print(f"      group {gid} ({npeople} people): "
                  f"reached {gr}/{len(g)}, mean error {gm:.3f} m")

    print("\n" + "=" * 78)
    print(f"  {'policy':<12} {'reached':>12} {'mean err':>10} {'median err':>12}")
    print("  " + "-" * 50)
    for policy in sorted(by_policy):
        rows = by_policy[policy]
        n = len(rows)
        reached = sum(r["reached"] for r in rows)
        errs = sorted(r["nearest_slot_error_m"] for r in rows)
        print(f"  {policy:<12} {reached:>4}/{n:<3} {100 * reached / n:>4.0f}% "
              f"{sum(errs) / n:>9.3f}m {errs[n // 2]:>11.3f}m")
    print("=" * 78)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"Written: {args.csv}")


if __name__ == "__main__":
    main()
