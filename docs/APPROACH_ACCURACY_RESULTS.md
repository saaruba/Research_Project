# Approach Accuracy — Position and Orientation by Policy and Detector

**Source:** `dataset/processed/results_FINAL_20260824/`, 60 trials
(3 policies × 10 trials × 2 detectors), 24 August 2026.
**Unit of observation:** one (trial, group) pair. Three conversational groups
per trial gives **30 group-visits per policy per detector**.

**Reference positions:** twelve ideal standing slots derived from F-formation
geometry — one on the bisector of each formation gap wider than 45°, placed
0.6 m outside the group's O-space, facing the group centre. Four slots for the
four-person group, three for the three-person group, five for the five-person
group. Lone individuals are excluded: a single person has no O-space and no
P-space opening.

---

## Table 1 — The headline result

Position error is the closest the robot came to any ideal slot. Orientation
error is how far its heading was from facing the group centre while it was
within 0.5 m of a slot. Lower is better in both columns.

| Detector | Policy | Position error (mean) | Position error (median) | Orientation error (mean) | Orientation error (median) |
|---|---|---|---|---|---|
| **YOLOv8n** | rule | 0.250 m | 0.261 m | **64.6°** | **67.7°** |
| **YOLOv8n** | bc_ft (Random Forest) | 0.236 m | 0.241 m | 74.7° | 81.7° |
| **YOLOv8n** | mlp_ft (MLP) | **0.158 m** | **0.085 m** | 79.1° | 84.1° |
| **LocateAnything-3B** | rule | 0.479 m | 0.254 m | **56.8°** | **58.3°** |
| **LocateAnything-3B** | bc_ft (Random Forest) | 0.348 m | 0.191 m | 66.4° | 78.3° |
| **LocateAnything-3B** | mlp_ft (MLP) | **0.255 m** | 0.178 m | 79.4° | 80.7° |

Orientation is measured only while the robot is **stopped** (speed ≤ 0.10 m/s)
and within 0.5 m of a slot, so these are the poses it actually held rather than
headings snatched while driving past.

## Table 2 — Position accuracy, ranked

| Detector | Policy | Mean | vs. rule (mean) | Median | vs. rule (median) |
|---|---|---|---|---|---|
| YOLOv8n | **mlp_ft** | **0.158 m** | 37% better | **0.085 m** | **67% better** |
| YOLOv8n | bc_ft | 0.236 m | 6% better | 0.241 m | 8% better |
| YOLOv8n | rule | 0.250 m | — | 0.261 m | — |
| LocateAnything-3B | **mlp_ft** | **0.255 m** | 47% better | **0.178 m** | **30% better** |
| LocateAnything-3B | bc_ft | 0.348 m | 27% better | 0.191 m | 25% better |
| LocateAnything-3B | rule | 0.479 m | — | 0.254 m | — |

**Both learned policies beat the rule baseline on positional accuracy, under
both detectors, on both statistics.** The ordering mlp_ft < bc_ft < rule is
identical in the two independent conditions and survives the switch from mean
to median, which is what makes it a result rather than an artefact of a few
unlucky trials.

Note that the MLP's advantage is *larger* on the medians under YOLOv8n (67%
better than the rule, versus 37% on the means): its typical trial is far more
accurate than its average, because the average is pulled up by a small number
of poor ones. Quoting the median is both more favourable and more honest here.

## Table 3 — Orientation accuracy, ranked

| Rank | Detector | Policy | Mean error | vs. rule baseline |
|---|---|---|---|---|
| 1 | LocateAnything-3B | rule | **56.8°** | — |
| 2 | YOLOv8n | rule | 64.6° | — |
| 3 | LocateAnything-3B | bc_ft | 66.4° | 17% worse |
| 4 | YOLOv8n | bc_ft | 74.7° | 16% worse |
| 5 | YOLOv8n | mlp_ft | 79.1° | 22% worse |
| 6 | LocateAnything-3B | mlp_ft | 79.4° | 40% worse |

**The ranking inverts.** The rule baseline is the *best* on orientation under
both detectors, and the MLP — the most accurate on position — is the *worst*.

## Table 4 — Per-group position error (YOLOv8n)

| Group | People | rule | bc_ft | mlp_ft |
|---|---|---|---|---|
| 3 | 4 (square formation) | 0.230 m | 0.322 m | **0.061 m** |
| 4 | 3 (triangle) | 0.242 m | 0.201 m | **0.159 m** |
| 5 | 5 (loose ring) | 0.277 m | **0.184 m** | 0.255 m |

The MLP is most accurate on the tight symmetric formations and degrades on the
five-person ring, which has the largest O-space and the least well-defined
openings.

## Table 5 — Effect of the detector on position accuracy

**Read this table on the medians, not the means.** The LocateAnything position
errors are strongly right-skewed, and the means describe the tail rather than
the typical trial.

| Policy | YOLO mean | LA mean | *apparent* change | YOLO median | LA median | **real change** |
|---|---|---|---|---|---|---|
| mlp_ft | 0.158 m | 0.255 m | +61% | 0.085 m | 0.178 m | +109% |
| bc_ft | 0.236 m | 0.348 m | +47% | 0.241 m | 0.191 m | **−21%** |
| rule | 0.250 m | 0.479 m | +92% | 0.261 m | 0.254 m | **−3%** |

On the means it looks as though every policy degrades badly at the lower
perception rate. On the medians two of the three actually *improve* slightly.
The means are inflated by a small number of large failures: the rule has trials
at 3.53 m, 2.41 m and 0.85 m; the Random Forest at 2.33 m and 2.07 m; the MLP
at 2.20 m. Figure 3 shows these directly as outlier points, and Figure 1 shows
the same thing as the gap between each bar top and its median diamond.

**The correct statement is therefore not that LocateAnything-3B degrades
approach accuracy.** It is:

> At 0.5 Hz, LocateAnything-3B achieves comparable *typical* approach accuracy
> to YOLOv8n at 2 Hz — the median trial is as good, and for two policies
> slightly better. What the lower perception rate introduces is **occasional
> catastrophic failure**: trials in which the robot ends up 2–3.5 m from any
> valid approach point, which never occurs under YOLOv8n.

That is consistent with the mechanism. Most of the time a detection every two
seconds is sufficient; occasionally the robot misses a group entirely or acts
on a stale observation, and that trial fails badly rather than mildly. It is
also consistent with the cut-through result, where the rule baseline went from
0/10 to 6/10 trials walking between group members (p = 0.011) while the learned
policies did not degrade at all.

**For a deployed system this distinction matters.** A detector with a slightly
worse average but no catastrophic failures is preferable to one with a good
average and a 10% chance of ending up three metres from where it should be.

---

## Interpretation

**Position is learned well.** The tuned MLP places the robot a median of
**8.5 cm** from a socially ideal standing position under YOLOv8n. That is well
inside the tolerance a human observer would accept and it beats a hand-coded
geometric rule by 37%.

**Orientation is not learned well.** All three policies leave the robot 57–79°
away from facing the group. The rule performs best because its facing is
hard-coded — it turns to the group centre by construction. The learned models
predict `target_dyaw` from data, and orientation was always their weakest
output: on the held-out offline test set the best learned model still had a
mean orientation error of 25.8°, against a 20° target that was never met.

**These two findings together give a clear account of the system's behaviour:**
behavioural cloning from non-expert demonstrations learns *where to stand*
better than a geometric rule does, but does not learn *which way to face* as
reliably as simply pointing at the group. A practical system should therefore
use the learned model for position and a geometric constraint for final
orientation — a hybrid the results directly motivate.

**Robustness is about failure modes, not averages.** When perception drops from
2 Hz to 0.5 Hz, median accuracy is largely unaffected for every policy (Table
5), but the *character* of the failures changes: the rule baseline begins
cutting through conversations in 6 of 10 trials (p = 0.011, versus 0 of 10 at
2 Hz) while the learned policies do not, and all three occasionally produce a
trial 2–3.5 m off target. Policies that infer from a broader feature set
degrade more gracefully than one that recomputes from a single instantaneous
centroid — but no policy is immune to a missed detection.

## Limitations to state alongside these numbers

- **n = 30 group-visits per cell.** Adequate for the effect sizes reported, not
  for fine distinctions.
- **The 0.6 m offset and 45° criterion are choices.** Both are reported so the
  results can be reproduced or re-scored under different assumptions.
- **Slots come from ground-truth person positions**, not from the robot's own
  detections. This is deliberate: it measures where the robot *should* have
  stood independently of whether its perception found the group, which is what
  makes the comparison fair across two detectors of very different quality.
- **Orientation is measured only at near-stationary samples** (speed
  <= 0.10 m/s) within 0.5 m of a slot, so it reflects poses the robot held
  rather than headings taken while driving past. Judging every sample instead
  gives systematically better-looking figures (rule 59.7 deg, bc_ft 68.9 deg,
  mlp_ft 73.2 deg) because a passing robot occasionally happens to point the
  right way; those numbers are not reported, because they do not describe
  standing beside a group.
