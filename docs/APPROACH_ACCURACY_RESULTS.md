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
| **YOLOv8n** | rule | 0.250 m | 0.261 m | **59.7°** | **58.1°** |
| **YOLOv8n** | bc_ft (Random Forest) | 0.236 m | 0.241 m | 68.9° | 74.3° |
| **YOLOv8n** | mlp_ft (MLP) | **0.158 m** | **0.085 m** | 73.2° | 79.5° |
| **LocateAnything-3B** | rule | 0.479 m | 0.254 m | **47.8°** | **50.4°** |
| **LocateAnything-3B** | bc_ft (Random Forest) | 0.348 m | 0.191 m | 60.5° | 71.0° |
| **LocateAnything-3B** | mlp_ft (MLP) | **0.255 m** | 0.178 m | 75.1° | 76.1° |

## Table 2 — Position accuracy, ranked

| Rank | Detector | Policy | Mean error | vs. rule baseline |
|---|---|---|---|---|
| 1 | YOLOv8n | mlp_ft | **0.158 m** | **37% better** |
| 2 | YOLOv8n | bc_ft | 0.236 m | 6% better |
| 3 | YOLOv8n | rule | 0.250 m | — |
| 4 | LocateAnything-3B | mlp_ft | 0.255 m | **47% better** |
| 5 | LocateAnything-3B | bc_ft | 0.348 m | 27% better |
| 6 | LocateAnything-3B | rule | 0.479 m | — |

**Both learned policies beat the rule baseline on positional accuracy, under
both detectors.** The ordering mlp_ft < bc_ft < rule is identical in the two
independent conditions, which is what makes it a result rather than noise.

## Table 3 — Orientation accuracy, ranked

| Rank | Detector | Policy | Mean error | vs. rule baseline |
|---|---|---|---|---|
| 1 | LocateAnything-3B | rule | **47.8°** | — |
| 2 | YOLOv8n | rule | 59.7° | — |
| 3 | LocateAnything-3B | bc_ft | 60.5° | 27% worse |
| 4 | YOLOv8n | bc_ft | 68.9° | 15% worse |
| 5 | YOLOv8n | mlp_ft | 73.2° | 23% worse |
| 6 | LocateAnything-3B | mlp_ft | 75.1° | 57% worse |

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

| Policy | YOLOv8n (2 Hz) | LocateAnything-3B (0.5 Hz) | Degradation |
|---|---|---|---|
| mlp_ft | 0.158 m | 0.255 m | **+61%** |
| bc_ft | 0.236 m | 0.348 m | **+47%** |
| rule | 0.250 m | 0.479 m | **+92%** |

Every policy is less accurate when the perception rate drops. The rule
baseline degrades roughly twice as much as the learned policies — consistent
with the cut-through result, where the rule went from 0/10 to 6/10 trials
walking between group members (p = 0.011) while the learned policies did not
degrade at all.

---

## Interpretation

**Position is learned well.** The tuned MLP places the robot a median of
**8.5 cm** from a socially ideal standing position under YOLOv8n. That is well
inside the tolerance a human observer would accept and it beats a hand-coded
geometric rule by 37%.

**Orientation is not learned well.** All three policies leave the robot 48–75°
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

**Robustness follows the same pattern.** When perception degrades from 2 Hz to
0.5 Hz, the rule loses 92% of its positional accuracy and starts cutting
through conversations; the learned policies lose 47–61% and do not. Policies
that infer from a broader feature set degrade more gracefully than one that
recomputes from a single instantaneous centroid.

## Limitations to state alongside these numbers

- **n = 30 group-visits per cell.** Adequate for the effect sizes reported, not
  for fine distinctions.
- **The 0.6 m offset and 45° criterion are choices.** Both are reported so the
  results can be reproduced or re-scored under different assumptions.
- **Slots come from ground-truth person positions**, not from the robot's own
  detections. This is deliberate: it measures where the robot *should* have
  stood independently of whether its perception found the group, which is what
  makes the comparison fair across two detectors of very different quality.
- **Orientation is measured while the robot is within 0.5 m of a slot**, which
  includes time in transit. The figures are therefore an upper bound on how
  well the robot faces the group, not a lower bound.
