# Results Tables — Complete Experiment

**Learning Socially Appropriate Group-Approach Behaviour for a TIAGo Robot
from Non-Expert Human Demonstrations**

All tables below are ready to be inserted into Chapter 5. Table numbers assume
Results is Chapter 5; renumber if it sits elsewhere. Every figure is traceable
to a file listed in §7.

---

## Part A — Offline evaluation (held-out test sessions)

### Table 5.1 — Approach-pose prediction accuracy on held-out sessions

Test sessions 5, 9 and 59 (11,921 rows), never seen during training or
hyperparameter selection. Thresholds are those stated in the project proposal:
position < 0.4 m, orientation < 20°.

| Policy | Mean position error (m) | Median position error (m) | Mean orientation error (°) | Median orientation error (°) | Within position threshold (%) | Within orientation threshold (%) | Within both (%) |
|---|---|---|---|---|---|---|---|
| Naive (predict mean) | 0.410 | 0.365 | 28.97 | 12.53 | 68.43 | 63.24 | 47.19 |
| **Rule-based (geometric)** | **0.305** | **0.164** | 29.13 | 16.60 | **70.09** | 56.82 | 43.13 |
| Random Forest (untuned) | 0.401 | 0.294 | 29.27 | 16.72 | 61.35 | 55.78 | 38.23 |
| MLP (untuned) | 0.466 | 0.353 | 42.22 | 28.78 | 55.26 | 37.49 | 21.78 |
| **Random Forest (tuned)** | 0.365 | 0.267 | **25.78** | **13.15** | 66.83 | 61.28 | **43.29** |
| MLP (tuned) | 0.395 | 0.305 | 30.99 | 21.36 | 61.92 | 46.95 | 31.75 |

**Reading of Table 5.1.** The rule-based baseline gives the lowest position
error (0.305 m). The tuned Random Forest gives the lowest orientation error
(25.78°) and is the only learned policy to beat the baseline on any metric.
**No policy met the proposal's 20° orientation threshold.** Tuning improved both
model families substantially — the untuned MLP was the worst configuration
tested at 42.22°.

Note also that the naive mean predictor attains 47.19% within both thresholds,
higher than any learned model. This is the first indication that the dataset,
not the model, is the binding constraint.

---

### Table 5.2 — Retraining sensitivity check

Six configurations trained to test whether further tuning could improve the
models. Session-level split; a different random assignment from Table 5.1, so
absolute values are not directly comparable — the comparison **within** this
table is what matters.

| Configuration | Test position error (m) | Test orientation error (°) |
|---|---|---|
| **Naive (predict training mean)** | 0.349 | **25.1** |
| Random Forest, unregularised | 0.351 | 30.4 |
| Random Forest, leaf = 5, depth = 20 | 0.341 | 28.9 |
| Random Forest, leaf = 20, depth = 15 | **0.331** | 27.1 |
| MLP 128–64 | 0.507 | 38.0 |
| MLP 64–32, alpha = 1e-2 | 0.401 | 34.6 |

**Reading of Table 5.2.** Heavier regularisation yields a marginal position
improvement, but **every learned configuration is worse than the naive
predictor on orientation**. Six configurations across two model families failed
to move the result. This is the ceiling of what 462 independent approach events
support for a seven-feature regression; the constraint is demonstration volume,
not model capacity.

---

## Part B — Simulation experiment

### Table 5.3 — Experimental configuration

| Parameter | Value |
|---|---|
| Environment | `restaurant_testing.world`, 20 × 15 m |
| People | 15 actors: 3 conversational groups (4, 3, 5), 2 lone individuals, 1 walker |
| Valid targets | Groups of ≥ 2 people only (`min_group_size = 2`) |
| Static obstacles | 5 tables, buffet, 5 plants, stage, kitchen partitions |
| Patrol route | (-5,5) → (3,6) → (8,1) → (8,-6) → (-8,-4) → (-5,5) |
| Start pose | (0, 0), reset before every trial |
| Trials | 10 per policy per detector; **60 total** |
| Localisation | Simulator ground truth, 30 Hz |
| Sampling | 10 Hz trajectory recording |
| Success criterion | Distance ∈ [0.5, 2.0] m of a real group centre **and** heading within 45°, in the same sample |
| Trial termination | Route completed, 60 s stationary, or 30 min timeout |

---

### Table 5.4 — Primary results: YOLOv8n detector (n = 10 per policy)

| Metric | Rule-based | BC – Random Forest | BC – MLP | Better |
|---|---|---|---|---|
| Task success | 10/10 (100%) | **10/10 (100%)** | 4/10 (40%) | higher |
| Collision free | 8/10 (80%) | **10/10 (100%)** | **10/10 (100%)** | higher |
| O-space intrusion | 10/10 (100%) | 7/10 (70%) | **3/10 (30%)** | lower |
| Cut-through runs | **0/10 (0%)** | **0/10 (0%)** | 2/10 (20%) | lower |
| Mean distance to nearest person (m) | 0.29 | 0.41 | **1.01** | higher |
| Mean path length (m) | 72.56 | **33.22** | 31.47 | lower |
| Mean navigation time (s) | 319.81 | **280.17** | 345.77 | lower |

---

### Table 5.5 — Statistical significance, YOLOv8n batch

Two-sided Fisher exact test on 2 × 2 contingency tables, n = 10 per policy.

| Comparison | Counts | p | Result |
|---|---|---|---|
| O-space intrusion: Rule vs BC–MLP | 10/10 vs 3/10 | **0.003** | **Significant** |
| Task success: BC–RF vs BC–MLP | 10/10 vs 4/10 | **0.011** | **Significant** |
| Task success: Rule vs BC–MLP | 10/10 vs 4/10 | **0.011** | **Significant** |
| O-space intrusion: Rule vs BC–RF | 10/10 vs 7/10 | 0.211 | Not significant |
| O-space intrusion: BC–RF vs BC–MLP | 7/10 vs 3/10 | 0.179 | Not significant |
| Collision free: Rule vs BC–RF | 8/10 vs 10/10 | 0.474 | Not significant |
| Cut-through: BC–MLP vs BC–RF | 2/10 vs 0/10 | 0.474 | Not significant |

**Only differences reaching p < 0.05 are claimed as findings.** Path length is a
continuous measure and the 72.56 m vs 33.22 m gap is a large effect rather than
sampling noise.

---

### Table 5.6 — Detector benchmark

Measured on the lab workstation during the experimental runs.

| Property | YOLOv8n | LocateAnything-3B |
|---|---|---|
| Architecture | Single-shot CNN | Vision-language model, autoregressive decoding |
| Parameters | ~3.2 million | ~3 billion |
| Hardware | NVIDIA RTX 4070, 12 GB, CUDA 13.0 | Same |
| Mean inference | ~0.005 s/frame | **8.40 s/frame** |
| Median inference | — | 9.64 s |
| Range | — | 0.35 – 11.37 s |
| Effective rate | ~200 Hz | **0.12 Hz** |
| Relative cost | 1× | **~1,700×** |
| Inferences over 30-trial batch | continuous (2 Hz throttled) | **31 (≈ 1 per trial)** |
| Detector recall (sessions 1 & 3) | 99.7% | evaluated offline on 30 frames |

---

### Table 5.7 — Results with LocateAnything-3B (n = 10 per policy)

| Metric | Rule-based | BC – Random Forest | BC – MLP |
|---|---|---|---|
| Task success | 10/10 (100%) | 10/10 (100%) | 9/10 (90%) |
| Collision free | 10/10 (100%) | 9/10 (90%) | 10/10 (100%) |
| O-space intrusion | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Cut-through runs | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Mean distance to nearest person (m) | 0.27 | 0.24 | 0.25 |
| Mean path length (m) | 68.61 | 64.36 | 60.75 |
| Mean navigation time (s) | 327.53 | 297.35 | 301.33 |

---

### Table 5.8 — Effect of detector, pooled across policies (n = 30 each)

| Metric | YOLOv8n | LocateAnything-3B | p |
|---|---|---|---|
| O-space intrusion | 20/30 (67%) | 30/30 (100%) | **0.001** |
| Task success | 24/30 (80%) | 29/30 (97%) | 0.103 |
| Between-policy spread, distance to person | 0.29 – 1.01 m | 0.24 – 0.27 m | — |
| Between-policy spread, path length | 31.5 – 72.6 m | 60.8 – 68.6 m | — |

**Reading of Tables 5.6–5.8.** At 8.40 s/frame the perception node necessarily
operates in single-shot mode, issuing approximately one detection per trial.
The approach policies therefore barely act, and behaviour is governed by the
navigation stack. All three policies converge to near-identical values and
every trial records an O-space intrusion, because the robot follows its patrol
route past groups it can no longer perceive.

**Table 5.7 is a feasibility measurement, not a second policy comparison.**
It should not be read as evidence about the policies.

---

### Table 5.9 — Target selection before restricting to conversational groups

Earlier trials with `min_group_size = 1`, re-scored to count only approaches to
groups of two or more people.

| Policy | Trials | Approached a conversational group | Approached only lone individuals |
|---|---|---|---|
| Rule-based | 12 | **12/12 (100%)** | 0 |
| BC – Random Forest | 26 | 21/26 (81%) | 4 |
| BC – MLP | 23 | **4/23 (17%)** | **11** |

**Reading of Table 5.9.** The MLP was not avoiding groups; perception was
supplying it with lone individuals, because targets of size 1 were admissible.
Its apparently strong O-space performance in those runs was partly an artefact
of rarely approaching a group at all. This motivated restricting valid targets
to groups of two or more for the final experiment (Tables 5.4–5.8).

---

## Part C — Consolidated summary

### Table 5.10 — All policies, all conditions

| | Offline position (m) | Offline orientation (°) | Sim success | Sim O-space intrusion | Sim distance to person (m) | Sim path (m) |
|---|---|---|---|---|---|---|
| Naive | 0.410 | 28.97 | — | — | — | — |
| **Rule-based** | **0.305** | 29.13 | **100%** | 100% | 0.29 | 72.56 |
| **BC – Random Forest** | 0.365 | **25.78** | **100%** | 70% | 0.41 | **33.22** |
| **BC – MLP** | 0.395 | 30.99 | 40% | **30%** | **1.01** | 31.47 |

*Simulation columns are the YOLOv8n condition (Table 5.4).*

**The central observation.** The offline and live rankings disagree. Offline,
the rule-based baseline gives the lowest position error and the Random Forest
the lowest orientation error. Live, the Random Forest matches the baseline on
task success while intruding on fewer O-spaces and travelling less than half the
distance, and the MLP — the weakest offline model on orientation — produces the
most socially cautious behaviour of the three.

Static prediction error and live social appropriateness are therefore **not the
same quantity**, and a policy cannot be selected on offline error alone.

---

## Part D — Notes for the write-up

**Claims supported at p < 0.05:**

1. The rule-based baseline intruded on group O-space significantly more often
   than BC–MLP (100% vs 30%, p = 0.003).
2. BC–Random Forest and the rule-based baseline achieved significantly higher
   task success than BC–MLP (100% vs 40%, p = 0.011 in both cases).
3. Detector choice significantly affected O-space intrusion (67% vs 100%,
   p = 0.001).

**Claims supported by effect size but not by significance testing:**

4. BC–Random Forest travelled less than half the distance of the baseline
   (33.22 m vs 72.56 m) for equal task success.
5. BC–MLP maintained more than three times the baseline's clearance to the
   nearest person (1.01 m vs 0.29 m).

**Claims that must NOT be made:**

- That the LocateAnything-3B condition compares the policies. It does not:
  approximately one detection per trial means the policies were largely inactive.
- That BC–Random Forest intrudes significantly less than the baseline. p = 0.211.
- That any policy met the proposal's 20° orientation threshold. None did.

---

## Part E — Data provenance

| Table | Source |
|---|---|
| 5.1 | `dataset/processed/models/approach_pose_evaluation.json` |
| 5.2 | Retraining sensitivity check, 24 Aug 2026 |
| 5.4, 5.5 | `dataset/processed/results_yolo/` — 30 trial JSONs + `summary.csv` |
| 5.6 | `/tmp/la3b_service.log`, 31 timed inferences; `scripts/validate_detector_recall.py` |
| 5.7, 5.8 | `dataset/processed/results_locateanything/` — 30 trial JSONs + `summary.csv` |
| 5.9 | Earlier trial set, re-scored with `rescore_sim_results.py --min-group-size 2` |

Every trial JSON contains its full 10 Hz trajectory, so all simulation metrics
can be recomputed offline without re-running the experiment.
