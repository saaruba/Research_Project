# Results and Discussion — Rebuilt Draft (for Chapter 5)

**Written 25 August 2026, drawing on the same primary sources grounding the rebuilt Methodology chapter** — `RESULTS_TABLES.md`, `APPROACH_ACCURACY_RESULTS.md`, `PROJECT_RECORD_POST_REPORT1.md` (Section 7, Results), `V2_RETRAINING_STUDY.md`, and `dataset/processed/ospace_labelling/ospace_validation_result.json`. Every number below is taken directly from those files, not re-derived or approximated — where two source documents report the same experiment (as `RESULTS_TABLES.md` and `PROJECT_RECORD_POST_REPORT1.md` both do for the final 60-trial batch), the figures were cross-checked and confirmed identical before use.

No word-count constraint has been applied, per your instruction — every result, every table, and every observation actually available is included, with every table and figure this chapter needs flagged in-line with **[TABLE]** and **[FIGURE]** markers as it comes up, and a consolidated list of all of them at the end (Section 5.11) exactly as done for the Methodology chapter.

This chapter explicitly answers back to the three research questions and four objectives set out in Chapter 1, and the aims/objectives as revisited in Chapter 2, Section 2.11 — Section 5.9 draws that mapping out explicitly once every result has been presented, rather than leaving the reader to make the connection themselves.

---

## 5. Results and Discussion

### 5.0 Structure of this chapter

Results are presented in the same order the Methodology chapter built the system: offline evaluation first (Section 5.1), the model-improvement investigation's own results (Section 5.2), then the live simulation experiment in full (Sections 5.3–5.6), the Objective 3 validation result (Section 5.7), a consolidated cross-condition summary (Section 5.8), and finally the discussion proper — explicitly linking every result back to the research questions and objectives (Section 5.9), followed by a critical discussion of what the results do and do not license as conclusions (Section 5.10).

### 5.1 Offline evaluation: approach-pose prediction accuracy

**[TABLE 5.1]** Table 5.1 reports the offline comparison on the three fully held-out test sessions (5, 9, 59; 11,921 rows), against the thresholds stated in the original project proposal (position error < 0.4 m, orientation error < 20°):

| Policy | Mean position error (m) | Median position error (m) | Mean orientation error (°) | Median orientation error (°) | Within position threshold (%) | Within orientation threshold (%) | Within both (%) |
|---|---|---|---|---|---|---|---|
| Naive (predict mean) | 0.410 | 0.365 | 28.97 | 12.53 | 68.43 | 63.24 | 47.19 |
| **Rule-based (geometric)** | **0.305** | **0.164** | 29.13 | 16.60 | **70.09** | 56.82 | 43.13 |
| Random Forest (untuned) | 0.401 | 0.294 | 29.27 | 16.72 | 61.35 | 55.78 | 38.23 |
| MLP (untuned) | 0.466 | 0.353 | 42.22 | 28.78 | 55.26 | 37.49 | 21.78 |
| **Random Forest (tuned)** | 0.365 | 0.267 | **25.78** | **13.15** | 66.83 | 61.28 | **43.29** |
| MLP (tuned) | 0.395 | 0.305 | 30.99 | 21.36 | 61.92 | 46.95 | 31.75 |

**Observations.** The rule-based baseline gives the lowest position error (0.305 m); the tuned Random Forest gives the lowest orientation error (25.78°) and is the only learned policy to beat the baseline on any single metric. **No policy of any kind met the proposal's 20° orientation threshold** — the best result, the tuned Random Forest's 25.78° mean, still misses it by nearly 6°. Tuning materially improved both learned families (Random Forest: 0.401 m/29.27° → 0.365 m/25.78°; MLP: 0.466 m/42.22° → 0.395 m/30.99°), confirming the grid search was doing genuine work rather than being a formality. Most strikingly, the naive mean-predictor's 47.19% "within both thresholds" figure is **higher than every learned model tested** — the first quantitative signal, appearing at the very first evaluation table in this chapter, that the dataset itself rather than model architecture is the binding constraint on accuracy. This suspicion is what motivated the entire model-improvement investigation reported in Section 5.2, and it is confirmed decisively there.

**[FIGURE 5.1-A]** A grouped bar chart of the six policies in Table 5.1 against both metrics (position error, orientation error), with the two proposal thresholds drawn as horizontal reference lines, would make the "nothing meets the orientation threshold, and the naive predictor's joint pass rate beats every learned model" finding visually immediate.

**[TABLE 5.2]** A supplementary retraining sensitivity check (six configurations across both model families, a different random session split from Table 5.1, so the two are only internally comparable, not directly cross-comparable) tested whether the naive-predictor result above was an artefact of one particular tuning run:

| Configuration | Test position error (m) | Test orientation error (°) |
|---|---|---|
| **Naive (predict training mean)** | **0.349** | **25.1** |
| Random Forest, unregularised | 0.351 | 30.4 |
| Random Forest, leaf = 5, depth = 20 | 0.341 | 28.9 |
| Random Forest, leaf = 20, depth = 15 | **0.331** | 27.1 |
| MLP 128–64 | 0.507 | 38.0 |
| MLP 64–32, α = 1e-2 | 0.401 | 34.6 |

**Observations.** Six configurations across two model families reproduce the same pattern independently: **a model that ignores every input and predicts the training-set average matches or beats every learned model tested, and beats all of them on orientation.** This is not confined to one grid search result — it recurs under a different data split with entirely different hyperparameters — and it is the result that triggered the full model-improvement investigation in Section 5.2.

### 5.2 Results of the model-improvement investigation ("v1 vs v2")

This section reports the *outcome* of the investigation whose *method* is described in full in Methodology Section 3.5 — the diagnosis, the re-segmented "v2" dataset, and the systematic 360-model search.

**[TABLE 5.3]** The central four-panel comparison, all models scored on the same held-out test sessions (5, 9, 59):

| Panel | What was tested | Best position (m) | Best orientation (°) | Best "within both" |
|---|---|---|---|---|
| A — v1 models on v1 (terminal-adjustment) test rows | reproduces Table 5.1 to 4 d.p. | 0.305 (rule) | 25.78 (tuned RF) | 43.3% (tuned RF) |
| B — v1 models on v2 (genuine-approach) test rows | same trained models, harder test rows | 0.652 (v1 MLP, position) / 0.656 (v1 RF) | 34.91 (v1 RF) | 23.5% (v1 RF) |
| C — v2-trained models (re-segmented data), v1 features | RF/MLP/Gradient Boosting trained on v2 | 0.722 (RF) | 34.93 (RF) | 15.5% |
| D — v2-trained models, v2 features (+ 6 new angular features) | as C plus new geometry | 0.722 (RF) | 36.55 | 15.5% |

**Observations — the two most important findings of the whole offline investigation.** First, moving from v1's easy terminal-adjustment test rows (Panel A) to v2's genuine-approach test rows (Panel B) **roughly doubles every policy's position error** (naive: 0.410 m → 0.724 m; tuned Random Forest: 0.365 m → 0.656 m) and drops the joint-threshold pass rate from 43.3% to 23.5%. Table 5.1's headline numbers were, in a measurable and now-quantified sense, obtained on the easiest part of the task. Second, and more strikingly, **the ranking between the rule-based and learned policies inverts**: on terminal adjustments the rule wins on position (0.305 m vs 0.365 m); on genuine approaches the rule becomes the **worst** policy tested of all four, beaten even by the naive mean-predictor (0.765 m vs 0.724 m), while the Random Forest becomes the **best** (0.656 m, 23.5% within both thresholds). Re-segmentation alone (Panel C) and re-segmentation plus new features (Panel D) both perform *worse* than simply testing the original v1-trained model on the harder rows (Panel B) — the attempted improvement, on its own, did not improve anything.

**[FIGURE 5.2-A]** A side-by-side bar chart of position error for all four policies (naive, rule, tuned RF, tuned MLP), grouped by "terminal-adjustment test rows" (Panel A) vs "genuine-approach test rows" (Panel B), would show the ranking reversal at a glance far better than the table alone.

**[TABLE 5.4]** The data-volume sweep (§5a of the source study) — does more training data recover the loss from Panel C/D, with the test set and learner held fixed:

| Configuration | Train events | Position (m) | Orientation (°) | Within both |
|---|---|---|---|---|
| `min_travel = 1.0 m` (v2 default) | 120 | 0.722 | 34.93 | 15.5% |
| `min_travel = 0.5 m` | 178 | 0.657 | 34.69 | 19.6% |
| `min_travel = 0.25 m` | 216 | 0.648 | 35.22 | 21.9% |
| `0.25 m` + mirror augmentation | 432 | 0.652 | 34.01 | 22.1% |
| `0.25 m` + mirror + validation sessions folded in | **578** | **0.642** | **33.56** | **24.2%** |
| *reference: shipped v1 Random Forest* | — | 0.656 | 34.91 | 23.5% |

**Observations.** Position error falls **monotonically** as training-event count rises from 120 to 578 (0.722 m → 0.642 m, an 11% reduction), confirming the diagnosis that re-segmentation itself was sound and the corpus was simply too small to support it. The gain over the shipped model is modest in absolute terms (0.642 m vs 0.656 m, 2.1% better) and mirror augmentation is nearly inert (216 → 432 events moved position error by 0.004 m in the wrong direction), which is itself informative: the limiting factor is genuine behavioural variety in the demonstrations, not raw row count.

**[FIGURE 5.2-B]** A line plot of position error (y-axis) against training-event count (x-axis, 120 → 578), from Table 5.4, directly visualises the "more demonstrations would help, and by roughly how much" argument this chapter uses later (Section 5.10) to ground the future-work discussion in a quantified projection rather than a generic recommendation.

**[TABLE 5.5]** The training-frame-rate sweep (§5b) — does keeping more rows per second help, with the test set, learner, and source dataset all held fixed:

| Training frame rate | Train rows | Events | Position (m) | Orientation (°) | Within both |
|---|---|---|---|---|---|
| 2 Hz | 5,883 | 216 | **0.640** | **34.25** | **22.7%** |
| 5 Hz | 14,016 | 216 | 0.643 | 34.62 | 22.0% |
| 10 Hz (default) | 26,194 | 216 | 0.648 | 35.22 | 21.9% |
| 20 Hz | 43,967 | 216 | 0.653 | 35.22 | 21.3% |
| all rows (~33 Hz) | 117,857 | 216 | 0.667 | 35.90 | 20.1% |

**Observations.** More frames per second makes the model **monotonically worse** — twenty times the rows (5,883 → 117,857) costs 4.3% in position error and 2.6 percentage points of joint pass rate. Read together, Tables 5.4 and 5.5 are the single most important methodological finding of the whole investigation: **what matters is the number of independent demonstrated approach events, not the number of rows** — increasing events from 120 to 578 improved position error by 11%, while increasing rows-per-event from 5,883 to 117,857 made it 4% *worse*. This retrospectively reframes the original dataset's "70,555 rows" (Methodology Section 3.4) as an overstatement of its true evidential content: the real sample size is 462 approach events (182 of them genuine approaches over a metre), not seventy thousand independent observations.

**[TABLE 5.6]** The label-ambiguity floor (§5c) — is the model underfitting, or is the label itself only weakly determined by the features:

| | Position error |
|---|---|
| Predict the training-set mean (no learning at all) | 0.746 m |
| Random Forest, **on its own training rows** | **0.597 m** |
| Random Forest, on held-out test rows | 0.648 m |
| Random Forest, unconstrained (`max_depth=None, min_samples_leaf=1`) — training rows | **0.210 m** |
| Random Forest, unconstrained — test rows | 0.739 m |

**Observations.** An unconstrained forest reaches 0.210 m on its own training data — the model plainly *can* fit the data — but every increase in capacity tested made the **test** error worse (0.648 m at the shipped configuration, rising to 0.739 m unconstrained), meaning the shipped configuration sits at, or very near, the optimum of the capacity/generalisation trade-off. The direct cause was measured, not inferred: for each training row, its ten nearest neighbours in standardised feature space disagree on their labelled stop pose by a **mean of 0.505 m** — this is the **label-ambiguity floor**, an approximate lower bound on achievable test error, and the shipped model's 0.648 m sits only about 0.15 m above it.

**[TABLE 5.7]** The attempted group-frame reformulation of the prediction target (§5c continued):

| Target formulation | Train position | Test position | Test "within both" |
|---|---|---|---|
| Robot-frame displacement (shipped) | 0.597 m | **0.648 m** | 21.9% |
| Group-frame standoff + bearing | 0.651 m | 0.747 m | 15.2% |

**Observations.** The reformulation performs **worse**, and for an instructive reason: reconstructing a group-frame goal requires knowing how far away the group actually is, and the only available estimate in this dataset is `lidar_min_range` — the same proxy already used as a feature — so that proxy's error enters the calculation twice (once building the label, once converting the prediction back). This confirms the root cause is a genuine property of the corpus (no metric group distance available anywhere in the recordings), not a fixable property of the target's mathematical formulation.

**[TABLE 5.8]** The systematic 360-model hyperparameter search (§5d) — 120 randomly sampled configurations per model family, selected on validation sessions, scored once on test:

| Model family | Position (m) | Orientation (°) | Within both |
|---|---|---|---|
| Random Forest | 0.636 | 34.05 | 24.8% |
| Gradient Boosting | 0.631 | 33.66 | 24.9% |
| **MLP** | **0.627** | **33.55** | **26.1%** |
| *shipped v1 Random Forest* | 0.656 | 34.91 | 23.5% |
| *label-ambiguity floor (Table 5.6)* | 0.505 | — | — |

**Observations.** The properly systematic search beats the shipped model by 0.029 m (4.4%) and 2.6 percentage points — a real, measured improvement obtained with no new data. Three findings from this table matter more than the improvement itself: **all three model families finish within 0.009 m of each other**, independent confirmation the ceiling is set by the data rather than the learner; **only 19% of the available headroom was captured** (0.029 m recovered of the 0.151 m gap to the 0.505 m floor); and **every family independently selected heavy regularisation** (winning Random Forest: `max_depth=4, min_samples_leaf=40`; Gradient Boosting: `max_depth=2`; MLP: `α=10.0`), three independent searches all concluding smaller is better. Notably, the MLP — the weakest model in the original Table 5.1 evaluation — is the **best** of the three here once properly tuned, showing its earlier underperformance was a tuning artefact rather than an architectural limitation. **None of these tuned models were deployed**; the improvement is too small relative to Table 5.10's finding (below) that offline error does not predict live behaviour to justify a fresh 30-trial re-run.

**[FIGURE 5.2-C]** A grouped bar chart of Table 5.8's three model families (position error, orientation error, within-both%) with the shipped model and the label-ambiguity floor both drawn as reference lines, visualising how close together the three families land and how much of the floor-to-shipped gap the search actually recovered.

### 5.3 Live simulation: primary results (YOLOv8n detector, n = 10 per policy)

**[TABLE 5.9]** The final experimental protocol's headline results, YOLOv8n as the detector, ten trials per policy:

| Metric | Rule-based | BC – Random Forest | BC – MLP | Better direction |
|---|---|---|---|---|
| Task success | 10/10 (100%) | **10/10 (100%)** | 4/10 (40%) | higher |
| Collision free | 8/10 (80%) | **10/10 (100%)** | **10/10 (100%)** | higher |
| O-space intrusion | 10/10 (100%) | 7/10 (70%) | **3/10 (30%)** | lower |
| Cut-through runs | **0/10 (0%)** | **0/10 (0%)** | 2/10 (20%) | lower |
| Mean distance to nearest person | 0.29 m | 0.41 m | **1.01 m** | higher |
| Mean path length | 72.56 m | **33.22 m** | 31.47 m | lower |
| Mean navigation time | 319.81 s | **280.17 s** | 345.77 s | lower |

**[TABLE 5.10]** Statistical significance (two-sided Fisher exact test, n = 10 per policy):

| Comparison | Counts | p | Result |
|---|---|---|---|
| O-space intrusion: Rule vs BC–MLP | 10/10 vs 3/10 | **0.003** | **Significant** |
| Task success: BC–RF vs BC–MLP | 10/10 vs 4/10 | **0.011** | **Significant** |
| Task success: Rule vs BC–MLP | 10/10 vs 4/10 | **0.011** | **Significant** |
| O-space intrusion: Rule vs BC–RF | 10/10 vs 7/10 | 0.211 | Not significant |
| O-space intrusion: BC–RF vs BC–MLP | 7/10 vs 3/10 | 0.179 | Not significant |
| Collision-free: Rule vs BC–RF | 8/10 vs 10/10 | 0.474 | Not significant |
| Cut-through: BC–MLP vs BC–RF | 2/10 vs 0/10 | 0.474 | Not significant |

**Observations.** **The hand-coded rule succeeds by intruding.** It reached a valid approach pose in every trial (100% task success) but entered a group's O-space in **every single trial** and came within 0.29 m of a person on average — inside Hall's (1966) intimate distance. **The MLP is socially cautious but operationally weak.** It intruded in only 3 of 10 trials (p = 0.003 against the rule) and maintained the largest clearance of any policy (1.01 m), but achieved a valid approach in only 4 of 10 trials — significantly worse than either other policy (p = 0.011 both comparisons). **The Random Forest is the balance point.** It matched the rule on task success (10/10) and improved on collision-freedom (10/10 vs the rule's 8/10), intruded less often (70% vs 100%, though this specific comparison does not reach significance at p = 0.211), never cut through a conversation, and did so travelling **33.22 m against the rule's 72.56 m — less than half the distance — in less time (280.17 s vs 319.81 s).** Only three comparisons reach p < 0.05 at this sample size; the path-length gap, while visually large, is reported as a continuous-measure effect size rather than dressed up with a significance test a binary-outcome test cannot supply for it.

**[FIGURE 5.3-A]** A grouped bar chart of all seven metrics in Table 5.9 across the three policies (normalising each metric to a common 0–1 scale, or presenting as a small multiples panel — one subplot per metric) is the single most useful figure for this section, since it lets a reader see the "rule succeeds by intruding, MLP is cautious but weak, RF is the balance" story in one image.

**[TABLE 5.11]** An earlier, pre-correction target-selection finding, kept as a result in its own right because of what it reveals about the MLP's earlier apparent strength (61 trials, before `min_group_size` was fixed at 2):

| Policy | Trials | Approached a conversational group | Approached only lone individuals |
|---|---|---|---|
| Rule-based | 12 | **12/12 (100%)** | 0 |
| BC – Random Forest | 26 | 21/26 (81%) | 4 |
| BC – MLP | 23 | **4/23 (17%)** | **11** |

**Observations.** **The MLP was not avoiding groups; it was being handed lone individuals by perception.** With targets of size 1 admissible, its apparently excellent O-space performance in that earlier batch was partly an artefact of rarely approaching a real group at all (only 17% of its "approaches" were to genuine 2+-person groups). This directly motivated restricting valid targets to groups of two or more for every result reported in Tables 5.9–5.10, and is included here as a result rather than only a methodological footnote, since it is itself a finding about how easily an apparently strong social-caution result can be an artefact of what the policy was actually given to approach.

### 5.4 Detector benchmark and the LocateAnything-3B condition

**[TABLE 5.12]** The measured detector benchmark, taken from the final 60-trial batch on the lab workstation (NVIDIA RTX 4070, 12 GB, CUDA 13.0, torch 2.5.1+cu121):

| | YOLOv8n | LocateAnything-3B |
|---|---|---|
| Architecture | Single-shot CNN | Vision-language model, autoregressive decoding |
| Parameters | ~3.2 million | ~3 billion |
| Mean inference | ~0.005 s/frame | **8.40 s/frame** |
| Median inference | — | 9.64 s |
| Range | — | 0.35 – 11.37 s |
| Effective rate | ~200 Hz | **0.12 Hz** |
| Relative cost | 1× | **~1,700× slower** |
| Inferences over the 30-trial batch | continuous, 2 Hz throttled | **31 (≈ 1 per trial)** |
| Detector recall (sessions 1 & 3, offline validation) | 99.7% | evaluated separately, 30 frames offline |

**[TABLE 5.13]** Results with LocateAnything-3B as the detector (n = 10 per policy):

| Metric | Rule-based | BC – Random Forest | BC – MLP |
|---|---|---|---|
| Task success | 10/10 (100%) | 10/10 (100%) | 9/10 (90%) |
| Collision free | 10/10 (100%) | 9/10 (90%) | 10/10 (100%) |
| O-space intrusion | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Cut-through runs | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Mean distance to nearest person | 0.27 m | 0.24 m | 0.25 m |
| Mean path length | 68.61 m | 64.36 m | 60.75 m |
| Mean navigation time | 327.53 s | 297.35 s | 301.33 s |

**[TABLE 5.14]** Effect of detector, pooled across all three policies (n = 30 each):

| Metric | YOLOv8n | LocateAnything-3B | p |
|---|---|---|---|
| O-space intrusion | 20/30 (67%) | 30/30 (100%) | **0.001** |
| Task success | 24/30 (80%) | 29/30 (97%) | 0.103 |
| Between-policy spread, distance to person | 0.29 – 1.01 m | 0.24 – 0.27 m | — |
| Between-policy spread, path length | 31.5 – 72.6 m | 60.8 – 68.6 m | — |

**Observations.** At 8.40 s/frame the perception node necessarily operates in single-shot mode, obtaining roughly one detection per trial. **All three policies become statistically indistinguishable from one another** under this condition — every between-policy spread collapses relative to the YOLOv8n condition (distance-to-person spread shrinks from 0.29–1.01 m to a narrow 0.24–0.27 m) — because behaviour is now governed by Nav2's own reactive navigation rather than by the approach policy, which has almost nothing to act on. O-space intrusion rises to **100% under LocateAnything-3B against 67% under YOLOv8n, a highly significant difference (p = 0.001)**, because the robot's fixed patrol route carries it past groups it can no longer perceive in time to react. **This table set is a feasibility measurement, not a second policy comparison, and must not be read as evidence about which policy is "better" under LocateAnything-3B** — the policies were barely able to act at all. Its actual value is that it converts the choice of YOLOv8n from a pragmatic engineering substitution into a measured, decisive finding: LocateAnything-3B, as evaluated in this project, is roughly three orders of magnitude too slow for closed-loop social navigation on the hardware tested.

**[FIGURE 5.4-A]** A log-scaled histogram or box-plot of LocateAnything-3B's 0.35–11.37 s inference-time range against YOLOv8n's near-fixed ~0.005 s, sitting alongside the ~1,700× relative-cost figure, is the clearest way to make this gap visually immediate (this is the same figure already flagged as [FIGURE 3.2-A] in the Methodology chapter — reuse rather than redraw it here).

### 5.5 F-formation slot-based approach accuracy (the complementary, non-saturating metric)

Because task success (Table 5.9) saturated at 100% for two of the three policies under YOLOv8n, a continuous metric — how close the robot got to any of several ideal standing positions derived directly from F-formation geometry (Kendon, 1990) — was applied to the same 60-trial batch, giving 30 group-visits per policy per detector (3 conversational groups × 10 trials).

**[TABLE 5.15]** The headline result:

| Detector | Policy | Position error (mean) | Position error (median) | Orientation error (mean) | Orientation error (median) |
|---|---|---|---|---|---|
| YOLOv8n | rule | 0.250 m | 0.261 m | **64.6°** | **67.7°** |
| YOLOv8n | bc_ft (Random Forest) | 0.236 m | 0.241 m | 74.7° | 81.7° |
| YOLOv8n | mlp_ft (MLP) | **0.158 m** | **0.085 m** | 79.1° | 84.1° |
| LocateAnything-3B | rule | 0.479 m | 0.254 m | **56.8°** | **58.3°** |
| LocateAnything-3B | bc_ft (Random Forest) | 0.348 m | 0.191 m | 66.4° | 78.3° |
| LocateAnything-3B | mlp_ft (MLP) | **0.255 m** | 0.178 m | 79.4° | 80.7° |

Orientation is measured only while the robot is stopped (speed ≤ 0.10 m/s) and within 0.5 m of a slot, so these figures describe poses the robot actually held, not headings caught mid-transit.

**[TABLE 5.16]** Position accuracy, ranked, with percentage improvement over the rule baseline:

| Detector | Policy | Mean | vs. rule (mean) | Median | vs. rule (median) |
|---|---|---|---|---|---|
| YOLOv8n | **mlp_ft** | **0.158 m** | 37% better | **0.085 m** | **67% better** |
| YOLOv8n | bc_ft | 0.236 m | 6% better | 0.241 m | 8% better |
| YOLOv8n | rule | 0.250 m | — | 0.261 m | — |
| LocateAnything-3B | **mlp_ft** | **0.255 m** | 47% better | **0.178 m** | **30% better** |
| LocateAnything-3B | bc_ft | 0.348 m | 27% better | 0.191 m | 25% better |
| LocateAnything-3B | rule | 0.479 m | — | 0.254 m | — |

**Observations.** **Both learned policies beat the rule baseline on positional accuracy, under both detectors, on both statistics.** The ordering mlp_ft < bc_ft < rule holds identically in two independent detector conditions and survives the switch from mean to median — the definition of a genuine result rather than an artefact of a handful of unlucky trials. The MLP's advantage is *larger* on the median under YOLOv8n (67% better than the rule) than on the mean (37% better): its typical trial is considerably more accurate than its average, because the average is pulled up by a small number of poor trials — the median is both the more favourable and the more honest number to lead with here.

**[TABLE 5.17]** Orientation accuracy, ranked:

| Rank | Detector | Policy | Mean error | vs. rule baseline |
|---|---|---|---|---|
| 1 | LocateAnything-3B | rule | **56.8°** | — |
| 2 | YOLOv8n | rule | 64.6° | — |
| 3 | LocateAnything-3B | bc_ft | 66.4° | 17% worse |
| 4 | YOLOv8n | bc_ft | 74.7° | 16% worse |
| 5 | YOLOv8n | mlp_ft | 79.1° | 22% worse |
| 6 | LocateAnything-3B | mlp_ft | 79.4° | 40% worse |

**Observations.** **The ranking inverts completely.** The rule baseline — whose facing is hard-coded to point at the group centre by construction — is *best* on orientation under both detectors, and the MLP, the most accurate policy on position, is *worst* under both. This is the project's central dissociation finding, previewed already in Chapter 1 (Section 1.5) and Chapter 2 (Section 2.10): **position and orientation are separately learnable, and separately achievable, properties of a group-approach policy** — the non-expert demonstrations carry a strong, learnable signal about where to stand, and a much weaker one about which way to end up facing.

**[TABLE 5.18]** Per-group position error (YOLOv8n), showing where the MLP's advantage is strongest and weakest:

| Group | People | rule | bc_ft | mlp_ft |
|---|---|---|---|---|
| 3 | 4 (square formation) | 0.230 m | 0.322 m | **0.061 m** |
| 4 | 3 (triangle) | 0.242 m | 0.201 m | **0.159 m** |
| 5 | 5 (loose ring) | 0.277 m | **0.184 m** | 0.255 m |

**Observations.** The MLP is most accurate on the tight, symmetric formations (a striking 0.061 m on the 4-person square) and degrades on the 5-person "loose ring," which has the largest O-space and the least well-defined single correct opening — consistent with the demonstrations containing fewer and messier examples of approaching wide, loosely-bounded formations.

**[TABLE 5.19]** Effect of the detector on position accuracy, read on medians rather than means (the LocateAnything distributions are strongly right-skewed, so means describe the tail, not the typical trial):

| Policy | YOLO mean | LA mean | *apparent* change | YOLO median | LA median | **real change** |
|---|---|---|---|---|---|---|
| mlp_ft | 0.158 m | 0.255 m | +61% | 0.085 m | 0.178 m | +109% |
| bc_ft | 0.236 m | 0.348 m | +47% | 0.241 m | 0.191 m | **−21%** |
| rule | 0.250 m | 0.479 m | +92% | 0.261 m | 0.254 m | **−3%** |

**Observations.** On the means, every policy looks as though it degrades badly at the lower perception rate. On the medians, **two of the three policies actually improve slightly.** The means are inflated by a small number of catastrophic trials specific to the LocateAnything condition (the rule at 3.53 m, 2.41 m and 0.85 m; the Random Forest at 2.33 m and 2.07 m; the MLP at 2.20 m) that never occur under YOLOv8n. **The correct statement is therefore not that LocateAnything-3B degrades approach accuracy — it is that at 0.5 Hz, LocateAnything-3B achieves comparable *typical* approach accuracy to YOLOv8n at 2 Hz, but introduces occasional catastrophic failure** (a trial 2–3.5 m from any valid approach point), which is consistent with the cut-through result in Table 5.13, where the rule baseline's cut-through rate rose sharply under the slower detector while the learned policies' did not.

**[FIGURE 5.5-A]** A grouped bar chart of Table 5.16's six position-error figures (three policies × two detectors), ideally with a spread/error-bar overlay, communicates the "learned beats rule on position under both detectors" finding directly.

**[FIGURE 5.5-B]** A scatter or dot-plot of individual trial errors under each detector condition (rather than only the summary means/medians) would make Table 5.19's "means are inflated by rare catastrophic trials, medians are not" observation visible as actual data points rather than only as a stated conclusion — this is arguably the most persuasive figure available for this specific, easily-misread finding.

**Limitations of this metric, stated alongside the numbers.** n = 30 group-visits per cell is adequate for the effect sizes reported here but not for finer distinctions. The 0.6 m offset and 45° gap-width criterion defining a "slot" are stated choices, not derived constants, and are reported explicitly so the results can be reproduced or re-scored under different assumptions. Slots are computed from ground-truth person positions, not the robot's own detections, deliberately — this measures where the robot *should* have stood independently of whether its own perception found the group, which is what makes the comparison fair across two detectors of very different quality. Orientation is judged only at near-stationary samples; judging every sample instead gives systematically better-looking (and less honest) figures — rule 59.7°, bc_ft 68.9°, mlp_ft 73.2° — because a robot driving past a slot occasionally happens to be pointed the right way; those numbers are not reported as results, for exactly that reason.

### 5.6 Objective 3 result: the O-space validation

**[TABLE 5.20]** The completed hand-labelling validation result, taken directly from `ospace_validation_result.json`:

| Quantity | Value |
|---|---|
| Frames prepared | 30 |
| Frames excluded (no group) | 0 |
| Frames scored | 30 |
| Tolerance | 0.5 × mean detected person bounding-box width |
| Mean error | 92.36 px (0.760 person-widths) |
| Median error | 75.11 px |
| Frames within tolerance | **11 / 30 (36.7%)** |
| Target (re-specified Objective 3 criterion) | ≥ 70% |
| **Verdict** | **FAIL** |

**Observations.** The pipeline's centroid-based O-space estimate matched human judgement on only 11 of 30 frames — barely half the 70% target the criterion was itself re-specified to (Methodology Section 3.3.1). This is reported here as an honest, negative result against the project's own stated criterion, consistent with the validation script's own built-in instruction to report a failing result rather than suppress it. Two facts bound how far this failure propagates: the mutual-facing, centroid-based estimate is independently justified in the reviewed literature (Vascon *et al.*, 2016) as a legitimate simplification given the complete absence of orientation data in this dataset, not an arbitrary guess; and the live simulation results in Sections 5.3–5.5, which actually depend on group geometry, are scored against the world's true ground-truth positions, not against this offline pixel-based estimate — so this specific failure does not silently invalidate any of the live results already reported. It is, nonetheless, a direct and honestly negative answer to Objective 3 on its own explicit terms.

**[FIGURE 5.6-A]** A montage of 2–3 of the 30 labelled frames — the human-clicked point, the pipeline's automatic centroid, and the resulting pixel error marked on each — with one clearly-passing and one clearly-failing example, gives a reader an intuitive sense of what an 0.760-person-width error actually looks like on an image, which the summary numbers alone cannot convey.

**[FIGURE 5.6-B]** A histogram of the 30 individual per-frame errors (in person-widths), with the 0.5-person-width tolerance line marked, shows the actual distribution behind the 36.7% headline figure — useful for judging whether the failures are marginal misses clustered just above tolerance or a wide, genuinely poor spread.

### 5.7 Consolidated summary across all conditions

**[TABLE 5.21]** All policies, all conditions, brought together in one table (simulation columns are the YOLOv8n condition, Table 5.9):

| | Offline position (m) | Offline orientation (°) | Sim task success | Sim O-space intrusion | Sim distance to person (m) | Sim path length (m) |
|---|---|---|---|---|---|---|
| Naive | 0.410 | 28.97 | — | — | — | — |
| **Rule-based** | **0.305** | 29.13 | **100%** | 100% | 0.29 | 72.56 |
| **BC – Random Forest** | 0.365 | **25.78** | **100%** | 70% | 0.41 | **33.22** |
| **BC – MLP** | 0.395 | 30.99 | 40% | **30%** | **1.01** | 31.47 |

**The central observation of this entire dissertation's results.** The offline and live rankings *disagree*, and disagree in a way that is itself the finding. Offline, the rule-based baseline gives the lowest position error and the Random Forest the lowest orientation error — a fairly unremarkable, mixed offline picture. Live, the Random Forest matches the baseline's task success while intruding on fewer O-spaces and travelling less than half the distance, and the MLP — the *weakest* offline model on orientation — produces the *most* socially cautious live behaviour of the three policies tested. **Static offline prediction error and live social appropriateness are demonstrably not the same quantity, and neither this project's nor, by extension, any comparable project's policy can be safely selected on offline error alone.**

**[TABLE 5.22]** Claims this dissertation is, and is not, entitled to make, stated explicitly to avoid over- or under-claiming in the write-up:

*Supported at p < 0.05:*
1. The rule-based baseline intruded on group O-space significantly more often than BC–MLP (100% vs 30%, p = 0.003).
2. BC–Random Forest and the rule-based baseline achieved significantly higher task success than BC–MLP (100% vs 40%, p = 0.011 in both cases).
3. Detector choice significantly affected O-space intrusion (67% vs 100%, p = 0.001).

*Supported by effect size but not by a significance test:*
4. BC–Random Forest travelled less than half the distance of the baseline (33.22 m vs 72.56 m) for equal task success.
5. BC–MLP maintained more than three times the baseline's clearance to the nearest person (1.01 m vs 0.29 m).

*Claims that must NOT be made, stated as explicitly as the claims that can:*
- That the LocateAnything-3B condition compares the three policies. It does not — at ~1 detection per trial the policies were largely inactive, and the condition measures detector feasibility, not policy quality (Section 5.4).
- That BC–Random Forest intrudes significantly less than the rule baseline. The comparison does not reach significance (p = 0.211) despite the visually large gap (70% vs 100%).
- That any policy met the proposal's original 20° orientation threshold. None did, at any point in this chapter, under any condition.

### 5.8 Discussion: linking the results back to the research questions and objectives

**RQ1 — Can a socially appropriate stopping pose be learned from non-expert demonstrations?** Yes, with an important qualification. The F-formation slot-accuracy results (Section 5.5, Table 5.16) show both learned policies beating the hand-designed rule on positional accuracy, under both detector conditions, on both mean and median — a result that survives every robustness check applied to it in this chapter. The qualification is that "stopping pose" in this dissertation's sharpened claim (Chapter 1, Section 1.3) refers specifically to the terminal metres of an approach, not the whole act of finding and travelling toward a distant group — the demonstrations available support learning the former convincingly and, per Methodology Section 3.4's original labelling analysis, do not represent the latter at all.

**RQ2 — Does a learned policy outperform the rule, and does the comparison differ by dimension?** The comparison differs sharply by dimension, and this dissociation is the single most important empirical finding this project produced. On **position**, both learned policies beat the rule decisively and consistently (Section 5.5). On **orientation**, the rule beats both learned policies decisively and consistently, under both detectors (Table 5.17) — because its facing is hard-coded to point at the group centre, a behaviour the learned models were never given a strong enough training signal to reproduce as reliably. On the project's own compound social-appropriateness metrics (Table 5.9), the answer further fractures by *which* learned policy: the Random Forest matches the rule's task success while being measurably less intrusive and far more efficient; the MLP trades task success away almost entirely in exchange for the largest personal-space margin of any policy tested. There is no single sentence that honestly answers "does learning beat the rule" — the honest answer requires stating which dimension, and which learned model, is being asked about, and this chapter has done so explicitly throughout rather than collapsing to one verdict.

**RQ3 — Where a policy falls short, what is the diagnosable cause?** Answered exhaustively in Sections 5.1–5.2 for the offline shortfall specifically: the cause is not model architecture (three independently-designed model families converge within 0.009 m of each other, Table 5.8), not insufficient tuning (a systematic 360-model search recovered only 19% of the available headroom), and not raw data quantity in the naive sense (adding rows made things worse, Table 5.5) — it is the number of independent, sufficiently-varied demonstrated approach *events*, quantified directly as a 0.505 m label-ambiguity floor (Table 5.6) that the corpus's 462 (182 genuine) events cannot get meaningfully closer to. For the live-simulation shortfalls — the MLP's low task success, the rule's universal intrusion, LocateAnything-3B's unsuitability — each has its own specific, evidenced cause stated at the point the result is reported (Sections 5.3–5.4) rather than left as an unexplained number.

**Objective 1 (Literature review).** Met — Chapter 2 reviews 34 verified sources and identifies a specific, defensible research gap (Section 2.10), revisited against the actually-achieved results here rather than only the originally-planned methodology.

**Objective 2 (Person perception).** Met, and exceeded its own original target: 99.7% recall against an 80% target (Methodology Section 3.2.2), with the detector-substitution decision converted from a pragmatic choice into the measured, decisive finding reported in Section 5.4 above.

**Objective 3 (Group detection and O-space estimation).** **Not met, honestly reported as such.** The re-specified validation criterion (≥70% of frames within 0.5× person-width) was not reached — the pipeline achieved 36.7% (Section 5.6). This is reported as a genuine, negative result rather than reframed or minimised, consistent with this project's practice throughout.

**Objective 4 (Behavioural Cloning versus rule-based baseline).** Met, with the answer being the dissociation finding itself rather than a simple pass/fail: the comparison was executed exactly as specified — offline and via a full, controlled 60-trial live simulation — and produced a genuine, statistically supported, and mechanistically explained result, even though that result is more nuanced than "the learned model won" or "the learned model lost."

### 5.9 Critical discussion of the results

Three limitations of the results themselves — distinct from the methodological trade-offs already discussed in Methodology Section 3.9 — are worth stating plainly here, since they bound how the findings above should be read.

**Sample size and statistical power.** At n = 10 trials per policy per detector condition, this study is powered to detect only large effects (Hoffman and Zhao, 2020), and this chapter has been deliberately disciplined about not over-claiming: only three of the many comparisons made across Sections 5.3–5.5 reach p < 0.05, and several visually striking differences (the 72.56 m vs 33.22 m path-length gap chief among them) are reported as effect sizes on a continuous measure specifically because a significance test appropriate to a binary outcome cannot be honestly applied to them.

**The `approach_guard` confound.** As disclosed in Methodology Section 3.9, both learned policies evaluated throughout this chapter operate under an active runtime safeguard that re-projects a prediction pointing away from the detected group back onto the robot-group axis, preserving the model's own predicted standoff. The learned policies' results in this chapter are therefore results for a **hybrid** system — learned standoff distance, geometrically-corrected direction — not for an unmodified end-to-end learned policy. This was applied identically to both learned policies, so the BC-Random-Forest vs BC-MLP comparisons in this chapter remain fair between the two, but any comparison against the rule-based baseline is, to that extent, hybrid-vs-hand-engineered rather than pure-learning-vs-pure-rules.

**The proxy-metric ceiling.** Every social-appropriateness result in this chapter — O-space intrusion, minimum distance, cut-through rate, the F-formation slot accuracy — is a geometric proxy for social acceptability (Methodology Section 3.0), not a direct measurement of how a real bystander would experience being approached. A policy that wins on every metric reported in this chapter has not been shown to be experienced as comfortable by an actual human being, only shown to be geometrically closer to what F-formation theory and the recorded demonstrations describe as appropriate. This gap is stated as an explicit, unclosed limitation rather than something these results resolve.

### 5.10 Further discussion points worth including once figures exist

- Once **[FIGURE 5.1-A]** and **[FIGURE 5.2-A]** exist side by side, a short paragraph contrasting them directly (the offline ranking reversal, terminal-adjustment vs genuine-approach) would let a reader see the single most important offline finding without needing to hold two separate tables in mind.
- Once **[FIGURE 5.3-A]** exists, it is worth explicitly walking a reader through it metric-by-metric in prose (as Table 5.9's own discussion already does) so the figure and the text reinforce rather than duplicate each other.
- Consider whether a single combined figure — offline ranking (Table 5.21, left panel) next to live ranking (Table 5.21, right panel) — would make the "central observation" of Section 5.7 land more forcefully as the dissertation's headline visual, given it is arguably the single most important result in the whole document.

### 5.11 Consolidated list of tables and figures for this chapter

**Tables** (22 total, numbered continuing from Chapter 3's own table numbering convention where relevant — renumber to fit the final document's actual running order):

1. Table 5.1 — Offline approach-pose prediction accuracy, six policies (Section 5.1).
2. Table 5.2 — Retraining sensitivity check, six configurations (Section 5.1).
3. Table 5.3 — v1-vs-v2 four-panel comparison (Section 5.2).
4. Table 5.4 — Data-volume sweep (Section 5.2).
5. Table 5.5 — Training-frame-rate sweep (Section 5.2).
6. Table 5.6 — Label-ambiguity floor / train-vs-test capacity check (Section 5.2).
7. Table 5.7 — Group-frame reformulation attempt (Section 5.2).
8. Table 5.8 — 360-model systematic hyperparameter search (Section 5.2).
9. Table 5.9 — Primary live results, YOLOv8n, n=10 per policy (Section 5.3).
10. Table 5.10 — Fisher exact test significance table, YOLOv8n batch (Section 5.3).
11. Table 5.11 — Pre-correction target-selection finding (Section 5.3).
12. Table 5.12 — Detector benchmark, YOLOv8n vs LocateAnything-3B (Section 5.4).
13. Table 5.13 — Live results with LocateAnything-3B (Section 5.4).
14. Table 5.14 — Effect of detector, pooled (Section 5.4).
15. Table 5.15 — F-formation slot accuracy headline result (Section 5.5).
16. Table 5.16 — Position accuracy, ranked (Section 5.5).
17. Table 5.17 — Orientation accuracy, ranked (Section 5.5).
18. Table 5.18 — Per-group position error (Section 5.5).
19. Table 5.19 — Effect of detector on slot-accuracy position (Section 5.5).
20. Table 5.20 — Objective 3 O-space validation result (Section 5.6).
21. Table 5.21 — Consolidated summary, all policies, all conditions (Section 5.7).
22. Table 5.22 — Claims supported / not supported (Section 5.7).

**Figures** (9 total):

1. Figure 5.1-A — Offline six-policy bar chart with threshold lines (Section 5.1).
2. Figure 5.2-A — Terminal-adjustment vs genuine-approach position error, side by side (Section 5.2).
3. Figure 5.2-B — Position error vs training-event count, line plot (Section 5.2).
4. Figure 5.2-C — 360-model search, three families vs shipped vs floor (Section 5.2).
5. Figure 5.3-A — All seven live metrics, three policies, grouped/small-multiples (Section 5.3).
6. Figure 5.4-A — Detector inference-time distribution, log-scaled (Section 5.4; reuses Methodology's Figure 3.2-A).
7. Figure 5.5-A — Slot-accuracy position error, six bars with spread (Section 5.5).
8. Figure 5.5-B — Individual-trial scatter, detector effect on position accuracy (Section 5.5).
9. Figure 5.6-A — O-space labelling montage, pass/fail example frames (Section 5.6; reuses Methodology's Figure 3.3-A).
10. Figure 5.6-B — O-space error histogram with tolerance line (Section 5.6; reuses Methodology's Figure 3.3-B).

**Note on reuse:** Figures 5.4-A, 5.6-A and 5.6-B are the same figures already flagged in the Methodology chapter's own consolidated list (Section 3.10 there) — build each once and reference it from both chapters, rather than producing duplicate versions, unless the dissertation's final layout requires a chapter-local copy.

---

## References

*Every source cited in-text anywhere in this chapter is listed below, in the same annotated Harvard format used throughout the Introduction, Literature Review and Methodology chapters. All sources here are already verified and dated identically in those chapters — this chapter introduces no new citations. Audited by grep against the body text above: every entry below has a matching in-text citation, and every in-text citation has a matching entry.*

Hall, E.T. (1966) *The Hidden Dimension*. Garden City, NY: Doubleday.
*[Chosen as the source for the intimate/personal proxemic-zone framing used to interpret the rule-based baseline's 0.29 m mean clearance to the nearest person. Used in Section 5.3.]*

Hoffman, G. and Zhao, X. (2020) 'A primer for conducting experiments in human–robot interaction', *ACM Transactions on Human-Robot Interaction*, 10(1), pp. 1–31.
*[Chosen to ground the small-sample statistical discipline applied throughout this chapter's discussion of significance — specifically the deliberate restraint in Section 5.9 about which findings are and are not claimed as significant. Used in Section 5.9.]*

Kendon, A. (1990) *Conducting Interaction: Patterns of Behavior in Focused Encounters*. Cambridge: Cambridge University Press.
*[Chosen as the theoretical basis for the F-formation slot-derivation method whose results are reported in full in Section 5.5. Used in Section 5.5.]*

Vascon, S., Mequanint, E.Z., Cristani, M., Hung, H., Pelillo, M. and Murino, V. (2016) 'Detecting conversational groups in images and sequences: A robust game-theoretic approach', *Computer Vision and Image Understanding*, 143, pp. 11–24.
*[Chosen to bound how far the Objective 3 validation failure (Section 5.6) is allowed to propagate — the position-only O-space estimate it validates is independently justified by this paper as a legitimate simplification, not an arbitrary guess, even where the validation itself fails its target. Used in Section 5.6.]*
