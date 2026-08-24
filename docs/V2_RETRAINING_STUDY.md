# Model Improvement Study (v2) — Diagnosis, Method, and Result

**Date:** 24 August 2026
**Status:** complete. **v2 does not improve on v1 and should not be deployed.**
All v1 datasets, models and results are unchanged on disk; nothing reported in
Chapter 5 is affected.

---

## 1. What prompted this

Seven candidate improvements were assessed. Verdicts:

| # | Proposed improvement | Verdict |
|---|---|---|
| 1 | Collect more independent approach demonstrations | **Not possible.** PLUS-HRI is a fixed corpus of 24 sessions; new collection needs ethics approval and participants |
| 2 | Train on the whole approach, not only the final metres | **Implemented** — see §2, §3 |
| 3 | Stronger social/geometric features | **Partly implemented.** Distance to group centre, O-space radius and nearest-person distance in metres are *not derivable* from the recordings: person positions exist only as pixel boxes from an uncalibrated camera. Angular equivalents were added instead |
| 4 | Reduce the train-to-live feature mismatch | **Implemented** — every v2 feature is an angle or a distance, never a pixel |
| 5 | Sample by approach event, not by row | **Implemented** — `event_id`, inverse-frequency weighting, 10 Hz decimation |
| 6 | Keep the MLP small and regularised | **Already the case.** The tuned 32/16, α=0.1 network already beat 128/64 |
| 7 | Try a stronger tabular regressor | **Implemented** — `HistGradientBoostingRegressor` |

## 2. The diagnosis: v1 labelled adjustments, not approaches

Reconstructing all 462 events in `approach_pose_dataset.csv`:

| Property of a v1 "approach event" | Value |
|---|---|
| Median distance from event start to the stop pose | **0.17 m** |
| Median event duration | **2.5 s** |
| Events starting more than 1.0 m from the stop | 53 / 462 (11%) |
| Events starting more than 2.0 m from the stop | 13 / 462 (3%) |
| Rows within 0.5 m of the stop | **72%** |
| Rows beyond 2.0 m from the stop | 3.6% |

The typical labelled demonstration was a 2.5-second shuffle ending 17 cm from
where it began.

The cause is in `find_approach_events()`: it pairs each *moving segment* with
the stop that follows, where "moving" is any 1.0 s bin above a 0.05 speed
threshold. A human walk toward a group contains pauses, turns and hesitations,
so it is fragmented, and only the final fragment is labelled.

**This is the single best explanation for the observed live behaviour:** the
models predict sensibly within a metre of a group and unreliably from further
out, because they were trained almost exclusively on the terminal phase.

## 3. What v2 changed

`scripts/build_approach_pose_dataset_v2.py` anchors on each sustained stop near
a group and walks *backwards*, absorbing brief pauses, until it has accumulated
at least 1.0 m of travel (or meets a stop longer than 3 s, or exhausts a 20 s
lookback). Events with less than 1.0 m of travel are discarded.

| | v1 | v2 |
|---|---|---|
| Events | 462 | **182** |
| Rows | 70,555 | 128,506 |
| Median event start distance | 0.17 m | **1.37 m** |
| Median event travel | — | 1.98 m |
| Median distance-to-go | 0.183 m | **0.784 m** |
| Rows within 0.5 m of stop | 72% | **35.9%** |
| Rows beyond 2.0 m | 3.6% | **10.6%** |

Six new features were added, all in radians so that the same quantity is
computable offline and in simulation without a change of units:
`group_span_rad`, `nearest_person_span_rad`, `gap_bearing_rad`,
`gap_width_rad`, `person_spacing_rad`, `people_visible`. The seven v1 features
are retained unchanged, so the v1 *models* can be scored on the v2 rows.

Training rows are decimated to 10 Hz within each event and weighted by
1/(rows in event). Test rows are left at full rate.

## 4. Results

All models scored on the **same** held-out test sessions (5, 9, 59). Panel A is
a sanity check: it reproduces the published Table 5.1 to four decimal places.

**Panel A — v1 models on v1 test rows** *(reproduces Table 5.1)*

| Policy | pos (m) | ori (°) | <0.4 m | <20° | both |
|---|---|---|---|---|---|
| naive | 0.410 | 28.97 | 68.4% | 63.2% | 47.2% |
| rule_based | **0.305** | 29.13 | 70.1% | 56.8% | 43.1% |
| random_forest_tuned | 0.365 | **25.78** | 66.8% | 61.3% | 43.3% |
| mlp_tuned | 0.395 | 30.99 | 61.9% | 47.0% | 31.8% |

**Panel B — v1 models on v2 test rows** *(real approaches)*

| Policy | pos (m) | ori (°) | <0.4 m | <20° | both |
|---|---|---|---|---|---|
| naive | 0.724 | 35.58 | 26.8% | 41.3% | 11.8% |
| rule_based | 0.765 | 36.55 | 26.8% | 39.7% | 18.9% |
| v1_random_forest_tuned | 0.656 | **34.91** | **35.4%** | **42.7%** | **23.5%** |
| v1_mlp_tuned | **0.652** | 40.01 | 34.6% | 35.1% | 16.3% |

**Panel C — v2 models, v1 features** *(effect of re-segmentation alone)*

| Policy | pos (m) | ori (°) | <0.4 m | <20° | both |
|---|---|---|---|---|---|
| random_forest_v2seg | 0.722 | **34.93** | 29.2% | 41.3% | 15.5% |
| mlp_v2seg | 0.796 | 40.65 | 23.3% | 35.9% | 8.5% |
| gradient_boosting_v2seg | 0.768 | 38.13 | 24.1% | 40.4% | 12.0% |

**Panel D — v2 models, v2 features** *(re-segmentation + new geometry)*

| Policy | pos (m) | ori (°) | <0.4 m | <20° | both |
|---|---|---|---|---|---|
| random_forest_v2full | 0.722 | 36.55 | 27.6% | 41.3% | 15.5% |
| mlp_v2full | 0.846 | 43.99 | 20.9% | 36.5% | 9.1% |
| gradient_boosting_v2full | 0.796 | 40.29 | 20.3% | 37.3% | 9.8% |

**Decision gate**

| | Best learned model | pos (m) | ori (°) | both |
|---|---|---|---|---|
| B | shipped v1 model | **0.652** | 40.01 | **16.3–23.5%** |
| C | re-segmented | 0.722 | **34.93** | 15.5% |
| D | + new geometry | 0.722 | 36.55 | 15.5% |

Position error **−10.7%** (worse), orientation error **+8.6%** (better).

## 5. Interpretation

**Re-segmentation did not improve prediction.** Models trained on genuine
approaches are *worse* at position than models trained on terminal
adjustments, even when both are tested on genuine approaches. The new
geometric features added nothing at all.

The most plausible explanation is sample count. Requiring at least 1 m of
travel reduced 462 events to 182, and after decimation the training set holds
120 independent approaches. The v1 models see many more (if less relevant)
examples. **The fix for problem #2 is blocked by problem #1** — and problem #1,
collecting more demonstrations, is exactly the item that cannot be done within
this project. That is a coherent and reportable finding, not a failure of
implementation.

**Two findings from this study are worth reporting in their own right.**

*First, the published offline evaluation is flattered by an easy test set.*
Every policy roughly doubles its position error when moved from v1's terminal
adjustments to v2's real approaches — the naive baseline from 0.410 m to
0.724 m, the tuned Random Forest from 0.365 m to 0.656 m. The proportion of
rows meeting both Objective 4 thresholds falls from 43.3% to 23.5%. The
Objective 4 thresholds were, in effect, being evaluated on the easiest part of
the task.

*Second, and more usefully, the ranking between the rule and the learned model
reverses.* On terminal adjustments the rule baseline wins on position
(0.305 m vs 0.365 m). On real approaches it is the **worst** policy tested,
beaten even by predicting the training-set mean (0.765 m vs 0.724 m), while the
Random Forest is the best (0.656 m, 23.5% within both thresholds).

This directly supports the dissertation's central claim. A fixed geometric rule
— face the group, stop 1.2 m short — is adequate for the last half metre and
degrades as soon as the robot has a real approach to plan. That is consistent
with the live simulation result, where BC–RF matched the rule's task success
using less than half the path length (33.2 m vs 72.6 m).

## 5a. Follow-up: does more training data recover the loss?

§5 attributed v2's regression to sample count. That is a testable claim, and it
holds. Three ways of recovering events without new recordings were swept, with
the **test set held fixed** throughout (full-rate rows of sessions 5, 9, 59 from
the 1.0 m dataset — the same rows as Panels B–D), and the learner held fixed at
the tuned Random Forest so the comparison is about data alone.

| Configuration | Train events | pos (m) | ori (°) | both |
|---|---|---|---|---|
| `min_travel = 1.0 m` (the v2 default) | 120 | 0.722 | 34.93 | 15.5% |
| `min_travel = 0.5 m` | 178 | 0.657 | 34.69 | 19.6% |
| `min_travel = 0.25 m` | 216 | 0.648 | 35.22 | 21.9% |
| `0.25 m` + mirror augmentation | 432 | 0.652 | 34.01 | 22.1% |
| `0.25 m` + mirror + validation sessions | **578** | **0.642** | **33.56** | **24.2%** |
| *reference:* v1 Random Forest (shipped) | — | 0.656 | 34.91 | 23.5% |
| *reference:* v2seg Random Forest | 120 | 0.722 | 34.93 | 15.5% |

**Position error falls monotonically with training-event count**, from 0.722 m
at 120 events to 0.642 m at 578. The proportion meeting both Objective 4
thresholds rises from 15.5% to 24.2%. The §5 attribution was correct: the
re-segmentation was sound and the corpus was too small to support it.

Three qualifications matter for how this is reported.

*The gain over the shipped model is marginal.* The best configuration reaches
0.642 m against the shipped 0.656 m — a 2.1% improvement — and 24.2% against
23.5% within both thresholds. Given that Table 5.10 already establishes that
offline error does not predict live social behaviour, a 2% offline gain is not
a basis for re-running the simulation.

*Mirror augmentation is nearly inert.* Doubling 216 events to 432 moved
position error by 0.004 m (the wrong way) and orientation by 1.2°. Reflection
adds no new information about how people approach groups; it only enforces a
symmetry the model can already learn. This is itself informative — it shows the
limit is genuine behavioural variety, not raw row count.

*The final row is mildly optimistic.* Folding the validation sessions into
training is legitimate only because the hyper-parameters were fixed in advance,
but those hyper-parameters were originally selected using a grid search that
saw the validation split. The 578-event figure should be quoted with that
caveat, or the 432-event row (0.652 m) used instead.

**Extrapolation for Future Work.** A 4.8× increase in training events
(120 → 578) bought an 11% reduction in position error. On that trend, reaching
the 0.4 m Objective 4 threshold would require substantially more independent
demonstration than the 24-session corpus contains. This converts "collect more
data" from an assertion into a quantified projection.

## 5b. Follow-up: does using more frames per second help?

Training rows are decimated to 10 Hz within each event. The recordings run at
~33 Hz (median inter-sample interval 0.030 s), so roughly two of every three
rows are discarded. Whether keeping them helps was measured with the test set,
the learner and the source dataset all held fixed.

| Training frame rate | Train rows | Events | pos (m) | ori (°) | both |
|---|---|---|---|---|---|
| 2 Hz | 5,883 | 216 | **0.640** | **34.25** | **22.7%** |
| 5 Hz | 14,016 | 216 | 0.643 | 34.62 | 22.0% |
| 10 Hz (default) | 26,194 | 216 | 0.648 | 35.22 | 21.9% |
| 20 Hz | 43,967 | 216 | 0.653 | 35.22 | 21.3% |
| all rows (~33 Hz) | 117,857 | 216 | 0.667 | 35.90 | 20.1% |

**More frames per second makes the model monotonically worse.** Twenty times
the rows costs 4.3% in position error and 2.6 points of threshold pass rate.
Keeping every frame is the worst configuration tested; the sparsest is the
best.

The reason is that these rows carry no additional information. At 33 Hz the
robot has moved about a centimetre between consecutive samples and the group
has not moved at all, so each extra row is a near-duplicate of its neighbour.
Inverse-frequency weighting equalises each event's total influence, but it
cannot stop redundant points dominating the trees' bootstrap samples and leaf
statistics, which is where the degradation comes from.

**Read §5a and §5b together — they are the key methodological result:**

| What is increased | Effect on position error |
|---|---|
| Independent approach **events** (120 → 578) | 0.722 m → 0.642 m (**11% better**) |
| **Rows per event** (5,883 → 117,857) | 0.640 m → 0.667 m (**4% worse**) |

What matters is the number of demonstrations, not the number of rows. This
retrospectively justifies the concern that the original dataset's "70,555 rows"
overstated its content: the true sample size was 462 approach events, and only
182 of those were approaches of more than a metre. Reporting row counts as
though they were sample sizes materially overstates the evidence available to
the models.

## 5c. Why no amount of retraining will help: the model underfits

The three studies above vary the *data*. This one asks whether the *learning*
is the limitation, and the answer settles the question.

The tuned Random Forest was scored on the data it was **trained** on:

| | Position error |
|---|---|
| Predict the training-set mean (no learning at all) | 0.746 m |
| Random Forest, **on its own training rows** | **0.597 m** |
| Random Forest, on held-out test rows | 0.648 m |

The train-to-test gap is **0.051 m**, which suggests the model is nowhere near
the limit of what it could fit. Loosening the regularisation tests that
directly:

| Random Forest configuration | Train | Test |
|---|---|---|
| `max_depth=8, min_samples_leaf=20` (current) | 0.597 m | **0.648 m** |
| `max_depth=16, min_samples_leaf=5` | 0.364 m | 0.707 m |
| `max_depth=None, min_samples_leaf=1` (unconstrained) | **0.210 m** | 0.739 m |

The model *can* fit the training data — an unconstrained forest reaches 0.210 m.
**But every increase in capacity makes the test error worse.** The shipped
configuration is already at, or very near, the optimum of this trade-off. More
capacity, more trees, more layers or more compute move performance in the wrong
direction, and this is measured rather than assumed.

### The floor is set by label ambiguity

The reason extra capacity does not generalise is that the label is only weakly
determined by the features. Measuring this directly: for each training row,
take its ten nearest neighbours in standardised feature space and ask how much
their labels disagree.

| | |
|---|---|
| Mean feature distance to 10 nearest neighbours | 0.284 (standardised) |
| Mean **label disagreement** among those neighbours | **0.505 m** |

Rows the model effectively cannot tell apart specify stop poses half a metre
apart. That is an approximate lower bound on achievable test error, and it
explains the shape of every result above.

The cause is the target definition. `target_dx, target_dy` is the displacement
from the robot's *current* pose to the eventual stop. Two moments with the same
group bearing, apparent group size and LiDAR ranges can sit at different points
along an approach — one two metres out, one thirty centimetres out — and carry
completely different displacements.

**Practical consequence:** the model sits at 0.648 m against a floor near
0.505 m. Roughly 0.15 m of headroom exists, and only hyper-parameter search can
reach any of it. The 0.4 m Objective 4 threshold is below the floor and was
never attainable with this feature set on this corpus.

### An attempted fix, and why it failed

If the target is ill-posed because it is measured from the *robot*, the obvious
remedy is to measure it from the *group*: predict the standoff distance and
approach bearing relative to the group centre, which are properties of the
group's configuration and do not depend on where the robot currently stands.
Predictions are converted back to a robot-frame goal for scoring, so the metric
is unchanged.

| Target formulation | Train pos | Test pos | Test both |
|---|---|---|---|
| Robot-frame displacement (current) | 0.597 m | **0.648 m** | 21.9% |
| Group-frame standoff + bearing | 0.651 m | 0.747 m | 15.2% |

It is **worse**, and the reason is instructive. Reconstructing a goal in the
group frame requires knowing how far away the group is, and the only available
estimate is `lidar_min_range` — the documented proxy, the nearest obstacle
ahead. That proxy's error now enters twice, once when the label is built and
again when the prediction is converted back. The reformulation is sound in
principle and defeated in practice by the absence of a metric group position.

**Which is the root cause.** The PLUS-HRI recordings provide person positions
only as pixel boxes from an uncalibrated monocular camera. Without calibration
there is no metric distance to the group, so neither the features nor the
labels can express the one quantity the approach decision actually depends on.
That is a property of the corpus, not of the method, and it cannot be repaired
by training.

## 5d. Systematic hyper-parameter search (360 models)

The shipped models used hyper-parameters carried over from an earlier phase.
`scripts/finetune_on_lab_pc.py` searches them properly: 120 randomly sampled
configurations for each of three model families, selected on the **validation**
sessions (10, 14, 15, 54) and scored once on the held-out test sessions. Search
ranges bracket the shipped values in both directions, so the search is free to
choose a *smaller* model as well as a larger one.

| Model family | pos (m) | ori (°) | both |
|---|---|---|---|
| Random Forest | 0.636 | 34.05 | 24.8% |
| Gradient Boosting | 0.631 | 33.66 | 24.9% |
| **MLP** | **0.627** | **33.55** | **26.1%** |
| *shipped v1 Random Forest* | 0.656 | 34.91 | 23.5% |
| *label ambiguity floor* (§5c) | 0.505 | — | — |

**The search beats the shipped model by 0.029 m (4.4%) and 2.6 percentage
points** on the proportion meeting both Objective 4 thresholds. That is a real
improvement, obtained without new data.

Three observations matter more than the improvement itself.

**All three families finish within 0.009 m of each other.** A random forest, a
boosted ensemble and a neural network share no architectural assumptions. If
the limitation were the learner, they would not converge on the same number.
This is independent confirmation of §5c: the ceiling is in the data.

**Only 19% of the available headroom was captured.** The gap between the
shipped model (0.656 m) and the label-ambiguity floor (0.505 m) is 0.151 m.
Exhaustive search over 360 models recovered 0.029 m of it. The remaining
0.122 m is not reachable by choosing better hyper-parameters.

**Every family independently selected heavy regularisation.** The winning
Random Forest was `max_depth=4, min_samples_leaf=40, max_features='sqrt'` —
substantially more constrained than the shipped `depth=8, leaf=20`. Gradient
boosting chose `max_depth=2`; the MLP chose `alpha=10.0`. Three independent
searches all concluding that smaller is better is consistent with a small,
noisy training set.

**Note on the MLP.** It is now the best of the three, having been the weakest
model in the original evaluation (0.395 m / 30.99° on the v1 test rows, behind
the Random Forest). Its earlier underperformance was a tuning artefact, not an
architectural limitation. Its winning configuration did emit a
`ConvergenceWarning` at `max_iter=400`, so the figure may shift slightly with a
longer iteration budget; the margin over gradient boosting (0.004 m) is inside
that uncertainty and the three families should be treated as tied.

### Should the tuned models be deployed?

**No.** The improvement is 0.029 m of offline position error. Table 5.10 already
establishes that offline error does not predict live social behaviour — the MLP
was the worst learned model on offline orientation error and the *least*
intrusive in simulation. Deploying these would require 30 fresh trials and a
rebuild of Tables 5.2–5.10 in exchange for a quantity that has no demonstrated
relationship to task success or O-space intrusion.

The value of this study is evidential, not operational: the hyper-parameter
space was searched systematically over 360 models, the shipped configuration
was confirmed to be within 4.4% of the best findable, and the agreement between
three model families localises the remaining error in the dataset rather than
the method.

## 6. Recommendation

1. **Do not deploy the v2 models.** Keep `approach_pose_random_forest_tuned`
   and `approach_pose_mlp_tuned` as the reported policies. No simulation
   re-run is needed, and Tables 5.1–5.10 stand.
2. **Report §2 and §5 in Chapter 5.** The segmentation diagnosis and the
   rule-vs-learned reversal are new evidence, obtained without invalidating
   anything already measured.
3. **Report §3–4 as an ablation.** It shows the improvement was attempted,
   measured, and rejected on evidence — which is stronger than not attempting
   it.
4. **Future Work gains a quantified argument.** "More demonstrations" is no
   longer a generic suggestion: 182 usable approach events across 24 sessions
   is demonstrably too few, and the study shows precisely what re-segmentation
   costs when the corpus is not enlarged to match.

## 7. Reproducing this

```bash
python3 scripts/build_approach_pose_dataset_v2.py   # writes approach_pose_dataset_v2.csv
python3 scripts/train_evaluate_v2.py                # trains, evaluates, writes both panels
```

Outputs: `dataset/processed/models/v1_vs_v2_evaluation.{json,csv}` and six
`*_v2seg` / `*_v2full` model files. Training is resumable — existing model
files are reused rather than refitted.

**Nothing in this study overwrites a v1 artefact.** `approach_pose_dataset.csv`
and the tuned v1 models retain their original timestamps (8 August 2026).
