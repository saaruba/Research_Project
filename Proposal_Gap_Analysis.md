# Proposal Gap Analysis — what's satisfied, what isn't

_Generated 8 August 2026, by direct check against `Proposal_Feedback_Action_Plan.md` and the current repo/results state. 22 days to submission (30 August 2026)._
_Updated later the same day, after running the grid search, building the O-space validation tooling, and preparing the detector comparison._

---

## The four formal objectives (with their committed numeric criteria)

### Objective 1 — Literature review
> _"Review at least 18–20 peer-reviewed sources across social navigation, F-formations, Learning from Demonstration, and Behavioural Cloning, producing a literature matrix that directly informs the system requirements and evaluation metrics."_

**Status: NOT SATISFIED.** Still the original 7 references from the proposal. No literature matrix exists. Nothing has been read and written up beyond the proposal itself.

**Gap:** ~11–13 more sources, plus the matrix artefact. This is pure reading/writing work — no code, no compute, no dependency on anything else in the project. It is also **the single largest unsatisfied objective by mark weight** and cannot be rushed in the final days.

---

### Objective 2 — People localisation
> _"...targeting a person-detection recall of at least 80% on sessions 1 and 3 before applying it to the remaining sessions."_

**Status: SATISFIED, and comfortably.** Measured recall **99.7%** vs. an 80% target (`scripts/validate_detector_recall.py`), using YOLOv8n as a documented substitution for LocateAnything-3B. Detections produced for all 24 sessions.

**Note on the substitution:** the proposal's own risk table names "keep a simpler person-detection baseline" as an approved mitigation, so this is legitimate as it stands. **However** — the objective is satisfied by *any* detector hitting ≥80%; it does not require LocateAnything-3B specifically. Running LA-3B would strengthen the *justification narrative* (a measured head-to-head beats "it was awkward to install"), but it cannot improve the objective's outcome, since only 0.3% of recall headroom remains.

---

### Objective 3 — Group / O-space detection
> _"...validated against a manually hand-labelled subset of at least 30 frames, targeting an O-space-centre estimate within 0.3m of the manual annotation for at least 70% of labelled frames."_

**Status: TOOLING NOW READY — awaiting your labelling. Criterion re-specified.** Progress since first writing:

1. ~~No hand-labelled validation set exists.~~ **Frames now exported.** `scripts/prepare_ospace_validation.py` produced 30 frames spanning 18 sessions (each containing a genuine 2+ person group), with the estimated O-space centre drawn on each, in `dataset/processed/ospace_labelling/`, alongside `labels_template.csv`. `scripts/validate_ospace_estimate.py` scores them and has been tested end-to-end. **Remaining: you label them (~20-30 min).**
2. **The "within 0.3m" criterion still cannot be evaluated** — 0.3m is a *metric* tolerance and this project has only 2D pixel coordinates from uncalibrated video. **Re-specified as: within 0.5 × mean person bounding-box width.** Adult shoulder width ≈ 0.45–0.50 m, so this is ≈ 0.22–0.25 m equivalent — the same order of magnitude as intended, expressed in a unit the data can actually measure, and it self-corrects for perspective (distant people get a proportionally tighter pixel tolerance).

**Still to do:** label the frames, run the scorer, and **raise the criterion change with your supervisor before submission** — a re-specification agreed in advance is a methodological adaptation; the same change discovered by a marker is a problem.

---

### Objective 4 — Behavioural Cloning vs. rule-based baseline
> _"...achieves a mean approach-position error below 0.4m and mean approach-orientation error below 20° on unseen test-session group configurations, and compare this against a rule-based baseline on the same metrics."_

**Status: MEASURED. POSITION THRESHOLD NOW MET BY THE TUNED BC MODEL; ORIENTATION THRESHOLD MET BY NOTHING.** The comparison was run properly (`scripts/evaluate_approach_pose.py`, held-out sessions 5/9/59, 11,921 rows), then re-run after grid-search tuning:

| policy | mean position err | mean orientation err | meets <0.4m? | meets <20°? |
|---|---|---|---|---|
| naive (predict mean) | 0.410 m | 29.0° | no | no |
| **rule_based (Phase E)** | **0.305 m** | 29.1° | **yes** | no |
| random_forest (untuned) | 0.401 m | 29.3° | no (marginal) | no |
| mlp (untuned) | 0.465 m | 42.2° | no | no |
| **random_forest (TUNED)** | 0.365 m | **25.8°** | **yes** | no |
| mlp (TUNED, primary) | 0.395 m | 31.0° | **yes** (marginal) | no |

The *comparison requirement* is satisfied — run correctly, session-level splits, no leakage. On *thresholds*: after tuning, the Random Forest and MLP both meet the <0.4 m position threshold, and the tuned Random Forest achieves the **lowest orientation error of any policy including the rule-based baseline** (25.8°). But the rule-based baseline still wins outright on position (0.305 m), and **no policy comes close to the 20° orientation threshold.**

Split result, honestly stated: **BC learned orientation better than the rule; the rule stops in better positions; nobody achieved socially-acceptable orientation accuracy.** This is a legitimate, defensible finding.

---

## Methodological commitments made in the proposal that haven't been honoured

These are things the proposal explicitly promises in its Research Methods section. They're smaller than the objectives but a marker can check them directly against the text.

| Commitment (from proposal §4) | Status |
|---|---|
| MLP (2 hidden layers, 128/64, ReLU) as primary architecture | **Done** — built and evaluated |
| Random Forest as documented lower-variance alternative | **Done** — built and evaluated |
| **"Hyper-parameters ... tuned via grid search on the validation split"** | **DONE (8 Aug 2026)** — `scripts/grid_search_approach_pose.py`, selection on validation only, test untouched until final refit. It materially helped: RF 0.401→0.365 m and 29.3°→25.8°; MLP 0.465→0.395 m. Winning MLP config (32/16, alpha=0.1) is far smaller and more regularised than the proposal's specified 128/64 — worth reporting: **the proposed architecture was too large for this dataset.** |
| Test split held out entirely until final evaluation | **Done** — honoured throughout |
| Feature set incl. "estimated group centroid and O-space boundary" | **Partial.** Group centroid and bearing are used; **no O-space boundary feature** was ever included in the model input. |
| "Pose-reachability checks against the ROS planner" | **NOT DONE** — requires Nav2 running |

---

## Simulation / integration work still outstanding

| Item | Status |
|---|---|
| TIAGo sim in Docker (Gazebo, RViz2) | Done |
| Custom restaurant world | Done |
| Head-control node | Done |
| **Nav2 confirmed working with TIAGo** | **NOT DONE — this is the critical blocker.** Never verified even once. |
| Rule-based baseline node (`tiago_group_approach`) | Written, **never run or tested** |
| Simulated scenes with groups of people (actor models) | **NOT DONE** |
| Live perception → policy → Nav2 loop (Phase G) | **NOT DONE** |
| ROS1 bag → ROS2 bridge/conversion plan | **NOT DONE** |

**6 of the 8 proposal evaluation metrics depend entirely on this section working:** O-space intrusion rate, min distance to group, and group cut-through rate need metric group positions; collision-free rate, task success rate, path length, and navigation time need an actually-executed trajectory in simulation. None can be computed from recorded data.

---

## Repo / submission hygiene

| Item | Status |
|---|---|
| `README.md` | **Empty (0 bytes)** |
| Commit history | 3 commits, informal messages |
| `dataset/` excluded from version control | Needs confirming |

---

## Honest summary of where the risk actually is

Ranked by "how much does this cost you if it stays unfinished". Updated after the 8 August work session:

1. **Literature review (Objective 1)** — still completely unstarted, large mark weight, cannot be compressed, zero technical dependencies. **This is now clearly the highest risk item in the project** and should be started immediately.
2. **Nav2 never verified** — blocks 6 of 8 evaluation metrics, the entire integration phase, and the rule-based baseline test. Everything in simulation is stacked behind this one unverified step. Only you can do this (devcontainer).
3. **O-space frames not yet labelled** — tooling is built and tested; needs ~20-30 min of your time, plus a supervisor conversation about the re-specified criterion. Closes Objective 3.
4. ~~Grid search never run~~ **DONE** — and it helped (see Objective 4 above).
5. **Objective 4 orientation threshold not met by any policy** — already measured and defensible, provided it's written up honestly as a split result.

**LocateAnything-3B remains low priority, and the comparison is now prepared rather than pending.** Objective 2 is already satisfied at 99.7% against an 80% target, so the model cannot improve an objective that has already passed, and it addresses none of the risks above. Its single real benefit is evidential: `scripts/export_detector_comparison_frames.py` has already exported 30 ground-truth-positive frames, and `scripts/run_locateanything_comparison.py` will run LA-3B on them and print a recall comparison — turning "the install was awkward" into "we measured both and YOLOv8n was sufficient", which is a materially stronger viva answer. Roughly half a day on the GPU machine. Do it only once items 1-3 are on track; it will not change any result in this project.
