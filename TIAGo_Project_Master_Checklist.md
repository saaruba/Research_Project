# TIAGo Group-Approach Project — Master Checklist

_Learning Socially Appropriate Group-Approach Behaviour for a TIAGo Robot from Non-Expert Human Demonstrations_
Last updated: 6 August 2026, based on a direct review of the `Research_Project` repo and dataset.

---

## 0. Timeline Reality Check

Your own Gantt chart targets: M2 (dataset ready) week 3, M3 (group/O-space module) week 5, M4 (end-to-end pipeline) week 8, M5 (evaluation) week 9, final submission 30 August 2026.

Today sits around week 6 of that plan. M2 is not yet fully done, and M3/M4 haven't started. That's roughly four weeks left for group detection, the simulation environment, the learning model, integration, evaluation, and the write-up. This is tight but recoverable **only if you stop expanding scope now** and follow the priority order below. Anything not in "Core Path" should be treated as a stretch goal you mention as future work, not something you attempt this month.

---

## 1. Mistakes & Corrections (be honest with yourself about these)

1. **The "nested folder" theory was wrong.** I checked every session folder directly. `facial_landmarks_uniface.csv` and `gaze_uniface.csv` exist **only** in sessions `1` and `3`, at the top level. The other 22 sessions were recorded with a different, later pipeline: segmented video clips (`X.1`, `X.2`, ...) containing `cmd_vel.csv`, `gaze_data.csv` (raw eye coordinates), `head_data.json` (this is the **robot's own head pan/tilt joint state**, not a human head pose), and `joystick_data.json`. There is no recursive-search bug to fix — those 22 sessions genuinely have no face/person annotation. Don't spend time "fixing" the folder search; spend it on item 2 instead.
2. **Objective 2 (people localisation) is not optional groundwork — it's now your only way to unlock the other 22 sessions.** Since face landmarks only exist for 2 sessions, you need your own person detector running on the video (LocateAnything or a lighter fallback) to get any human-position signal from the rest of the dataset. This was always in your plan, but now it's also the fix for your small-sample-size problem, not just a "nice to have."
3. **`extract_training_table.py` is actually working now** — I found real `training_table.csv` output for sessions 1 (16,742 rows) and 3 (11,454 rows) with the ROS1_NOETIC typestore fix already applied. Good — that part of your worry is resolved. Don't re-do it.
4. ~~No train/val/test split script exists yet~~ **RESOLVED (4 Aug 2026).** `split_dataset.py` now splits all 24 sessions by whole session (17 train / 4 val / 3 test, sessions 1 & 3 both forced into train since they're the only ones with face data). `train_baseline_model.py` trained a first Behavioural Cloning baseline (LiDAR + previous action → linear_x/angular_z) and it beats the naive "always predict the mean" baseline by ~25-30% on held-out validation and test sessions (test MAE 0.196/0.086 vs naive 0.270/0.114). Note the first attempt used absolute robot (x, y) as a feature and it was *worse* than the naive baseline, because raw position doesn't generalise across sessions recorded in different rooms — LiDAR distance and recent action history do generalise. Keep that lesson in mind for any future feature engineering.
5. **Repo hygiene is weak for a submission artifact.** `README.md` is empty (0 bytes). Only 3 commits, with messages like "Installing the TIAGo in the Docker and testing anf playing with the bot in the simulation" — fine for your own use, not fine for something a supervisor or marker might open. Budget time to write a real README and commit more descriptively from here on.
6. **No baseline exists yet.** Your proposal requires comparing Behavioural Cloning against a rule-based baseline (Objective 4). Right now neither exists. Build the rule-based one first — it's simpler, and gives you an end-to-end working pipeline early, which de-risks the rest of the project.
7. **Literature review is still just the 7 references from the proposal.** The categories/keywords you drafted are a good research plan, but nothing has been read and written up yet beyond what's already in the proposal.
8. **Sanity-check the training table before trusting it.** In the first rows of `session_1/training_table.csv`, `robot_x/y/yaw` stay identical for several consecutive rows while `linear_x`/`angular_z` are 0 — plausible if the robot is stationary at the start, but worth plotting the full `robot_x`/`robot_y` trajectory once to confirm the robot actually moves meaningfully through the session before you build anything on top of it.
9. **Two dataset generations exist and need a deliberate strategy, not an accidental one.** Sessions 1 & 3 (rich: full face/gaze processing) vs. the other 22 (segmented, only raw gaze + robot head joints + joystick). Decide explicitly: sessions 1 & 3 become your validation reference for a custom detector: run your detector on their frames and compare against the provided facial landmarks as a sanity check, then trust the same detector on the other 22 sessions.
10. **`gaze_uniface.csv` is empty for both sessions 1 and 3 — checked directly (6 Aug 2026).** Every column (`yaw`, `pitch`, `roll`, `gaze_x/y/z`, `screen_x/y`) is 100% null across all ~32,000 combined rows. The checklist previously assumed sessions 1/3 had usable facing-orientation data for validating an O-space estimate — they don't. **There is no real per-person orientation ground truth anywhere in this dataset.** Don't spend time trying to recover or re-parse this file; it's genuinely empty, not a parsing bug. Consequence: proper F-formation O-space estimation (which normally works by finding where people's gaze vectors converge) isn't measurable against ground truth here. Adopted fallback: the **mutual-facing assumption** (standard in F-formation literature when true orientation is unavailable) — each group's O-space centre is approximated as the group centroid, which `cluster_groups.py` already computes (`group_center_x/y` in `detected_groups.csv`). Document this as a stated limitation, not a gap — it's a legitimate, literature-backed simplification given the data available.

---

## 2. Phase Checklist

### Phase A — Dataset & extraction pipeline
- [x] Inspect all 24 sessions, catalogue available topics/files
- [x] Build master session summary (`master_session_summary.csv/json`)
- [x] Fix ROS1 typestore bug in `extract_training_table.py`
- [x] Produce working `training_table.csv` for **all 24 sessions** (not just 1 and 3 — confirmed 4 Aug 2026, sessions 1/3 have face features, the other 22 have robot state + LiDAR only)
- [x] Build the train/val/test split script (`split_dataset.py`, session-level, no leakage) — done 4 Aug 2026
- [x] Train a first pipeline-validation baseline (`train_baseline_model.py`) — proves extraction → split → training works end to end; NOT the final approach-pose model (see Phase F)
- [ ] Sanity-check robot trajectory (plot `robot_x`/`robot_y`/`robot_yaw` over time) for sessions 1 and 3 — lower priority now that the baseline model result already shows the data behaves sensibly
- [ ] Decide final feature set and lock it (don't keep adding columns)
- [ ] Write a script to extract frames from `.mp4`/segment videos at a fixed rate, ready for the detector in Phase B

### Phase B — People/person perception (Objective 2) — COMPLETE (6 Aug 2026)
- [x] **DECISION: YOLOv8n adopted as the project's perception model, not LocateAnything-3B.** Documented, justified substitution — proposal's own risk table names "keep a simpler person-detection baseline" as approved mitigation. LocateAnything-3B needs a separate GPU venv (numpy version conflict) and wasn't worth the setup time given the deadline. `scripts/test_locateanything_detector.py` still exists if ever revisited, but it's not on the critical path — don't second-guess this again without a real reason to.
- [x] Validated detector against ground truth (`scripts/validate_detector_recall.py`): **99.7% recall** on sessions 1 & 3 vs. the proposal's 80% target. Cite this number in the dissertation.
- [x] Ran detection across all 24 sessions (`scripts/extract_person_detections.py` → `detected_people.csv` per session)
- [x] Wired detections into the training pipeline (`extract_training_table.py` falls back to `detected_people.csv` when `facial_landmarks_uniface.csv` is absent)
- [x] Fixed a merge-tolerance bug that was dropping 80%+ of detected-people rows (0.1s → 0.6s tolerance) — recovered real coverage across all sessions
- [ ] (Optional, low priority) Add tracking (Supervision + ByteTrack) so people persist identity across frames — only worth doing if Phase C group-clustering needs it
- [x] **LocateAnything-3B comparison RUN AND COMPLETE (9 Aug 2026). The proposal's named model has now actually been executed on this project's data — Objective 2's substitution is evidence-based, not assumed.** Results on 30 ground-truth-positive frames from sessions 1 and 3:

  | detector | recall | mean people/frame | exact count match vs GT | mean abs count error |
  |---|---|---|---|---|
  | LocateAnything-3B | **100.0%** (30/30) | 2.87 | 17/30 | 0.87 |
  | YOLOv8n | 96.7% (29/30) | 3.20 | 14/30 | 1.27 |

  **How to report this honestly — three caveats that matter:**
  1. **The recall difference is not statistically meaningful.** It is a *single frame* (frame_028_s3, where YOLO found nobody and LA-3B found 2). Wilson 95% CIs: LA-3B 88.6–100%, YOLOv8n 83.3–99.4% — they overlap heavily at n=30. Do not claim LA-3B "beats" YOLOv8n on this evidence.
  2. **The count comparison is confounded and should be reported with a caveat, not as an accuracy result.** Ground truth counts *annotated faces*, not people — a person facing away has no face but is still a person. frame_000 is the clearest case: both detectors found 9 people where GT says 1 face. So LA-3B's lower count error may reflect it being more conservative rather than more accurate.
  3. **The headline YOLOv8n figure remains 99.7%**, from `validate_detector_recall.py` scoring every annotated moment across sessions 1 and 3 (thousands of frames). The 96.7% here is a 30-frame spot-check and should not replace it. Cite 99.7%.

  **What this does for the dissertation:** both detectors clear the ≥80% Objective 2 target by a wide margin, and they agree exactly on 21/30 frames (mean count difference 0.53). The substitution justification therefore no longer rests on "LA-3B was awkward to install" — it rests on measured equivalence in accuracy plus a large, measured speed difference. That is a materially stronger position in a viva.
  - Timing instrumentation has been added to `run_locateanything_comparison.py` (`la3b_seconds` column + a speed summary). **Re-run it once to capture a citable s/frame figure** — the speed argument should be backed by a measured number from your own hardware, not an estimate.
- [x] ~~LocateAnything-3B head-to-head comparison PREPARED (8 Aug 2026)~~ — superseded by the completed run above. Setup notes retained below for reference. Two new scripts: `scripts/export_detector_comparison_frames.py` (already run — exported 30 ground-truth-positive frames from sessions 1 and 3 to `dataset/processed/detector_comparison/` with a manifest holding the GT count and YOLO count per frame) and `scripts/run_locateanything_comparison.py` (runs LA-3B on those frames, fills in `la3b_count`, prints the recall comparison). Copy the folder to the GPU machine, set up the separate `la3b_env` venv per the docstring, run it, copy the manifest back.
  - **Scope this honestly in the write-up:** Objective 2 is *already* satisfied (YOLOv8n at 99.7% vs an 80% target), so LA-3B **cannot improve any result** — there is 0.3% of headroom, and every group/O-space/training table/model in the project is already built on existing detections. Its only value is evidential: "we measured both and YOLOv8n was sufficient" is a much stronger viva answer than "the install was awkward". Roughly half a day, not days.
  - Reminder: LA-3B is used for **inference only** — nothing is trained here. Neither detector in this project is trained by you; both are pretrained models being applied.

### Phase C — Group detection / F-formation / O-space (Objective 3)
- [x] Cluster tracked people into candidate groups (6 Aug 2026, `scripts/cluster_groups.py`) — per-frame connected-components clustering on 2D detection centres, distance normalised by average bounding-box width (a size-relative proxy that partially corrects for camera perspective, since real metric/depth coordinates aren't available). Run across all 24 sessions → `detected_groups.csv` per session (`group_id`, `num_people`, `group_center_x/y`, `group_bbox` extent, `is_largest_group`). Group sizes look sensible everywhere (mix of solo detections and 2-12 person clusters).
- [x] ~~Estimate orientation per person from `gaze_uniface.csv`~~ **Not possible — confirmed 6 Aug 2026 that `gaze_uniface.csv` is 100% empty for both sessions 1 and 3 (no yaw/pitch/roll/gaze data anywhere in the dataset).** See Mistakes & Corrections item 10.
- [x] O-space estimation, simplified: adopted the **mutual-facing assumption** (group members assumed to orient toward each other — standard literature fallback when true orientation is unavailable). O-space centre = group centroid, already produced by `cluster_groups.py` (`group_center_x/y`). No further modelling step needed — this **is** the O-space estimate for this project.
- [x] Generate candidate approach locations outside the O-space (6 Aug 2026, `scripts/estimate_approach_points.py`) — for each group, draws a line from an assumed robot viewpoint (bottom-centre of camera frame) through the group centroid, and places `approach_x/y` just outside the group's extent plus a standoff buffer, with `approach_facing_deg` pointing back at the group. Added to `detected_groups.csv` for all 24 sessions. **Scope note (important, written into the script's docstring too): these are 2D image-pixel coordinates, not real-world metres — there's no camera calibration/depth in this dataset. This closes Phase C for the recorded dataset (feature engineering / write-up), but Phase E's live Nav2 baseline needs the same formula re-applied to real-world (x, y) positions from the simulation itself, not these pixel values directly.**
- [x] **O-space validation tooling built (8 Aug 2026) — Objective 3's hand-labelled validation set is ready for you to label.** `scripts/prepare_ospace_validation.py` exported 30 frames (spanning 18 sessions, each containing a real 2+ person group) with the estimated O-space centre drawn on them → `dataset/processed/ospace_labelling/`, plus `labels_template.csv`. `scripts/validate_ospace_estimate.py` scores them once labelled (tested end-to-end with synthetic labels — the scorer works). **YOU still need to do the actual labelling — ~20-30 min, instructions in the script docstring.**
  - **IMPORTANT — the proposal's criterion had to be re-specified, and this must be disclosed:** the proposal targets "within 0.3 m" of the manual label, but 0.3 m is a *metric* tolerance and this dataset is uncalibrated video with no depth, so metres are not recoverable from pixels. Any figure in metres would be fabricated. Replacement: **within 0.5 × mean person bounding-box width** (adult shoulder width ≈ 0.45–0.50 m, so ≈ 0.22–0.25 m equivalent — same order as intended, in a unit the data can actually measure, and it self-corrects for perspective). **Agree this with your supervisor**, and state it in Methodology/Limitations.
  - Two bugs found and fixed while building this, both worth remembering: (1) `cv2.CAP_PROP_POS_FRAMES` seeking decodes **corrupt solid-green frames** on these videos — must read sequentially, as `extract_person_detections.py` does; (2) `np.isclose` defaults to `rtol=1e-5`, which on a ~1.76e9 UNIX timestamp is a tolerance of **~17,000 seconds** — every detection in the session matched and buried the frame under thousands of boxes. Always pass `rtol=0` when comparing timestamps.
- [ ] Document explicitly in your report that this module is custom-built and simplified because the dataset provides (a) no group/O-space ground truth and (b) no orientation ground truth at all — both are legitimate, stated limitations, not gaps to apologise for

### Phase D — Simulation environment
- [x] TIAGo simulation installed in Docker (ROS 2 Humble, Gazebo Classic, RViz2) per `docs/tiago_setup_progress.md`
- [x] Basic head-control test node (`head_scan_node.py`) working
- [ ] Bridge or conversion plan for ROS1 bag data → ROS2 sim environment (decide now, don't discover this gap late)
- [x] **Simulated scenes with groups of people — DONE (9 Aug 2026, `scripts/generate_social_world.py`).** Parametric generator rather than hand-written SDF, so new configurations are one command. Produces three scenarios: `default` (2 groups / 5 people), `unseen` (3 groups / 9 people — for the Phase H "unseen group configuration" requirement), and `adjacent` (two close groups, to stress-test whether clustering separates them). People are built from primitives (no `model://` downloads, so nothing breaks in the container) standing on a circle facing inward, with a red nose marker making facing direction visible.
  - **Critically, each world ships a `.groundtruth.json` giving every person's exact position in METRES.** This is what finally makes the O-space intrusion / min-distance / cut-through metrics measurable — the recorded PLUS-HRI video never could, having no depth or calibration. The simulation is not just where results are demoed; it is where several metrics become computable for the first time.
  - Fixed along the way: `restaurant_humans.launch.py` referenced `restaurant_humans.world`, which **did not exist** — the launch would have failed. It exists now.
- [x] **Nav2 map — DONE (9 Aug 2026), computed exactly rather than SLAM-mapped.** `src/tiago_social_worlds/maps/restaurant.{pgm,yaml}`, 320×240 @ 0.05 m/px covering the 16×12 m room, origin `[-8, -6, 0]`. Because the room is a known rectangle, the occupancy grid is derived directly from the wall geometry — no SLAM run, no drift, no re-mapping after each world change. People are deliberately *not* baked into the static map (they are dynamic obstacles for the local costmap).
- [ ] Set up Nav2 for TIAGo and confirm basic goal-sending works

### Phase E — Rule-based baseline (Objective 4, build before ML)
- [x] Implement simple rule: stand N metres from group centroid, outside O-space, facing group centre (6 Aug 2026, new package `src/tiago_group_approach`, node `group_approach_baseline_node.py`) — subscribes to `/group_centroid` (map-frame `PointStamped`), reads the robot's position from TF (`map` -> `base_link`), computes a standoff point 1.2 m short of the centroid (Hall's proxemics social-space boundary, exposed as a ROS 2 parameter), facing back toward the group, and sends it to Nav2 via `NavigateToPose`.
- [ ] **NOT YET RUN OR TESTED — needs you, in your devcontainer, not doable from here.** This Cowork sandbox has no ROS2/Gazebo/TIAGo install, so I could only write and syntax-check the code, not execute it. To test: `colcon build --packages-select tiago_group_approach`, source, launch the restaurant world + Nav2 for TIAGo, run `ros2 run tiago_group_approach group_approach_baseline_node`, then in another terminal publish a fake group centroid (`ros2 topic pub --once /group_centroid geometry_msgs/msg/PointStamped "{header: {frame_id: 'map'}, point: {x: 2.0, y: 1.0, z: 0.0}}"`) and confirm TIAGo drives to the right standoff pose. Full command block is in the node's own docstring.
- [ ] **Blocker to check first: Nav2 itself has never been confirmed working for TIAGo in this project (Phase D item still unchecked).** If `navigate_to_pose` action server isn't up, this node will just log an error and do nothing — bring up Nav2 before testing this node, not after.
- [x] This becomes your comparison point for everything in Phase F

### Phase F — Behavioural Cloning model (Objective 4)
- [x] Architecture decided (4 Aug 2026, per supervisor feedback response): MLP (2 hidden layers, 128/64 units) as primary, Random Forest as documented comparison — see `Proposal_Feedback_Action_Plan.md`
- [x] Target thresholds set: approach-pose error <0.4m position / <20° orientation on held-out sessions; person-detection recall ≥80% vs sessions 1/3 ground truth; O-space estimate within 0.3m for ≥70% of a hand-labelled validation set
- [x] Decide target: predict approach pose (x, y, yaw), RELATIVE to the robot's own pose/heading at prediction time, not absolute map coordinates (6 Aug 2026) — absolute (x,y) was already shown to hurt generalisation as a *feature* in the first baseline, so it was never going to work as a *target* either, for the same reason (doesn't transfer across sessions recorded in different rooms).
- [x] Built the actual labelled dataset for this target (6 Aug 2026, `scripts/build_approach_pose_dataset.py`) — there is no ground-truth "this is a demonstrated approach" anywhere in the raw data, so it's inferred: find (moving → genuine stop) transitions in each session (speed binned into 0.5s windows to smooth out the very spiky raw cmd_vel signal) where a detected group is present near the stop, treat that stop's pose as the demonstrated label, and label every row in the lead-up to it. Found 462 such events across all 24 sessions -> 70,555 labelled rows -> `dataset/processed/approach_pose_dataset.csv`.
- [x] Trained Random Forest on this target (6 Aug 2026, `scripts/train_approach_pose_model.py`), using group-level features (`num_people`, `group_center_x/y_norm`, `group_scale_norm` from Phase C's clustering) instead of raw scattered person positions. **Result is a real, reportable negative-ish finding, not a clean win:** train MAE is well below naive baseline (as expected), but on held-out validation/test sessions it roughly ties or loses to naive-mean prediction (test: dx 0.33m vs naive 0.35m barely better; dy 0.16m vs naive 0.13m worse; dyaw 0.50 rad/28.5° vs naive 0.51 rad, roughly tied; validation is worse across the board). Only the test-set forward-distance number clears your <0.4m threshold - orientation misses <20° on both val and test. Likely causes to write up: only 8 input features may not carry enough signal, and the 70k rows come from just 462 independent approach events, so there's much less real information than the row count suggests. Metrics saved to `dataset/processed/models/approach_pose_metrics.json`.
- [x] Built the MLP (6 Aug 2026, `scripts/train_approach_pose_mlp.py`) — 2 hidden layers (128/64 units) per the decided architecture, features standardised via `StandardScaler` (MLPs are scale-sensitive, unlike Random Forest), `early_stopping=True` and L2 regularisation (`alpha=1e-3`) to fight overfitting. Same dataset, features, target, and session split as the Random Forest, so directly comparable.
- [x] **Compared MLP vs Random Forest (6 Aug 2026) — Random Forest wins clearly, but neither is good yet.** On held-out test: Random Forest beats the MLP by 26% (dx), 23% (dy), and 48% (dyaw). But recall from the entry above: the Random Forest itself only roughly ties/barely beats the naive baseline on held-out sessions, and the MLP is *worse than naive* across every target on both validation and test. **Selected model for now: Random Forest** (`approach_pose_random_forest.joblib`) — better of the two, but "better of two weak options" is the honest framing, not "solved."
- [x] **Tried replacing raw pixel position with group bearing (angle off robot-forward, 6 Aug 2026) — did NOT close the gap.** Swapped `group_center_x_norm/y_norm` for `group_bearing_rad` (computed from pixel x-position assuming a 58° camera FOV, documented assumption) in both models. Result: essentially unchanged - Random Forest test MAE moved by <1% on every target (dx 0.330 vs 0.327 before, dy 0.159 vs 0.159, dyaw 0.517 vs 0.497), MLP also barely moved and is still clearly worse than the Random Forest. **This is itself a useful finding: the bottleneck isn't how the group position is encoded as a feature - it's more likely the small number of independent demonstrated events (462) or the label-inference method itself (inferring "demonstrated approach" from stop-near-group heuristics rather than having real ground truth).** Don't spend more time on feature re-encoding without addressing one of those two first.
- [x] **Ran the hyper-parameter grid search the proposal promised (8 Aug 2026, `scripts/grid_search_approach_pose.py`) — and it MATERIALLY IMPROVED both models.** The proposal's Research Methods section explicitly commits to grid-search tuning on the validation split; until now both models used hand-picked defaults, so that promise was unfulfilled. Selection done on validation only; test untouched until the final refit. Results on held-out test:

  | model | before tuning | after tuning |
  |---|---|---|
  | Random Forest | 0.401 m / 29.27° | **0.365 m / 25.78°** |
  | MLP | 0.465 m / 42.22° | 0.395 m / 30.99° |

  Winning configs: RF `max_depth=8, min_samples_leaf=20, n_estimators=150` — note *shallower* trees won, independently confirming the overfitting diagnosis; MLP `hidden_layer_sizes=(32,16), alpha=0.1` — a much smaller, far more heavily regularised network than the proposal's 128/64 default, which is itself a reportable finding (the proposed architecture was too large for this dataset). The tuned Random Forest now **meets the <0.4 m position threshold** and has the **lowest orientation error of any policy, including the rule-based baseline** — the first metric on which Behavioural Cloning beats the rule. Script is resumable (checkpoints after each config) since a full grid exceeds one run's time budget. Full grid: `dataset/processed/models/grid_search_results.csv`.
- [ ] Remaining gap to close (or accept): the rule-based baseline still wins on **position** (0.305 m vs 0.365 m), and **nothing** meets the <20° orientation threshold. Two feature-encoding attempts have already failed to move it, and tuning has now been exhausted, so the remaining candidates are structural: a longer feature-history window, or a simpler position-only target. Given ~3 weeks left, accepting and writing this up honestly is the time-safe option.
- [x] Compare final chosen model against Phase E's rule-based baseline — done, see Phase H

### Phase G — Integration
- [ ] Full pipeline: camera → detection → tracking → group/O-space → policy (baseline or BC) → pose validation → Nav2 execution

### Phase H — Evaluation
- [x] **Built the offline evaluation harness (6 Aug 2026, `scripts/evaluate_approach_pose.py`)** — compares 4 policies (naive / rule-based / Random Forest / MLP) on the held-out test sessions using the proposal's Objective 4 metrics and thresholds, reporting mean AND median error (median matters: a few huge errors dominate the mean) plus % of rows meeting each threshold. Results saved to `dataset/processed/models/approach_pose_evaluation.{json,csv}`.
- [x] **RAN IT — and the result is a genuine negative finding for the BC hypothesis. Read this carefully before writing anything up.** On held-out test sessions (5, 9, 59; 11,921 rows):

  **FINAL RESULTS (updated 8 Aug 2026 after grid-search tuning — these are the numbers to put in the dissertation):**

  | policy | mean pos err | median pos err | mean orient err | % meeting BOTH thresholds |
  |---|---|---|---|---|
  | naive (predict mean) | 0.410 m | 0.365 m | 29.0° | 47.2% |
  | **rule_based (Phase E)** | **0.305 m** | **0.164 m** | 29.1° | 43.1% |
  | random_forest (untuned) | 0.401 m | 0.294 m | 29.3° | 38.2% |
  | mlp (untuned) | 0.465 m | 0.353 m | 42.2° | 21.8% |
  | **random_forest (TUNED)** | 0.365 m | 0.267 m | **25.8°** | 43.3% |
  | mlp (TUNED, primary) | 0.395 m | 0.305 m | 31.0° | 31.8% |

  **The nuanced, honest reading — this is a split result, not a clean loss:**
  - **Position: the rule-based baseline wins** (0.305 m vs the tuned RF's 0.365 m). Both meet the <0.4 m threshold; the rule is simply better, and its median error (0.164 m) is less than half the naive baseline's.
  - **Orientation: the tuned Random Forest wins** (25.8°) — beating the rule-based baseline, the naive baseline, and every other policy. This is the one metric where Behavioural Cloning demonstrably adds value.
  - **Nothing meets the <20° orientation threshold.** Orientation prediction is the clear failure across all policies.
  - Tuning mattered: it moved the RF from 0.401→0.365 m and 29.3°→25.8°, and the MLP from 0.465→0.395 m. Notably the winning MLP config (32/16 units, alpha=0.1) is far *smaller* and more regularised than the proposal's specified 128/64 — the proposed architecture was simply too large for 462 independent demonstration events.

  So the defensible dissertation claim is: **"Behavioural Cloning from non-expert demonstrations learned group orientation better than a geometric rule, but did not learn stopping position better, and neither approach achieved socially-acceptable orientation accuracy on this dataset."**
- [ ] Metrics that CANNOT be computed offline and still need the running simulation (Phase D/G) — do not fabricate these from recorded data: O-space intrusion rate, min distance to group, group cut-through rate (all need group positions in *metres*, which this uncalibrated-video dataset cannot provide), plus collision-free rate, task success rate, path length, navigation time (all properties of an executed trajectory).
- [ ] Test on at least one unseen group configuration if time allows

### Phase I — Literature review & writing
- [ ] Expand from 7 to ~15-20 references using your existing category list (prioritise: social navigation surveys, LfD/BC surveys, F-formation, group-approach specifically)
- [ ] Draft literature review section
- [ ] Draft methodology, results, discussion incrementally — don't leave all writing to the last week

### Phase J — Repo hygiene
- [ ] Write a real `README.md` (what the repo is, how to run the scripts, current status)
- [ ] Commit more frequently with descriptive messages
- [ ] Keep `dataset/`, `build/`, `install/`, `log/` out of version control if not already (check `.gitignore` covers `build/`, `install/`, `log/` — it does for build artifacts, confirm `dataset/` is excluded too given its size)

---

## 3. What to do next, in order

_Updated 6 Aug 2026: Phases A, B, and C are done. Phase E's rule-based baseline node is written but untested (needs your devcontainer). Phase F now has both models trained on a real approach-pose dataset - Random Forest is the better of the two, but both currently only roughly tie or underperform a naive "always predict the average" baseline on held-out sessions. This is the actual state of the ML side right now - not broken, but not yet a result you can present as "the model works." Next up, in order:_

_Re-ordered 8 August 2026, after the grid search, O-space tooling and detector-comparison prep. The ML side is now essentially exhausted - what remains is mostly work only YOU can do (labelling, devcontainer, writing)._

1. **Literature review — start this now, it is the biggest unmitigated risk in the project.** Objective 1 requires 18-20 sources plus a literature matrix; you still have the original 7 and no matrix. It carries real mark weight, cannot be compressed into the final days, and is the one item with zero technical dependencies. Everything else on this list is smaller than it.
2. **Label the 30 O-space frames** (`dataset/processed/ospace_labelling/`, ~20-30 min) and run `validate_ospace_estimate.py`. This closes Objective 3, which is currently completely unsatisfied. Also: **raise the 0.3 m → person-width criterion change with your supervisor** — do not let that surface for the first time at submission.
3. **In the devcontainer: confirm Nav2 works with TIAGo** (Phase D), then test the rule-based baseline node (Phase E). This single unverified step blocks 6 of your 8 evaluation metrics, all of Phase G integration, and any simulation results at all.
4. **Optional, ~half a day, on the GPU machine:** run the LocateAnything-3B comparison (Phase B). Strengthens the substitution justification; changes no result. Do this only if items 1-3 are on track.
5. Write the README and tidy commits - 30 minutes, removes an easy source of lost marks.

**On the ML side specifically: stop optimising.** Two feature-encoding attempts and a full grid search have now been run. The tuned Random Forest meets the position threshold and beats the rule-based baseline on orientation; the rule-based baseline still wins on position; nothing meets the 20° orientation threshold. That is a complete, defensible, honestly-measured Objective 4 result. Further tuning has low expected value against 3 weeks of remaining deadline.
