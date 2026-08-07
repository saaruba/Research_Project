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

### Phase C — Group detection / F-formation / O-space (Objective 3)
- [x] Cluster tracked people into candidate groups (6 Aug 2026, `scripts/cluster_groups.py`) — per-frame connected-components clustering on 2D detection centres, distance normalised by average bounding-box width (a size-relative proxy that partially corrects for camera perspective, since real metric/depth coordinates aren't available). Run across all 24 sessions → `detected_groups.csv` per session (`group_id`, `num_people`, `group_center_x/y`, `group_bbox` extent, `is_largest_group`). Group sizes look sensible everywhere (mix of solo detections and 2-12 person clusters).
- [x] ~~Estimate orientation per person from `gaze_uniface.csv`~~ **Not possible — confirmed 6 Aug 2026 that `gaze_uniface.csv` is 100% empty for both sessions 1 and 3 (no yaw/pitch/roll/gaze data anywhere in the dataset).** See Mistakes & Corrections item 10.
- [x] O-space estimation, simplified: adopted the **mutual-facing assumption** (group members assumed to orient toward each other — standard literature fallback when true orientation is unavailable). O-space centre = group centroid, already produced by `cluster_groups.py` (`group_center_x/y`). No further modelling step needed — this **is** the O-space estimate for this project.
- [x] Generate candidate approach locations outside the O-space (6 Aug 2026, `scripts/estimate_approach_points.py`) — for each group, draws a line from an assumed robot viewpoint (bottom-centre of camera frame) through the group centroid, and places `approach_x/y` just outside the group's extent plus a standoff buffer, with `approach_facing_deg` pointing back at the group. Added to `detected_groups.csv` for all 24 sessions. **Scope note (important, written into the script's docstring too): these are 2D image-pixel coordinates, not real-world metres — there's no camera calibration/depth in this dataset. This closes Phase C for the recorded dataset (feature engineering / write-up), but Phase E's live Nav2 baseline needs the same formula re-applied to real-world (x, y) positions from the simulation itself, not these pixel values directly.**
- [ ] Document explicitly in your report that this module is custom-built and simplified because the dataset provides (a) no group/O-space ground truth and (b) no orientation ground truth at all — both are legitimate, stated limitations, not gaps to apologise for

### Phase D — Simulation environment
- [x] TIAGo simulation installed in Docker (ROS 2 Humble, Gazebo Classic, RViz2) per `docs/tiago_setup_progress.md`
- [x] Basic head-control test node (`head_scan_node.py`) working
- [ ] Bridge or conversion plan for ROS1 bag data → ROS2 sim environment (decide now, don't discover this gap late)
- [ ] Build or source simulated scenes with groups of people standing around (actor/pedestrian models)
- [ ] Set up Nav2 for TIAGo and confirm basic goal-sending works

### Phase E — Rule-based baseline (Objective 4, build before ML)
- [x] Implement simple rule: stand N metres from group centroid, outside O-space, facing group centre (6 Aug 2026, new package `src/tiago_group_approach`, node `group_approach_baseline_node.py`) — subscribes to `/group_centroid` (map-frame `PointStamped`), reads the robot's position from TF (`map` -> `base_link`), computes a standoff point 1.2 m short of the centroid (Hall's proxemics social-space boundary, exposed as a ROS 2 parameter), facing back toward the group, and sends it to Nav2 via `NavigateToPose`.
- [ ] **NOT YET RUN OR TESTED — needs you, in your devcontainer, not doable from here.** This Cowork sandbox has no ROS2/Gazebo/TIAGo install, so I could only write and syntax-check the code, not execute it. To test: `colcon build --packages-select tiago_group_approach`, source, launch the restaurant world + Nav2 for TIAGo, run `ros2 run tiago_group_approach group_approach_baseline_node`, then in another terminal publish a fake group centroid (`ros2 topic pub --once /group_centroid geometry_msgs/msg/PointStamped "{header: {frame_id: 'map'}, point: {x: 2.0, y: 1.0, z: 0.0}}"`) and confirm TIAGo drives to the right standoff pose. Full command block is in the node's own docstring.
- [ ] **Blocker to check first: Nav2 itself has never been confirmed working for TIAGo in this project (Phase D item still unchecked).** If `navigate_to_pose` action server isn't up, this node will just log an error and do nothing — bring up Nav2 before testing this node, not after.
- [x] This becomes your comparison point for everything in Phase F

### Phase F — Behavioural Cloning model (Objective 4)
- [x] Architecture decided (4 Aug 2026, per supervisor feedback response): MLP (2 hidden layers, 128/64 units) as primary, Random Forest as documented comparison — see `Proposal_Feedback_Action_Plan.md`
- [x] Target thresholds set: approach-pose error <0.4m position / <20° orientation on held-out sessions; person-detection recall ≥80% vs sessions 1/3 ground truth; O-space estimate within 0.3m for ≥70% of a hand-labelled validation set
- [ ] Decide target: predict approach pose (x, y, yaw) — recommended over raw `cmd_vel`, per your own proposal's error metrics
- [ ] Train MLP + Random Forest, compare on validation split, select final model
- [ ] Train, validate on held-out sessions
- [ ] Compare against Phase E's baseline

### Phase G — Integration
- [ ] Full pipeline: camera → detection → tracking → group/O-space → policy (baseline or BC) → pose validation → Nav2 execution

### Phase H — Evaluation
- [ ] Implement the 8 metrics from your proposal: O-space intrusion rate, min distance to group, group cut-through rate, approach-pose error, collision-free rate, task success rate, path length, navigation time
- [ ] Run BC-vs-baseline comparison across test scenarios
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

_Updated 6 Aug 2026: Phases A, B, and C are done (detection, grouping, O-space, approach-point geometry). Phase E's rule-based baseline node is now written (`tiago_group_approach` package) but **not yet run or tested** - that has to happen in your own devcontainer, since this assistant's sandbox has no ROS2/Gazebo. Next up, in order:_

1. **In your devcontainer: confirm Nav2 actually works for TIAGo in the sim first (Phase D).** This is the real blocker right now - the baseline node sends goals to Nav2's `navigate_to_pose` action, and if that server isn't up, nothing will move, no matter how correct the node's logic is. If you haven't brought up Nav2 with TIAGo before, do that in isolation first (send it one manual goal via RViz2's "Nav2 Goal" tool) before touching the new node at all.
2. **Then test the new baseline node** (`src/tiago_group_approach`) using the fake-centroid publish command in the node's own docstring. Confirm TIAGo drives to a point ~1.2m short of the fake group position, facing it. This proves the rule-based end-to-end pipeline shape (goal computation -> Nav2 -> robot motion) works, independent of perception.
3. **Only after step 2 works:** connect real input to `/group_centroid` instead of the fake test publish - i.e. bridge your live camera feed in the sim through the same detect -> group -> centroid logic you already built for the recorded dataset (Phase G integration). This is the piece that turns "I can send one test goal" into "TIAGo actually reacts to people it sees."
4. **In parallel, back on the ML side:** retrain the BC model using the new group-level features (`detected_groups.csv`) instead of raw scattered person positions, and shift the training target from raw `cmd_vel` toward approach-pose (x, y, yaw), per your proposal's actual evaluation metrics.
5. Write the README and tidy commits - 30 minutes, removes an easy source of lost marks.

Everything else follows from these five.
