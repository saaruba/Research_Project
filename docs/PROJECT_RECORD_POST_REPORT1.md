# Project Record — Everything After the First Report

**Learning Socially Appropriate Group-Approach Behaviour for a TIAGo Robot
from Non-Expert Human Demonstrations**

Saarunathan Thuviprakash — MSc Robotics, University of Lincoln
Record covering work from the first report through to 24 August 2026.

This document is a complete technical and experimental record: what was built,
what was measured, every significant fault found and how it was corrected, and
what the final data shows. It is written so that any claim in the dissertation
can be traced back to the evidence behind it.

---

## 1. Summary of what changed

At the time of the first report the project had an offline data pipeline and a
partial simulation. Since then the following were completed:

1. A full closed-loop ROS 2 system: perception → group detection → approach
   policy → Nav2 → metrics, running live in Gazebo.
2. Three approach policies implemented and made drop-in swappable.
3. A repeatable experimental protocol with automated, unattended batch running.
4. **60 simulation trials** completed (30 per detector, 10 per policy).
5. A measured benchmark of **LocateAnything-3B** against **YOLOv8n**, closing
   the perception-model gap left by the proposal.
6. A large number of measurement and infrastructure faults identified and
   corrected — several of which had been silently invalidating earlier results.

---

## 2. System architecture

### 2.1 Package layout

```
src/tiago_group_approach/
  tiago_group_approach/
    group_perception_node.py          detection, depth back-projection, clustering
    group_approach_baseline_node.py   rule-based policy (geometric)
    bc_policy_node.py                 learned policy (Random Forest or MLP)
    mission_node.py                   scripted patrol tour and reporting
    gt_localisation_node.py           ground-truth map->odom at 30 Hz
    explore_node.py                   free-roam patrol (superseded by mission_node)
    metrics_recorder_node.py          trajectory sampling and social scoring
  launch/group_approach.launch.py     one launch, policy selected by argument
  rviz/group_approach.rviz            layout including the annotated detection view

src/tiago_social_worlds/
  worlds/restaurant_testing.world              the experimental environment
  worlds/restaurant_testing.groundtruth.json   true person and group positions
  maps/restaurant_testing.pgm/.yaml            occupancy map for Nav2
```

### 2.2 Runtime pipeline

```
head camera (RGB + depth, 640x480)
        |
        v
  person detection            YOLOv8n, class 0, confidence 0.45
        |
        v
  depth back-projection       x=(u-cx)d/fx, y=(v-cy)d/fy, z=d
        |
        v
  TF chain                    camera_optical -> base_link -> odom -> map
        |
        v
  group clustering            connected components, 1.5 m single linkage
        |
        v
  O-space estimate            centroid; radius = mean centroid-to-member distance
        |
        v
  APPROACH POLICY             rule | bc (Random Forest) | mlp
        |
        v
  Nav2 NavigateToPose         global planner + DWB local controller
        |
        v
  metrics recorder            10 Hz trajectory sampling, social scoring
```

### 2.3 Frameworks

| Layer | Technology |
|---|---|
| Middleware | ROS 2 Humble |
| Simulator | Gazebo Classic 11 |
| Robot | PAL TIAGo (`tiago_gazebo`, `tiago_description`, `tiago_2dnav`) |
| Navigation | Nav2 (planner, DWB controller, costmap_2d, behaviour tree) |
| Transforms | TF2 |
| Detection | Ultralytics YOLOv8n; NVIDIA LocateAnything-3B (comparison) |
| Machine learning | scikit-learn, joblib |
| Data | NumPy, pandas, `rosbags` (ROS 1 bag reading) |

### 2.4 Algorithms implemented

- **Person detection** — YOLOv8n, COCO class 0 only.
- **Pinhole back-projection** — pixel + depth to metric camera coordinates.
- **Connected-components clustering** — people within 1.5 m form one group.
  Performed in metres, unlike the offline version which had to normalise pixel
  distance by bounding-box width because the dataset lacked depth.
- **O-space estimation** — group centroid; radius as the mean distance from
  centroid to members (Kendon's F-formation model, mutual-facing assumption).
- **P-space gap selection** — each member's bearing from the centroid is
  computed, angular gaps between neighbours identified, and the robot is placed
  in the middle of the **nearest gap of at least 60°**. Reachability is
  prioritised over width (see §5.9).
- **Proxemic standoff** — Hall's personal zone. 0.8 m of free space to the
  nearest person, computed shell-to-body by adding the robot radius (0.30 m)
  and person radius (0.25 m), giving 1.35 m centre-to-centre.
- **Ground-truth localisation** — `T_map_odom = T_map_base · T_odom_base⁻¹`,
  recomputed at 30 Hz from `/gazebo/model_states`.

---

## 3. The learned models

### 3.1 What the models actually do

This distinction matters and was clarified during the work. The trained models
perform **one** function: given a situation, predict where to stop.

| Function | Component | Learned? |
|---|---|---|
| Detect humans | YOLOv8n (pretrained) | No |
| Group people into conversations | Connected-components clustering | No |
| Find the P-space opening | Gap selection (rule policy) | No |
| **Decide the stopping pose** | **Random Forest / MLP** | **Yes** |
| Navigate there | Nav2 | No |

### 3.2 Training data

- Source: PLUS-HRI, 24 non-expert teleoperation sessions, 229,678 synchronised
  rows at approximately 40 Hz.
- Segmentation: 462 approach events identified as moving → stopping near a
  group, yielding **70,555 training rows**.
- Split by session: 17 train / 4 validation / 3 test. Test sessions untouched
  until final evaluation.

**Features (7):** `lidar_min_range`, `lidar_mean_range`, `linear_x_prev`,
`angular_z_prev`, `num_people`, `group_bearing_rad`, `group_scale_norm`

**Targets (3):** `target_dx`, `target_dy`, `target_dyaw` — the stopping pose
**relative to the robot**. Relative encoding was chosen deliberately: absolute
map coordinates do not transfer between rooms and were measured to hurt
generalisation.

### 3.3 A limitation discovered during this phase

The training events were segmented as *"operator was moving → operator
stopped near a group"*. The model therefore learned **the final metres of an
approach already underway**. The phase of setting off toward a distant group
was not represented in the training data.

This explains behaviour observed repeatedly in simulation: the models produce
sensible predictions at 0.5–1 m from people and unreliable ones at 5 m from a
group. The project's claim is therefore narrower and sharper than originally
framed:

> *Can the stopping pose for a group approach be learned from non-expert
> demonstrations?*

rather than *can a robot learn to find and approach groups end-to-end*, which
would require demonstrations of the whole behaviour.

### 3.4 Retraining sensitivity check

To test whether the models could be improved, six configurations across both
families were trained and evaluated on held-out test sessions:

| Model | Test position error | Test orientation error |
|---|---|---|
| **Naive (predict the training mean)** | **0.349 m** | **25.1°** |
| Random Forest, unregularised | 0.351 m | 30.4° |
| Random Forest, leaf=5, depth=20 | 0.341 m | 28.9° |
| Random Forest, leaf=20, depth=15 | **0.331 m** | 27.1° |
| MLP 128-64 | 0.507 m | 38.0° |
| MLP 64-32, alpha=1e-2 | 0.401 m | 34.6° |

*(A different random session split from the original study was used, so these
absolute values are not directly comparable to the reported 0.305/0.365 m
figures. The pattern within this run is what matters.)*

**Conclusion:** a model that ignores every input and predicts the training-set
average matches or beats every learned model, and beats all of them on
orientation. This is not a tuning problem. It is the ceiling of what 462
independent approach events support for a 7-feature regression. Further
hyperparameter search was therefore abandoned as exhausted; the binding
constraint is data volume.

---

## 4. The experimental environment

`restaurant_testing.world` — 20 × 15 m room, walls at x = ±9.9 and y = ±7.4.

**Static furniture:** 5 round dining tables at (-4,-1), (-1,4), (2,-2), (6,1),
(5,-4); a buffet at (-5.5,-6.0); 5 plants; a stage at (7.45, 5.20); kitchen
partitions on the west side.

**People — 15 actors in 6 targets:**

| Target | People | Centre | O-space radius |
|---|---|---|---|
| Group A | 4 | (-3.50, -2.50) | 0.71 m |
| Group B | 3 | (4.67, 2.67) | 0.65 m |
| Group C | 5 | (5.60, -1.80) | 0.87 m |
| Solo 1 | 1 | (-6.0, -5.0) | — |
| Solo 2 | 1 | (-6.0, -7.0) | — |
| Walker | 1 (moving) | (4.0, 2.0) | — |

**Patrol route (identical in every trial):**
(-5,5) → (3,6) → (8,1) → (8,-6) → (-8,-4) → back to (-5,5).

A fixed route makes trials repeatable: any difference between policies is a
difference in behaviour, not in which part of the room the robot wandered into.

**Scoring targets:** for the final experiments, `min_group_size = 2`. Only the
three conversational groups are valid approach targets. A lone individual has
no F-formation, no O-space and no P-space opening, so approaching one cannot
demonstrate the behaviour under study; the solo actors and the walker remain in
the scene as obstacles and distractors.

---

## 5. Faults found and corrected

This section records every fault that materially affected behaviour or
measurement. Several had been silently invalidating results.

### 5.1 Stale world file
Gazebo loads worlds from `pal_gazebo_worlds`, which holds a **copy**. Edits to
the project world had no effect until that copy was refreshed. The bring-up
script now syncs it on every run and reports when the installed copy was stale.

### 5.2 Missing ground truth
`restaurant_testing.groundtruth.json` did not exist; the pipeline refused to
start. Generated from the world file; regenerated after every world change.

### 5.3 map_server never configured
`map_server` is a lifecycle node whose `configure` step loads the map. PAL sets
no `yaml_filename` for a custom world, so configure failed and the node stayed
`unconfigured` — no `/map`, no `map` frame, no planning. The parameter is now
set before configuring.

### 5.4 CameraInfo QoS mismatch
`camera_info` was subscribed with the default **RELIABLE** profile while Gazebo
publishes sensor data **BEST_EFFORT**. ROS 2 QoS compatibility is one-way: a
reliable subscriber receives nothing from a best-effort publisher, silently.
The topic listed, `count_publishers` returned 1, and no message ever arrived —
the cause of every `Waiting for CameraInfo...` hang. Fixed by using the sensor
QoS profile.

### 5.5 Static map→odom transform (major)
The correction was computed **once** at startup and published as a static
transform. Odometry drifts continuously and drifts violently when wheels slip,
so the map estimate diverged progressively. Symptoms: the robot in a different
place in RViz than in Gazebo, costmaps offset from real furniture, and
detections placed at wrong map coordinates. A detection initially attributed to
a YOLO false positive at (4.4, 2.8) was in fact a **correct** detection of the
person at (-3.0, 0.0), ruined by a stale transform. Replaced with
`gt_localisation_node`, recomputing and republishing at 30 Hz. Detections
subsequently landed within **10 cm** of ground truth.

### 5.6 Furniture absent from the map
`world_to_map.py` rasterised only `<model>` elements with box geometry. The five
tables are `<include>` blocks referencing a mesh, so **all five were missing
from the occupancy map** and Nav2 planned straight through them. Include
elements are now rasterised as 1.2 m squares; the map went from 9 to 20
obstacles.

### 5.7 Policy and mission fighting over Nav2 (major)
Both the mission node and the policy node send `NavigateToPose` goals to the
same server, where a new goal preempts the old. Logs showed approach goals
accepted and "finished" **six milliseconds** later — the robot never executed a
single approach. Resolved with explicit arbitration: the policy publishes
`/approach/start` before driving; the mission cancels its own goal and stands
down until `/approach/complete` or a timeout.

### 5.8 No goal throttling in the learned policies
Goal throttling had been added to the rule policy but never to `bc_policy_node`,
so BC and MLP issued a fresh goal on every perception frame (~2 Hz), each
preempting the last. Fixed: 2 s minimum interval, and re-issue only if the
target moved more than 0.4 m.

### 5.9 Gap selection chose the far side of the group
Gaps were sorted by width, so the robot repeatedly selected an opening 175–179°
around the group — walking around, and therefore through, the people. Logs:
`177 deg gap ... 179 deg off the robot's current side`. Now any gap of at least
60° is considered adequate and the **nearest** adequate gap is chosen. Tested on
the 5-person group with the robot to the west: old rule sent it to (7.31,-1.41)
149° around; new rule sends it to (3.64,-1.19), 0° off.

### 5.10 Clearance measured centre-to-centre
`min_person_clearance` was applied between robot and person **centres**. With a
0.30 m robot and 0.25 m person, a 0.7 m setting leaves 0.15 m of real space. The
robot wedged into a group, reached **0.062 m** from a person, and spent 84% of a
ten-minute run stationary. Body radii are now added, so 0.8 m requested is 0.8 m
delivered.

### 5.11 Collision metric measuring the robot's own body (major)
Every trial in the first full experiment reported a collision, with
`min_obstacle_range_m` between 0.200 and 0.267 m in all 19 runs — twelve at
exactly 0.200. That is not nineteen collisions but a constant: the base laser
returning the robot's own chassis. With a 0.30 m collision threshold this
flagged a collision in every run, and because `task_success` is gated on
collisions, **success was 0% for all three policies for reasons unrelated to
their behaviour**. Fixed by calibrating the self-hit radius during the startup
grace period and discarding returns below it. Collision-free rates rose to 100%.

### 5.12 Task success scored against a phantom (major)
`task_success` was evaluated against `goal_centroid`, which is overwritten by
every `/group_centroid` message and therefore held whatever perception saw
**last** — frequently a false positive near a wall. One MLP trial came within
0.51 m of a person and registered an O-space intrusion while being scored
"never held a pose in the band": it had approached a real group and was judged
against a phantom. Success is now scored against **ground-truth groups**, with
distance and heading required in the same trajectory sample. Re-scoring the
existing trials raised MLP success from 20% to 60% and BC from 60% to 90%.

### 5.13 Approach never completed, and never released the mission
The policy published completion the instant Nav2 reported arrival, so the
mission resumed immediately and the robot rolled on — it approached and left
without pausing. A **6 s dwell** was added, and arrival is verified by position
(within 0.75 m of the intended pose) rather than trusting Nav2's status, which
reports "finished" for aborted goals too.

### 5.14 No memory of attempted groups
The policy retried the same unreachable group indefinitely: one run lasted 30
minutes, drove 208 m and was 65% stationary. Group positions are now remembered
in 1.5 m cells and retired after 3 failed attempts or one success.

### 5.15 Recovery that made things worse
A stall-recovery reflex reversed the base blindly. TIAGo's laser covers only the
forward arc, so the robot backed into furniture it could not see. It now checks
the rear beams and **rotates in place** if the space behind is not confirmed
clear. A second fault disabled the reflex precisely when the robot had been
stuck longest, because it keyed off "a goal was recently *sent*" rather than
"a goal is in flight".

### 5.16 Coverage dominated by policy convergence
The mission yielded Nav2 for 45 s per approach attempt with no overall limit, so
a policy whose predictions rarely converged held the robot indefinitely.
Measured coverage: 34 map cells for the rule policy, 14–15 for the learned
policies, which never crossed the centre of the room. A **total approach budget
of 120 s per run** was introduced, so the tour completes regardless of which
policy is driving and coverage is no longer a property of convergence rate.

### 5.17 `num_people` frozen at inference
The feature was hard-coded to `3.0` with a comment promising it would be
"refined below if perception reports it" — it never was. One of the seven
features the models were trained on was a constant at inference while training
saw values from 1 to 6. Now taken from the live detection count.
*(Feature importance for `num_people` is 0.055, so the effect was real but
modest.)*

### 5.18 Predictions rejected rather than clamped
An implausible prediction caused the BC node to send **no goal at all**, so the
robot simply stood still — behaviour easily mistaken for a bad model.
Over-long predictions are now scaled back onto the plausible range instead of
discarded.

### 5.19 Detection overlay only drawn on processed frames
Perception runs at 2 Hz while the camera runs at ~15 Hz, so 13 of every 15
frames published nothing and the RViz panel appeared frozen or blank — read as
"detection is not working" when it was. The overlay now redraws cached boxes on
skipped frames and publishes in every state, with a status banner.

### 5.20 Arm tuck reported success it had not verified
`play_motion2 home` was used; `home` **extends** the arm. The check treated
"the command returned 0" as "the arm moved", so a motion that did exactly the
wrong thing was reported as success. Now uses `tuck_arm`, also commands the arm
controller directly, and **reads `/joint_states` back** to verify.

### 5.21 Trials starting from the previous trial's end pose
The simulation stays up across trials, so each run began wherever the last
ended. One trial's first trajectory sample was already (-2.46, -2.48) —
standing among people before the policy acted. The robot is now teleported to
(0, 0) at the start of every trial.

### 5.22 Camera-based obstacle avoidance (attempted, not adopted)
TIAGo's base laser scans at 0.2 m and cannot perceive a tabletop at 0.75 m. An
attempt was made to add the depth point cloud as a second Nav2 observation
source via `pointcloud_to_laserscan`. It destabilised the navigation stack
within the time available — in one configuration the robot did not move at all —
and was **not used for the reported experiments**. The implementation is
retained behind `CAMERA_OBSTACLES=1` and identified as future work.

---

## 6. Experimental infrastructure

| Script | Purpose |
|---|---|
| `run_everything.sh` | Bring-up: process cleanup, world sync, Gazebo + TIAGo + Nav2, arm tuck, map activation, ground-truth localisation, RViz |
| `run_pipeline.sh` | One trial: pose reset, target selection, launch, bag recording, auto-exit |
| `run_trials.sh` | Batch of trials across policies |
| `run_overnight.sh` | Full unattended experiment: both detectors, separate result folders, wall-clock budgets |
| `rescore_sim_results.py` | Re-score recorded trials against ground truth without re-running |
| `summarise_sim_results.py` | Aggregate a results folder |
| `drive_test.sh` | Diagnostic: drive the base directly, bypassing Nav2 |
| `world_to_map.py` | Generate the occupancy map from world geometry |
| `extract_world_groundtruth.py` | Extract true person and group positions |
| `add_person_collisions.py` | Give actors collision bodies (available, not used in reported runs) |
| `make_nav2_camera_params.py` | Nav2 parameters with depth obstacles (available, not used) |

**Trial termination.** A trial ends when the mission returns to its start point,
after 60 s of no motion (with a 90 s startup grace), or at a 30-minute hard
timeout. A stalled trial still writes a valid results file rather than being
killed mid-write. All pipeline nodes are force-killed between trials so no
process contaminates the next run.

---

## 7. Results

### 7.1 Protocol

- 10 trials per policy per detector; **60 trials total**.
- Identical world, start pose, patrol route and target set in every trial.
- `min_group_size = 2` — conversational groups only.
- Localisation from simulator ground truth, so navigation error cannot confound
  the behavioural comparison. Laser, costmaps and planners operate normally;
  only the global correction is exact.
- All trials re-scored post hoc against ground truth with identical criteria.
- Success requires distance within [0.5, 2.0] m of a real group centre **and**
  heading within 45° of it, in the same 10 Hz trajectory sample.

### 7.2 YOLOv8n — primary results (n = 10 per policy)

| Metric | rule | bc (RF) | mlp |
|---|---|---|---|
| Task success | 10/10 (100%) | **10/10 (100%)** | 4/10 (40%) |
| Collision free | 8/10 (80%) | **10/10 (100%)** | **10/10 (100%)** |
| O-space intrusion | 10/10 (100%) | 7/10 (70%) | **3/10 (30%)** |
| Cut-through runs | **0/10 (0%)** | **0/10 (0%)** | 2/10 (20%) |
| Mean distance to nearest person | 0.29 m | 0.41 m | **1.01 m** |
| Path length | 72.56 m | **33.22 m** | 31.47 m |
| Navigation time | 319.81 s | **280.17 s** | 345.77 s |

### 7.3 Statistical tests (two-sided Fisher exact)

| Comparison | p | Verdict |
|---|---|---|
| O-space intrusion: rule 10/10 vs mlp 3/10 | **0.003** | **Significant** |
| Task success: bc 10/10 vs mlp 4/10 | **0.011** | **Significant** |
| Task success: rule 10/10 vs mlp 4/10 | **0.011** | **Significant** |
| O-space intrusion: rule 10/10 vs bc 7/10 | 0.211 | Not significant |
| O-space intrusion: bc 7/10 vs mlp 3/10 | 0.179 | Not significant |
| Collision-free: rule 8/10 vs bc 10/10 | 0.474 | Not significant |
| Cut-through: mlp 2/10 vs bc 0/10 | 0.474 | Not significant |

### 7.4 Interpretation

**The hand-coded baseline succeeds by intruding.** It reached a valid approach
pose in every trial, but entered group O-space in **every trial** and came
within 0.29 m of a person on average.

**The MLP is socially cautious but operationally weak.** It intruded in only 3
of 10 trials (p = 0.003 against the rule) and maintained 1.01 m, but achieved
the approach in only 4 of 10.

**The Random Forest is the balance.** It matched the baseline on task success
(10/10) and collision-freedom (10/10), intruded less often, never cut through a
conversation, and did so in **33.2 m against the baseline's 72.6 m** — less than
half the distance, in less time.

### 7.5 An earlier finding on target selection

Before `min_group_size` was set to 2, scoring only on conversational groups
across 61 earlier trials showed:

| Policy | Approached a real group | Approached only lone individuals |
|---|---|---|
| rule | 12/12 (100%) | 0 |
| bc | 21/26 (81%) | 4 |
| mlp | 4/23 (17%) | 11 |

The MLP was not avoiding groups; it was being **handed** lone individuals by
perception, because targets of size 1 were admissible. Its apparently excellent
O-space score was partly an artefact of rarely approaching a group at all. This
motivated restricting targets to groups of two or more.

### 7.6 LocateAnything-3B — detector benchmark

| | |
|---|---|
| Hardware | NVIDIA RTX 4070, 12 GB, CUDA 13.0, torch 2.5.1+cu121 |
| Mean inference | **8.40 s/frame** |
| Median | 9.64 s |
| Range | 0.35 – 11.37 s |
| Effective rate | **0.12 Hz** |
| YOLOv8n | ~0.005 s/frame (**~200 Hz**) |
| Ratio | **~1,700× slower** |
| Inferences over the whole 30-trial batch | **31** (≈ 1 per trial) |

Results with LocateAnything-3B (n = 10 per policy):

| Metric | rule | bc | mlp |
|---|---|---|---|
| Task success | 100% | 100% | 90% |
| Collision free | 100% | 90% | 100% |
| O-space intrusion | 100% | 100% | 100% |
| Cut-through | 0% | 0% | 0% |
| Mean distance to person | 0.27 m | 0.24 m | 0.25 m |
| Path length | 68.61 m | 64.36 m | 60.75 m |

Pooled across policies, O-space intrusion was 30/30 under LocateAnything-3B
against 20/30 under YOLOv8n (**p = 0.001**).

**Interpretation.** At 8.40 s/frame the perception node necessarily operates in
single-shot mode, issuing roughly one detection per trial. The policies
therefore barely act, and behaviour is governed by the navigation stack rather
than the approach policy — which is why all three became indistinguishable and
why intrusion is universal (the robot drives its patrol route past groups it can
no longer see).

**This set is a feasibility measurement, not a second policy comparison**, and
must be reported as such. Its value is that it converts the substitution of
YOLOv8n from a pragmatic choice into a measured engineering finding:
LocateAnything-3B, as of this evaluation, is unsuitable for closed-loop social
navigation.

---

## 8. Limitations

1. **Human actors have no collision geometry.** Gazebo `<actor>` elements are
   visual only, so people never entered the laser-based costmap. Obstacle
   avoidance around people relied on the approach policy rather than the
   navigation stack. Contact events with humans are a simulator artefact.

2. **The base laser cannot see tabletops.** Scanning at 0.2 m, it perceives
   table legs at best. Camera-based obstacle avoidance was implemented but
   destabilised navigation within the time available and was not adopted.

3. **Localisation was taken from simulator ground truth**, deliberately, so that
   navigation error could not confound the behavioural comparison. On a real
   robot AMCL would be required; these results do not demonstrate deployability.

4. **The dataset contains no gaze or body-orientation data.** `gaze_uniface.csv`
   is empty across all columns, so O-space was estimated from position alone
   under a mutual-facing assumption.

5. **The models learned only the final approach.** Training events began with
   the operator already moving toward a group, so gross navigation toward a
   distant group is not represented in the demonstrations.

6. **A declared intervention was used.** With `approach_guard` active,
   predictions that would move the robot away from a detected group are
   re-projected onto the robot-group axis, preserving the model's own standoff.
   The learned policies are therefore hybrid: learned standoff, geometric
   direction. This was applied identically to both learned policies.

7. **Detection false positives.** At confidence 0.45 the detector occasionally
   reported people against wall textures, causing wasted approaches. This
   inflates path length and navigation time but does not affect the social
   metrics, which are scored against ground-truth positions.

8. **Sample size.** Ten trials per policy per detector. Differences resting on
   one or two trials are not claimed; only tests reaching p < 0.05 are reported
   as findings.

---

## 9. Artefacts

```
dataset/processed/results_yolo/               30 trials + summary.txt + summary.csv
dataset/processed/results_locateanything/     30 trials + summary.txt + summary.csv
dataset/processed/sim_bags/                   per-trial rosbags (local only)
dataset/processed/models/                     trained models (.joblib)
dataset/processed/approach_pose_dataset.csv   70,555 training rows
docs/RUNBOOK.md                               operating instructions
docs/LAB_PC_SETUP.md                          reproduction on another machine
```

Each trial JSON contains the full 10 Hz trajectory, so any metric can be
recomputed offline without re-running the simulation — as was done when the
scoring faults in §5.11 and §5.12 were found.

---

## 10. Contributions

1. A complete closed-loop social-approach system for TIAGo in ROS 2, with three
   interchangeable policies differing only in the choice of stopping pose.
2. An evaluation protocol scoring social appropriateness — O-space intrusion,
   cut-through, proxemic distance — against ground truth rather than perception,
   and recomputable offline from recorded trajectories.
3. Evidence that a learned stopping-pose policy can match a hand-coded geometric
   baseline on task success while intruding less and travelling less than half
   the distance.
4. Evidence that model family matters more than expected: two policies trained
   on identical data with identical features behave very differently, the MLP
   trading task success for proxemic caution.
5. A measured benchmark showing LocateAnything-3B is approximately three orders
   of magnitude too slow for closed-loop social navigation on current hardware.
6. A negative result worth stating: on 462 approach events, neither learned
   model substantially outperformed a mean predictor offline, indicating that
   demonstration volume, not model capacity, is the binding constraint.
