# The Research Journey — Documentary Script, Phases 4 to 13

**Project:** Learning Socially Appropriate Group-Approach Behaviour for a TIAGo Robot from Non-Expert Human Demonstrations
**Author:** Saarunathan Thuviprakash (30276855)
**Course:** MSc Robotics, University of Lincoln — CMP9140M
**Document purpose:** Complete narrative record for the YouTube documentary. Nothing omitted — every failure, every wrong turn, every bug, and every fix, with the papers and tools that belong to each stage.

---

## How to read this document

Each phase carries six blocks:

| Block | What it holds |
|---|---|
| **The story** | What was actually happening, in plain language — the narration spine |
| **The papers** | Every piece of literature that genuinely belongs to this stage |
| **The tools** | Software, libraries, repositories, assets |
| **What went wrong** | Every failure and bug, with the real numbers |
| **How it was tackled** | The fix, and why that fix rather than another |
| **The handoff** | The question this phase leaves open, which the next phase answers |

**A note on attribution.** Where work was done before or outside my involvement, it is marked inline in square brackets. The full breakdown is in Appendix A. This matters: the documentary's whole credibility rests on being honest about failures, so it has to be honest about authorship too.

---

# PHASE 4 — DATASET FORENSICS
## "What Is Actually Inside These Bag Files?"

### The story

Up to this point the dataset had been an abstraction — "the recordings my supervisor gave me." Phase 4 is where it becomes 24 folders of ROS 1 bag files, videos, CSVs and timestamps that do not agree with each other.

PLUS-HRI consists of **24 recorded non-expert teleoperation sessions**. Ordinary people, not roboticists, drove a robot around and approached groups. The premise of the whole project is that their choices contain useful judgement about where it is socially appropriate to stand.

Direct inspection — rather than assuming a uniform format — established something that shaped everything afterward: **the dataset spans two distinct recording generations.**

- **Sessions 1 and 3** carry full per-frame facial-landmark annotations (`facial_landmarks_uniface.csv`) from an earlier pipeline.
- **The other 22 sessions** were recorded later in a segmented-clip format carrying only `cmd_vel.csv`, the operator's own raw eye-gaze coordinates (the *driver's* eyes, not the people being approached), the robot's head pan/tilt joint state, and joystick input — **with no person-position ground truth of any kind.**

Then the finding that redirected the entire methodology: `gaze_uniface.csv` was confirmed by direct inspection to be **entirely empty — 100% null across every column, in every session.**

There is no per-person facing-direction ground truth anywhere in this corpus.

That single fact removes the possibility of doing classical F-formation detection the way the literature does it, because the standard methods need to know which way people are facing. Phase 5 exists because of this discovery.

Extraction (`extract_training_table.py`) read each session's ROS 1 bags through the `rosbags` library's ROS 1 reader and typestore, and merged pose (`/robot_pose`, `/mobile_base_controller/odom`, `/dlo_node/odom`), LiDAR (`/scan`) and person-detection topics onto a common timestamp index with `pandas.merge_asof`. Across all 24 sessions this produced **229,678 synchronised rows at approximately 40 Hz**.

*[The initial extraction pipeline was built before my involvement in this project — that work was done with GPT. What is described here is reconstructed from the code, the Methodology chapter and the dataset itself, not from having been in the room for it.]*

### The papers

| Reference | Why it belongs here |
|---|---|
| **Argall, Chernova, Veloso & Browning (2009)** — *A Survey of Robot Learning from Demonstration* | Stops being abstract. You are now physically looking at what a "demonstration" consists of: state, action, timestamp |
| **Ravichandar et al.** — *Recent Advances in Robot Learning from Demonstration* | Demonstration *quality* and *coverage*, not quantity, is what determines what can be learned |
| **Faris, Kucukyilmaz, Polydoros & Del Duchetto (2025)** — PLUS-HRI recovery policies | The precedent that imperfect non-expert demonstrations still contain usable behavioural signal |

### The tools

`rosbags` (ROS 1 reader + typestore) · pandas (`merge_asof`) · NumPy · OpenCV · the 24-session PLUS-HRI corpus (~70 GB raw, deliberately excluded from git)

### What went wrong

**1. Corrupt green video frames.** `cv2.CAP_PROP_POS_FRAMES`-based seeking decoded solid-green images at certain offsets. Silently. A frame that looks like a rendering glitch produces detections of nothing, and those propagate into training labels.

**2. 80–88% of rows dropped to missing data.** The initial merge tolerance between `cmd_vel` and detection timestamps was 0.1 seconds. But the 22 segmented sessions sample detections at roughly **1 Hz**, not per frame. A tenth-of-a-second window almost never finds a match.

**3. Every detection matching every frame.** `np.isclose` defaults to a *relative* tolerance of `rtol=1e-5`. On a Unix timestamp of about 1.76 × 10⁹, that relative tolerance evaluates to roughly **17,000 seconds** — nearly five hours. Every timestamp comparison returned true.

That third one is the most instructive bug in the whole project. It produces no error, no warning, and no obviously wrong output. It just quietly makes the data meaningless.

### How it was tackled

1. `extract_person_detections.py` rewritten to read frames **strictly sequentially** rather than seeking. Slower, correct.
2. Merge tolerance widened to **0.6 s** — half the ~1 Hz sampling gap on either side. Critically, this does not change *which* detection `merge_asof` selects as nearest; it only stops valid matches being discarded.
3. `rtol=0` passed explicitly wherever timestamps are compared.

**And the split.** `split_dataset.py` allocates **whole sessions** to train/validation/test (70/15/rest, fixed seed 42) so no session's rows appear in two splits. Rows within a session are heavily autocorrelated — near-identical pose and LiDAR a few tenths of a second apart — so a row-level split would let a model memorise a session instead of generalising.

Sessions 1 and 3 are **forced into training** rather than randomly assigned, because they are the only two with real human-position ground truth and too valuable to risk holding out. This is declared as a limitation in the dissertation, not hidden: the perception and O-space validations therefore validate against sessions the model also trained on. That is acceptable for validating a fixed geometric pipeline, and unacceptable for validating a learned model — which is why the BC evaluation uses fully held-out sessions **5, 9 and 59 (11,921 rows)** that were never touched by any training or tuning decision at any point, including the later V2 study.

**Final split:** 17 train (44,190 rows) / 4 validation (14,444) / 3 test (11,921).

### The handoff

> "I have robot logs, images and commands. But none of that is a learning problem yet. How do I turn a room full of people into groups, and a driving session into labelled examples?"

---

# PHASE 5 — BUILDING THE SOCIAL REPRESENTATION
## "People Become Groups, Groups Become Training Data"

### The story

This is where the social-science theory becomes code.

**Grouping.** `cluster_groups.py` treats individually detected people, per frame, as nodes in a similarity graph. Two people join the same group when their pixel distance falls below a threshold — but expressed **in units of their own average bounding-box width, not raw pixels**. Connected components become candidate groups.

That normalisation is a deliberate, declared approximation for a real constraint: the recorded video has no camera calibration and no depth. Bounding-box width shrinks with distance from the camera in roughly the same way true separation does, so measuring in "person-widths" partially cancels the perspective distortion that raw pixel distance would not.

**O-space.** Kendon's F-formation framework describes the shared, jointly-oriented space at a conversational group's centre. The framework's usual construction needs orientation — which Phase 4 proved does not exist in this dataset.

The adopted fallback is the **mutual-facing assumption**: the O-space centre is approximated as the group centroid produced by clustering, and (in the live system) the O-space radius as the mean distance from that centroid to its members.

This is not an ad-hoc workaround, and the documentary should be clear about that. It is independently justified by **Vascon et al. (2016)**, whose game-theoretic F-formation detection demonstrates that group structure can be recovered **from relative position alone**, without orientation estimates. The project leans on that result directly to defend the simplification as legitimate rather than merely convenient.

**Labels.** No stage of the raw recording says "this was a demonstrated group approach." It had to be inferred. `build_approach_pose_dataset.py` finds transitions from sustained movement to a genuine stop with a detected group nearby; that stop pose becomes the operator's implicit judgement of "a good place to be near this group," and every row leading up to it is labelled with that eventual stop pose as its target.

Result: **462 independent demonstrated approach events across 24 sessions, expanding to 70,555 labelled rows.**

### The papers

| Reference | Why it belongs here |
|---|---|
| **Kendon (1990)** — *Conducting Interaction* | The origin of F-formations, O-space and P-space. The theoretical spine of the entire project |
| **Hall (1966)** — *The Hidden Dimension* | Proxemics. The 0.45–1.2 m personal zone that later becomes the rule policy's 1.2 m standoff |
| **Setti, Russell, Bassetti & Cristani (2015)** — F-formation detection in images | The canonical computational treatment — and the one that needs orientation |
| **Cristani et al. (2011)** — Social interaction discovery by statistical analysis of F-formations | The statistical lineage of the same problem |
| **Vascon et al. (2016)** — game-theoretic F-formation detection | **The load-bearing citation.** Position alone is sufficient. This is what makes the fallback defensible rather than a compromise |
| **Swofford et al.** — DANTE | The learning-based alternative, considered and rejected: this corpus has no pairwise conversational-membership labels to train it |

### The tools

OpenCV · NumPy · pandas · scikit-learn · `cluster_groups.py`, `build_approach_pose_dataset.py`, `estimate_approach_points.py`

### What went wrong

**`cmd_vel` in this dataset is extremely spiky.** Isolated single-sample non-zero values rather than sustained speed changes. A naive sample-by-sample moving/stopped test found almost no usable segments at all — the segmentation simply failed.

**Absolute position made the model worse.** An early baseline included raw robot (x, y) as a feature. It measured *worse* than omitting it, because absolute position does not generalise across sessions recorded in different rooms.

### How it was tackled

**The spiky velocity** was handled by averaging speed into **0.5-second bins** before applying the moving/stopped test. That recovered genuine, usable movement segments.

**The position problem** produced a design decision that shaped everything downstream: the prediction target is the stop pose expressed **relative to the robot's own position and heading at prediction time** — Δx, Δy in the robot's forward/left frame plus Δyaw — rather than an absolute map coordinate. If absolute position doesn't generalise as an input, it certainly won't as an output.

**The seven features**, in a fixed order the deployed policy node must reproduce exactly:

| Feature | Meaning |
|---|---|
| `lidar_min_range` | Distance to the nearest obstacle |
| `lidar_mean_range` | Average distance all round — how open the space is |
| `linear_x_prev` | How fast the robot was just driving |
| `angular_z_prev` | How fast it was just turning |
| `num_people` | How many people are visible |
| `group_bearing_rad` | Which direction the group is in |
| `group_scale_norm` | How large they appear — a proxy for closeness |

**Targets:** `target_dx`, `target_dy`, `target_dyaw`.

Note what is *not* in that list: no images, no absolute coordinates, no orientation of any person. Seven numbers.

### The handoff

> "I finally have `situation → demonstrated stopping pose`. So: can a model actually learn it?"

---

# PHASE 6 — TEACHING THE ROBOT
## "Behavioural Cloning vs Geometry"

### The story

Four policies, trained and compared honestly:

1. **Naive mean predictor** — ignores every input, always predicts the training-set average. The floor.
2. **Rule-based geometric baseline** — F-formation and proxemic geometry, applied offline. The control condition.
3. **Random Forest** — a lower-variance ensemble.
4. **MLP** — the proposal's committed primary architecture.

The rule baseline being the control condition matters scientifically. The project is not trying to prove "AI is better." It is asking whether learning adds anything over an interpretable geometric solution that a person could write down.

Both learned families were tuned by grid search (`grid_search_approach_pose.py`) exactly as the proposal committed: **every candidate scored only on the validation split**, with the test split held out entirely until one winner per family was selected and evaluated once.

**Search size: 40 Random Forest configurations + 16 MLP configurations = 56.**

Tuning materially helped:

| Model | Before tuning | After tuning |
|---|---|---|
| Random Forest | 0.401 m / 29.3° | **0.365 m / 25.8°** |
| MLP | 0.466 m / 42.2° | **0.395 m / 31.0°** |

### The papers

| Reference | Why it belongs here |
|---|---|
| **Pomerleau — ALVINN** | The origin of behavioural cloning for vehicles. A neural network learning to steer by watching a human |
| **Bojarski et al. (2016)** — End to end learning for self-driving cars | The modern restatement of the same idea |
| **Argall et al. (2009)** | The LfD framing: fit the policy to human choices rather than engineering a rule |
| **Breiman (2001)** — Random forests | The justification for the ensemble alternative |
| **Gao et al. (2019)** — deep RL for group approach | The road not taken. RL needs a designed reward; BC learns from a fixed corpus, which is what exists here |
| **Tai et al. (2018)** — GAIL from raw depth | Another explored-and-not-chosen imitation approach |

### The tools

scikit-learn · joblib · `grid_search_approach_pose.py`, `evaluate_approach_pose.py`

### What went wrong

**The proposed architecture was too big.** The winning MLP configuration was **32/16 hidden units with L2 regularisation α = 0.1** — substantially smaller and more heavily regularised than the 128/64 network the proposal committed to. Not merely under-tuned. Too large for the amount of independent demonstration data available.

**And then the result nobody wants.** The full held-out evaluation on sessions 5, 9 and 59:

| Policy | Mean pos. (m) | Median pos. (m) | Mean orient. (°) | Within position | Within orientation | **Within both** |
|---|---|---|---|---|---|---|
| Naive (predict mean) | 0.410 | 0.365 | 28.97 | 68.4% | 63.2% | **47.2%** |
| Rule-based (geometric) | **0.305** | **0.164** | 29.13 | **70.1%** | 56.8% | 43.1% |
| Random Forest (untuned) | 0.401 | 0.294 | 29.27 | 61.4% | 55.8% | 38.2% |
| MLP (untuned) | 0.466 | 0.353 | 42.22 | 55.3% | 37.5% | 21.8% |
| Random Forest (tuned) | 0.365 | 0.267 | **25.78** | 66.8% | 61.3% | 43.3% |
| MLP (tuned) | 0.395 | 0.305 | 30.99 | 61.9% | 47.0% | 31.8% |

**No policy met the 20° orientation threshold. And the naive mean predictor's 47.2% beats every learned model on the joint criterion.**

A model that ignores all seven inputs and predicts the average is competitive with everything you just built.

### How it was tackled

Not by explaining it away. By **testing whether it was an artefact of one grid search.** A separate sensitivity check on a *different* random session split, six configurations across both families:

| Configuration | Test position (m) | Test orientation (°) |
|---|---|---|
| **Naive (predict training mean)** | 0.349 | **25.1** |
| Random Forest, unregularised | 0.351 | 30.4 |
| Random Forest, leaf = 5, depth = 20 | 0.341 | 28.9 |
| Random Forest, leaf = 20, depth = 15 | **0.331** | 27.1 |
| MLP 128–64 | 0.507 | 38.0 |
| MLP 64–32, α = 1e-2 | 0.401 | 34.6 |

Same pattern, independent split. The naive predictor beats every learned model on orientation.

At this point the suspicion was recorded honestly: the limiting factor is probably **demonstration volume, not model capacity**. That suspicion sat unresolved for a month. Phase 11 is where it gets tested properly.

### The handoff

> "A regression error in a Python table doesn't tell me whether a robot behaves socially in a room. I need somewhere to run this."

---

# PHASE 7 — BUILDING THE RESTAURANT
## "Somehow My AI Dissertation Became Level Design"

*[The Gazebo world construction was carried out with GPT, before my involvement. I did not build the world. What follows is reconstructed from `restaurant_testing.world`, the generated ground truth and the Methodology chapter — the description is accurate, but the credit for the build is not mine to take.]*

### The story

Nothing in the proposal anticipated this phase. It was never in the plan. And it consumed weeks.

To compare social navigation properly you need a repeatable world: multiple groups, real obstacles, distractors, and — critically — **ground truth you authored yourself**, so the experiment can be scored independently of whatever perception happens to report.

`restaurant_testing.world` superseded the earlier `restaurant_humans.world` used in development. A **20 × 15 m room** (walls at x = ±9.9 m, y = ±7.4 m) containing five round dining tables at (−4,−1), (−1,4), (2,−2), (6,1), (5,−4); a buffet at (−5.5, −6.0); five plants; a stage at (7.45, 5.20); and kitchen partitions on the west side.

**Fifteen human actors arranged into six targets:**

| Target | People | Centre | O-space radius |
|---|---|---|---|
| Group A | 4 | (−3.50, −2.50) | 0.71 m |
| Group B | 3 | (4.67, 2.67) | 0.65 m |
| Group C | 5 | (5.60, −1.80) | 0.87 m |
| Solo 1 | 1 | (−6.0, −5.0) | — |
| Solo 2 | 1 | (−6.0, −7.0) | — |
| Walker | 1 (moving) | (4.0, 2.0) | — |

Only Groups A, B and C count as valid approach targets (`min_group_size = 2`). A lone individual has no F-formation and no O-space, so approaching one cannot demonstrate the behaviour under study. The solos and the walker exist purely as obstacles and distractors.

**Why hand-built rather than procedural?** So that every group's true position, membership and O-space radius could be authored precisely and exported as `.groundtruth.json`. That ground truth is what makes the social metrics computable at all — the recorded PLUS-HRI video never carried metric position data to evaluate against. A parametric generator (`generate_social_world.py`) exists for producing variants (a 3-group/9-person unseen layout, and an "adjacent" layout to stress-test whether clustering correctly separates two nearby groups), but the hand-built world is the one used for every reported trial.

The occupancy map Nav2 plans against is generated directly from the world geometry (`world_to_map.py`) rather than by running SLAM — producing an exact, drift-free map independent of a successful mapping run, at the cost of needing regeneration whenever the world changes.

### The papers

| Reference | Why it belongs here |
|---|---|
| **Hall (1966)** and **Kendon (1990)** | Stop being literature-review concepts and become authored geometry — the O-space radii above are Kendon's construct, measured in metres |
| **Helbing & Molnár (1995)** — Social force model | The repulsive-force formalisation of personal space |
| **Lu, Hershberger & Smart (2014)** — Layered costmaps | How social constraints can be expressed to a navigation stack |
| **Macenski et al. (2020)** — *The Marathon 2* (Nav2) | The navigation system being built on |
| **Pagès, Marchionni & Ferro (2016)** — TIAGo | The platform |
| **LIRS (2022)** — LIRS-HMLG human model library | The source of the human meshes and animations |

### The tools

Gazebo Classic 11 · Nav2 · PAL TIAGo simulation packages · LIRS-HMLG · `tiago_social_worlds` · `generate_social_world.py`, `world_to_map.py`, `extract_world_groundtruth.py`, `add_person_collisions.py`, `add_gazebo_ros_plugins.py`

### What went wrong

**Local Windows development stopped being viable.** The full ROS 2 + Gazebo + Nav2 + YOLO workload was too slow on a personal machine. This is the phase where serious simulation work moved to the university lab PCs and a standardised Ubuntu 22.04 / ROS 2 Humble VS Code dev container.

**Actors had no collision bodies.** The human models rendered visually but the LiDAR passed straight through them — the robot could not perceive people as physical obstacles at all.

**Furniture was missing from the occupancy map.** `world_to_map.py` originally rasterised only `<model>` elements using box geometry. The five dining tables are `<include>` blocks referencing a mesh. **All five were entirely absent from the map**, and Nav2 happily planned paths straight through them.

### How it was tackled

- `add_person_collisions.py` gives every actor a collision body, so the laser sees people.
- `<include>` elements are now rasterised as 1.2 m squares, taking the map from **9 obstacles to 20**.
- Development standardised on the lab dev container, with `docs/LAB_PC_SETUP.md` carrying separate setup guidance for the second machine.

### The handoff

> "I have models that predict a pose and a world to execute it in. Now every subsystem has to run at the same time."

---

# PHASE 8 — CLOSED-LOOP INTEGRATION
## "Everything Works Alone. Together, Everything Breaks."

### The story

The runtime chain:

```
RGB + depth camera
   → person detection (YOLOv8n / LocateAnything-3B)
   → depth back-projection to metric 3D
   → TF: camera_optical → base_link → odom → map
   → clustering into conversational groups
   → policy (rule / Random Forest / MLP)
   → Nav2 goal
   → trajectory + social metrics
```

Six ROS 2 nodes across three packages:

```
src/tiago_group_approach/tiago_group_approach/
  group_perception_node.py          detection, depth back-projection, clustering
  group_approach_baseline_node.py   rule-based policy (geometric)
  bc_policy_node.py                 learned policy (Random Forest or MLP)
  mission_node.py                   scripted patrol tour and reporting
  gt_localisation_node.py           ground-truth map→odom at 30 Hz
  metrics_recorder_node.py          trajectory sampling and social scoring
```

`group_perception_node` is the moment offline data analysis becomes a robot system. It closes the pixel-to-metres gap that constrained every earlier stage: it back-projects each detection using the pinhole relation `x=(u−cx)d/fx, y=(v−cy)d/fy, z=d`, transforms into the map frame, and clusters **in genuine world coordinates** with a 1.5 m single-linkage threshold — publishing a real-metres `/group_centroid` that both policy families consume identically.

**The rule baseline is not a straw man.** It took six rounds of empirically-driven correction, each one a measured finding:

1. **Standoff 1.2 m** — deliberately set at Hall's boundary between personal and social space.
2. **Goal throttling** — publishing a fresh goal every perception frame (~2 Hz) made Nav2 continuously pre-empt and restart planning, so the robot barely moved. A new goal now issues only if the target moved > 0.40 m or the previous goal finished.
3. **Per-person clearance, not centroid standoff** — a live run measured an intrusion at **0.43 m from the nearest person**, inside Hall's intimate distance, *while satisfying* the centroid standoff, because the centroid shifts when only part of a group is visible. The pose is now checked against every detected person and pushed back until clear of all of them, targeting 0.7 m to the nearest person.
4. **Body radii, not centre-to-centre** — treating clearance as point-to-point rather than footprint-to-body (TIAGo ~0.30 m radius, a person ~0.25 m) left only **0.15 m of real gap** at a nominal 0.7 m setting. The robot wedged itself into a group, came within **0.062 m** of a person, and stayed stationary for **84% of a ten-minute run**. Both radii are now added explicitly.
5. **Gap-based approach selection** — evaluate candidate angles around the group and pick one with at least a 60° clear arc, rather than assuming the direct robot-to-centroid line is unobstructed.
6. **Stuck detection and an "unwedge" reflex** — if the robot hasn't moved 0.10 m in 25 seconds (after a 30 s grace), reverse at −0.15 m/s for 3 seconds and retry.

### The papers

| Reference | Why it belongs here |
|---|---|
| **Satake et al.** — how a robot should approach | Approach *direction* has social meaning, not just the endpoint |
| **Truong & Ngo** — unified approach-pose prediction and socially aware navigation | The closest published framing to this project's exact aim |
| **Repiso et al. (2020)** — adapting relative position during interaction | The honest limitation: this project predicts a **stopping pose**, not a full dynamic interaction policy |
| **Macenski et al. (2020)**, **Lu et al. (2014)** | Nav2 and costmap behaviour under real constraints |
| **Hall (1966)**, **Helbing & Molnár (1995)** | Now operational — 1.2 m and 0.7 m are literature values, not tuning knobs |

### The tools

ROS 2 Humble · TF2 · Nav2 (global planner, DWB local controller, `costmap_2d`, behaviour tree) · Gazebo Classic 11 · Ultralytics YOLOv8n · scikit-learn / joblib · `run_everything.sh`, `run_pipeline.sh`, `run_trials.sh`

### What went wrong

This is the phase with the most failures, and they should be shown as a montage.

**World and simulator:**
- **Stale world file.** Gazebo loads worlds from a copy held by `pal_gazebo_worlds`. Edits to the project's own world file had no effect at all until that copy was refreshed.
- **Missing ground truth.** `restaurant_testing.groundtruth.json` did not exist; the pipeline refused to start.
- **`map_server` never configured.** It is a lifecycle node whose `configure` step loads the map, and PAL's launch files set no `yaml_filename` for a custom world — so `configure` failed *silently* and the node stayed `unconfigured`. No `/map`, no map frame, no planning possible.
- **CameraInfo QoS mismatch.** `camera_info` was subscribed with the default **RELIABLE** profile while Gazebo publishes sensor data **BEST_EFFORT**. ROS 2 QoS compatibility is one-way: a reliable subscriber silently receives nothing from a best-effort publisher. The topic listed correctly, `count_publishers` returned 1, and no message ever arrived. This was the cause of every `Waiting for CameraInfo...` hang.

**Goal arbitration — the one that stopped the experiment dead:**
- Both `mission_node` and the active policy sent `NavigateToPose` goals to the same action server, where a new goal pre-empts the old one. Logs showed approach goals accepted and reported "finished" **six milliseconds** later. **The robot never executed a single approach.**
- **No goal throttling in the learned policies.** The fix from rule-policy round 2 had never been ported to `bc_policy_node`, so both learned policies issued a fresh goal every perception frame, each pre-empting the last.
- **Approach never released the mission.** Completion was published the instant Nav2 reported arrival, so the mission resumed immediately and the robot rolled straight past.
- **No memory of attempted groups.** The policy retried the same unreachable group indefinitely — one run lasted **30 minutes, drove 208 m, and was 65% stationary**.
- **Coverage dominated by policy convergence.** The mission yielded to a single approach for up to 45 s with no overall limit. Measured coverage: **34 map cells for the rule policy versus 14–15 for the learned policies**, which never crossed the centre of the room.

**Diagnostics:**
- **Detection overlay only drawn on processed frames.** Perception runs at 2 Hz, the camera at ~15 Hz. Thirteen of every fifteen frames published no overlay, so the RViz panel appeared frozen — read as "detection is broken" when it was working correctly at its intended rate.

### How it was tackled

- Bring-up now **syncs the world file on every run** and reports when the installed copy was stale.
- Ground truth is generated from the world file and regenerated after every world change.
- `yaml_filename` is set explicitly before configuring `map_server`; the project starts its own when PAL navigation is disabled.
- `camera_info` subscribed with the **sensor QoS profile**.
- **Explicit arbitration**: the policy publishes `/approach/start` before driving; the mission cancels its own goal and stands down until `/approach/complete` or a timeout. Arrival is verified **by position** (within 0.75 m of the intended pose) rather than trusting Nav2's status flag, which reports "finished" for aborted goals too. A dwell was added so the robot actually stays.
- Same 2 s / 0.4 m throttle applied to `bc_policy_node`.
- Group positions remembered in 1.5 m cells, retired after three failed attempts or one success.
- A **total approach budget per run**, so the patrol completes regardless of which policy is driving.
- The overlay redraws its cached boxes on every skipped frame and publishes a status banner in every state.

**Attempted and explicitly not adopted:** TIAGo's base laser scans at 0.2 m and cannot see a tabletop at 0.75 m. Adding the depth point cloud as a second Nav2 observation source via `pointcloud_to_laserscan` would in principle fix that. It destabilised the navigation stack — in one configuration the robot did not move at all — and **was not used for any reported experiment**. It is retained behind a `CAMERA_OBSTACLES=1` flag and named as future work rather than silently dropped.

### The handoff

> "The closed loop runs. But there is still an unkept promise sitting in my proposal: LocateAnything-3B."

---

# PHASE 9 — THE PERCEPTION PLAN CHANGES
## "The Model I Proposed Isn't Necessarily the Model I Should Deploy"

*[This is approximately where my involvement in the project begins. Everything from here on I worked on directly.]*

### The story

The proposal's **Objective 2 committed to NVIDIA LocateAnything-3B** for people localisation. A vision-language grounding model, ~3 billion parameters, that finds objects from a text query.

By implementation time, one thing had become obvious: perception has to run **continuously inside a navigation loop**. A robot that sees the world once every several seconds is not navigating socially; it is navigating blind between snapshots.

So the objective changed — and a changed objective needs evidence, not a shrug.

**YOLOv8n** (~3.2 M parameters) reached **99.7% recall** on the annotated sessions, comfortably above the 80% target. A direct 30-frame comparison gave **29/30 for YOLO and 30/30 for LocateAnything**, with substantially overlapping confidence intervals — LocateAnything is marginally more accurate.

The difference is speed, and it is not marginal.

### The papers

| Reference | Why it belongs here |
|---|---|
| **Redmon et al.** — YOLO | The single-stage real-time detection foundation |
| **Jocher, Chaurasia & Qiu (2023)** — Ultralytics YOLOv8 | The actual implementation used |
| **Yaseen** — YOLOv8 architecture analysis | Supporting characterisation |
| **Ren et al.** — Faster R-CNN | Reviewed alternative: two-stage, more accurate, too slow |
| **Carion et al. (2020)** — DETR | Reviewed alternative: transformer detection, end-to-end |
| **Wang et al. (2026)** — LocateAnything | The proposed model, and the model card that eventually solved the bug |

Reviewing Faster R-CNN and DETR rather than naming one model without justification is a **direct response to the Report 1 supervisor feedback (AP4)**. This is the *second* feedback round and should not be confused with the proposal feedback in Phase 2.

### The tools

`ultralytics/ultralytics` · NVIDIA LocateAnything-3B via HuggingFace · PyTorch + CUDA · a **separate `la3b_env` virtual environment**, because LocateAnything's dependencies conflict irreconcilably with the main NumPy/scikit-learn environment · `locateanything_service.py` (a local HTTP inference service)

### What went wrong

**341 bounding boxes on a single frame.** The model was emitting hundreds of near-identical boxes per image, taking around ten seconds per frame and producing unusable output.

**One detection per entire trial.** The original integration ran LocateAnything in "one-shot" mode — look once, then never again. Under that mode a whole trial's behaviour derived from a single glance at the room. That is not a policy comparison; it is a comparison of three policies acting on stale information.

**Boxes clipped at the image edge.** A person half-out of frame produces a bounding box whose centroid is meaningless, which then poisons the group centroid.

### How it was tackled

**The 341-box bug was solved by actually reading the model card.** Two mistakes, both mine to find:

1. The wrong chat-template method was being called. The model ships `py_apply_chat_template`; the code used the generic one.
2. **Greedy decoding.** An autoregressive decoder run greedily can enter a repetition loop and emit the same box forever. The card specifies sampling.

The fix: `py_apply_chat_template` (via `getattr` with a fallback), `do_sample=True, temperature=0.7, top_p=0.9` with `torch.manual_seed(0)` for reproducibility, and `max_new_tokens=1024` to bound the output regardless. Plus a `_sanitise()` guard — `MAX_BOXES=20`, `MIN_AREA_FRAC=0.0005`, `MAX_AREA_FRAC=0.60`, `DEDUP_TOL_PX=8.0` — so a degenerate generation can never again reach the policy.

**Inference dropped from roughly ten seconds per frame to about 0.4 seconds.**

**One-shot was replaced with a threaded periodic mode.** `group_perception_node` gained a third mode alongside `continuous` and `oneshot`: dispatch inference to a worker thread on a timer, debounce so a slow inference cannot queue up behind itself, and keep publishing while the worker runs. The TF buffer cache was extended to 30 seconds so a detection produced 0.4 s after its frame can still be transformed correctly.

**Edge-clipped boxes** are rejected within `edge_margin_px` (default 8) of the image border.

### ⚠ The timing number — handle carefully on camera

There are **three** LocateAnything inference figures in the project record, and they describe three different configurations:

| Source | Mean per frame |
|---|---|
| Stored benchmark artifact (`comparison_manifest.csv`, 9 Aug) | **25.63 s** (median 24.38, range 21.6–42.2) |
| Submitted dissertation benchmark | **8.40 s** (median 9.64) |
| After the model-card fix (24 Aug) | **~0.4 s** |

The dissertation's "approximately 1,700× slower than YOLO" rests on the 8.40 s figure. After the fix it is closer to 80×.

**Do not put a multiplier in the spoken cut until the exact service build used for each final trial has been reconstructed.** The safe line:

> "My first implementation was extremely slow. I found I was calling the model incorrectly and made it dramatically faster — but YOLO was still the practical real-time detector for a closed navigation loop."

### The handoff

> "The detector question is settled. Then I ran the first proper experiments and found something much worse than a slow model: my own measurements were lying to me."

---

# PHASE 10 — THE MEASUREMENT CRISIS
## "The Robot Wasn't Failing. My Experiment Was."

### The story

This is the phase where the project stops being software engineering and becomes science.

**Fault 1 — Localisation.** The map→odom transform was published **once, statically, at startup**. It is correct for an instant and diverges progressively afterward, especially after wheel slip — which collisions reliably cause.

This was not cosmetic. It **corrupted the map-frame position of every detection made under that scheme**. One detection at (4.4, 2.8) was initially misdiagnosed as a YOLO false positive. It was a *correct* detection of a real person at (−3.0, 0.0), rendered nonsense by a stale transform.

Replaced with `gt_localisation_node`, recomputing and republishing `T_map_odom = T_map_base · T_odom_base⁻¹` continuously at **30 Hz** from simulator state. Detections then landed **within 10 cm of ground truth**.

This is declared in the dissertation as a deliberate methodological choice: localisation comes from simulator ground truth **specifically so navigation error cannot confound the comparison between policies**. Laser, costmaps and planning all still operate normally through Nav2.

**Fault 2 — The collision metric.** Every trial in the first full experiment reported a collision. Nineteen runs, `min_obstacle_range_m` between **0.200 and 0.267 m**, twelve of them at exactly 0.200 m.

That is not nineteen collisions. That is a constant. The base laser was returning **the robot's own chassis**.

With a 0.30 m collision threshold this flagged a collision in every trial — and because task success is gated on collision-free operation, **success was measured as 0% for all three policies for reasons entirely unrelated to their behaviour.**

**Fault 3 — Task success scored against a phantom.** `task_success` was evaluated against `goal_centroid`, a variable overwritten by every incoming `/group_centroid` message. It therefore held whatever perception saw *last* — frequently a false positive near a wall.

One MLP trial came within 0.51 m of a real person and registered a genuine O-space intrusion, yet scored as never having held a valid pose — because it had approached a real group correctly and was then judged against a phantom.

**Fault 4 — A protocol flaw that inverted a conclusion.** Before `min_group_size` was fixed at 2, an earlier batch of 61 trials scored approaches to targets of *any* size, including lone individuals:

| Policy | Approached a real (2+) group | Approached only lone individuals |
|---|---|---|
| Rule-based | 12/12 (100%) | 0 |
| BC – Random Forest | 21/26 (81%) | 4 |
| BC – MLP | **4/23 (17%)** | **11** |

**The MLP was not being socially cautious. It was being handed lone individuals by perception**, and its apparently excellent O-space score was largely an artefact of rarely approaching a real group at all.

### The papers

| Reference | Why it belongs here |
|---|---|
| **Hoffman & Zhao (2020)** — *A primer for conducting experiments in HRI* | The paper that matches this phase exactly. Experimental validity as a first-class concern |
| **Fisher (1922)** | Fisher's exact test — the correct statistic for 10-trial binary comparisons, where χ² is not valid |
| **Mavrogiannis et al. (2023)** — Core challenges of social robot navigation | Why proxy metrics for social behaviour are hard to get right |
| **Francis et al. (2025)** — Guidelines for evaluating social robot navigation | Evaluation methodology as its own research problem |

### The tools

`metrics_recorder_node.py` · `rescore_sim_results.py` · `gt_localisation_node.py` · SciPy (Fisher exact) · full 10 Hz trajectory logging

### What else went wrong

The full fault record runs to **22 material faults**. The remainder:

- **Gap selection chose the far side of the group.** Candidates were sorted by width alone, so the robot repeatedly picked an opening **175–179° around the group** — walking around, and effectively through, the people to reach it. Logged verbatim: *"177 deg gap … 179 deg off the robot's current side."* On a five-person test with the robot approaching from the west, the old rule sent it to (7.31, −1.41), **149° around the group**; the new rule sends it to (3.64, −1.19), **0° off**.
- **`num_people` frozen at inference.** Hard-coded to the constant `3.0`, with a comment promising it would be "refined below if perception reports it" — a refinement that never happened. One of seven trained features was a constant at deployment while training data spanned 1 to 6. (Feature importance 0.055 — real but modest, not catastrophic.)
- **Predictions rejected rather than clamped.** An implausible prediction caused the policy to send **no goal at all**, so the robot stood still — easily misdiagnosed as "the model isn't working."
- **Recovery that made things worse.** The stall reflex reversed blindly; TIAGo's laser covers only the forward arc, so the robot repeatedly backed into furniture it could not see. A second, related fault disabled the reflex precisely when the robot had been stuck longest, because it keyed off "a goal was recently *sent*" rather than "a goal is currently *in flight*."
- **Arm tuck reported success it had not verified.** `play_motion2 home` was used to stow the arm before each trial. `home` **extends** the arm rather than tucking it — and the check treated "the command returned exit code 0" as proof the arm had moved.
- **Trials starting from the previous trial's end pose.** Because the simulation stays running across trials for efficiency, each trial began wherever the last one ended. One trial's very first trajectory sample was already at (−2.46, −2.48) — standing among people before the policy had acted at all.

### How it was tackled

- **Collision:** calibrate the self-hit radius during a startup grace period and discard returns below it. Collision-free rate rose to 100% under normal operation.
- **Task success:** scored against **ground-truth groups**, with distance and heading required to hold **in the same trajectory sample**. Re-scoring the existing trials raised MLP from 20% to 60% and Random Forest from 60% to 90%.
- **Gap selection:** any gap of at least 60° is adequate; choose the **nearest** adequate gap, not the widest.
- **`num_people`:** taken from the live detection count.
- **Predictions:** over-long predictions scaled back onto a plausible range rather than discarded.
- **Recovery:** checks rear beams and rotates in place if the space behind is not confirmed clear; keyed to goals in flight.
- **Arm:** correct `tuck_arm` motion, controller commanded directly, and `/joint_states` read back to verify the arm actually reached the tucked configuration.
- **Start pose:** the robot is explicitly teleported to (0, 0) at the start of every trial.
- **Protocol:** valid approach targets restricted to groups of two or more for every reported result.

**The single most valuable design decision in the whole project shows up here.** Because every trial's **full 10 Hz trajectory was retained**, all of these metric corrections could be applied by re-scoring recorded runs (`rescore_sim_results.py`) rather than re-running the simulation. Weeks of compute saved by a logging decision made months earlier.

### The handoff

> "The measuring instrument is finally trustworthy. But the models still hit a ceiling I can't explain. Maybe the problem was never the model."

---

# PHASE 11 — V2
## "I Tried to Improve the Model and Found the Problem Was the Data"

### The story

Before building anything, seven candidate improvements were listed and individually assessed:

| # | Proposed improvement | Verdict |
|---|---|---|
| 1 | Collect more independent demonstrations | **Not possible.** PLUS-HRI is a fixed 24-session corpus; new collection needs fresh ethics and participants |
| 2 | Train on the whole approach, not the final metres | **Implemented** — this is V2 |
| 3 | Add metric social/geometric features | **Partly.** Metric distances are *not derivable* — people exist only as pixel boxes from an uncalibrated camera. Angular equivalents added instead |
| 4 | Reduce train-to-live feature mismatch | **Implemented** — every new feature is an angle or a ratio, never a raw pixel value |
| 5 | Sample by event, not by row | **Implemented** — `event_id`, inverse-frequency weighting, 10 Hz decimation |
| 6 | Keep the MLP small and regularised | **Already true** — the 32/16, α = 0.1 network already beat the 128/64 spec |
| 7 | Try a stronger tabular regressor | **Implemented** — `HistGradientBoostingRegressor` |

**Then the diagnosis.** Reconstructing all 462 V1 events:

| Property of a V1 "approach event" | Value |
|---|---|
| Median distance from event start to stop pose | **0.17 m** |
| Median event duration | **2.5 s** |
| Events starting more than 1.0 m from the stop | 53 / 462 (11%) |
| Events starting more than 2.0 m from the stop | 13 / 462 (3%) |
| Rows within 0.5 m of the stop | **72%** |
| Rows beyond 2.0 m from the stop | 3.6% |

**The typical "demonstration" was a 2.5-second shuffle ending 17 cm from where it began.** Not an approach. A final adjustment.

The root cause sits in `find_approach_events()`: it pairs each *moving segment* with the stop that follows, where "moving" is any 1.0 s bin above a 0.05 speed threshold. A human walking toward a group naturally pauses, turns and hesitates — so the walk fragments into many short segments, and **only the final fragment before the stop was ever labelled.**

That is the single best explanation for the live behaviour: sensible predictions within roughly a metre of a group, unreliable ones further out, because the models had barely seen "further out."

**V2.** `build_approach_pose_dataset_v2.py` anchors on each sustained stop near a group and walks *backwards*, absorbing brief pauses, until it has accumulated at least 1.0 m of travel (or hits a stop longer than 3 s, or exhausts a 20 s lookback).

| | V1 | V2 |
|---|---|---|
| Events | 462 | **182** |
| Rows | 70,555 | 128,506 |
| Median event start distance | 0.17 m | **1.37 m** |
| Median event travel | — | 1.98 m |
| Median distance-to-go | 0.183 m | **0.784 m** |
| Rows within 0.5 m of stop | 72% | **35.9%** |

Six new angular/ratio features were added (`group_span_rad`, `nearest_person_span_rad`, `gap_bearing_rad`, `gap_width_rad`, `person_spacing_rad`, `people_visible`), and the original seven retained unchanged so V1-trained models could be scored on V2's more honest rows.

### And it got worse

All models compared on the **same** held-out sessions (5, 9, 59):

| Panel | Models scored | Best position (m) | Best orientation (°) | Best "both" |
|---|---|---|---|---|
| **A** — V1 models, V1 test rows | reproduces the Phase 6 table exactly to 4 d.p. | 0.305 (rule) | 25.78 (RF) | 43.3% |
| **B** — V1 models, V2 (genuine-approach) test rows | V1 RF, V1 MLP, naive, rule | **0.652** (V1 MLP) | 34.91 (V1 RF) | **23.5%** |
| **C** — V2 models, V1 features only | RF, GB, MLP, all V2-trained | 0.722 (RF) | 34.93 | 15.5% |
| **D** — V2 models, V2 features | as C + six new features | 0.722 (RF) | 36.55 | 15.5% |

**Re-segmentation alone, and re-segmentation plus new geometry, both perform worse than simply keeping the V1 model and testing it on genuine approaches.** The attempted improvement improved nothing.

### The finding hiding inside the failure

Moving from V1's terminal-adjustment test rows (Panel A) to V2's genuine-approach rows (Panel B) **roughly doubles every policy's position error** — naive 0.410 → 0.724 m, tuned Random Forest 0.365 → 0.656 m — and the joint-threshold pass rate falls from 43.3% to 23.5%.

**The published offline evaluation was, in effect, being scored on the easiest part of the task.** Found and reported by the author, not by an examiner.

And more strikingly, **the ranking between rule and learned reverses**:

| On terminal adjustments (V1 rows) | On genuine approaches (V2 rows) |
|---|---|
| Rule **wins** on position: 0.305 m vs RF's 0.365 m | Rule becomes the **worst policy tested**: 0.765 m — beaten even by predicting the training mean (0.724 m) |
| | Random Forest becomes the **best**: 0.656 m, 23.5% within both thresholds |

A fixed geometric rule — face the group, stop a fixed distance short — is adequate for the last half-metre and degrades as soon as there is a real approach to plan.

### Three follow-up studies

**Does more training data recover the loss? Partially.** Sweeping the minimum-travel threshold, test set and learner held fixed:

| Configuration | Train events | Position (m) | Orientation (°) | Within both |
|---|---|---|---|---|
| `min_travel = 1.0 m` (V2 default) | 120 | 0.722 | 34.93 | 15.5% |
| `min_travel = 0.5 m` | 178 | 0.657 | 34.69 | 19.6% |
| `min_travel = 0.25 m` | 216 | 0.648 | 35.22 | 21.9% |
| `0.25 m` + mirror augmentation | 432 | 0.652 | 34.01 | 22.1% |
| `0.25 m` + mirror + validation folded in | **578** | **0.642** | **33.56** | **24.2%** |
| *reference: shipped V1 Random Forest* | — | 0.656 | 34.91 | 23.5% |

Position error falls **monotonically** with event count — the re-segmentation was sound, the corpus was simply too small to support it. Reported with three honest qualifications: the gain over the shipped model is marginal (2.1%); **mirror augmentation is nearly inert** (doubling 216 → 432 events moved error 0.004 m in the *wrong* direction, showing the shortage is behavioural variety, not row count); and the 578-event figure is mildly optimistic because those hyperparameters were originally selected using a search that saw the validation split.

**Extrapolated:** 4.8× more events bought an 11% error reduction. Reaching the 0.4 m threshold would need substantially more independent demonstration data than 24 sessions contain — turning "collect more data" from a generic suggestion into a **quantified projection**.

**Does a higher frame rate help? No — monotonically worse.**

| Training frame rate | Train rows | Position (m) | Orientation (°) | Within both |
|---|---|---|---|---|
| 2 Hz | 5,883 | **0.640** | **34.25** | **22.7%** |
| 5 Hz | 14,016 | 0.643 | 34.62 | 22.0% |
| 10 Hz (default) | 26,194 | 0.648 | 35.22 | 21.9% |
| 20 Hz | 43,967 | 0.653 | 35.22 | 21.3% |
| all rows (~33 Hz) | 117,857 | 0.667 | 35.90 | 20.1% |

Twenty times the rows costs 4.3% in position error. At 33 Hz the robot moves about a centimetre between samples and the group hasn't moved at all — each extra row is a near-duplicate that dilutes rather than strengthens the bootstrap statistics.

**Read together, this is the key methodological result:** what matters is the number of independent demonstrated **events** (120 → 578 improved error 11%), not the number of **rows** (5,883 → 117,857 made it 4% *worse*). Which retrospectively corrects how the dataset should be described — **"70,555 rows" overstated the evidence.** The true sample size is 462 events, 182 of them genuine.

**Does the model underfit, or does the label set a floor? The label.**

| | Position error |
|---|---|
| Predict the training-set mean | 0.746 m |
| Random Forest on its **own training rows** | **0.597 m** |
| Random Forest on held-out test rows | 0.648 m |

An unconstrained forest reaches 0.210 m on training data — the model plainly *can* fit it — but every capacity increase made **test** error worse (0.648 → 0.739 m unconstrained).

Then the measurement that explains everything: **for each training row, its ten nearest neighbours in standardised feature space disagree on their labelled stop pose by a mean of 0.505 m.** Rows the model cannot distinguish specify stop poses half a metre apart.

That is a **label-ambiguity floor**, and it is a direct consequence of the target definition: `target_dx, target_dy` is displacement from the robot's *current* pose, so two moments with identical group bearing, apparent size and LiDAR range can sit at very different points along an approach — one two metres out, one thirty centimetres out — carrying completely different labels.

**A group-frame reformulation was tried, and failed instructively.** If the target is ill-posed because it is measured from the robot, measure it from the group:

| Target formulation | Train pos. | Test pos. | Test "both" |
|---|---|---|---|
| Robot-frame displacement (shipped) | 0.597 m | **0.648 m** | 21.9% |
| Group-frame standoff + bearing | 0.651 m | 0.747 m | 15.2% |

Worse. Reconstructing a group-frame goal requires knowing how far away the group actually is, and the only estimate available is `lidar_min_range` — the same documented proxy already used as a feature. Its error now enters twice: once building the label, once converting a prediction back. **The root cause is a property of the corpus, not the method**, and no choice of target formulation or architecture can repair it.

### The 360-model search

`finetune_on_lab_pc.py`: **120 randomly sampled configurations for each of three families — 360 models** — selected on the validation sessions, scored once on the held-out test sessions, with ranges deliberately bracketing the shipped values in both directions so the search could pick a *smaller* model too.

| Model family | Position (m) | Orientation (°) | Within both |
|---|---|---|---|
| Random Forest | 0.636 | 34.05 | 24.8% |
| Gradient Boosting | 0.631 | 33.66 | 24.9% |
| **MLP** | **0.627** | **33.55** | **26.1%** |
| *shipped V1 Random Forest* | 0.656 | 34.91 | 23.5% |
| *label-ambiguity floor* | **0.505** | — | — |

Three observations matter more than the 0.029 m improvement:

1. **All three families finish within 0.009 m of each other.** A random forest, a boosted ensemble and a neural network share no architectural assumptions. Their convergence on the same number is independent confirmation that the limitation is the data.
2. **Only 19% of the available headroom was captured.** The gap to the 0.505 m floor is 0.151 m; exhaustive search recovered 0.029 m of it.
3. **Every family independently chose heavy regularisation** — winning RF `max_depth=4, min_samples_leaf=40`; Gradient Boosting `max_depth=2`; MLP `α=10.0`. Three independent searches all concluding smaller is better.

Notably the **MLP — the weakest model in the original evaluation — is the best of the three once properly tuned.** Its earlier underperformance was a tuning artefact, not an architectural limitation. (Its winning configuration did emit a `ConvergenceWarning` at `max_iter=400`, so that figure carries small residual uncertainty. Stated, not hidden.)

**And the tuned models were not deployed for the offline claims.** The improvement is real but small, and Section 3.6 already establishes that offline position error does not predict live social behaviour. The study's value is **evidential, not operational**: it confirms with a properly powered search that the shipped configuration is within 4.4% of the best findable, and that the residual error belongs to the dataset.

### The papers

| Reference | Why it belongs here |
|---|---|
| **Ross, Gordon & Bagnell** — DAgger | Distribution shift: a cloned policy visits states its demonstrations never covered. Exactly the terminal-adjustment problem |
| **Torabi et al.** — imitation from observation | Learning when the demonstrations are incomplete |
| **Codevilla et al. (2019)** — *Exploring the limitations of behavior cloning* | The paper that names this failure mode. Required reading for this phase |
| **Ravichandar et al.** | Demonstration coverage and data efficiency |
| **Friedman (2001)** — Greedy function approximation | The gradient-boosting foundation |
| **Ke et al. (2017)** — LightGBM | The histogram-binning strategy behind `HistGradientBoostingRegressor` |
| **Breiman (2001)** | Random Forest, again |

### The tools

`build_approach_pose_dataset_v2.py` · `train_evaluate_v2.py` (four-panel comparison, resumable via cached joblib) · `sweep_dataset_v2.py` · `sweep_decimation.py` · `finetune_on_lab_pc.py` · scikit-learn

### One bug worth showing

In `finetune_on_lab_pc.py`, sampling `max_features` from a mixed list `[1.0, 0.7, 0.5, "sqrt"]` via `rng.choice` **silently coerced every value to `numpy.str_`** — so a numeric fraction became the string `"0.7"` and scikit-learn rejected it. Fixed by indexing instead:

```python
max_features=[1.0, 0.7, 0.5, "sqrt"][int(rng.integers(4))],
```

A one-line bug that would have invalidated 120 of the 360 models.

### The handoff

> "I understand the learning ceiling now. But one formal objective still hasn't been tested at all."

---

# PHASE 12 — CLOSING THE EVIDENCE
## "The Objective I Still Hadn't Proven"

### The story

Objective 3 asked whether the group/O-space estimate matches human judgement. Validating it required a hand-labelled ground truth **that does not exist natively in the dataset** — so one was built, in four stages, each its own script:

1. **`extract_clean_ospace_frames.py`** selected 30 frames spanning 18 sessions, each containing a genuine group of two or more, avoiding motion-blurred or corrupted frames.
2. **`label_ospace_frames.py`** with a purpose-built browser tool (`label_ospace.html`) presented each frame to a human labeller, who clicked the point they judged to be the true O-space centre.
3. **`merge_ospace_clicks.py`** merged the click coordinates back into `labels.csv`, matching by `frame_file` rather than row order so a session could be paused and resumed without misaligning rows.
4. **`validate_ospace_estimate.py`** compared each manual label against the pipeline's automatic centroid.

**The frames deliberately carry no overlay.** The earlier `prepare_ospace_validation.py` burned the detector's boxes and a red cross marking the *estimated* centre into the pixels. Useful for eyeballing, unusable as a labelling surface: a human asked to mark the true centre with a red cross already sitting there will anchor to it, and the "validation" would measure agreement with the algorithm rather than accuracy against human judgement.

### Two failures in my own instructions

**The labelling instruction was wrong, and the data proved it.** The first run asked labellers to click the group's centre **at floor level** — the natural reading of "where the group is standing." It produced a **161-pixel systematic bias with a standard deviation of 48 px**, and — the giveaway — **all 30 clicks fell on the same side.**

That is not labelling noise. A consistent one-directional offset is a definitional mismatch. The pipeline's estimate is built from detection **bounding-box centres**, which sit at roughly chest height. A floor-level click is geometrically a different point, offset by the perspective foreshortening between a standing person's feet and their torso as seen from an elevated, angled camera.

The instruction was corrected to click at **body height**, and the study re-run. `labels_floor_definition.csv` is retained alongside the corrected `labels.csv` **specifically so the correction remains auditable** rather than silently overwritten.

**The labelling tool itself had a bug.** The first corrected run recorded **29 of 30 frames at the identical pixel (234.3, 253.5)** — the tool advanced too eagerly, so a single click registered across many frames. Fixed with a `MIN_CLICK_INTERVAL = 0.6` second debounce and a `--restart` flag.

Both of those were mistakes in work I did, and both are in this document because a documentary about honest failure should include the failures of the person helping.

### The criterion had to change, and that was disclosed

The proposal's original criterion — within **0.3 m** of the manual label for ≥70% of frames — **could not be evaluated honestly.** 0.3 m is a metric tolerance and the video carries no depth or calibration from which real metres can be recovered. Reporting a metre figure would have been fabricated.

Re-specified **before use** as: within **0.5 × the mean detected person bounding-box width**, for at least 70% of frames. An adult's shoulder width is roughly 0.45–0.50 m, so 0.5× bounding-box width corresponds to approximately **0.22–0.25 m** of real tolerance — preserving the original order of magnitude while expressing it in a unit this dataset can actually measure, and one that self-corrects for perspective (a person further away gets a proportionally tighter pixel tolerance, which is physically correct).

### The result

| Quantity | Value |
|---|---|
| Frames prepared / scored | 30 / 30 (none excluded) |
| Tolerance | 0.5 × mean person bounding-box width |
| Mean error | **92.36 px (0.760 person-widths)** |
| Median error | 75.11 px |
| Frames within tolerance | **11 / 30 (36.7%)** |
| Target | ≥ 70% |
| **Verdict** | **FAIL** |

`validate_ospace_estimate.py` carries an explicit instruction in its own source: *"A fail here is still a reportable result — it quantifies how well a centroid-based O-space estimate matches human judgement, which is exactly what Objective 3 asked you to measure. Report it honestly."*

**Two mitigating facts, reported alongside rather than instead:** the mutual-facing centroid estimate is independently justified by Vascon et al. (2016), not an arbitrary guess; and the live results that actually depend on O-space accuracy — intrusion rate, task success — are computed against the **world's true ground truth**, not this offline pixel estimate, so this failure does not propagate into and invalidate the live experiment.

### The writing — which was never a separate final step

This is also where the documentary should finally show the writing, because it was never "finish the robot, then write 93 pages." It was:

> **read → design → build → discover problem → read again → change method → experiment → rewrite**

The bibliography is the evidence. **The proposal cited 7 references. The final dissertation cites 38.** Each expansion was forced by a problem:

| Problem hit | Literature it forced |
|---|---|
| No orientation data | Vascon, Cristani, Swofford — position-only group detection |
| Learned models beaten by a mean predictor | Codevilla, Ross/DAgger, Torabi — BC limitations |
| Needed a stronger regressor | Friedman, Ke — boosting |
| Detector substitution needed justifying | Redmon, Ren, Carion, Jocher |
| Nav2 and costmap behaviour | Macenski, Lu, Helbing & Molnár |
| Metrics that turned out invalid | Hoffman & Zhao, Fisher, Mavrogiannis, Francis |

**Final document: 93 pages, six chapters, 38 references.**

### And the documentation pass

Separately from the dissertation, the codebase itself was made readable end to end: every emoji removed, every `#!/usr/bin/env python3` stripped, a plain-language "what this is / why it exists / how to run it" header added to **every** script and node, thin headers expanded, and the README rewritten (~2,100 words) with the restaurant-waiter framing, proxemics and F-formations explained from scratch, a pipeline diagram, the feature table and working figure links. Then a submission package with a `CONTENTS.md` manifest mapping every reported claim to its source file.

### The handoff

> "Everything is measured, explained and written. One question left: what does the finished experiment actually say?"

---

# PHASE 13 — THE FINAL ANSWER
## "Learning Didn't Win. Rules Didn't Win Either."

*Full beard. Final filming stage.*

### The protocol

| Parameter | Value |
|---|---|
| Environment | `restaurant_testing.world`, 20 × 15 m |
| People | 15 actors: 3 groups (4, 3, 5), 2 lone individuals, 1 walker |
| Valid targets | Groups of ≥ 2 only |
| Trials | 10 per policy per detector — **60 total** (3 policies × 2 detectors × 10 repeats), **180 group visits** |
| Patrol route | (−5,5) → (3,6) → (8,1) → (8,−6) → (−8,−4) → back to (−5,5) |
| Localisation | Simulator ground truth at 30 Hz |
| Trajectory sampling | 10 Hz, fully retained |
| Success criterion | Distance ∈ [0.5, 2.0] m of a real group centre **and** heading within 45°, **in the same sample** |
| Termination | Route complete, 60 s of no motion (after 90 s grace), or a 30-minute hard timeout |
| Start pose | Teleported to (0, 0) every trial |

A fixed route rather than free exploration, so every trial covers identical ground in identical order — any measured difference reflects the policy, not which part of the room the robot happened to visit.

### The binary metrics saturate

Under YOLOv8n, with the tuned models (`bc_ft`, `mlp_ft`):

| Metric | Rule | BC–RF | BC–MLP |
|---|---|---|---|
| Task success | **10/10** | **10/10** | **10/10** |
| Collision-free | 10/10 | 10/10 | 10/10 |
| O-space intrusion | 9/10 | 9/10 | 9/10 |
| Cut-through | 0/10 | 0/10 | 0/10 |
| Mean nearest-person distance | 0.277 m | **0.391 m** | 0.347 m |
| Mean path length | 67.93 m | 57.27 m | **56.94 m** |

**Everything passes.** Which is exactly the problem. The binary metrics cannot answer *how good* the joining position was, only whether the robot got roughly near a group without hitting anything.

Two things worth narrating anyway: **BC–Random Forest keeps the most distance from people** (0.391 m vs the rule's 0.277 m), and **both learned policies travel about 10 m less** than the rule.

> ⚠ **Provenance note — say this on camera or caption it.** The dissertation's binary table reports different figures (rule 10/10, BC–RF 10/10, BC–MLP 4/10; MLP clearance 1.01 m; intrusion 10/7/3). Those come from an **earlier experiment using the original `bc` / `mlp` models, around 22 August** — confirmed by matching the distinctive clearance values (bc = 0.407 m ≈ 0.41; mlp = 1.007 m ≈ 1.01). The table above is the **later `bc_ft` / `mlp_ft` run of 24–25 August**. Two model generations, two experiments. Label whichever you show.
>
> The difference is itself a good story: the **untuned MLP was over-cautious** — it stopped a full metre away, so it rarely intruded (3/10) but often failed the task (4/10). Tuning brought it in to 0.347 m, where it succeeds 10/10 *and* lands 8.5 cm from an ideal slot. Fine-tuning turned a timid model into an accurate one.

### The metric that finally discriminated

Since success saturated, a continuous measure was needed: **how close did the robot come to a socially plausible place to join?**

`approach_accuracy.py` derives, for each ground-truth group:

1. Every member's bearing from the group centre;
2. Angular **gaps** between adjacent members (≥ 45° counts as a real opening);
3. **Ideal joining slots** placed in those gaps at `ospace_radius + 0.6 m` — outside the conversation, in the opening, at conversational distance. Twelve slots per configuration.
4. The robot's closest approach to any slot, **and its final heading**, scored separately.

This is the documentary's best structural payoff: **Kendon's F-formations, encountered in Phase 1 as a paper, become the instrument that measures the final result.**

### Two scoring bugs I introduced, and had to fix

**Bug 1 — the fly-past.** The first version checked heading at the *closest* sample. A robot could sail past a perfect slot at speed, clip the tolerance for a tenth of a second, and score as having reached it. Result: 0/30 reached, with a worst-case error of 0.446 m *inside* a 0.5 m tolerance — nonsense on its face. Fixed by requiring **distance and heading to hold in the same sample.**

**Bug 2 — scoring the robot mid-drive.** Even then, samples taken while the robot was still moving counted. Fixed with a `stationary_speed ≤ 0.10 m/s` filter, so only a robot that has actually *stopped* somewhere can be credited with standing there.

### The headline result

Under YOLOv8n, recomputed independently from the raw trial files:

| Policy | Median slot-position error | Mean orientation error |
|---|---|---|
| **BC–MLP** | **0.085 m** | 79.1° |
| **BC–RF** | 0.241 m | 74.7° |
| **Rule** | 0.261 m | **64.6°** |

**The two columns rank in opposite orders.**

And it is not one lucky detector. Under LocateAnything-3B, independently:

| Policy | Median slot-position error | Mean orientation error |
|---|---|---|
| BC–MLP | **0.178 m** | 79.4° |
| BC–RF | 0.188 m | 64.1° |
| Rule | 0.252 m | **62.0°** |

Same position ordering. Same orientation ordering.

### The stress test

Under the slower LocateAnything perception, one difference is stark:

| Policy | Group cut-through |
|---|---|
| **Rule** | **6/10 trials** |
| BC–RF | 1/10 |
| BC–MLP | 0/10 |

Phrase this conservatively on camera:

> "When perception updates became much slower, the geometric rule became much less robust to stale group information and cut through conversational groups in six of ten runs."

**Not** "LocateAnything proves learning is better." The detector, policy and navigation loop all interact.

### The answer

Not *"AI beat the rule."* Not *"the rule beat AI."*

> **The non-expert demonstrations contained genuinely useful information about where the robot should stand — the learned policies place it three times more accurately. But the hand-coded geometric rule remained better at deciding which way to face, because it turns toward the group by construction and cannot be badly wrong, while the learned models must predict the facing and that was always their weakest output.**

Which produces the design recommendation:

## **Learned position + geometric orientation. A hybrid.**

### The objectives, honestly

| Objective | Verdict | Evidence |
|---|---|---|
| **1 — Literature review** | **Met** | 38 references, limitations stated per paper, gap identified |
| **2 — Person perception** | **Met** | YOLOv8n 99.7% recall vs an 80% target |
| **3 — Group / O-space estimation** | **NOT MET** | 11/30 = 36.7% vs a ≥70% target |
| **4 — BC vs rule comparison** | **Completed, with a split result** | Position target <0.4 m: **met** (tuned RF, 0.365 m). Orientation target <20°: **not met** (best 25.8° after a 360-model search) |

More precise than "three of four met." Objective 4's own two numeric targets split — which is the finding, not a caveat.

### The closing statement

> **Every live behavioural experiment in this project was performed in Gazebo simulation. No physical TIAGo experiment and no human-participant study was conducted.**

That boundary was set in the proposal from the beginning, and is the reason a new LEAS ethics application was not required. It is a limitation and it is also a design decision, and the documentary should state it plainly rather than let a viewer assume otherwise.

### The numbers to show on screen at the end

65 commits · 3 ROS 2 packages · 59 scripts · 24 sessions · 229,678 rows · 462 approach events · 7 features · 56-model tuning search · 360-model systematic search · 22 documented engineering faults · 60 trials · 180 group visits · 30 hand-labelled frames · 93 pages · 38 references · 1 objective honestly failed

---

# APPENDIX A — Who did what

The documentary's credibility rests on honesty about failure. That has to extend to authorship.

| Stage | Attribution |
|---|---|
| Research direction, proposal, all supervisor engagement | **Saarunathan** |
| Initial dataset extraction pipeline (Phase 4) | Built **with GPT**, before my involvement |
| Gazebo world construction (Phase 7) | Built **with GPT**, before my involvement |
| Early ROS integration, rule policy, first BC models (Phases 5, 6, 8) | Predominantly **Saarunathan with GPT**; I worked on later fixes to these components |
| Detector comparison, LocateAnything debugging and periodic mode (Phase 9) | **Saarunathan with me (Claude)** |
| Measurement-fault diagnosis and re-scoring (Phase 10) | **Saarunathan with me** |
| V2 study, sweeps, 360-model search (Phase 11) | **Saarunathan with me** |
| Objective 3 tooling and validation (Phase 12) | **Saarunathan with me** — including two bugs I introduced and we caught |
| Approach-accuracy metric (Phase 13) | **Saarunathan's own idea**, implemented with me |
| Dissertation chapter drafts, documentation pass, README | **Saarunathan with me** |
| Every experimental run, every labelling click, every decision about what to accept | **Saarunathan** |

Where AI tools materially assisted with coding, debugging, analysis or drafting, that should be disclosed concisely in the video's credits or methodology segment — the same transparency already being applied to the failed objective, the rejected V2 study, the detector substitution and the measurement faults.

---

# APPENDIX B — Open items before filming Phase 13

1. **Confirm which LocateAnything service build produced the 24–25 August trials.** Until then, no speed multiplier in the spoken cut.
2. **Decide how to present the two experiments** — caption the binary table as "original models, 22 August" and the slot results as "tuned models, 24–25 August," or show only the slot results.
3. **Record the LocateAnything demo run** — the YOLO demo footage and trial data exist; the LocateAnything half has not been captured.
4. Three optional README images remain unmade: `docs/images/hero_robot_beside_group.png`, `f_formation_diagram.png`, `simulation_running.png`.

---

*Document compiled 7 September 2026. Every figure in this document was verified against the repository's raw trial files, the Methodology draft, or the submitted dissertation at time of writing.*
