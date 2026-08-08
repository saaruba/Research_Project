# Learning Socially Appropriate Group-Approach Behaviour for a TIAGo Robot from Non-Expert Human Demonstrations

MSc Robotics dissertation project — University of Lincoln, School of Computer Science.

A TIAGo robot learns, via Behavioural Cloning, where to stop and which way to face when approaching a small conversational group, trained on the PLUS-HRI dataset of 24 non-expert teleoperation sessions, and compared against a rule-based geometric baseline.

---

## Quick start

```bash
git clone https://github.com/saaruba/Research_Project.git
cd Research_Project

chmod +x setup_environment.sh
./setup_environment.sh --cpu-torch      # or plain ./setup_environment.sh
```

Verify an existing install without changing anything:

```bash
./setup_environment.sh --check
```

Install into an isolated virtual environment instead of system Python:

```bash
./setup_environment.sh --venv
```

### What the setup script does *not* install (deliberately)

| Not installed | Why | How to get it |
|---|---|---|
| ROS 2 packages (`rclpy`, `nav2_msgs`, `tf2_ros`, …) | Not on PyPI — `pip install rclpy` will not work | Provided by the ROS 2 Humble system install; build the project's nodes with `colcon build --packages-select tiago_group_approach` |
| LocateAnything-3B (`torch`, `transformers`, `peft`, …) | Pins `numpy==1.25.0`, which conflicts with the `numpy 2.x` the data pipeline needs | Separate venv + `requirements-locateanything.txt` |

**Note on install size:** `ultralytics` depends on PyTorch, and on Linux pip resolves that to the CUDA build (~3–4 GB with the NVIDIA libraries). You probably don't need it — person detection has already been run across all 24 sessions and cached, so every downstream step works without it. Use `--cpu-torch` (~200 MB) or comment out the `ultralytics` line in `requirements.txt`.

---

## Repository layout

```
scripts/          data pipeline: extraction → detection → clustering → training → evaluation
src/
  tiago_head_control/    ROS 2 node: head scanning (working)
  tiago_group_approach/  ROS 2 node: rule-based approach baseline (written, UNTESTED)
  tiago_social_worlds/   Gazebo restaurant world + launch files
docs/             setup notes
dataset/          PLUS-HRI sessions (git-ignored — ~67 GB, not distributed here)
```

Key planning documents:

- `TIAGo_Project_Master_Checklist.md` — phase-by-phase status, mistakes, decisions
- `Proposal_Gap_Analysis.md` — what's satisfied vs. outstanding against the proposal's own criteria

---

## Pipeline

Run in order. Steps 1–3 are already done and their outputs are cached per session, so you can usually start at step 4.

```bash
# 1. Extract robot state + LiDAR + actions from the ROS 1 bags
python3 scripts/extract_training_table.py --session dataset/9

# 2. Detect people (YOLOv8n) — needs ultralytics
python3 scripts/extract_person_detections.py --session dataset/9

# 3. Cluster people into groups, estimate O-space + approach points
python3 scripts/cluster_groups.py --session dataset/9
python3 scripts/estimate_approach_points.py --session dataset/9

# 4. Build splits and the approach-pose training set
python3 scripts/split_dataset.py
python3 scripts/build_approach_pose_dataset.py

# 5. Train, tune, evaluate
python3 scripts/train_approach_pose_model.py     # Random Forest
python3 scripts/train_approach_pose_mlp.py       # MLP (proposal's primary)
python3 scripts/grid_search_approach_pose.py     # resumable; re-run to continue
python3 scripts/evaluate_approach_pose.py        # full policy comparison
```

Validation:

```bash
python3 scripts/validate_detector_recall.py      # Objective 2 — person detection
python3 scripts/prepare_ospace_validation.py     # Objective 3 — exports frames to label
python3 scripts/validate_ospace_estimate.py      # Objective 3 — scores your labels
```

---

## Results

Approach-pose prediction on held-out test sessions (5, 9, 59 — 11,921 rows). Thresholds are the proposal's own: position < 0.4 m, orientation < 20°.

| policy | mean position error | mean orientation error |
|---|---|---|
| naive (predict mean) | 0.410 m | 29.0° |
| **rule-based baseline** | **0.305 m** | 29.1° |
| Random Forest (tuned) | 0.365 m | **25.8°** |
| MLP (tuned) | 0.395 m | 31.0° |

**A split result, stated honestly:** the rule-based baseline predicts stopping *position* better than either learned model, while the tuned Random Forest predicts *orientation* better than every other policy including the rule. No policy meets the 20° orientation threshold.

Grid-search tuning materially helped (Random Forest 0.401 → 0.365 m; MLP 0.465 → 0.395 m). The winning MLP configuration — 32/16 hidden units with strong regularisation — is far smaller than the 128/64 originally proposed, indicating the proposed architecture was too large for the 462 independent demonstration events available.

---

## Known limitations

These are properties of the dataset, not bugs, and are reported rather than worked around:

- **No orientation ground truth anywhere.** `gaze_uniface.csv` is 100% empty across both sessions that have it. F-formation O-space estimation therefore uses the mutual-facing assumption (O-space centre ≈ group centroid).
- **No camera calibration or depth.** All perception output is in 2D image pixels, so real-world metres are not recoverable. Objective 3's "within 0.3 m" criterion is re-specified as "within 0.5 × mean person bounding-box width" (≈ 0.22–0.25 m equivalent).
- **Approach events are inferred, not annotated.** The dataset never marks "this is a group approach"; 462 events were derived from moving→stop transitions near a detected group.
- **Only 2 of 24 sessions** have face/gaze annotations; the other 22 rely on this project's own detector output.

---

## Two gotchas worth knowing if you extend this code

1. **Do not seek video frames with `cv2.CAP_PROP_POS_FRAMES`** — it lands on a non-keyframe and decodes corrupt solid-green frames on these videos. Read sequentially instead.
2. **Always pass `rtol=0` to `np.isclose` when comparing timestamps** — the default `rtol=1e-5` on a ~1.76e9 UNIX timestamp is a tolerance of roughly 17,000 seconds.

---

## Status

Complete: data extraction, person detection (99.7% recall vs. an 80% target), group clustering, O-space estimation, approach-pose dataset construction, both BC models, hyperparameter tuning, and offline evaluation.

Outstanding: literature review, O-space manual labelling, Nav2 verification with TIAGo, live perception→policy→Nav2 integration, and the six evaluation metrics that require a running simulation.

See `TIAGo_Project_Master_Checklist.md` for detail.
