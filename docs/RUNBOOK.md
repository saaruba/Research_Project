# Runbook — running the group-approach experiment

Everything needed to take `restaurant_testing.world` from a static scene to a
measured rule-vs-BC comparison. Follow the stages in order; each one is a
checkpoint, so a failure tells you exactly which part broke.

---

## Stage 0 — one-time setup

```bash
deactivate 2>/dev/null            # never do ROS work inside la3b_env
source /opt/ros/humble/setup.bash

# Installs Nav2, TIAGo, slam_toolbox, and copies the project's worlds into
# pal_gazebo_worlds so world_name:= can find them. Idempotent - safe to re-run.
bash scripts/install_sim_stack.sh

# YOLO must be installed for the SYSTEM python; the ROS node runs under
# /usr/bin/python3, NOT the la3b_env venv.
pip install ultralytics

cd /workspaces/Research_Project
colcon build --packages-select tiago_group_approach tiago_social_worlds
source install/setup.bash
```

Generated already for `restaurant_testing.world` — regenerate only if you edit
the world:

```bash
python3 scripts/extract_world_groundtruth.py \
    --world src/tiago_social_worlds/worlds/restaurant_testing.world
python3 scripts/world_to_map.py \
    --world src/tiago_social_worlds/worlds/restaurant_testing.world \
    --output src/tiago_social_worlds/maps/restaurant_testing
```

---

## Stage 1 — does the robot appear in your world?

Do this before anything else. It separates "the world loads" from "navigation
works" from "my nodes work", so you are only ever debugging one thing.

```bash
ros2 launch tiago_gazebo tiago_gazebo.launch.py \
    is_public_sim:=True \
    world_name:=restaurant_testing \
    navigation:=False \
    moveit:=False \
    arm_type:=no-arm
```

Open the VNC desktop (forwarded port **5801**). You should see the restaurant,
six people in two groups, and TIAGo.

Check the robot is alive:

```bash
ros2 topic list | grep -E "scan_raw|rgb/image_raw|depth/image_raw|cmd_vel"
ros2 topic hz /head_front_camera/rgb/image_raw     # should be ~15-30 Hz
```

| Symptom | Cause |
|---|---|
| World loads, no robot | `is_public_sim:=True` missing |
| People invisible | actor meshes not found — check the `file:///workspaces/...` paths in the world resolve |
| "world file not found" | re-run `install_sim_stack.sh` to copy worlds into `pal_gazebo_worlds` |
| Everything very slow | expected: Gazebo is software-rendering in the container |

**TIAGo spawns at the world origin (0, 0) by default.** Both your groups are at
(-3.3, -0.3) and (-0.3, 4.7), so the robot starts a few metres away with them
roughly in view — good for a first test.

---

## Stage 2 — does Nav2 work?

This is the step that was blocked for most of the project. Nav2 needs to
localise, and your world has no PAL-supplied map, so use SLAM:

```bash
ros2 launch tiago_gazebo tiago_gazebo.launch.py \
    is_public_sim:=True \
    world_name:=restaurant_testing \
    navigation:=True \
    slam:=True \
    moveit:=False \
    arm_type:=no-arm
```

In RViz: click **Nav2 Goal** and pick a point a few metres away on clear floor.

**If TIAGo drives there, Phase D is genuinely complete** and six of the eight
proposal metrics become reachable. Do not proceed until this works — every
later stage sends goals to Nav2 and will fail silently if Nav2 is not up.

```bash
ros2 action list | grep navigate_to_pose     # must exist
```

*(A prebuilt map is available at `src/tiago_social_worlds/maps/restaurant_testing.yaml`
— generated directly from the world geometry, no SLAM drift. Use it with
`map_server` if you prefer AMCL over SLAM; SLAM is simply less setup for a
first run.)*

---

## Stage 3 — the full pipeline

Leave Stage 2 running. In a **second terminal**:

```bash
source /opt/ros/humble/setup.bash
source /workspaces/Research_Project/install/setup.bash

ros2 launch tiago_group_approach group_approach.launch.py \
    policy:=rule \
    groundtruth:=/workspaces/Research_Project/src/tiago_social_worlds/worlds/restaurant_testing.groundtruth.json
```

What should happen:

1. `group_perception_node` detects people, back-projects them to **metres**
   using the depth image, and publishes `/group_centroid`
2. the policy node computes an approach pose and sends it to Nav2
3. TIAGo drives to a standoff position facing the group
4. `metrics_recorder_node` scores the run

Watch it:

```bash
ros2 topic echo /group_centroid            # should show map-frame coordinates
ros2 topic echo /detected_people           # every person found
```

In RViz add a **MarkerArray** on `/group_markers` — blue cylinders are detected
people, the red sphere is the group centre, the translucent disc is the
estimated O-space.

Stop the run to write results:

```bash
ros2 topic pub --once /metrics/finish std_msgs/msg/Empty {}
```

---

## Stage 4 — the actual experiment

Objective 4 is a controlled comparison: same world, same start pose, same
groups, **one variable — the policy**.

```bash
# several runs of each, resetting the robot between them
ros2 launch tiago_group_approach group_approach.launch.py policy:=rule ...
ros2 launch tiago_group_approach group_approach.launch.py policy:=bc   ...

python3 scripts/summarise_sim_results.py
```

Reset the robot between runs so every trial starts identically:

```bash
ros2 service call /reset_simulation std_srvs/srv/Empty
```

Aim for **at least 5 runs per policy**. With fewer, a single failure swings the
rate by 20 percentage points and the comparison carries no weight.

---

## Optional — LocateAnything-3B as the detector

The proposal's named model, running in the live pipeline:

```bash
# terminal 3 - the model, in ITS OWN venv (numpy 1.25 vs ROS's numpy 2.x)
source la3b_env/bin/activate
python3 scripts/locateanything_service.py

# terminal 2 - point the pipeline at it
ros2 launch tiago_group_approach group_approach.launch.py \
    policy:=rule detector:=locateanything
```

At a measured **25.6 s/frame** this automatically switches to one-shot mode:
the robot looks once, decides, then drives under Nav2's own obstacle avoidance.
Take another look with:

```bash
ros2 topic pub --once /perception/trigger std_msgs/msg/Empty {}
```

Use YOLO (the default) for the actual experiment runs — 200 FPS vs 0.04 FPS is
not a close call. LocateAnything is there to demonstrate the proposal's model
end-to-end and to support the substitution argument.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `/group_centroid` never publishes | No people detected, or TF is failing. Check `ros2 topic echo /detected_people`; if that is empty too, the detector sees nobody — drive closer or check the camera topic. |
| "TF optical_frame -> map failed" | Nav2/SLAM is not publishing a map frame yet. Finish Stage 2 first. |
| "Nav2 navigate_to_pose unavailable" | Nav2 is not running — you launched Stage 1 (`navigation:=False`) instead of Stage 2. |
| Robot drives into the group | Check the `standoff_distance` parameter (default 1.2 m, Hall's social-space boundary). |
| BC node rejects predictions | Expected on out-of-distribution input — the node refuses absurd goals rather than sending them. Reported honestly rather than silently clamped. |
| `ultralytics` ImportError | Installed into the wrong Python. `deactivate`, then `pip install ultralytics`. |
| Metrics all null | `groundtruth:=` not passed, so social metrics cannot be scored. |
