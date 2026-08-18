# Running this project on another machine (lab PC)

Everything needed to reproduce the **simulation experiment**. You do not need
the raw dataset.

---

## What actually has to move

| | Size | How | Needed to run? |
|---|---|---|---|
| Source code, launch files, worlds, maps, ground truth | ~5 MB | **git** | yes |
| Trained BC models (`*_tuned.joblib`) | 3 MB | bundle | yes, for `policy:=bc` |
| Gazebo assets the world references | ~135 MB | bundle | yes |
| Existing simulation results | <10 MB | bundle | only to aggregate across machines |
| **Raw PLUS-HRI recordings** (`dataset/1` … `dataset/60`) | **~70 GB** | **do not move** | **no** |
| **Rest of the LIRS-HMLG actor library** | **7.8 GB** | **do not move** | **no** |

The raw dataset exists to *derive* the offline results — extracting detections,
building the approach-pose dataset, training and grid-searching the models.
That work is finished and its output is the 3 MB of `.joblib` files. Running the
robot never touches the recordings.

Same logic for the actors: `models/actors/LIRS-HMLG` is 7.8 GB, and
`restaurant_testing.world` references exactly one mesh from it
(`Male/m_suit/talk.dae`).

**Only move the dataset if** you intend to re-run training from scratch on the
lab PC — for example if a marker asks you to demonstrate the pipeline
end-to-end. In that case an external drive is far less painful than a network
copy.

---

## Steps

### 1. On this machine — build the bundle

```bash
cd /workspaces/Research_Project
git add -A && git commit -m "Simulation pipeline" && git push
bash scripts/make_transfer_bundle.sh
```

Produces `/tmp/tiago_sim_bundle.tar.gz`, roughly 140 MB. Copy it across on a
USB stick, or with `scp`.

### 2. On the lab PC — system packages

Requires **Ubuntu 22.04**. ROS 2 Humble does not support other versions without
a container.

```bash
# ROS 2 Humble, if not already installed
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions \
    python3-pip python3-venv
```

### 3. Clone, unpack, install the robot stack

```bash
git clone <your repo url> Research_Project
cd Research_Project
tar xzf ~/tiago_sim_bundle.tar.gz

source /opt/ros/humble/setup.bash
bash scripts/install_sim_stack.sh      # TIAGo, Nav2, Gazebo, world install
```

`install_sim_stack.sh` installs `ros-humble-tiago-*`, `navigation2`,
`nav2-bringup`, `gazebo-ros-pkgs`, `slam-toolbox`, and copies the project's
worlds into `pal_gazebo_worlds`.

### 4. Python packages

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-simulation.txt --break-system-packages
```

The CPU-only torch first is deliberate: without it, pip pulls the CUDA build
and the entire NVIDIA stack, about 3–4 GB. YOLOv8n runs fine on CPU at the
~2 Hz this pipeline uses. **If the lab PC has an NVIDIA GPU**, skip the first
line and let pip install the CUDA build — Gazebo will also run far faster with
real hardware rendering than the software rendering in the container.

### 5. Fix the absolute path (important)

`restaurant_testing.world` references the actor mesh by absolute path:

```
file:///workspaces/Research_Project/models/actors/LIRS-HMLG/Male/m_suit/talk.dae
```

If the lab PC clones anywhere other than `/workspaces/Research_Project`, Gazebo
will not find the human and will stall looking for it online:

```bash
NEW=$(pwd)
sed -i "s|/workspaces/Research_Project|${NEW}|g" \
    src/tiago_social_worlds/worlds/*.world
grep -rn "file://" src/tiago_social_worlds/worlds/restaurant_testing.world
```

Also update `PROJECT_ROOT` / `PROJECT` at the top of
`scripts/setup_gazebo_env.sh`, `scripts/run_everything.sh`,
`scripts/run_pipeline.sh`, and `PROJECT_ROOT` in
`src/tiago_group_approach/launch/group_approach.launch.py`.

### 6. Build and run

```bash
colcon build --packages-select tiago_group_approach
source install/setup.bash

# terminal 1
bash scripts/run_everything.sh restaurant_testing rule --no-pipeline

# terminal 2
bash scripts/run_pipeline.sh restaurant_testing rule
```

---

## Expect it to be faster

The devcontainer runs Gazebo with software rendering at roughly 0.3–0.6× real
time and 3–4 FPS. A lab PC with a real GPU should manage close to 1.0× — each
trial drops from ~5 minutes to ~2, which matters when you need 10 of them.

---

## Sanity check after setup

```bash
bash scripts/check_sim_setup.sh
```

Confirms ROS 2, TIAGo packages, Nav2, Gazebo binaries and the Python
environment before you spend time debugging a launch failure that is really a
missing package.
