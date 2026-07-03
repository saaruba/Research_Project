# Design and Inverse Dynamics Analysis of a ROS2 Based Robotic Manipulator using Pinocchio and MoveIt2

This repository contains the complete implementation of a custom ROS2 based robotic manipulator developed for Forward Kinematics, Inverse Kinematics and Inverse Dynamics analysis using MoveIt2 and the Pinocchio robotics library.

The project was developed as part of an Advanced Robotics assignment using ROS2 Humble inside a Docker Dev Container environment.

---

# Project Overview

This project includes:

- Custom robotic manipulator design using URDF/Xacro
- Robot visualization using RViz2
- MoveIt2 integration
- Forward Kinematics (FK)
- Inverse Kinematics (IK)
- Inverse Dynamics using Pinocchio
- Real-time torque monitoring
- CSV based torque logging
- Torque graph plotting

The manipulator was tested under multiple robot joint configurations to analyse torque behaviour across different robot states.

---

# Robot Features

- 4-DOF robotic manipulator
- Revolute joint structure
- Parallel gripper system
- MoveIt2 motion planning
- Pinocchio RNEA inverse dynamics
- Live joint torque monitoring
- Torque graph generation

---

# Technologies Used

| Software / Library | Purpose |
|---|---|
| ROS2 Humble | Robotics Middleware |
| MoveIt2 | Motion Planning |
| RViz2 | Visualization |
| Pinocchio | Dynamics Computation |
| Python | Scripting |
| NumPy | Mathematical Operations |
| Matplotlib | Graph Plotting |
| Docker Dev Container | Development Environment |
| VS Code | IDE |


---

# Building the Workspace

Open the ROS2 Dev Container and run:

```bash
cd /workspaces/Adv._Robotics_Asssignment

source /opt/ros/humble/setup.bash

colcon build

source install/setup.bash
```

---

# Running the Full Project

Use separate terminals for each step.

---

# Terminal 1 — Launch Robot and MoveIt2

```bash
cd /workspaces/Adv._Robotics_Asssignment

source install/setup.bash

ros2 launch my_arm_moveit_config demo.launch.py
```

This will open:

- RViz2
- MoveIt2
- Robot Visualization
- Motion Planning Interface

You can manually move the robot joints inside RViz.

---

# Terminal 2 — Check Joint States

```bash
cd /workspaces/Adv._Robotics_Asssignment

source install/setup.bash

ros2 topic echo /joint_states
```

This verifies live robot joint values.

---

# Terminal 3 — Check Available Services

```bash
cd /workspaces/Adv._Robotics_Asssignment

source install/setup.bash

ros2 service list
```

You should see:

```bash
/compute_fk
/compute_ik
```

---

# Testing Forward Kinematics (FK)

```bash
ros2 service call /compute_fk moveit_msgs/srv/GetPositionFK "
header:
  frame_id: 'world'
fk_link_names:
- gripper_base
robot_state:
  joint_state:
    name:
    - joint1
    - joint2
    - joint4
    - gripper_base_joint6
    position:
    - 0.5
    - 0.3
    - 0.2
    - 0.1
"
```

This computes the end-effector pose from joint values.

---

# Running the Inverse Dynamics Experiment

Open a new terminal:

```bash
cd /workspaces/Adv._Robotics_Asssignment

source install/setup.bash

ros2 run pinocchio_dynamics dynamics_experiment
```

This will:

- Run multiple predefined robot states
- Compute inverse dynamics torques
- Save outputs into:

```bash
dynamics_results.csv
```

---

# Example Output

```bash
State 1
q   = [0.0 0.2 0.3 0.0 0.0 0.0 0.0]
tau = [0.0 -11.04 -4.26 -0.86 0.0 0.0]
```

---

# Plotting Torque Graphs

Run:

```bash
ros2 run pinocchio_dynamics plot_torque_results
```

This generates:

```bash
torque_results_plot.png
```

The graph visualizes torque changes across robot states.

---

# Running Live Inverse Dynamics Monitoring

Open another terminal:

```bash
cd /workspaces/Adv._Robotics_Asssignment

source install/setup.bash

ros2 run pinocchio_dynamics dynamics_node
```

Now move the robot manually in RViz.

The terminal will print:

- Joint positions
- Velocities
- Accelerations
- Live torque values

Example:

```bash
----- Live Inverse Dynamics -----

q: [ ... ]

tau: [ ... ]
```

---

# Live CSV Logging

The live dynamics node automatically stores data inside:

```bash
live_dynamics_results.csv
```

This file contains:

- timestamps
- joint values
- torque values

---

# Output Files

| File | Description |
|---|---|
| dynamics_results.csv | Predefined robot state torque outputs |
| live_dynamics_results.csv | Live torque monitoring outputs |
| torque_results_plot.png | Torque graph plot |

---

# Important Concepts Implemented

## Forward Kinematics

Computes end-effector pose from known joint angles.

---

## Inverse Kinematics

Computes required joint angles for a desired pose.

---

## Inverse Dynamics

Computes required joint torques using:

- Recursive Newton Euler Algorithm (RNEA)
- Pinocchio robotics library

---

# Robot DOF

The manipulator is considered as a:

# 4-DOF Robotic Manipulator

Active joints:

- joint1
- joint2
- joint4
- gripper_base_joint6

---

# Developed By

Saarunathan Thuviprakash

University of Lincoln  
MSc Robotics and AI
