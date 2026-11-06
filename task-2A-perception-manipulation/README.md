# Task 2A: Integrated Vision-Guided Cobot Manipulation & Sorting

## Overview
Task 2A integrates 3D computer vision perception with advanced 6-DOF robotic arm manipulation. The UR7e cobot autonomously scans the conveyor belt, identifies Martian ore samples by mineral type, moves to grasp each ore using an electromagnetic end-effector, and precisely places it into designated sorting bins.

## Key Technical Highlights
- **Perception Pipeline (`task2A_perception.py`):**
  - Continuous RealSense RGB-D stream acquisition.
  - Multi-class ore segmentation (Azurite, Malachite, Vanadinite).
  - Spatial noise filtering and dynamic TF2 frame broadcasting relative to the robot base.
- **Manipulation Pipeline (`task2A_manipulation.py`):**
  - **Optimization-Based Inverse Kinematics:** Custom numerical inverse kinematics formulated as an objective minimization problem (using SciPy optimization `minimize` with joint boundary constraints) to compute singularity-free joint angles for target 3D Cartesian coordinates.
  - **Closed-Loop Joint Space Control:** Drives the 6 joints via `/delta_joint_cmds` with feedback from `/joint_states`.
  - **Electromagnetic Grasping:** Interfaces with end-effector actuation services (`/ur7e/set_magnet`) and monitors contact force feedback (`/ur7e/end_effector/force_magnitude`) to confirm successful pick-up before trajectory execution.
  - **Multi-Phase State Machine:** Implements a finite-state machine (Home $\to$ Pre-Grasp $\to$ Grasp $\to$ Lift $\to$ Bin Transport $\to$ Release $\to$ Return) ensuring safe transit corridors and zero collisions with the environment.

## Topics & Interfaces
| Type | Topic / Service Name | Message / Service Type |
|---|---|---|
| Sub | `/tcp_pose_raw` | `geometry_msgs/msg/PoseStamped` |
| Sub | `/joint_states` | `sensor_msgs/msg/JointState` |
| Sub | `/arm_status` | `std_msgs/msg/Int32` |
| Sub | `/ur7e/end_effector/force_magnitude` | `std_msgs/msg/Float32` |
| Pub | `/delta_joint_cmds` | `control_msgs/msg/JointJog` |
| Srv Client | `/ur7e/set_magnet` | `std_srvs/srv/SetBool` |
| Srv Client | `/controller_manager/switch_controller` | `controller_manager_msgs/srv/SwitchController` |
