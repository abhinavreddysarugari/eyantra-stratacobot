# Task 1B: UR7e Cobot Velocity Control & Waypoint Trajectory Tracking

## Overview
Task 1B implements closed-loop velocity-based trajectory control of a 6-DOF Universal Robots UR7e industrial cobot. The end-effector is commanded to navigate through an ordered sequence of 3D Cartesian coordinates in space, coming to a dead stop and stably holding each waypoint position for at least 2.0 seconds.

## Key Technical Highlights
- **Controller Interface Switching:** Leverages ROS 2 `controller_manager_msgs/srv/SwitchController` to dynamically switch between `delta_twist_controller` and `delta_joint_controller`.
- **Proportional Closed-Loop Cartesian Servo:** Calculates real-time 3D displacement vectors $\vec{e} = \vec{p}_{\text{target}} - \vec{p}_{\text{current}}$ from raw TCP pose feedback (`/tcp_pose_raw`).
- **Velocity Clamping & Safety Limiting:** Clamps Cartesian linear velocity to $\le 0.15 \text{ m/s}$ and angular velocity to $\le 0.35 \text{ rad/s}$ to comply with cobot hardware safety constraints and prevent command dropping.
- **Dead-man Switch & Steady-State Hold:** Features high-rate control loop execution (100 Hz / 10 ms) with a dead-man watchdog timer (timeout 0.15s) and position error thresholding ($\le 1 \text{ cm}$) to enforce stable waypoint hold before transitioning to the next objective.

## Topics & Interfaces
| Type | Topic / Service Name | Message / Service Type |
|---|---|---|
| Sub | `/tcp_pose_raw` | `geometry_msgs/msg/PoseStamped` |
| Sub | `/joint_states` | `sensor_msgs/msg/JointState` |
| Sub | `/arm_status` | `std_msgs/msg/Int32` |
| Pub | `/delta_twist_cmds` | `geometry_msgs/msg/TwistStamped` |
| Pub | `/delta_joint_cmds` | `control_msgs/msg/JointJog` |
| Srv Client | `/controller_manager/switch_controller` | `controller_manager_msgs/srv/SwitchController` |
