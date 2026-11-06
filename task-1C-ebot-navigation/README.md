# Task 1C: Autonomous eBot Mobile Base Navigation & Obstacle Avoidance

## Overview
Task 1C develops a robust trajectory tracking and collision avoidance controller for the differential-drive eBot mobile robot navigating complex obstacle-laden terrain.

## Key Technical Highlights
- **Transient Local Path Reception:** Subscribes to the latched global trajectory `/ebot_path` using matched ROS 2 Quality of Service (QoS) profiles (`TRANSIENT_LOCAL` durability and `RELIABLE` policy).
- **Pure Pursuit & Heading Orientation Controller:**
  - Computes heading angle error $\Delta \theta = \text{atan2}(y_t - y, x_t - x) - \theta_{\text{robot}}$ with angular normalization into $[-\pi, \pi]$.
  - Implements proportional orientation alignment: turns in-place when heading error is large, blending into smooth linear forward motion ($v \le 0.5 \text{ m/s}$, $\omega \le 1.0 \text{ rad/s}$) as the robot aligns with the path tangent.
- **2D LiDAR Reactive Safety Obstacle Avoidance:**
  - Processes laser scan depth returns (`/scan`) over forward-facing angular sectors ($[-30^\circ, +30^\circ]$).
  - Automatically modulates robot velocity and injects corrective steering torque to bypass static and dynamic obstacles without losing the global path reference.

## Topics & Interfaces
| Type | Topic Name | Message Type |
|---|---|---|
| Sub | `/ebot_path` | `nav_msgs/msg/Path` (Transient Local QoS) |
| Sub | `/odom` | `nav_msgs/msg/Odometry` |
| Sub | `/scan` | `sensor_msgs/msg/LaserScan` |
| Pub | `/cmd_vel` | `geometry_msgs/msg/Twist` |
