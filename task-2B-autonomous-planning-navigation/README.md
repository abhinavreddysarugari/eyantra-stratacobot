# Task 2B: Global Grid Path Planning & Autonomous Mobile Logistics

## Overview
Task 2B implements a complete autonomous mobile logistics system in the Martian colony Kepler world. The differential-drive eBot autonomously computes optimal collision-free paths across a large 2D occupancy grid, receives ore packages, and navigates between colony mining stations and drop-off zones.

## Key Technical Highlights
- **Global Path Planning (`task2B_planning.py`):**
  - **Occupancy Grid Processing & Inflation:** Converts raw `/map` data into a 2D binary grid and applies morphological dilation (`cv2.dilate`) to expand obstacle boundaries by the robot's physical radius plus safety inflation margin.
  - **A* / Dijkstra Shortest Path Search:** Employs an efficient graph search with 8-connectivity and Euclidean distance heuristics, guaranteeing optimal paths while avoiding hazardous colony terrain.
  - **Path Pruning & Smoothing:** Eliminates collinear and redundant intermediate waypoints to generate smooth trajectories and reduce unnecessary stop-and-turn motions.
  - **Mission Execution & Spawning:** Communicates with the colony logistics service (`/spawn_ore_package`) to synchronize loading and dispatching.
- **Trajectory Tracking & Navigation (`task2B_navigation.py`):**
  - High-frequency odometry feedback (`/odom`) tracking waypoints on `/ebot_path`.
  - Dynamic lookahead pure pursuit controller with smooth angular velocity modulation and precise stopping within tolerance at goal waypoints.

## Topics & Interfaces
| Type | Topic / Service Name | Message / Service Type |
|---|---|---|
| Sub | `/map` | `nav_msgs/msg/OccupancyGrid` |
| Sub | `/odom` | `nav_msgs/msg/Odometry` |
| Pub | `/ebot_path` | `nav_msgs/msg/Path` |
| Pub | `/cmd_vel` | `geometry_msgs/msg/Twist` |
| Srv Client | `/spawn_ore_package` | `std_srvs/srv/Trigger` |
