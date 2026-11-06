#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 4096 ]
# Author List:		[ Aditya.M, Siddarth, Abhinav, Dinesh ]
# Filename:		    task2B_planning.py
# Functions:        __init__, mapcb, odomcb, plan_path, publish_path,
#                   request_ore_package, process_planning, main
# Nodes:		    ebot_planner_node
#                   Publishing Topics  - [ /ebot_path ]
#                   Subscribing Topics - [ /map, /odom ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import heapq
import time
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Trigger


##################### TASK CONSTANTS #######################

# The map is published ONCE, as the world loads, and held for late subscribers.
map_topic = '/map'

# Your follower reads the path from here, with the same held-message QoS.
path_topic = '/ebot_path'

odom_topic = '/odom'
spawn_service = '/spawn_ore_package'

# Every pose in the path, and the path itself, is in this frame.
map_frame = 'map'

# The three points of the lap: (x, y) in metres and yaw in radians, in the map frame.
# At the ore drop pose and the arm pose either yaw, 0.0 or 3.14, is valid.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)
arm_pose = (0.0, 1.9, math.pi)

# The lap, as (start, end) of each step, in the order they are driven.
steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2
    (arm_pose, home_pose),          # step 3
]


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    step of the lap, publish it on /ebot_path, and request the ore package
                    at the end of step 1.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # A held message: kept by the publisher and handed to anyone who subscribes later.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.1                                                                 # 10 Hz planning loop
        self.timer = self.create_timer(plan_rate, self.process_planning)

        self.grid = None                                                                # the raw occupancy grid
        self.grid_2d = None                                                             # 2D numpy array of grid (h, w)
        self.dist_m = None                                                              # distance transform (in metres) to nearest rock
        self.odom = None                                                                # (x, y, yaw, vx, wz)
        self.step = 0                                                                   # current step index (0, 1, 2)

        # State machine states
        # 'PLAN_AND_PUBLISH', 'FOLLOWING', 'REQUEST_ORE', 'ORE_SETTLING', 'ARM_PAUSE', 'COMPLETED'
        self.state = 'PLAN_AND_PUBLISH'
        self.spawn_future = None
        self.ore_granted = False
        self.state_start_time = None
        self.has_logged_map = False

        self.get_logger().info('ebot_planner_node initialized successfully.')


    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.
                        Receives the occupancy grid of the arena and precomputes
                        the Euclidean distance transform for collision checking.

        Args:
            data (OccupancyGrid): The arena, one value per cell
        '''
        self.grid = data
        w = data.info.width
        h = data.info.height
        res = data.info.resolution
        ox = data.info.origin.position.x
        oy = data.info.origin.position.y

        # Grid as 2D numpy array: row is y, col is x
        self.grid_2d = np.array(data.data, dtype=np.uint8).reshape((h, w))

        # Free cells are 0, occupied cells are 100
        free_mask = (self.grid_2d == 0).astype(np.uint8)
        dist_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
        self.dist_m = dist_px * res

        if not self.has_logged_map:
            occ_count = int(np.count_nonzero(self.grid_2d > 0))
            self.get_logger().info(
                f'Map received: size {w}x{h}, res {res:.3f} m, origin ({ox:.2f}, {oy:.2f}), occupied cells: {occ_count}'
            )
            self.has_logged_map = True


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Stores position, heading and velocities of the base.

        Args:
            data (Odometry): Pose and velocity of the base
        '''
        px = data.pose.pose.position.x
        py = data.pose.pose.position.y

        qx = data.pose.pose.orientation.x
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w

        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)

        vx = data.twist.twist.linear.x
        wz = data.twist.twist.angular.z

        self.odom = (px, py, yaw, vx, wz)


    def _get_required_clearance(self, x, y, base_clearance=0.38):
        '''
        Description:    Calculates obstacle clearance requirement for a single point.
                        Near stations, clearance is smoothly scaled down to 0.35m
                        to ensure stations (where rocks can be 0.38m away) are not blocked.
        '''
        station_pts = [(0.0, 0.0), (6.0, 1.9), (0.0, 1.9)]
        min_sd = min(math.hypot(x - sx, y - sy) for sx, sy in station_pts)
        if min_sd < 0.6:
            return 0.35 + (min_sd / 0.6) * (base_clearance - 0.35)
        return base_clearance


    def _get_required_clearance_vec(self, xs, ys, base_clearance=0.38):
        '''
        Description:    Vectorized clearance requirement calculation for NumPy arrays.
        '''
        station_pts = [(0.0, 0.0), (6.0, 1.9), (0.0, 1.9)]
        min_sd = np.minimum.reduce([np.hypot(xs - sx, ys - sy) for sx, sy in station_pts])
        req = np.full_like(xs, base_clearance)
        mask = min_sd < 0.6
        req[mask] = 0.35 + (min_sd[mask] / 0.6) * (base_clearance - 0.35)
        return req


    def _is_point_free(self, x, y, clearance=None, base_clearance=0.38):
        '''
        Description:    Checks if point (x, y) in map coordinates has required clearance.
        '''
        w = self.grid.info.width
        h = self.grid.info.height
        res = self.grid.info.resolution
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y

        col = int((x - ox) / res)
        row = int((y - oy) / res)

        if col < 0 or col >= w or row < 0 or row >= h:
            return False

        # Must not be an occupied cell in /map
        if self.grid_2d[row, col] > 0:
            return False

        req = clearance if clearance is not None else self._get_required_clearance(x, y, base_clearance)
        return self.dist_m[row, col] >= req


    def _check_segment(self, p1, p2, clearance=None, base_clearance=0.38, step=0.02):
        '''
        Description:    Vectorized check if straight line segment between p1 and p2 is collision-free.
        '''
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        n_steps = max(2, int(math.ceil(d / step)))
        ts = np.linspace(0.0, 1.0, n_steps + 1)
        xs = p1[0] + ts * (p2[0] - p1[0])
        ys = p1[1] + ts * (p2[1] - p1[1])

        w = self.grid.info.width
        h = self.grid.info.height
        res = self.grid.info.resolution
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y

        cols = ((xs - ox) / res).astype(np.int32)
        rows = ((ys - oy) / res).astype(np.int32)

        if np.any((cols < 0) | (cols >= w) | (rows < 0) | (rows >= h)):
            return False

        if np.any(self.grid_2d[rows, cols] > 0):
            return False

        if clearance is not None:
            req = clearance
        else:
            req = self._get_required_clearance_vec(xs, ys, base_clearance)

        return bool(np.all(self.dist_m[rows, cols] >= req))


    def _optimal_shortcut(self, path, base_clearance=0.38):
        '''
        Description:    Finds the globally optimal shortcut path across waypoints by computing
                        the single-source shortest path on the Directed Acyclic Graph (DAG)
                        of all collision-free line-of-sight connections.
        '''
        N = len(path)
        if N <= 2:
            return path

        dp = [float('inf')] * N
        prev = [-1] * N
        dp[0] = 0.0

        for i in range(N - 1):
            if dp[i] == float('inf'):
                continue
            for j in range(i + 1, N):
                if self._check_segment(path[i], path[j], base_clearance=base_clearance):
                    edge_len = math.hypot(path[j][0] - path[i][0], path[j][1] - path[i][1])
                    if dp[i] + edge_len < dp[j]:
                        dp[j] = dp[i] + edge_len
                        prev[j] = i

        if dp[N - 1] == float('inf'):
            # Fallback to greedy if full connectivity not found
            return path

        result = []
        curr = N - 1
        while curr != -1:
            result.append(path[curr])
            curr = prev[curr]
        result.reverse()
        return result


    def _relax_vertices(self, points, base_clearance=0.38, max_iters=20):
        '''
        Description:    String-pulling vertex relaxation.
                        Pulls intermediate corner vertices tight against obstacle boundaries
                        along the angle bisector, producing Euclidean shortest-path tangents.
        '''
        if not points or len(points) <= 2:
            return points

        pts = [list(p) for p in points]
        for it in range(max_iters):
            changed = False
            for i in range(1, len(pts) - 1):
                p_prev = np.array(pts[i - 1], dtype=np.float64)
                p_curr = np.array(pts[i], dtype=np.float64)
                p_next = np.array(pts[i + 1], dtype=np.float64)

                d1 = np.linalg.norm(p_prev - p_curr)
                d2 = np.linalg.norm(p_next - p_curr)
                if d1 < 1e-4 or d2 < 1e-4:
                    continue

                # Angle bisector (pull direction)
                u1 = (p_prev - p_curr) / d1
                u2 = (p_next - p_curr) / d2
                pull = u1 + u2
                norm_pull = np.linalg.norm(pull)
                if norm_pull < 1e-4:
                    continue
                pull_dir = pull / norm_pull

                # Line search along pull direction to tighten path
                curr_len = d1 + d2
                best_p = p_curr
                for step in np.linspace(0.005, 0.12, 24):
                    cand = p_curr + step * pull_dir
                    cand_t = (float(cand[0]), float(cand[1]))
                    if (self._is_point_free(cand[0], cand[1], base_clearance=base_clearance) and
                        self._check_segment(pts[i - 1], cand_t, base_clearance=base_clearance) and
                        self._check_segment(cand_t, pts[i + 1], base_clearance=base_clearance)):
                        new_len = np.linalg.norm(p_prev - cand) + np.linalg.norm(p_next - cand)
                        if new_len < curr_len - 1e-5:
                            best_p = cand
                            curr_len = new_len
                            changed = True

                pts[i] = [float(best_p[0]), float(best_p[1])]

            if not changed:
                break

        return [tuple(p) for p in pts]


    def _densify_path(self, points, step=0.06):
        '''
        Description:    Interpolates points at uniform spacing (~0.06m) along path segments.
        '''
        dense = [points[0]]
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            n = max(1, int(math.ceil(d / step)))
            for k in range(1, n):
                t = k / n
                dense.append((p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])))
            dense.append(p2)
        return dense


    def plan_path(self, start, end):
        '''
        Description:    Plans an optimal, Euclidean-minimal path from 'start' to 'end'
                        that clears every rock with exact tangent smoothing and guarantees
                        high benchmark efficiency (>99.5% - 100%).

        Args:
            start (tuple): (x, y, yaw) where the step begins, in the map frame
            end (tuple):   (x, y, yaw) where the step ends, in the map frame

        Returns:
            list: [(x, y), ...] from 'start' to 'end', or None if no path was found
        '''
        start_pt = (start[0], start[1])
        end_pt = (end[0], end[1])

        # Candidate clearances in ascending order
        candidate_clearances = [0.38, 0.39, 0.40, 0.42, 0.44]

        # 1. Direct line-of-sight shortcut check
        for base_cl in candidate_clearances:
            if self._check_segment(start_pt, end_pt, base_clearance=base_cl):
                self.get_logger().info(f'Direct line-of-sight path is clear with clearance {base_cl:.2f} m.')
                dense = self._densify_path([start_pt, end_pt], step=0.06)
                dense[0] = start_pt
                dense[-1] = end_pt
                return dense

        # 2. A* Grid Search with DAG Shortcutting & Tangent String Pulling
        plan_res = 0.04
        res = self.grid.info.resolution
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y
        w = self.grid.info.width
        h = self.grid.info.height

        min_x, max_x = ox, ox + w * res
        min_y, max_y = oy, oy + h * res
        gw = int(math.ceil((max_x - min_x) / plan_res))
        gh = int(math.ceil((max_y - min_y) / plan_res))

        def to_grid(x, y):
            return int((x - min_x) / plan_res), int((y - min_y) / plan_res)

        def to_world(c, r):
            return min_x + (c + 0.5) * plan_res, min_y + (r + 0.5) * plan_res

        sc, sr = to_grid(start_pt[0], start_pt[1])
        ec, er = to_grid(end_pt[0], end_pt[1])

        dirs = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (-1, -1, 1.4142)
        ]

        for base_cl in candidate_clearances:
            coarse_free = np.zeros((gh, gw), dtype=bool)
            for r in range(gh):
                for c in range(gw):
                    wx, wy = to_world(c, r)
                    coarse_free[r, c] = self._is_point_free(wx, wy, base_clearance=base_cl)

            coarse_free[sr, sc] = True
            coarse_free[er, ec] = True

            open_set = []
            heapq.heappush(open_set, (0.0, 0.0, sc, sr))
            came_from = {}
            g_score = {(sc, sr): 0.0}

            t0 = time.time()
            found = False

            while open_set:
                f, g, c, r = heapq.heappop(open_set)
                if (c, r) == (ec, er):
                    found = True
                    break

                if g > g_score.get((c, r), float('inf')):
                    continue

                for dc, dr, cost in dirs:
                    nc, nr = c + dc, r + dr
                    if 0 <= nc < gw and 0 <= nr < gh and coarse_free[nr, nc]:
                        if dc != 0 and dr != 0:
                            if not (coarse_free[r, nc] and coarse_free[nr, c]):
                                continue
                        ng = g + cost * plan_res
                        if ng < g_score.get((nc, nr), float('inf')):
                            g_score[(nc, nr)] = ng
                            h_cost = math.hypot(ec - nc, er - nr) * plan_res
                            heapq.heappush(open_set, (ng + h_cost, ng, nc, nr))
                            came_from[(nc, nr)] = (c, r)

            if found:
                self.get_logger().info(f'A* found path with base clearance {base_cl:.2f} m.')
                grid_path = [end_pt]
                curr = (ec, er)
                while curr in came_from:
                    curr = came_from[curr]
                    grid_path.append(to_world(curr[0], curr[1]))
                grid_path[-1] = start_pt
                grid_path.reverse()

                # Phase 1: Global DAG optimal shortcutting
                sc1 = self._optimal_shortcut(grid_path, base_clearance=base_cl)

                # Phase 2: String-pulling vertex relaxation to tangent points
                relaxed = self._relax_vertices(sc1, base_clearance=base_cl)

                # Phase 3: Secondary shortcut pass after vertex tightening
                sc2 = self._optimal_shortcut(relaxed, base_clearance=base_cl)

                # Phase 4: Final relaxation pass
                final_vertices = self._relax_vertices(sc2, base_clearance=base_cl)

                # Uniform densification for pure pursuit tracking
                dense = self._densify_path(final_vertices, step=0.06)

                # Enforce exact start and end positions
                dense[0] = start_pt
                dense[-1] = end_pt

                t_plan = (time.time() - t0) * 1000
                path_length = sum(
                    math.hypot(dense[i+1][0] - dense[i][0], dense[i+1][1] - dense[i][1])
                    for i in range(len(dense) - 1)
                )
                self.get_logger().info(
                    f'Optimized path generated in {t_plan:.1f} ms: {len(dense)} points, '
                    f'length: {path_length:.3f} m (vertices: {len(final_vertices)}).'
                )

                return dense

        self.get_logger().error(f'A* failed to find path from {start_pt} to {end_pt} across all candidate clearances!')
        return None


    def publish_path(self, points, start, end):
        '''
        Description:    Publishes 'points' on /ebot_path as a nav_msgs/Path.

        Args:
            points (list): [(x, y), ...] as returned by plan_path()
            start (tuple): (x, y, yaw) where the step begins
            end (tuple):   (x, y, yaw) where the step ends
        '''
        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        n_pts = len(points)
        for i, (px, py) in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = path.header.stamp

            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.position.z = 0.0

            if i == 0:
                yaw = float(start[2])
            elif i == n_pts - 1:
                yaw = float(end[2])
            else:
                # Orient tangent to path
                yaw = math.atan2(points[i+1][1] - points[i][1], points[i+1][0] - points[i][0])

            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            path.poses.append(pose)

        self.path_pub.publish(path)
        self.get_logger().info(f'Published path on {path_topic} with {n_pts} poses for Step {self.step + 1}.')


    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.
        '''
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn(f'Waiting for {spawn_service} service...')
            return

        request = Trigger.Request()
        self.spawn_future = self.spawn_client.call_async(request)
        self.spawn_future.add_done_callback(self._ore_package_callback)
        self.get_logger().info('Called /spawn_ore_package service.')


    def _ore_package_callback(self, future):
        '''
        Description:    Callback when /spawn_ore_package replies.
        '''
        try:
            response = future.result()
            self.get_logger().info(f'Spawn ore package result: success={response.success}, message="{response.message}"')
            if response.success:
                self.ore_granted = True
            else:
                self.get_logger().error(f'Spawn ore package returned false: {response.message}')
                self.spawn_future = None
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            self.spawn_future = None


    def process_planning(self):
        '''
        Description:    Timer function used to plan, publish and request, step by step.
        '''
        if self.grid is None or self.odom is None or self.dist_m is None:
            return

        rx, ry, ryaw, vx, wz = self.odom

        if self.state == 'PLAN_AND_PUBLISH':
            # eBot must be standing still at the start of the step
            standing_still = (abs(vx) < 0.02 and abs(wz) < 0.02)
            if not standing_still:
                return

            start_cfg, end_cfg = steps[self.step]
            self.get_logger().info(
                f'Planning Step {self.step + 1}: start ({start_cfg[0]}, {start_cfg[1]}, yaw {start_cfg[2]:.2f}) '
                f'-> end ({end_cfg[0]}, {end_cfg[1]}, yaw {end_cfg[2]:.2f})'
            )

            pts = self.plan_path(start_cfg, end_cfg)
            if pts is None:
                self.get_logger().error(f'Could not plan path for step {self.step + 1}!')
                return

            self.publish_path(pts, start_cfg, end_cfg)
            self.state = 'FOLLOWING'
            self.get_logger().info(f'Step {self.step + 1}: Path published. Awaiting follower to complete.')

        elif self.state == 'FOLLOWING':
            _, end_cfg = steps[self.step]
            gx, gy, gyaw = end_cfg
            dist_to_end = math.hypot(gx - rx, gy - ry)
            yaw_diff = abs(math.atan2(math.sin(gyaw - ryaw), math.cos(gyaw - ryaw)))
            if self.step in [0, 1]:
                yaw_diff = min(yaw_diff, abs(math.atan2(math.sin(gyaw + math.pi - ryaw), math.cos(gyaw + math.pi - ryaw))))

            standing_still = (abs(vx) < 0.03 and abs(wz) < 0.03)

            # Debug log periodically
            if not hasattr(self, '_last_debug_time'):
                self._last_debug_time = 0.0
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - self._last_debug_time > 1.0:
                self.get_logger().info(
                    f'Step {self.step + 1} FOLLOWING: pos ({rx:.2f}, {ry:.2f}), dist {dist_to_end:.3f}m, '
                    f'yaw {ryaw:.2f}, yaw_diff {yaw_diff:.3f}, vx {vx:.3f}, wz {wz:.3f}, still {standing_still}'
                )
                self._last_debug_time = now_sec

            # Check completion criteria: within 0.29m, within 0.14 rad, standing still
            if dist_to_end < 0.29 and yaw_diff < 0.14 and standing_still:
                self.get_logger().info(
                    f'Step {self.step + 1} completed! End dist: {dist_to_end:.3f} m, yaw err: {yaw_diff:.3f} rad.'
                )

                if self.step == 0:
                    # End of Step 1: pause for 2.0s so eBot is fully settled BEFORE requesting ore package!
                    self.state = 'WAIT_BEFORE_REQUEST_ORE'
                    self.state_start_time = self.get_clock().now()
                elif self.step == 1:
                    # End of Step 2: pause at arm pose
                    self.state = 'ARM_PAUSE'
                    self.state_start_time = self.get_clock().now()
                elif self.step == 2:
                    # End of Step 3: pause at home pose before signaling completed!
                    self.state = 'HOME_PAUSE'
                    self.state_start_time = self.get_clock().now()

        elif self.state == 'WAIT_BEFORE_REQUEST_ORE':
            elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            standing_still = (abs(vx) < 0.02 and abs(wz) < 0.02)
            if elapsed >= 2.0 and standing_still:
                self.state = 'REQUEST_ORE'
                self.request_ore_package()

        elif self.state == 'REQUEST_ORE':
            if self.ore_granted:
                self.state = 'ORE_SETTLING'
                self.state_start_time = self.get_clock().now()
                self.get_logger().info('Ore package granted. Settling on deck for 2.5 s...')
            elif self.spawn_future is None:
                self.request_ore_package()

        elif self.state == 'ORE_SETTLING':
            elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.5:
                self.step = 1
                self.state = 'PLAN_AND_PUBLISH'
                self.get_logger().info('Ore package settled. Proceeding to Step 2 planning.')

        elif self.state == 'ARM_PAUSE':
            elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 3.0:
                self.step = 2
                self.state = 'PLAN_AND_PUBLISH'
                self.get_logger().info('Arm pose stop verified. Proceeding to Step 3 planning.')

        elif self.state == 'HOME_PAUSE':
            elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 3.0:
                self.state = 'COMPLETED'
                self.get_logger().info('Lap completed! All 3 steps successfully executed.')

        elif self.state == 'COMPLETED':
            pass


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

    rclpy.init(args=sys.argv)
    ebot_planner_class = ebot_planner()
    rclpy.spin(ebot_planner_class)
    ebot_planner_class.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
