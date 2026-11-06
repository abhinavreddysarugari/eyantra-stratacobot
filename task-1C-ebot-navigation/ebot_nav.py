#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    ebot_nav.py
# Functions:        __init__, pathcb, odomcb, scancb, process_navigation, main
# Nodes:		    ebot_nav_node
#                   Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


##################### TASK CONSTANTS #######################

# The route is published here ONCE, before your node starts, and held for late
# subscribers.
path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# Velocity limits for eBot mobile base
max_linear_mps = 0.5
max_angular_rps = 1.0


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''
        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        # Matched QoS for latched /ebot_path publisher
        path_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05
        self.timer = self.create_timer(control_rate, self.process_navigation)

        self.waypoints = []
        self.current_wp_idx = 0
        self.completed = False

        self.current_x = None
        self.current_y = None
        self.yaw = None

        self.laser_points = []
        self.lidar_offset_x = 0.2555


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''
        if not self.waypoints and data.poses:
            self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in data.poses]
            self.get_logger().info(f'Received route with {len(self.waypoints)} waypoints.')


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y

        qx = data.pose.pose.orientation.x
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w

        # Compute yaw from orientation quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)


    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data (LaserScan):    One lidar sweep

        Returns:
        '''
        pts = []
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        rmin = data.range_min
        rmax = data.range_max

        for i, r in enumerate(data.ranges):
            if math.isfinite(r) and rmin <= r <= rmax:
                angle_i = angle_min + i * angle_inc
                lx = r * math.cos(angle_i)
                ly = r * math.sin(angle_i)
                # Convert from laser frame to base link coordinates
                bx = lx + self.lidar_offset_x
                by = ly
                pts.append((bx, by))

        self.laser_points = pts


    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the route.

        Args:
        Returns:
        '''
        cmd = Twist()

        if self.completed:
            self.cmd_pub.publish(cmd)
            return

        if not self.waypoints or self.current_x is None or self.yaw is None:
            return

        target_x, target_y = self.waypoints[self.current_wp_idx]
        dist_to_target = math.hypot(target_x - self.current_x, target_y - self.current_y)

        # Waypoint sequence tracking
        # Each waypoint counts within 0.5m; using 0.35m ensures precision and accuracy bonus
        if self.current_wp_idx == 0:
            if dist_to_target < 0.40:
                self.get_logger().info(f'Reached waypoint {self.current_wp_idx} (dist: {dist_to_target:.2f} m)')
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.waypoints):
                    self.completed = True
                    self.cmd_pub.publish(cmd)
                    return
                target_x, target_y = self.waypoints[self.current_wp_idx]
                dist_to_target = math.hypot(target_x - self.current_x, target_y - self.current_y)
        elif self.current_wp_idx < len(self.waypoints) - 1:
            if dist_to_target < 0.35:
                self.get_logger().info(f'Reached waypoint {self.current_wp_idx} (dist: {dist_to_target:.2f} m)')
                self.current_wp_idx += 1
                target_x, target_y = self.waypoints[self.current_wp_idx]
                dist_to_target = math.hypot(target_x - self.current_x, target_y - self.current_y)
        else:
            # Stop precisely at the 10th (last) waypoint
            if dist_to_target < 0.20:
                self.get_logger().info(f'Final waypoint 9 reached! Stopping. (dist: {dist_to_target:.2f} m)')
                self.completed = True
                self.cmd_pub.publish(cmd)
                return

        # Target heading relative to robot
        goal_map_angle = math.atan2(target_y - self.current_y, target_x - self.current_x)
        goal_rel_angle = math.atan2(math.sin(goal_map_angle - self.yaw), math.cos(goal_map_angle - self.yaw))

        # Turn towards target first if angle is steep
        if abs(goal_rel_angle) > 1.3:
            cmd.linear.x = 0.0
            cmd.angular.z = max(-max_angular_rps, min(max_angular_rps, 1.2 * goal_rel_angle))
            self.cmd_pub.publish(cmd)
            return

        # Candidate angle evaluation for obstacle avoidance
        best_angle = goal_rel_angle
        best_score = -1e9
        corridor_half_width = 0.32
        max_lookahead = 2.0

        for step in range(-20, 21):
            cand_angle = step * 0.05
            cos_a = math.cos(cand_angle)
            sin_a = math.sin(cand_angle)

            min_dist = max_lookahead
            for bx, by in self.laser_points:
                x_rot = bx * cos_a + by * sin_a
                y_rot = -bx * sin_a + by * cos_a

                if x_rot > 0.15 and abs(y_rot) < corridor_half_width:
                    if x_rot < min_dist:
                        min_dist = x_rot

            if min_dist < 0.45:
                continue

            angle_diff = abs(cand_angle - goal_rel_angle)
            score = -angle_diff + 1.2 * (min_dist / max_lookahead)

            if score > best_score:
                best_score = score
                best_angle = cand_angle

        # Handle fully blocked situation
        if best_score == -1e9:
            left_space = sum(1 for bx, by in self.laser_points if by > 0 and math.hypot(bx, by) < 1.0)
            right_space = sum(1 for bx, by in self.laser_points if by < 0 and math.hypot(bx, by) < 1.0)
            cmd.linear.x = 0.0
            cmd.angular.z = max_angular_rps if left_space < right_space else -max_angular_rps
            self.cmd_pub.publish(cmd)
            return

        # Drive commands
        kp_w = 1.4
        w_cmd = kp_w * best_angle
        w_cmd = max(-max_angular_rps, min(max_angular_rps, w_cmd))

        speed_scale = max(0.1, math.cos(best_angle))
        if self.current_wp_idx == len(self.waypoints) - 1:
            v_cmd = min(max_linear_mps, max(0.15, dist_to_target * 0.5))
        else:
            v_cmd = max_linear_mps

        v_cmd = v_cmd * speed_scale
        v_cmd = max(0.0, min(max_linear_mps, v_cmd))

        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''

    rclpy.init(args=sys.argv)

    node = rclpy.create_node('ebot_nav_process')

    node.get_logger().info('Node created: eBot navigation process')

    ebot_nav_class = ebot_nav()

    rclpy.spin(ebot_nav_class)

    ebot_nav_class.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
