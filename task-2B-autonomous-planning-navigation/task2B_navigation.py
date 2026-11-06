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
# Filename:		    task2B_navigation.py
# Functions:        __init__, pathcb, odomcb, process_navigation, main
# Nodes:		    ebot_nav_node
#                   Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'

# Velocity limits for eBot mobile base
max_linear_mps = 0.45
max_angular_rps = 1.0


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the planned path
                    published on /ebot_path for each step of the lap.
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

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # 20 Hz control loop
        self.timer = self.create_timer(control_rate, self.process_navigation)

        self.waypoints = []
        self.current_path_id = None
        self.path_seq = 0
        self.target_end_yaw = 0.0
        self.closest_idx = 0
        self.completed = True                                                           # True until first path is received
        self.at_goal = False

        self.current_x = None
        self.current_y = None
        self.yaw = None

        self.lookahead = 0.30                                                           # Pure pursuit lookahead distance (m)

        self.get_logger().info('ebot_nav_node initialized successfully.')


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Receives the planned path poses for each step.

        Args:
            data (Path): The route, as a sequence of poses
        '''
        if not data.poses or len(data.poses) < 2:
            return

        # Unique identifier for the received path
        first_pose = data.poses[0].pose.position
        last_pose = data.poses[-1].pose.position
        new_path_id = (
            round(first_pose.x, 3), round(first_pose.y, 3),
            round(last_pose.x, 3), round(last_pose.y, 3),
            len(data.poses)
        )

        if new_path_id != self.current_path_id:
            self.current_path_id = new_path_id
            self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in data.poses]

            # Extract target orientation quaternion of the final pose
            last_ori = data.poses[-1].pose.orientation
            siny = 2.0 * (last_ori.w * last_ori.z + last_ori.x * last_ori.y)
            cosy = 1.0 - 2.0 * (last_ori.y * last_ori.y + last_ori.z * last_ori.z)
            self.target_end_yaw = math.atan2(siny, cosy)

            self.closest_idx = 0
            self.completed = False
            self.at_goal = False
            self.path_seq += 1

            self.get_logger().info(
                f'Received Path #{self.path_seq} with {len(self.waypoints)} waypoints. '
                f'Goal: ({last_pose.x:.2f}, {last_pose.y:.2f}), target yaw: {self.target_end_yaw:.2f} rad'
            )


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Stores the current position and heading of the base.

        Args:
            data (Odometry): Pose and velocity of the base
        '''
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y

        qx = data.pose.pose.orientation.x
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w

        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.yaw = math.atan2(siny, cosy)


    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the planned path
                        using pure pursuit path tracking.
        '''
        cmd = Twist()

        if self.completed:
            self.cmd_pub.publish(cmd)
            return

        if not self.waypoints or self.current_x is None or self.yaw is None:
            return

        rx, ry, ryaw = self.current_x, self.current_y, self.yaw
        gx, gy = self.waypoints[-1]
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        # 1. Goal arrival and alignment (latched once within 0.18 m)
        if dist_to_goal < 0.18:
            self.at_goal = True

        if getattr(self, 'at_goal', False):
            # Position reached; align yaw to target_end_yaw
            yaw_err = math.atan2(
                math.sin(self.target_end_yaw - ryaw),
                math.cos(self.target_end_yaw - ryaw)
            )

            # For ore drop and arm pose, +-pi is also a valid direction.
            # For home (0.0, 0.0) in Step 3, yaw MUST BE strictly 0.0!
            is_home = math.hypot(gx, gy) < 0.5
            if not is_home and (abs(abs(self.target_end_yaw) - math.pi) < 0.1 or abs(self.target_end_yaw) < 0.1):
                alt_yaw_err = math.atan2(
                    math.sin((self.target_end_yaw + math.pi) - ryaw),
                    math.cos((self.target_end_yaw + math.pi) - ryaw)
                )
                if abs(alt_yaw_err) < abs(yaw_err):
                    yaw_err = alt_yaw_err

            if abs(yaw_err) > 0.08:
                # Rotate gently on the spot to align within tolerance
                w_cmd = max(-0.5, min(0.5, 1.2 * yaw_err))
                cmd.linear.x = 0.0
                cmd.angular.z = w_cmd
                self.cmd_pub.publish(cmd)
                return
            else:
                # Position and heading aligned: stop completely
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                self.completed = True
                self.at_goal = False
                self.get_logger().info(
                    f'Path #{self.path_seq} reached end pose! Dist: {dist_to_goal:.3f} m, yaw err: {yaw_err:.3f} rad. Standing still.'
                )
                return

        # 2. Update closest index along waypoints (monotonic forward search)
        search_start = self.closest_idx
        search_end = min(len(self.waypoints), self.closest_idx + 50)
        best_d = float('inf')
        best_i = self.closest_idx

        for i in range(search_start, search_end):
            d = math.hypot(self.waypoints[i][0] - rx, self.waypoints[i][1] - ry)
            if d < best_d:
                best_d = d
                best_i = i

        self.closest_idx = best_i

        # 3. Find lookahead target point
        lookahead_pt = self.waypoints[-1]
        for i in range(self.closest_idx, min(len(self.waypoints), self.closest_idx + 60)):
            d = math.hypot(self.waypoints[i][0] - rx, self.waypoints[i][1] - ry)
            if d >= self.lookahead:
                lookahead_pt = self.waypoints[i]
                break


        # 4. Pure pursuit steering
        tx, ty = lookahead_pt
        target_angle = math.atan2(ty - ry, tx - rx)
        alpha = math.atan2(math.sin(target_angle - ryaw), math.cos(target_angle - ryaw))

        # Corridor safety: in narrow corridor driving south (Step 3), strictly align before moving forward
        is_south_corridor = (rx < 0.35 and ty < ry and dist_to_goal > 0.45)
        turn_threshold = 0.18 if is_south_corridor else 0.80

        if abs(alpha) > turn_threshold:
            # Turn on the spot with solid torque to face the path
            cmd.linear.x = 0.0
            cmd.angular.z = math.copysign(max(0.7, min(max_angular_rps, 1.8 * abs(alpha))), alpha)
        else:
            # Speed regulation: smoothly slow down when approaching goal
            if dist_to_goal < 0.8:
                v_target = max(0.12, dist_to_goal * 0.5)
            else:
                v_target = max_linear_mps

            # Modulate linear speed by angular alignment
            v_cmd = v_target * max(0.15, math.cos(alpha))

            # Curvature calculation: w = 2 * v * sin(alpha) / L
            w_cmd = (2.0 * v_cmd * math.sin(alpha)) / self.lookahead
            w_cmd = max(-max_angular_rps, min(max_angular_rps, w_cmd))

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
    ebot_nav_class = ebot_nav()
    rclpy.spin(ebot_nav_class)
    ebot_nav_class.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
