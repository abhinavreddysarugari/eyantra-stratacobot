#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    arm_waypoints.py
# Functions:        __init__, tcpposecb, jointstatecb, armstatuscb, switch_controller, process_waypoints, main
# Nodes:		    arm_waypoints_node
#                   Publishing Topics  - [ /delta_joint_cmds, /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status ]


################### IMPORT MODULES #######################

import rclpy
import sys
import numpy as np
from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32


##################### TASK CONSTANTS #######################

# Tool positions in base_link, in metres, in the order they must be reached.
# The tool stops at each one and holds it for at least two seconds.
waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430,  0.5280, 0.1967),   # 3
    (-0.4097,  0.5280, 0.1967),   # 4
    (-0.0763,  0.5280, 0.1967),   # 5
]

# The two command interfaces. Only ONE is active at a time; messages to the other are
# accepted and ignored.
servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

# The only frame a twist may be stamped with; any other is refused, not converted.
base_frame = 'base_link'

# JointJog velocities are matched to these names, in this order.
joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# What the servo accepts. A command above one of these is dropped WHOLE, not clamped.
cap_linear_mps = 0.15     # magnitude of a twist's linear part
cap_angular_rps = 0.35    # magnitude of its angular part
cap_joint_rps = 0.35      # per joint

# Dead-man switch: the arm stops this long after the last message it received.
command_timeout_s = 0.15


##################### CLASS DEFINITION #######################

class arm_waypoints(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the UR7e's tool through the
                    given waypoints using the arm's velocity command interfaces.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class arm_waypoints
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic PUBLISHERS ############

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)    # end-effector velocity, in base_link
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)        # per-joint velocity

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.switch_cli = self.create_client(                                           # client used to pick which command topic is live
            SwitchController, f'{servo_ns}/switch_controller')
        self.timer = self.create_timer(control_rate, self.process_waypoints)            # timer-based function called every 0.05s

        self.tcp_pose = None                                                            # tool pose variable (from tcpposecb())
        self.joint_angles = None                                                        # joint feedback variable (from jointstatecb())
        self.arm_status = None                                                          # arm state code variable (from armstatuscb())

        self.active_controller_ready = False
        self.switch_future = None
        self.current_stage = 0
        self.is_holding = False
        self.hold_start_time = None
        self.completed_all = False

        # Intermediate via-points and waypoint joint configurations to avoid collisions
        self.stages = [
            # Waypoint 1
            {'target_q': np.array([-1.83588, -2.64421, -1.21901,  1.15432,  0.0,      1.13812]),
             'is_wp': True,  'target_p': np.array([-0.4085, -0.5379, 0.1967]), 'name': 'Waypoint 1'},

            # Via points to Waypoint 2
            {'target_q': np.array([-1.83788, -2.36212, -1.12235,  0.97590,  0.14471,  0.94388]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 1'},
            {'target_q': np.array([-2.85096, -2.58911, -0.62442,  0.59136,  0.65094,  0.87651]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 2'},

            # Waypoint 2
            {'target_q': np.array([-2.82653, -2.52743, -0.89956,  0.28540,  0.31435,  1.57079]),
             'is_wp': True,  'target_p': np.array([-0.8000, -0.0005, 0.3967]), 'name': 'Waypoint 2'},

            # Via points to Waypoint 3
            {'target_q': np.array([-2.83060, -2.46997, -0.79696,  0.18076,  0.38931,  1.52958]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 3'},
            {'target_q': np.array([-3.60132, -2.64336, -0.23453,  3.26065, -1.48562, -1.69357]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 4'},

            # Waypoint 3
            {'target_q': np.array([-3.59901, -2.73185, -0.50014,  3.23199, -1.46824, -1.57080]),
             'is_wp': True,  'target_p': np.array([-0.7430,  0.5280, 0.1967]), 'name': 'Waypoint 3'},

            # Via points to Waypoint 4
            {'target_q': np.array([-3.59912, -2.65951, -0.47135,  3.25807, -1.46905, -1.50807]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 5'},
            {'target_q': np.array([-3.66374, -2.29920, -1.12371,  2.99631,  0.0,     -1.14419]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 6'},

            # Waypoint 4
            {'target_q': np.array([-3.73602, -2.38236, -1.09312,  2.39514, -0.89905, -1.01487]),
             'is_wp': True,  'target_p': np.array([-0.4097,  0.5280, 0.1967]), 'name': 'Waypoint 4'},

            # Via points to Waypoint 5
            {'target_q': np.array([-3.72521, -2.29425, -1.03310,  2.34576, -0.82383, -0.93816]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 7'},
            {'target_q': np.array([-4.07411, -2.13540, -1.55470,  2.49994,  0.0,     -0.38064]),
             'is_wp': False, 'target_p': None,                                 'name': 'Via 8'},

            # Waypoint 5
            {'target_q': np.array([-4.07411, -2.29719, -1.59873,  2.47555,  0.0,     -0.15043]),
             'is_wp': True,  'target_p': np.array([-0.0763,  0.5280, 0.1967]), 'name': 'Waypoint 5'},
        ]

        self.get_logger().info('Arm waypoints node initialized')


    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.
                        Stores the tool's position and orientation reported in base_link.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link
        '''
        self.tcp_pose = data.pose


    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic.
                        Stores current joint angles, matched BY NAME.

        Args:
            data (JointState):    Joint feedback published by the arm
        '''
        try:
            d = {n: p for n, p in zip(data.name, data.position)}
            self.joint_angles = np.array([d[j] for j in joint_names])
        except KeyError:
            pass


    def armstatuscb(self, data):
        '''
        Description:    Callback function for the arm status topic.
                        Stores the arm's current state code.

        Args:
            data (Int32):    One state code describing what the arm is doing
        '''
        self.arm_status = data.data


    def switch_controller(self, controller):
        '''
        Description:    Function to make one of the arm's two command interfaces the active
                        one, so that commands published to it are acted on.

        Args:
            controller  (str):      Name of the controller to activate, either
                                    'twist_controller' or 'joint_controller'

        Returns:
            success     (bool):     Whether the controller was activated
        '''
        if not self.switch_cli.service_is_ready():
            return False

        req = SwitchController.Request()
        if controller == 'joint_controller':
            req.activate_controllers = [joint_controller]
            req.deactivate_controllers = [twist_controller]
        else:
            req.activate_controllers = [twist_controller]
            req.deactivate_controllers = [joint_controller]
        req.strictness = SwitchController.Request.STRICT

        self.switch_future = self.switch_cli.call_async(req)
        return True


    def process_waypoints(self):
        '''
        Description:    Timer function used to drive the tool through the waypoints.
        '''
        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            return

        if not self.active_controller_ready:
            if self.switch_future is None:
                self.switch_controller('joint_controller')
                return
            elif self.switch_future.done():
                result = self.switch_future.result()
                if result is not None and result.ok:
                    self.active_controller_ready = True
                    self.get_logger().info('Switched to joint controller')
                else:
                    self.switch_future = None
                    return
            else:
                return

        if self.completed_all:
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.velocities = [0.0] * 6
            self.joint_pub.publish(msg)
            return

        stage_info = self.stages[self.current_stage]
        target_q = stage_info['target_q']
        is_wp = stage_info['is_wp']
        target_p = stage_info['target_p']
        stage_name = stage_info['name']

        curr_p = np.array([self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z])
        q_err = target_q - self.joint_angles
        max_q_err = np.max(np.abs(q_err))
        p_err = np.linalg.norm(target_p - curr_p) if is_wp else 0.0

        if self.is_holding:
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.velocities = [0.0] * 6
            self.joint_pub.publish(msg)

            hold_elapsed = (self.get_clock().now() - self.hold_start_time).nanoseconds / 1e9
            if hold_elapsed >= 2.5:
                self.get_logger().info(f'Finished hold at {stage_name}')
                self.is_holding = False
                self.current_stage += 1
                if self.current_stage >= len(self.stages):
                    self.completed_all = True
                    self.get_logger().info('Finished all waypoints.')
            return

        if is_wp and p_err < 0.007 and max_q_err < 0.03:
            self.get_logger().info(f'Reached {stage_name}, holding...')
            self.is_holding = True
            self.hold_start_time = self.get_clock().now()

            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.velocities = [0.0] * 6
            self.joint_pub.publish(msg)
            return

        elif not is_wp and max_q_err < 0.05:
            self.get_logger().info(f'Passed {stage_name}')
            self.current_stage += 1
            if self.current_stage >= len(self.stages):
                self.completed_all = True
            return

        # P-controller on joint errors
        kp = 1.0
        v_joints = kp * q_err
        v_max = np.max(np.abs(v_joints))
        if v_max > 0.22:
            v_joints = v_joints * (0.22 / v_max)

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = [float(v) for v in v_joints]
        self.joint_pub.publish(msg)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    arm_waypoints class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('arm_waypoints_process')               # creating ROS node

    node.get_logger().info('Node created: Arm waypoints process')   # logging information

    arm_waypoints_class = arm_waypoints()                           # creating a new object for class 'arm_waypoints'

    rclpy.spin(arm_waypoints_class)                                 # spinning on the object to make it alive in ROS 2 DDS

    arm_waypoints_class.destroy_node()                              # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()
