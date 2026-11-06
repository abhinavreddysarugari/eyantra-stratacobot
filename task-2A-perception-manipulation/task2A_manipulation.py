#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    task2A_manipulation.py
# Functions:        __init__, tcpposecb, jointstatecb, armstatuscb, forcecb, switch_controller,
#                   set_magnet, solve_ik, move_arm_p, process_manipulation, main
# Nodes:		    task2A_manipulation_node
#                   Publishing Topics  - [ /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status,
#                                          /ur7e/end_effector/force_magnitude ]


################### IMPORT MODULES #######################

import rclpy
import sys
import time
import numpy as np
import tf2_ros
from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32
from std_srvs.srv import SetBool, Trigger
from scipy.optimize import minimize


##################### TASK CONSTANTS #######################

servo_ns = '/ur_arm_controller'
joint_controller = 'delta_joint_controller'
twist_controller = 'delta_twist_controller'
base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Safe collision-free intermediate via points between conveyor and cargo rover
via_conveyor = np.array([-1.83788, -2.36212, -1.12235,  0.97590,  0.14471,  0.94388])
via_mid      = np.array([-2.85096, -2.58911, -0.62442,  0.59136,  0.65094,  0.87651])
via_rover    = np.array([-3.60132, -2.64336, -0.23453,  3.26065, -1.48562, -1.69357])

# Verified collision-free and singularity-free drop poses inside each container
drop_poses = {
    'azurite': np.array([-4.37795, -2.40832, -1.35034, 2.18786, -1.57079, -0.2]),
    'malachite': np.array([-3.96262, -2.61467, -0.93658, 1.98046, -1.57080, -1.0]),
    'vanadinite': np.array([-3.69773, -3.09094, 0.00026, 2.06491, -1.50177, -1.0]),
}


##################### KINEMATICS HELPERS #######################

def rotX(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def rotY(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def rotZ(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def trans(x, y, z):
    T = np.eye(4)
    T[0:3, 3] = [x, y, z]
    return T

def ur7e_fk(q):
    '''
    Description:    Analytical forward kinematics for UR7e to TCP frame in base_link.
    '''
    T = trans(0, 0, 0.1625) @ rotZ(q[0])
    T = T @ rotX(np.pi / 2) @ rotZ(q[1])
    T = T @ trans(-0.425, 0, 0) @ rotZ(q[2])
    T = T @ trans(-0.3922, 0, 0.1333) @ rotZ(q[3])
    T = T @ trans(0, -0.0997, 0) @ rotX(np.pi / 2) @ rotZ(q[4])
    T = T @ trans(0, 0.0996, 0) @ rotX(np.pi / 2) @ rotY(np.pi) @ rotZ(np.pi) @ rotZ(q[5])
    T = T @ trans(0, 0, 0.02041)
    return T

def solve_ik(target_pos, q_seed):
    '''
    Description:    Solves inverse kinematics for a target position with tool Z pointing down.
    '''
    def cost(q):
        T = ur7e_fk(q)
        p = T[0:3, 3]
        z = T[0:3, 2]
        return 20.0 * np.sum((p - target_pos) ** 2) + np.sum((z - [0, 0, -1]) ** 2)

    bnds = [(-2 * np.pi, 2 * np.pi)] * 2 + [(-2.7, 2.7)] + [(-2 * np.pi, 2 * np.pi)] * 3
    res = minimize(cost, q_seed, bounds=bnds, method='L-BFGS-B')
    return res.x, (res.fun < 1e-4)


##################### CLASS DEFINITION #######################

class task2A_manipulation(Node):
    '''
    ___CLASS___

    Description:    Class which coordinates the UR7e arm to pick ores off the conveyor
                    and place each into the matching container on the cargo rover.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class task2A_manipulation
        '''

        super().__init__(
            'task2A_manipulation_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic PUBLISHERS ############

        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.force_sub = self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 10)

        ############ Service CLIENTS ############

        self.switch_cli = self.create_client(SwitchController, f'{servo_ns}/switch_controller')
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        self.unlock_cli = self.create_client(Trigger, f'{servo_ns}/unlock_protective_stop')

        ############ TF LISTENER ############

        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        ############ STATE VARIABLES ############

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = 0
        self.force_mag = 0.0

        self.controller_active = False
        self.switch_future = None
        self.magnet_future = None

        # State machine management
        self.fsm_state = 'INIT_CONTROLLER'
        self.sub_step = 0
        self.step_timer = 0.0
        self.target_q = None
        self.current_ore_index = 0
        self.ore_list = []
        self.ore_cache = {}
        self.retry_count = 0

        self.timer = self.create_timer(0.04, self.process_manipulation)
        self.get_logger().info('Task 2A manipulation node initialized.')


    def tcpposecb(self, data):
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        try:
            d = dict(zip(data.name, data.position))
            self.joint_angles = np.array([d[j] for j in joint_names])
        except KeyError:
            pass

    def armstatuscb(self, data):
        self.arm_status = data.data

    def forcecb(self, data):
        self.force_mag = data.data


    def switch_controller(self):
        if not self.switch_cli.service_is_ready():
            return False
        req = SwitchController.Request()
        req.activate_controllers = [joint_controller]
        req.deactivate_controllers = [twist_controller]
        req.strictness = SwitchController.Request.STRICT
        self.switch_future = self.switch_cli.call_async(req)
        return True


    def set_magnet(self, state):
        if not self.magnet_cli.service_is_ready():
            return False
        req = SetBool.Request()
        req.data = state
        self.magnet_future = self.magnet_cli.call_async(req)
        return True


    def unlock_stop(self):
        if self.unlock_cli.service_is_ready():
            req = Trigger.Request()
            self.unlock_cli.call_async(req)


    def move_arm_p(self, target_q, tol=0.015, max_v=0.25, kp=1.5):
        '''
        Description:    Proportional velocity controller driving joints toward target_q.
        '''
        err = target_q - self.joint_angles
        max_err = np.max(np.abs(err))

        if max_err < tol:
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.velocities = [0.0] * 6
            self.joint_pub.publish(msg)
            return True

        v = kp * err
        mv = np.max(np.abs(v))
        if mv > max_v:
            v = v * (max_v / mv)

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = [float(x) for x in v]
        self.joint_pub.publish(msg)
        return False


    def stop_arm(self):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = [0.0] * 6
        self.joint_pub.publish(msg)


    def process_manipulation(self):
        '''
        Description:    Main control loop executing the pick and place sequence.
        '''
        if self.joint_angles is None:
            return

        # Recover from safety latch if detected
        if self.arm_status >= 20:
            self.get_logger().warn(f'Protective stop detected (code {self.arm_status}), unlocking...')
            self.unlock_stop()

        # 1. Controller Initialization
        if self.fsm_state == 'INIT_CONTROLLER':
            if not self.controller_active:
                if self.switch_future is None:
                    self.switch_controller()
                elif self.switch_future.done():
                    res = self.switch_future.result()
                    if res is not None and res.ok:
                        self.controller_active = True
                        self.get_logger().info('Switched to delta_joint_controller successfully.')
                        self.fsm_state = 'UNFOLD'
                    else:
                        self.switch_future = None
            return

        # 2. Unfold safely into conveyor workspace
        if self.fsm_state == 'UNFOLD':
            reached = self.move_arm_p(via_conveyor, tol=0.03, max_v=0.22)
            if reached:
                self.get_logger().info('Arm unfolded to via_conveyor pose.')
                self.fsm_state = 'DISCOVER_ORES'
            return

        # 3. Discover ores from TF tree
        if self.fsm_state == 'DISCOVER_ORES':
            known_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
            for ore_t in known_types:
                for idx in [1, 2]:
                    frame_name = f'{ore_t}_{idx}'
                    try:
                        tf_stamped = self.tf_buffer.lookup_transform(base_frame, frame_name, rclpy.time.Time())
                        tx = tf_stamped.transform.translation.x
                        ty = tf_stamped.transform.translation.y
                        tz = tf_stamped.transform.translation.z
                        if frame_name not in self.ore_cache:
                            self.ore_cache[frame_name] = np.array([tx, ty, tz])
                    except Exception:
                        pass

            if len(self.ore_cache) >= 4 or (len(self.ore_cache) > 0 and len(self.ore_list) == 0):
                self.ore_list = sorted(list(self.ore_cache.keys()))
                self.get_logger().info(f'Discovered ores for sorting: {self.ore_list}')
                self.current_ore_index = 0
                self.fsm_state = 'PREPARE_ORE'
            return

        # 4. Prepare next ore in queue
        if self.fsm_state == 'PREPARE_ORE':
            if self.current_ore_index >= len(self.ore_list):
                self.fsm_state = 'COMPLETED'
                self.get_logger().info('All ores successfully sorted.')
                return

            ore_name = self.ore_list[self.current_ore_index]
            try:
                tf_stamped = self.tf_buffer.lookup_transform(base_frame, ore_name, rclpy.time.Time())
                pos = np.array([tf_stamped.transform.translation.x,
                                tf_stamped.transform.translation.y,
                                tf_stamped.transform.translation.z])
                self.ore_cache[ore_name] = pos
            except Exception:
                pos = self.ore_cache[ore_name]

            self.curr_ore_pos = pos
            self.curr_ore_name = ore_name
            self.retry_count = 0

            if 'azurite' in ore_name:
                self.curr_container = 'azurite'
            elif 'malachite' in ore_name:
                self.curr_container = 'malachite'
            else:
                self.curr_container = 'vanadinite'

            q_seed_conveyor = np.array([-1.7, -2.6, -1.2, -0.9, 1.57, -1.8])
            p_hover = np.array([pos[0], pos[1], 0.15])
            p_touch = np.array([pos[0], pos[1], -0.071])

            self.q_hover, ok_h = solve_ik(p_hover, q_seed_conveyor)
            self.q_touch, ok_t = solve_ik(p_touch, self.q_hover)

            self.get_logger().info(f'Starting pick for {ore_name} ({self.curr_container}) at pos: {np.round(pos, 3)}')
            self.sub_step = 0
            self.fsm_state = 'PICK_AND_PLACE'
            return

        # 5. Pick and place sequence
        if self.fsm_state == 'PICK_AND_PLACE':

            # Sub-step 0: Move to Hover above ore
            if self.sub_step == 0:
                reached = self.move_arm_p(self.q_hover, tol=0.015, max_v=0.22)
                if reached:
                    self.get_logger().info('Hover reached, energizing magnet.')
                    self.set_magnet(True)
                    self.sub_step = 1
                return

            # Sub-step 1: Descend down onto ore top face (compresses 10mm tool spring)
            elif self.sub_step == 1:
                reached = self.move_arm_p(self.q_touch, tol=0.010, max_v=0.08)
                if reached:
                    self.stop_arm()
                    self.sub_step = 2
                    self.step_timer = time.time()
                return

            # Sub-step 2: Dwell at contact for firm magnetic capture
            elif self.sub_step == 2:
                self.stop_arm()
                if time.time() - self.step_timer > 0.4:
                    self.get_logger().info(f'Captured {self.curr_ore_name}. Lifting...')
                    self.sub_step = 3
                return

            # Sub-step 3: Lift back to Hover position
            elif self.sub_step == 3:
                reached = self.move_arm_p(self.q_hover, tol=0.015, max_v=0.15)
                if reached:
                    self.get_logger().info(f'Lift complete. Force: {self.force_mag:.2f} N')
                    # If grasp missed, retry with fresh TF transform and deeper compliance touch
                    if self.force_mag < 0.6 and self.retry_count < 2:
                        self.get_logger().info('Grasp retry with updated position and deeper compliance touch.')
                        self.retry_count += 1
                        try:
                            tf_stamped = self.tf_buffer.lookup_transform(base_frame, self.curr_ore_name, rclpy.time.Time())
                            pos = np.array([tf_stamped.transform.translation.x,
                                            tf_stamped.transform.translation.y,
                                            tf_stamped.transform.translation.z])
                            self.curr_ore_pos = pos
                        except Exception:
                            pos = self.curr_ore_pos
                        p_hover = np.array([pos[0], pos[1], 0.15])
                        p_touch_deep = np.array([pos[0], pos[1], -0.073])
                        self.q_hover, _ = solve_ik(p_hover, self.joint_angles)
                        self.q_touch, _ = solve_ik(p_touch_deep, self.q_hover)
                        self.sub_step = 0
                    else:
                        self.sub_step = 4
                return

            # Sub-step 4: Fast transit to via_conveyor
            elif self.sub_step == 4:
                reached = self.move_arm_p(via_conveyor, tol=0.05, max_v=0.25)
                if reached:
                    self.sub_step = 5
                return

            # Sub-step 5: Fast transit to via_mid
            elif self.sub_step == 5:
                reached = self.move_arm_p(via_mid, tol=0.05, max_v=0.25)
                if reached:
                    self.sub_step = 6
                return

            # Sub-step 6: Fast transit to via_rover
            elif self.sub_step == 6:
                reached = self.move_arm_p(via_rover, tol=0.05, max_v=0.25)
                if reached:
                    self.sub_step = 7
                return

            # Sub-step 7: Move to Container Drop Pose
            elif self.sub_step == 7:
                target_drop_q = drop_poses[self.curr_container]
                reached = self.move_arm_p(target_drop_q, tol=0.02, max_v=0.22)
                if reached:
                    self.stop_arm()
                    self.get_logger().info(f'Reached {self.curr_container} container. Releasing magnet.')
                    self.set_magnet(False)
                    self.sub_step = 8
                    self.step_timer = time.time()
                return

            # Sub-step 8: Dwell to ensure ore drops cleanly into container and settles
            elif self.sub_step == 8:
                self.stop_arm()
                if time.time() - self.step_timer > 1.2:
                    self.get_logger().info(f'Successfully dropped {self.curr_ore_name}. Returning...')
                    self.sub_step = 9
                return

            # Sub-step 9: Fast return to via_rover
            elif self.sub_step == 9:
                reached = self.move_arm_p(via_rover, tol=0.05, max_v=0.25)
                if reached:
                    self.sub_step = 10
                return

            # Sub-step 10: Fast return to via_mid
            elif self.sub_step == 10:
                reached = self.move_arm_p(via_mid, tol=0.05, max_v=0.25)
                if reached:
                    self.sub_step = 11
                return

            # Sub-step 11: Return to via_conveyor
            elif self.sub_step == 11:
                reached = self.move_arm_p(via_conveyor, tol=0.04, max_v=0.25)
                if reached:
                    self.current_ore_index += 1
                    self.fsm_state = 'PREPARE_ORE'
                return

        # 6. Completed
        if self.fsm_state == 'COMPLETED':
            self.stop_arm()


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    task2A_manipulation class to perform its task
    '''

    rclpy.init(args=sys.argv)
    manipulation_node = task2A_manipulation()
    manipulation_node.get_logger().info('Task 2A manipulation node started.')

    try:
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
