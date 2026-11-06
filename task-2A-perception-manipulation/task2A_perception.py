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
# Filename:		    task2A_perception.py
# Functions:        detect_ores, depthimagecb, colorimagecb, caminfocb, process_image, main
# Nodes:		    task2A_perception_node
#                   Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                                          /camera/camera/aligned_depth_to_color/image_raw,
#                                          /camera/camera/color/camera_info ]


################### IMPORT MODULES #######################

import rclpy
import sys
import os
import cv2
import math
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image


##################### TASK CONSTANTS #######################

ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

base_frame = 'base_link'


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image                   (numpy.ndarray): Input colour image frame (BGR format)

    Returns:
        center_ore_list         (list): Center pixel (cX, cY) of every ore detected in the frame
        ore_type_list           (list): Type of each ore detected, taken from 'ore_types'
    '''

    center_ore_list = []
    ore_type_list = []
    contour_map = {}

    if image is None or image.size == 0:
        return center_ore_list, ore_type_list

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    # Conveyor belt region of interest in camera frame
    roi = np.zeros(image.shape[:2], dtype=bool)
    roi[140:475, 420:635] = True

    # Color thresholding masks
    # 1. Azurite: dominant Blue channel, low Red
    mask_azurite = roi & (b > 130) & (b > g) & (r < 70)

    # 2. Malachite: dominant Green channel, low Red
    mask_malachite = roi & (g > 150) & (g > b) & (r < 100)

    # 3. Vanadinite: high Red, moderate Green, low Blue
    mask_vanadinite = roi & (r > 160) & (g > 30) & (g < 140) & (b < 50)

    type_masks = [
        ('azurite_ore', mask_azurite),
        ('malachite_ore', mask_malachite),
        ('vanadinite_ore', mask_vanadinite),
    ]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for ore_name, mask_bool in type_masks:
        mask_u8 = (mask_bool * 255).astype(np.uint8)
        mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 15 < area < 2000:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    center_ore_list.append((cX, cY))
                    ore_type_list.append(ore_name)
                    contour_map[(cX, cY)] = c

    detect_ores.last_contours = contour_map

    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################

class task2A_perception(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores on the conveyor
                    and broadcast a transform for each one relative to base_link.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class task2A_perception
        '''

        super().__init__('task2A_perception_node')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.1                                                              # 10 Hz
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(control_rate, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None
        self.latest_image_stamp = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.optical_frame_id = 'camera_color_optical_frame'

        # Tracked ores dictionary: {name: {'pos': np.array([x, y, z]), 'pixel': (cX, cY)}}
        self.tracked_ores = {}
        self.logged_ready = False


    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Converts ROS Image to CV2 float depth in metres.
        '''
        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                self.depth_image = depth.astype(np.float32) / 1000.0
            else:
                self.depth_image = depth.astype(np.float32)
        except CvBridgeError as error:
            self.get_logger().error(f'Depth conversion failed: {error}')


    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Converts ROS Image to CV2 BGR8 image.
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            self.latest_image_stamp = data.header.stamp
            if data.header.frame_id:
                self.optical_frame_id = data.header.frame_id
        except CvBridgeError as error:
            self.get_logger().error(f'Colour conversion failed: {error}')


    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Extracts focal lengths and principal point.
        '''
        self.cam_info = data
        self.fx = float(data.k[0])
        self.cx = float(data.k[2])
        self.fy = float(data.k[4])
        self.cy = float(data.k[5])
        if data.header.frame_id:
            self.optical_frame_id = data.header.frame_id


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.
        '''

        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        try:
            cam_to_base_tf = self.tf_buffer.lookup_transform(
                base_frame,
                self.optical_frame_id,
                rclpy.time.Time()
            )
        except Exception:
            return

        center_ore_list, ore_type_list = detect_ores(self.cv_image)
        contour_map = getattr(detect_ores, 'last_contours', {})

        detections_by_type = {t: [] for t in ore_types}
        for (cX, cY), ore_type in zip(center_ore_list, ore_type_list):
            if ore_type in detections_by_type:
                h, w = self.depth_image.shape
                patch = self.depth_image[max(0, cY - 3):min(h, cY + 4),
                                         max(0, cX - 3):min(w, cX + 4)]
                valid_patch = patch[np.isfinite(patch) & (patch > 0.1)]
                if len(valid_patch) == 0:
                    continue
                z_cam = float(np.median(valid_patch))

                x_cam = float((cX - self.cx) * z_cam / self.fx)
                y_cam = float((cY - self.cy) * z_cam / self.fy)

                pt_cam = PointStamped()
                pt_cam.header.frame_id = self.optical_frame_id
                pt_cam.point.x = x_cam
                pt_cam.point.y = y_cam
                pt_cam.point.z = z_cam

                pt_base = tf2_geometry_msgs.do_transform_point(pt_cam, cam_to_base_tf)
                pos_base = np.array([pt_base.point.x, pt_base.point.y, pt_base.point.z], dtype=float)

                detections_by_type[ore_type].append({
                    'pixel': (cX, cY),
                    'pos': pos_base,
                    'contour': contour_map.get((cX, cY), None)
                })

        # Process each ore kind and track consistent labels
        for t in ore_types:
            dets = detections_by_type[t]
            if len(dets) == 0:
                continue

            existing_for_type = [k for k in self.tracked_ores.keys() if k.startswith(f'{t}_')]

            if len(existing_for_type) == 0:
                # Initial frame: deterministic ordering by pixel Y coordinate along conveyor
                dets_sorted = sorted(dets, key=lambda d: d['pixel'][1])
                for idx, det in enumerate(dets_sorted):
                    track_name = f'{t}_{idx + 1}'
                    self.tracked_ores[track_name] = {
                        'pos': det['pos'],
                        'pixel': det['pixel'],
                        'contour': det['contour']
                    }
            else:
                # Subsequent frames: associate by nearest distance in base_link
                assigned_dets = set()
                for track_name in sorted(existing_for_type):
                    track_pos = self.tracked_ores[track_name]['pos']
                    best_i = None
                    best_dist = 1e9
                    for i, det in enumerate(dets):
                        if i in assigned_dets:
                            continue
                        dist = float(np.linalg.norm(det['pos'] - track_pos))
                        if dist < best_dist:
                            best_dist = dist
                            best_i = i

                    if best_i is not None and best_dist < 0.20:
                        assigned_dets.add(best_i)
                        det = dets[best_i]
                        old_pos = self.tracked_ores[track_name]['pos']
                        self.tracked_ores[track_name]['pos'] = 0.8 * old_pos + 0.2 * det['pos']
                        self.tracked_ores[track_name]['pixel'] = det['pixel']
                        self.tracked_ores[track_name]['contour'] = det['contour']

                # Any new detections not matched to existing tracks
                unassigned = [det for i, det in enumerate(dets) if i not in assigned_dets]
                for det in unassigned:
                    next_idx = 1
                    while f'{t}_{next_idx}' in self.tracked_ores:
                        next_idx += 1
                    track_name = f'{t}_{next_idx}'
                    self.tracked_ores[track_name] = {
                        'pos': det['pos'],
                        'pixel': det['pixel'],
                        'contour': det['contour']
                    }

        # Publish transforms for all tracked ores
        stamp = self.get_clock().now().to_msg()
        for ore_name, ore_data in self.tracked_ores.items():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = base_frame
            t.child_frame_id = ore_name
            t.transform.translation.x = float(ore_data['pos'][0])
            t.transform.translation.y = float(ore_data['pos'][1])
            t.transform.translation.z = float(ore_data['pos'][2])
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.br.sendTransform(t)

        if len(self.tracked_ores) == 4 and not self.logged_ready:
            self.logged_ready = True
            self.get_logger().info(f'All 4 ores detected and tracked: {list(self.tracked_ores.keys())}')


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    task2A_perception class to perform its task
    '''

    rclpy.init(args=sys.argv)
    perception_node = task2A_perception()
    perception_node.get_logger().info('Task 2A perception node started.')

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
