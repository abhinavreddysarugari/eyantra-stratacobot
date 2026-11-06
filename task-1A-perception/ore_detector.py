#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    ore_detector.py
# Functions:        detect_ores, depthimagecb, colorimagecb, caminfocb, process_image, main
# Nodes:		    ore_tf_publisher
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

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
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
    contour_map = {}  # stores contour corresponding to each (cX, cY)

    if image is None or image.size == 0:
        return center_ore_list, ore_type_list

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    # Conveyor belt region of interest (ROI) in camera frame
    roi = np.zeros(image.shape[:2], dtype=bool)
    roi[160:470, 430:635] = True

    # Color thresholding:
    # 1. Azurite (Blue face): dominant Blue channel, low Red
    mask_azurite = roi & (b > 130) & (b > g) & (r < 70)

    # 2. Malachite (Green face): dominant Green channel, low Red
    mask_malachite = roi & (g > 170) & (g > b) & (r < 85)

    # 3. Vanadinite (Orange face): high Red, moderate Green, minimal Blue (rejects Martian soil)
    mask_vanadinite = roi & (r > 175) & (g > 30) & (g < 135) & (b < 40)

    type_masks = [
        ('azurite_ore', mask_azurite),
        ('malachite_ore', mask_malachite),
        ('vanadinite_ore', mask_vanadinite),
    ]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for ore_name, mask_bool in type_masks:
        mask_u8 = (mask_bool * 255).astype(np.uint8)
        mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            # Area filter to reject noise and background structures
            if 40 < area < 1500:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    center_ore_list.append((cX, cY))
                    ore_type_list.append(ore_name)
                    contour_map[(cX, cY)] = c

    # Attach contour map to function object for annotation without breaking 2-tuple signature
    detect_ores.last_contours = contour_map

    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and
                    broadcast a transform for each one.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''

        super().__init__('ore_tf_publisher')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # 5 Hz
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(image_processing_rate, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None
        self.latest_image_stamp = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.optical_frame_id = 'camera_color_optical_frame'

        # Dictionary tracking persistent ores: {ore_transform_name: {'pos': [x, y, z], 'pixel': (cX, cY)}}
        self.tracked_ores = {}
        self.image_saved = False


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

        # Return early until images and camera intrinsics have arrived
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        # Lookup transform from camera optical frame to base_frame (base_link)
        try:
            cam_to_base_tf = self.tf_buffer.lookup_transform(
                base_frame,
                self.optical_frame_id,
                rclpy.time.Time()
            )
        except Exception:
            # TF tree not yet populated
            return

        # Detect ore centers and types
        center_ore_list, ore_type_list = detect_ores(self.cv_image)
        contour_map = getattr(detect_ores, 'last_contours', {})

        # Group detections by ore type
        detections_by_type = {t: [] for t in ore_types}
        for (cX, cY), ore_type in zip(center_ore_list, ore_type_list):
            if ore_type in detections_by_type:
                # Read median depth over small window centered at (cX, cY)
                h, w = self.depth_image.shape
                patch = self.depth_image[max(0, cY - 4):min(h, cY + 5),
                                         max(0, cX - 4):min(w, cX + 5)]
                valid_patch = patch[np.isfinite(patch) & (patch > 0.0)]
                if len(valid_patch) == 0:
                    continue
                z_cam = float(np.median(valid_patch))

                # Deproject (u, v, z) into 3D point in camera optical frame
                x_cam = float((cX - self.cx) * z_cam / self.fx)
                y_cam = float((cY - self.cy) * z_cam / self.fy)

                # Transform 3D point into base_link
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

        # Update persistent tracked ores with consistent ID assignment
        for t in ore_types:
            dets = detections_by_type[t]
            track_1_name = f'{t}_1'
            track_2_name = f'{t}_2'

            has_track_1 = track_1_name in self.tracked_ores
            has_track_2 = track_2_name in self.tracked_ores

            if not has_track_1 or not has_track_2:
                # Initial assignment: sort deterministically by conveyor position (pixel cY)
                if len(dets) >= 2:
                    dets_sorted = sorted(dets, key=lambda d: d['pixel'][1])
                    self.tracked_ores[track_1_name] = {
                        'pos': dets_sorted[0]['pos'],
                        'pixel': dets_sorted[0]['pixel'],
                        'contour': dets_sorted[0]['contour']
                    }
                    self.tracked_ores[track_2_name] = {
                        'pos': dets_sorted[1]['pos'],
                        'pixel': dets_sorted[1]['pixel'],
                        'contour': dets_sorted[1]['contour']
                    }
                elif len(dets) == 1:
                    # Single detection fallback
                    if not has_track_1:
                        self.tracked_ores[track_1_name] = {
                            'pos': dets[0]['pos'],
                            'pixel': dets[0]['pixel'],
                            'contour': dets[0]['contour']
                        }
            else:
                # Tracks already established: match each detection to nearest known track by pixel distance
                for d in dets:
                    d_pix = np.array(d['pixel'])
                    p1 = np.array(self.tracked_ores[track_1_name]['pixel'])
                    p2 = np.array(self.tracked_ores[track_2_name]['pixel'])
                    dist1 = np.linalg.norm(d_pix - p1)
                    dist2 = np.linalg.norm(d_pix - p2)

                    matched_name = track_1_name if dist1 < dist2 else track_2_name
                    # Exponential smoothing on 3D position to eliminate depth noise
                    old_pos = self.tracked_ores[matched_name]['pos']
                    smoothed_pos = 0.8 * old_pos + 0.2 * d['pos']
                    self.tracked_ores[matched_name]['pos'] = smoothed_pos
                    self.tracked_ores[matched_name]['pixel'] = d['pixel']
                    if d['contour'] is not None:
                        self.tracked_ores[matched_name]['contour'] = d['contour']

        # Broadcast transforms for all tracked ores on every cycle
        stamp = self.latest_image_stamp if self.latest_image_stamp is not None else self.get_clock().now().to_msg()
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

        # Generate annotated frame once all 6 ores are detected
        if len(self.tracked_ores) == 6:
            annotated_frame = self.cv_image.copy()

            for ore_name, ore_data in self.tracked_ores.items():
                cX, cY = ore_data['pixel']
                cnt = ore_data.get('contour', None)

                # Draw boundary around the ore
                if cnt is not None:
                    cv2.drawContours(annotated_frame, [cnt], -1, (255, 255, 255), 2)
                else:
                    cv2.circle(annotated_frame, (cX, cY), 15, (255, 255, 255), 2)

                # Draw center point
                cv2.circle(annotated_frame, (cX, cY), 3, (0, 0, 255), -1)

                # Draw transform name label
                label_pos = (cX + 12, cY + 4)
                cv2.putText(
                    annotated_frame,
                    ore_name,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame,
                    ore_name,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

            # Save annotated frame to file as required by submission
            out_paths = [
                '/home/aditya/ros2_ws/src/SC#4096_task1A_detection.png',
                'SC#4096_task1A_detection.png'
            ]
            for p in out_paths:
                try:
                    cv2.imwrite(p, annotated_frame)
                except Exception as e:
                    self.get_logger().warn(f'Could not write annotated image to {p}: {e}')

            if not self.image_saved:
                self.image_saved = True
                self.get_logger().info('All 6 ores detected and tracked. Annotated frame saved.')

            # Optionally display detections if GUI is available
            try:
                cv2.imshow('Ore Detections', annotated_frame)
                cv2.waitKey(1)
            except cv2.error:
                pass


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task
    '''

    rclpy.init(args=sys.argv)
    ore_tf_class = ore_tf()
    ore_tf_class.get_logger().info('Ore TF publisher node started.')

    try:
        rclpy.spin(ore_tf_class)
    except KeyboardInterrupt:
        pass
    finally:
        ore_tf_class.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
