import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped, Point
from std_msgs.msg import Header, ColorRGBA, Int32, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from max_camera_msgs.msg import PusherInfo, ObjectPose, ObjectPoseArray
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class LocalizerBridge(Node):
    def __init__(self):
        super().__init__('localizer_bridge')
        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.048, -0.01]) # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0]) # identity quaternion
        
        # Calibration offsets for camera position correction
        self.calibration_offset_x = +0.004 # 5mm correction
        self.calibration_offset_y = +0.002 # 0mm correction
        # --- Latest EE Pose (using values here if no ROS input - Home position) ---
        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.lock = threading.Lock()
        
        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            10)
        self.get_logger().info("TCPSubscriber node started.")
        
        # --- Publishers ---
        self.cam_pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)
        self.image_publisher = self.create_publisher(Image, 'intel_camera_rgb_raw', 10)
        self.annotated_image_publisher = self.create_publisher(Image, 'intel_camera_annotated', 10)
        self.bridge = CvBridge()
        
        # Clean approach: Single topic with proper structured data
        self.object_poses_pub = self.create_publisher(ObjectPoseArray, '/objects_poses', 10)
        
        # Blue dot targets publisher
        self.targets_poses_pub = self.create_publisher(ObjectPoseArray, '/targets_poses', 10)
        
        # Pusher publishers
        self.pusher_publishers = {}
        self.frame_num_publsher = self.create_publisher(Int32, '/camera_frame_number', 10)
        self.recommended_publishers = {"pusher_1_position": self.create_publisher(PointStamped, '/recommended_pusher_1/position', 10),
                                     "pusher_2_position": self.create_publisher(PointStamped, '/recommended_pusher_2/position', 10),
                                     "pusher_1_normal": self.create_publisher(Vector3Stamped, '/recommended_pusher_1/normal', 10),
                                     "pusher_2_normal": self.create_publisher(Vector3Stamped, '/recommended_pusher_2/normal', 10)}
        

    def publish_image(self, frame):
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_publisher.publish(img_msg)

    def publish_annotated_image(self, frame):
        """Publish the annotated/processed frame that's displayed in OpenCV window"""
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.annotated_image_publisher.publish(img_msg)

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                   msg.pose.orientation.z, msg.pose.orientation.w])

    def get_ee_pose(self):
        return self.ee_position, self.ee_quat

    def get_camera_pose(self):
        with self.lock:
            r_ee = R.from_quat(self.ee_quat)
            r_cam_offset = R.from_quat(self.cam_offset_quat)
            cam_pos_world = self.ee_position + r_ee.apply(self.cam_offset_position)
            
            # Apply calibration offsets in the camera frame
            calibration_offset = np.array([self.calibration_offset_x, self.calibration_offset_y, 0.0])
            cam_pos_world += r_ee.apply(calibration_offset)
            
            cam_quat_world = (r_ee * r_cam_offset).as_quat()
        return cam_pos_world, cam_quat_world

    def canonicalize_euler(self, orientation):
        """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
        roll, pitch, yaw = orientation
        if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
            return (0.0, 180.0, (yaw % 360)-180)
        else:
            return orientation

    def publish_camera_pose(self, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.cam_pose_pub.publish(msg)

    def publish_object_poses(self, object_data):
        """Publish all object poses in a single structured topic"""
        now = self.get_clock().now().to_msg()
        
        # Create ObjectPoseArray message
        msg = ObjectPoseArray()
        msg.header.stamp = now
        msg.header.frame_id = "base"
        
        # Add each object as an ObjectPose
        for obj in object_data:
            object_pose = ObjectPose()
            object_pose.header.stamp = now
            object_pose.header.frame_id = "base"
            object_pose.object_name = obj["name"]
            
            # Set pose
            object_pose.pose.position.x = obj["position"][0]
            object_pose.pose.position.y = obj["position"][1]
            object_pose.pose.position.z = obj["position"][2]
            object_pose.pose.orientation.x = obj["quaternion"][0]
            object_pose.pose.orientation.y = obj["quaternion"][1]
            object_pose.pose.orientation.z = obj["quaternion"][2]
            object_pose.pose.orientation.w = obj["quaternion"][3]
            
            # Calculate and set RPY angles from quaternion
            quat = obj["quaternion"]
            rot = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
            rpy = rot.as_euler('xyz', degrees=True)  # Convert to degrees
            
            # Apply canonicalization to match OpenCV display values
            rpy = self.canonicalize_euler(rpy)
            
            object_pose.roll = rpy[0]
            object_pose.pitch = rpy[1]
            object_pose.yaw = rpy[2]
            
            msg.objects.append(object_pose)
        
        # Publish the structured message
        self.object_poses_pub.publish(msg)

    def publish_target_poses(self, detected_color_points):
        """Publish all detected color dot poses to /targets_poses topic"""
        now = self.get_clock().now().to_msg()
        
        # Create ObjectPoseArray message for targets
        msg = ObjectPoseArray()
        msg.header.stamp = now
        msg.header.frame_id = "base"
        
        # Add all detected color points as target poses
        for color_name, world_points in detected_color_points.items():
            for i, point in enumerate(world_points):
                target_pose = ObjectPose()
                target_pose.header.stamp = now
                target_pose.header.frame_id = "base"
                target_pose.object_name = f"{color_name}_dot_{i}"
                
                # Set position (dots are assumed to be on table, so z=0.01m)
                target_pose.pose.position.x = float(point[0])
                target_pose.pose.position.y = float(point[1])
                target_pose.pose.position.z = 0.01  # Table height
                
                # Set orientation (identity quaternion for points)
                target_pose.pose.orientation.x = 0.0
                target_pose.pose.orientation.y = 0.0
                target_pose.pose.orientation.z = 0.0
                target_pose.pose.orientation.w = 1.0
                
                # Set RPY angles (identity quaternion = 0,0,0)
                target_pose.roll = 0.0
                target_pose.pitch = 0.0
                target_pose.yaw = 0.0
                
                msg.objects.append(target_pose)
        
        # Publish the targets message
        self.targets_poses_pub.publish(msg)

    def publish_contacts(self, pushers):
        """Publish pusher contact information"""
        now = self.get_clock().now().to_msg()
        
        for pusher in pushers:
            msg = PusherInfo()
            msg.header = Header()
            msg.header.stamp = now
            msg.header.frame_id = "base"
            msg.frame_num = pusher['frame_number']
            msg.pusher_name = pusher['pusher_name']
            if msg.pusher_name not in self.pusher_publishers:
                topic = f"/pusher_data_{msg.pusher_name}"
                self.pusher_publishers[msg.pusher_name] = self.create_publisher(PusherInfo, topic, 10)
            r, g, b = pusher['color']
            msg.color = ColorRGBA(r=r/255.0, g=g/255.0, b=b/255.0, a=1.0)
            msg.pusher_location = Point(
                x=float(pusher['pusher_location'][0]),
                y=float(pusher['pusher_location'][1]),
                z=float(pusher['pusher_location'][2])
            )
            msg.nearest_point = Point(
                x=float(pusher['nearest_point'][0]),
                y=float(pusher['nearest_point'][1]),
                z=float(pusher['nearest_point'][2])
            )
            msg.kappa = float(pusher['kappa'])
            msg.object_index = pusher['object_index']
            msg.local_contour_index = pusher['local_contour_index']
            self.pusher_publishers[msg.pusher_name].publish(msg)

    def publish_recommended_contacts(self, recommended):
        now = self.get_clock().now().to_msg()
        (pos_1, norm_1), (pos_2, norm_2) = recommended
        pos_1_msg = PointStamped()
        pos_1_msg.header.stamp = now
        pos_1_msg.header.frame_id = "base"
        pos_1_msg.point.x, pos_1_msg.point.y, pos_1_msg.point.z = pos_1
        self.recommended_publishers["pusher_1_position"].publish(pos_1_msg)
        pos_2_msg = PointStamped()
        pos_2_msg.header.stamp = now
        pos_2_msg.header.frame_id = "base"
        pos_2_msg.point.x, pos_2_msg.point.y, pos_2_msg.point.z = pos_2
        self.recommended_publishers["pusher_2_position"].publish(pos_2_msg)
        norm_1_msg = Vector3Stamped()
        norm_1_msg.header.stamp = now
        norm_1_msg.header.frame_id = "base"
        norm_1_msg.vector.x, norm_1_msg.vector.y, norm_1_msg.vector.z = norm_1
        self.recommended_publishers["pusher_1_normal"].publish(norm_1_msg)
        norm_2_msg = Vector3Stamped()
        norm_2_msg.header.stamp = now
        norm_2_msg.header.frame_id = "base"
        norm_2_msg.vector.x, norm_2_msg.vector.y, norm_2_msg.vector.z = norm_2
        self.recommended_publishers["pusher_2_normal"].publish(norm_2_msg)
