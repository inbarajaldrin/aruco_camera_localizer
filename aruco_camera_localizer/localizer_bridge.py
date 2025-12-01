import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped, Point, TransformStamped, Transform, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_msgs.msg import TFMessage
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
from aruco_camera_localizer.config_loader import get_config

class LocalizerBridge(Node):
    def __init__(self, camera_topic='/camera/image_raw', depth_topic=None):
        super().__init__('localizer_bridge')
        
        # Load configuration from YAML
        config = get_config()
        
        # Calculate camera-to-EE offset from default poses
        # This matches the logic in opencv_to_base_transform.py
        ee_pos = config.get_ee_default_position()
        ee_quat = config.get_ee_default_quaternion()
        cam_pos = config.get_camera_default_position()
        cam_quat = config.get_camera_default_quaternion()
        
        # Calculate offset position in EE frame
        R_ee_inv = R.from_quat(ee_quat).inv()
        offset_world = cam_pos - ee_pos
        self.cam_offset_position = R_ee_inv.apply(offset_world)
        
        # Calculate offset quaternion
        self.cam_offset_quat = (R_ee_inv * R.from_quat(cam_quat)).as_quat()
        
        self.get_logger().info(f"Camera-to-EE offset position: {self.cam_offset_position}")
        self.get_logger().info(f"Camera-to-EE offset quaternion: {self.cam_offset_quat}")
        
        # Calibration offsets for fine-tuning object positions
        self.calibration_offset_x, self.calibration_offset_y, self.calibration_offset_z = config.get_calibration_offset()
        self.get_logger().info(f"Calibration offset: x={self.calibration_offset_x}, y={self.calibration_offset_y}, z={self.calibration_offset_z}")
        
        # --- Latest EE Pose (from ROS topic, defaults used if no input) ---
        self.ee_position = ee_pos
        self.ee_quat = ee_quat
        self.lock = threading.Lock()
        
        # --- Latest Camera Pose (from ROS topic, defaults used if no input) ---
        self.cam_position = cam_pos
        self.cam_quat = cam_quat
        self.cam_lock = threading.Lock()
        
        # --- Latest camera frame ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # --- Latest depth image ---
        self.latest_depth = None
        self.depth_lock = threading.Lock()
        
        # Get TCP pose topic from configuration
        tcp_pose_topic = config.get_tcp_pose_topic()
        
        self.subscription = self.create_subscription(
            PoseStamped,
            tcp_pose_topic,
            self.ee_pose_callback,
            10)
        self.get_logger().info(f"LocalizerBridge started. Subscribing to: {tcp_pose_topic}")
        
        # Subscribe to camera pose topic
        self.camera_pose_subscription = self.create_subscription(
            PoseStamped,
            '/camera_pose',
            self.camera_pose_callback,
            10)
        self.get_logger().info(f"Subscribing to camera pose on: /camera_pose")
        
        # Subscribe to camera images
        self.camera_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10)
        self.get_logger().info(f"Subscribing to camera images on: {camera_topic}")
        
        # Subscribe to depth images (if depth topic provided)
        self.depth_subscription = None
        if depth_topic:
            self.depth_subscription = self.create_subscription(
                Image,
                depth_topic,
                self.depth_callback,
                10)
            self.get_logger().info(f"Subscribing to depth images on: {depth_topic}")
        else:
            self.get_logger().info("No depth topic provided, using config distance")
        
        # --- Publishers ---
        # Note: /camera_pose is published by external package, we only subscribe
        self.annotated_image_publisher = self.create_publisher(Image, 'camera_annotated', 10)
        self.bridge = CvBridge()
        
        # Use RELIABLE QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )
        self.object_poses_pub = self.create_publisher(TFMessage, '/objects_poses', qos_profile)
        self.aruco_poses_pub = self.create_publisher(TFMessage, '/aruco_poses', qos_profile)
        self.drop_poses_pub = self.create_publisher(TFMessage, '/drop_poses', qos_profile)
        

    def publish_annotated_image(self, frame):
        """Publish the annotated/processed frame that's displayed in OpenCV window"""
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.annotated_image_publisher.publish(img_msg)

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                   msg.pose.orientation.z, msg.pose.orientation.w])

    def camera_pose_callback(self, msg: PoseStamped):
        """Callback for receiving camera pose from /camera_pose topic"""
        with self.cam_lock:
            self.cam_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.cam_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def camera_callback(self, msg: Image):
        """Callback for receiving camera frames from the camera publisher"""
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")

    def depth_callback(self, msg: Image):
        """Callback for receiving depth images"""
        try:
            # Convert ROS Image message to OpenCV format
            # Try different depth encodings (Isaac Sim might use different formats)
            try:
                depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            except Exception as e:
                self.get_logger().warn(f"Failed passthrough, trying 16UC1: {e}")
                depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            
            with self.depth_lock:
                self.latest_depth = depth_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def get_ee_pose(self):
        return self.ee_position, self.ee_quat
    
    def get_latest_frame(self):
        """Get the latest camera frame (thread-safe)"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_depth(self):
        """Get the latest depth image (thread-safe)"""
        with self.depth_lock:
            return self.latest_depth.copy() if self.latest_depth is not None else None

    def apply_calibration_offset(self, position):
        """Apply calibration offset to a world position"""
        return np.array([
            position[0] + self.calibration_offset_x,
            position[1] + self.calibration_offset_y,
            position[2] + self.calibration_offset_z
        ])

    def get_camera_pose(self):
        """Get camera pose from the subscribed /camera_pose topic"""
        with self.cam_lock:
            return self.cam_position.copy(), self.cam_quat.copy()

    def canonicalize_euler(self, orientation):
        """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
        roll, pitch, yaw = orientation
        if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
            return (0.0, 180.0, (yaw % 360)-180)
        else:
            return orientation


    def publish_object_poses(self, object_data):
        """Publish all object poses as TF transforms using TFMessage"""
        now = self.get_clock().now().to_msg()
        
        # Create TFMessage with array of transforms
        msg = TFMessage()
        
        # Add each object as a TransformStamped
        for obj in object_data:
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = "base"  # Parent frame
            transform.child_frame_id = obj["name"]  # Object name as child frame (e.g., "aruco_3")
            
            # Set translation (position)
            transform.transform.translation = Vector3(
                x=float(obj["position"][0]),
                y=float(obj["position"][1]),
                z=float(obj["position"][2])
            )
            
            # Set rotation (orientation as quaternion)
            transform.transform.rotation = Quaternion(
                x=float(obj["quaternion"][0]),
                y=float(obj["quaternion"][1]),
                z=float(obj["quaternion"][2]),
                w=float(obj["quaternion"][3])
            )
            
            msg.transforms.append(transform)
        
        # Publish the TF message
        self.object_poses_pub.publish(msg)

    def publish_aruco_poses(self, object_data):
        """Publish ArUco marker poses as TF transforms using TFMessage to /aruco_poses topic"""
        now = self.get_clock().now().to_msg()
        
        # Create TFMessage with array of transforms
        msg = TFMessage()
        
        # Filter for ArUco markers only (names starting with "aruco_")
        aruco_objects = [obj for obj in object_data if obj["name"].startswith("aruco_")]
        
        # Add each ArUco marker as a TransformStamped
        for obj in aruco_objects:
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = "base"  # Parent frame
            transform.child_frame_id = obj["name"]  # Object name as child frame (e.g., "aruco_3")
            
            # Set translation (position)
            transform.transform.translation = Vector3(
                x=float(obj["position"][0]),
                y=float(obj["position"][1]),
                z=float(obj["position"][2])
            )
            
            # Set rotation (orientation as quaternion)
            transform.transform.rotation = Quaternion(
                x=float(obj["quaternion"][0]),
                y=float(obj["quaternion"][1]),
                z=float(obj["quaternion"][2]),
                w=float(obj["quaternion"][3])
            )
            
            msg.transforms.append(transform)
        
        # Publish the TF message to /aruco_poses
        self.aruco_poses_pub.publish(msg)

    def publish_drop_poses(self, drop_poses_data):
        """Publish drop poses as TF transforms using TFMessage to /drop_poses topic"""
        now = self.get_clock().now().to_msg()
        
        # Create TFMessage with array of transforms
        msg = TFMessage()
        
        # Add each drop pose as a TransformStamped
        for drop_pose in drop_poses_data:
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = "base"  # Parent frame
            transform.child_frame_id = drop_pose["name"]  # Drop pose name (e.g., "drop_3")
            
            # Set translation (position)
            transform.transform.translation = Vector3(
                x=float(drop_pose["position"][0]),
                y=float(drop_pose["position"][1]),
                z=float(drop_pose["position"][2])
            )
            
            # Set rotation (orientation as quaternion)
            transform.transform.rotation = Quaternion(
                x=float(drop_pose["quaternion"][0]),
                y=float(drop_pose["quaternion"][1]),
                z=float(drop_pose["quaternion"][2]),
                w=float(drop_pose["quaternion"][3])
            )
            
            msg.transforms.append(transform)
        
        # Publish the TF message to /drop_poses
        self.drop_poses_pub.publish(msg)

