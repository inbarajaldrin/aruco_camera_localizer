import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped, Point, TransformStamped, Transform
from std_msgs.msg import Header, ColorRGBA, Int32
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from max_camera_msgs.msg import PusherInfo
import numpy as np
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import quat_to_rpy
import threading

class LocalizerBridge(Node):
    def __init__(self, image_topic=None, robot_config=None):
        super().__init__('localizer_bridge')
        # Offset of camera from EE (in EE frame, Z points outward from tool flange)
        self.robot_config = robot_config
        if robot_config is not None:
            ee_pose_topic = robot_config.ee_pose_topic
        else:
            ee_pose_topic = '/tcp_pose_broadcaster/pose'

        # --- Latest EE Pose ---
        self.ee_position = None
        self.ee_quat = None
        self.ee_pose_received = False
        self.ee_speed = 0.0  # mm/s (EMA-filtered)
        self._prev_ee_pos = None
        self._prev_ee_time = None
        self._speed_ema_alpha = 0.1  # low alpha = heavy smoothing
        self.lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.latest_frame = None
        self.frame_available = False
        self.use_image_topic = image_topic is not None

        self.subscription = self.create_subscription(
            PoseStamped,
            ee_pose_topic,
            self.ee_pose_callback,
            10)
        self.get_logger().info("TCPSubscriber node started.")
        
        # Image subscription if topic is provided
        if image_topic:
            self.image_subscription = self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                10)
            self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        
        # --- Publishers ---
        self.cam_pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)
        # Only create image publisher for real camera mode (not sim mode)
        if not self.use_image_topic:
            self.image_publisher = self.create_publisher(Image, 'intel_camera_rgb_raw', 10)
        else:
            self.image_publisher = None
        self.annotated_stream_pub = self.create_publisher(Image, 'annotated_stream', 10)
        self.bridge = CvBridge()
        
        # Use TF2 message for object poses
        self.object_poses_pub = self.create_publisher(TFMessage, '/objects_poses_real', 10)
        
        self.pusher_publishers = {}
        self.frame_num_publsher = self.create_publisher(Int32, '/camera_frame_number', 10)

    def publish_image(self, frame):
        """Publish raw camera image only when not using an image topic (real camera mode)"""
        if not self.use_image_topic and self.image_publisher is not None:
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_publisher.publish(img_msg)

    def publish_annotated_stream(self, annotated_frame):
        """Publish the annotated frame that shows up in the OpenCV window"""
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
        self.annotated_stream_pub.publish(img_msg)

    def image_callback(self, msg: Image):
        """Callback for incoming image messages"""
        try:
            with self.image_lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.frame_available = True
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def get_latest_frame(self):
        """Get the latest frame from ROS topic"""
        with self.image_lock:
            if self.frame_available:
                return self.latest_frame.copy(), True
            else:
                return None, False

    def ee_pose_callback(self, msg: PoseStamped):
        with self.lock:
            new_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self._prev_ee_pos is not None and self._prev_ee_time is not None:
                dt = t - self._prev_ee_time
                if dt > 0:
                    raw_speed = np.linalg.norm(new_pos - self._prev_ee_pos) / dt * 1000.0  # mm/s
                    a = self._speed_ema_alpha
                    self.ee_speed = a * raw_speed + (1 - a) * self.ee_speed
            self._prev_ee_pos = new_pos
            self._prev_ee_time = t
            self.ee_position = new_pos
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                   msg.pose.orientation.z, msg.pose.orientation.w])
            if not self.ee_pose_received:
                self.ee_pose_received = True
                self.get_logger().info("Received first EE pose from /tcp_pose_broadcaster/pose")

    def get_ee_pose(self):
        return self.ee_position, self.ee_quat

    def get_ee_speed(self):
        with self.lock:
            return self.ee_speed

    def get_camera_pose(self):
        with self.lock:
            if self.ee_position is None or self.ee_quat is None:
                return None, None
            if self.robot_config is not None:
                offset_pos = np.array(self.robot_config.cam_offset_position)
                offset_quat = np.array(self.robot_config.cam_offset_quat)
            else:
                offset_pos = np.array([0.0, -0.07, 0.0825])
                offset_quat = np.array([0.0, 0.0, 0.0, 1.0])
            r_ee = R.from_quat(self.ee_quat)
            r_cam_offset = R.from_quat(offset_quat)
            cam_pos_world = self.ee_position + r_ee.apply(offset_pos)
            cam_quat_world = (r_ee * r_cam_offset).as_quat()
        return cam_pos_world, cam_quat_world
    
    def publish_camera_pose(self, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
        self.cam_pose_pub.publish(msg)

    def publish_object_poses(self, object_data):
        """Publish all object poses using TF2 message format - much cleaner than custom messages"""
        now = self.get_clock().now().to_msg()
        
        # Create TFMessage with all object transforms
        msg = TFMessage()
        
        # Add each object as a TransformStamped
        for obj in object_data:
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = "World"  # Parent frame
            transform.child_frame_id = obj["name"]  # Object name as child frame
            
            # Set translation
            transform.transform.translation.x = float(obj["position"][0])
            transform.transform.translation.y = float(obj["position"][1])
            transform.transform.translation.z = float(obj["position"][2])
            
            # Set rotation (quaternion)
            transform.transform.rotation.x = float(obj["quaternion"][0])
            transform.transform.rotation.y = float(obj["quaternion"][1])
            transform.transform.rotation.z = float(obj["quaternion"][2])
            transform.transform.rotation.w = float(obj["quaternion"][3])
            
            msg.transforms.append(transform)
        
        # Publish the TF2 message
        self.object_poses_pub.publish(msg)

    def publish_contacts(self, pushers):
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
