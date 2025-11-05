import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3Stamped, PointStamped, Point, TransformStamped, Transform
from std_msgs.msg import Header, ColorRGBA, Int32
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from max_camera_msgs.msg import PusherInfo, GraspPoint, GraspPointArray
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class LocalizerBridge(Node):
    def __init__(self, image_topic=None):
        super().__init__('localizer_bridge')
        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.048, -0.1]) # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0]) # identity quaternion
        
        # Object pose correction offsets (X and Y offsets to account for real vs simulated differences)
        # Based on measurements: fork_orange (X=11.8mm, Y=6.7mm), line_brown (X=11.5mm, Y=7.2mm)
        # Average: X=11.66mm, Y=6.96mm. Using rounded values:
        # sim_offset: applied when using image topic (simulated environment)
        self.sim_offset = np.array([-0.011086, -0.007811, -0.046685]) # meters (X, Y offsets, Z=0 to leave height unchanged)
        # real_world_offset: applied when using real world camera (no image topic)
        # Set to zero since real world measurements are already correct
        self.real_world_offset = np.array([0.0, 0.0, 0.0]) # meters
        
        # Rotation offsets for sim and real (quaternion format: [x, y, z, w])
        self.sim_offset_quat = np.array([0.0, 1.0, 0.0, 0.0]) # rotation offset for simulated environment
        self.real_world_offset_quat = np.array([0.0, 1.0, 0.0, 0.0]) # rotation offset for real world
        
        # --- Latest EE Pose (using values here if no ROS input - Home position) ---
        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.latest_frame = None
        self.frame_available = False
        self.use_image_topic = image_topic is not None
        
        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
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
        self.image_publisher = self.create_publisher(Image, 'intel_camera_rgb_raw', 10)
        self.annotated_stream_pub = self.create_publisher(Image, 'annotated_stream', 10)
        self.bridge = CvBridge()
        
        # Use TF2 message for object poses
        self.object_poses_pub = self.create_publisher(TFMessage, '/objects_poses_real', 10)
        
        # Grasp points publisher
        self.grasp_points_pub = self.create_publisher(GraspPointArray, '/grasp_points', 10)
        
        self.pusher_publishers = {}
        self.frame_num_publsher = self.create_publisher(Int32, '/camera_frame_number', 10)

    def publish_image(self, frame):
        """Publish raw camera image only when not using an image topic"""
        if not self.use_image_topic:
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
            cam_quat_world = (r_ee * r_cam_offset).as_quat()
        return cam_pos_world, cam_quat_world
    
    def get_object_pose_offset(self):
        """Get the appropriate object pose offset based on whether using image topic (sim) or real camera"""
        if self.use_image_topic:
            return self.sim_offset  # Use sim offset when using image topic (simulated)
        else:
            return self.real_world_offset  # Use real world offset when using real camera
    
    def get_object_rotation_offset(self):
        """Get the appropriate object rotation offset based on whether using image topic (sim) or real camera"""
        if self.use_image_topic:
            return self.sim_offset_quat  # Use sim rotation offset when using image topic (simulated)
        else:
            return self.real_world_offset_quat  # Use real world rotation offset when using real camera

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
            
            # Apply object pose correction offset (X and Y offsets for real vs simulated differences)
            # Use sim_offset when using image topic, real_world_offset when using real camera
            corrected_position = obj["position"] + self.get_object_pose_offset()
            
            # Set translation
            transform.transform.translation.x = float(corrected_position[0])
            transform.transform.translation.y = float(corrected_position[1])
            transform.transform.translation.z = float(corrected_position[2])
            
            # Apply rotation offset (quaternion format: [x, y, z, w])
            rotation_offset_quat = self.get_object_rotation_offset()
            r_obj = R.from_quat(obj["quaternion"])
            r_offset = R.from_quat(rotation_offset_quat)
            corrected_quat = (r_obj * r_offset).as_quat()
            
            # Set rotation (quaternion)
            transform.transform.rotation.x = float(corrected_quat[0])
            transform.transform.rotation.y = float(corrected_quat[1])
            transform.transform.rotation.z = float(corrected_quat[2])
            transform.transform.rotation.w = float(corrected_quat[3])
            
            msg.transforms.append(transform)
        
        # Publish the TF2 message
        self.object_poses_pub.publish(msg)

    def publish_grasp_points(self, identified_objects, model_data):
        """Publish grasp points for all identified objects"""
        now = self.get_clock().now().to_msg()
        
        # Create GraspPointArray message
        msg = GraspPointArray()
        msg.header.stamp = now
        msg.header.frame_id = "base"
        
        # Process each identified object
        for obj in identified_objects:
            model_name = obj["name"]
            
            # Check if this model has grasp points data
            if model_name not in model_data or model_data[model_name]['grasp_points'] is None:
                continue
                
            grasp_points = model_data[model_name]['grasp_points']
            # Apply object pose correction offset (X and Y offsets for real vs simulated differences)
            # Use sim_offset when using image topic, real_world_offset when using real camera
            object_pos = obj["position"] + self.get_object_pose_offset()
            
            # Apply rotation offset (quaternion format: [x, y, z, w])
            rotation_offset_quat = self.get_object_rotation_offset()
            r_obj = R.from_quat(obj["quaternion"])
            r_offset = R.from_quat(rotation_offset_quat)
            object_quat = (r_obj * r_offset).as_quat()
            
            # Transform object rotation to rotation matrix
            rot_matrix = R.from_quat(object_quat).as_matrix()
            
            # Coordinate system transformation matrix (same as wireframe)
            coord_transform = np.array([
                [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
                [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
                [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
            ])
            
            # Process each grasp point
            for grasp_point in grasp_points:
                # Get grasp point position relative to object center
                grasp_pos_local = np.array([
                    grasp_point['position']['x'],
                    grasp_point['position']['y'], 
                    grasp_point['position']['z']
                ])
                
                # Apply coordinate system transformation (same as wireframe)
                grasp_pos_transformed = coord_transform @ grasp_pos_local
                
                # Transform to world frame
                grasp_pos_world = object_pos + rot_matrix @ grasp_pos_transformed
                
                # Create GraspPoint message
                grasp_msg = GraspPoint()
                grasp_msg.header.stamp = now
                grasp_msg.header.frame_id = "base"
                grasp_msg.object_name = model_name
                grasp_msg.grasp_id = grasp_point['id']
                grasp_msg.grasp_type = grasp_point.get('type', 'unknown')
                
                # Set pose (position and orientation)
                grasp_msg.pose.position.x = float(grasp_pos_world[0])
                grasp_msg.pose.position.y = float(grasp_pos_world[1])
                grasp_msg.pose.position.z = float(grasp_pos_world[2])
                
                # Generate orientation from approach vector if available
                if 'approach_vector' in grasp_point:
                    approach_vec_local = np.array([
                        grasp_point['approach_vector']['x'],
                        grasp_point['approach_vector']['y'],
                        grasp_point['approach_vector']['z']
                    ])
                    
                    # Apply coordinate system transformation to approach vector
                    approach_vec_transformed = coord_transform @ approach_vec_local
                    
                    # Transform approach vector to world frame
                    approach_vec_world = rot_matrix @ approach_vec_transformed
                    
                    # Generate full orientation from approach vector
                    # The approach vector becomes the Z-axis of the gripper frame
                    z_axis = approach_vec_world / np.linalg.norm(approach_vec_world)
                    
                    # Create a perpendicular vector for X-axis (gripper opening direction)
                    # Use a default direction and make it perpendicular to approach vector
                    if abs(z_axis[0]) < 0.9:  # If not pointing along X
                        x_axis = np.array([1.0, 0.0, 0.0])
                    else:  # If pointing along X, use Y as reference
                        x_axis = np.array([0.0, 1.0, 0.0])
                    
                    # Make X-axis perpendicular to Z-axis
                    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    
                    # Y-axis is cross product of Z and X
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    
                    # Construct rotation matrix
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                    
                    # Convert to quaternion
                    grasp_quat = R.from_matrix(rotation_matrix).as_quat()
                    
                else:
                    # Default orientation (identity)
                    grasp_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
                
                # Set orientation in world frame
                grasp_msg.pose.orientation.x = float(grasp_quat[0])
                grasp_msg.pose.orientation.y = float(grasp_quat[1])
                grasp_msg.pose.orientation.z = float(grasp_quat[2])
                grasp_msg.pose.orientation.w = float(grasp_quat[3])
                
                # Convert quaternion to RPY degrees (same as object poses)
                grasp_euler = R.from_quat(grasp_quat).as_euler('xyz', degrees=True)
                grasp_msg.roll = float(grasp_euler[0])
                grasp_msg.pitch = float(grasp_euler[1])
                grasp_msg.yaw = float(grasp_euler[2])
                
                msg.grasp_points.append(grasp_msg)
        
        # Publish the structured message
        self.grasp_points_pub.publish(msg)

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
