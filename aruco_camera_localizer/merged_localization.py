import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.config_loader import get_config
from aruco_camera_localizer.geometric_functions import transform_points_world_to_img, rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines
from aruco_camera_localizer.detection_functions import detect_markers, estimate_pose
import threading
import rclpy
import argparse
import json
import os
from ultralytics import YOLOE
from ament_index_python.packages import get_package_share_directory

# Load configuration from YAML
config = get_config()

c_width = config.get_camera_width()
c_height = config.get_camera_height()
c_hfov = config.get_camera_hfov()
c_vfov = config.get_camera_vfov()

# OpenCV to camera frame transformation
OPENCV_TO_CAMERA_QUAT = config.get_opencv_to_camera_quaternion()
print(f"OpenCV-to-Camera quaternion: {OPENCV_TO_CAMERA_QUAT}")

# Detection distance
DETECTION_DISTANCE = config.get_detection_distance()
print(f"Detection distance: {DETECTION_DISTANCE}m")

fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

CAMERA_MATRIX = config.get_camera_matrix()
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)  # datasheet says <= 1.5%
MARKER_SIZE = 0.032  # meters
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
}

# YOLO detection settings
YOLO_PROMPTS = ["hand"]
YOLO_PROMPT_MAP = {
    "hand": "hand"
}

# Global variables for dynamic YOLO prompt management
yolo_prompts_lock = threading.Lock()
current_yolo_prompts = YOLO_PROMPTS.copy()
current_yolo_prompt_map = YOLO_PROMPT_MAP.copy()

# Generic color for all YOLO detections (cyan in BGR)
GENERIC_COLOR = (255, 255, 0)

# Store previous frame's objects for position-based tracking
previous_yolo_objects = {}  # color_name -> list of {name, position}

def start_ros_node(camera_topic='/camera/image_rgb', depth_topic=None):
    rclpy.init()
    node = LocalizerBridge(camera_topic=camera_topic, depth_topic=depth_topic)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def get_yolo_prompts():
    """Get current YOLO prompts (thread-safe)"""
    with yolo_prompts_lock:
        return current_yolo_prompts.copy(), current_yolo_prompt_map.copy()

def update_yolo_prompts(prompts, prompt_map, yolo_model=None):
    """Update YOLO prompts and prompt mapping (thread-safe)"""
    with yolo_prompts_lock:
        global current_yolo_prompts, current_yolo_prompt_map
        current_yolo_prompts = prompts.copy()
        current_yolo_prompt_map = prompt_map.copy()
        print(f"Updated YOLO prompts: {current_yolo_prompts}")
        print(f"Updated prompt mapping: {current_yolo_prompt_map}")
        
        # Update YOLO model if provided
        if yolo_model is not None:
            try:
                yolo_model.set_classes(current_yolo_prompts, yolo_model.get_text_pe(current_yolo_prompts))
                print(f"YOLO model updated with new prompts")
            except Exception as e:
                print(f"Failed to update YOLO model: {e}")

def yolo_prompts_callback(msg, yolo_model=None):
    """Topic callback for real-time YOLO prompt updates"""
    try:
        data = json.loads(msg.data)
        prompts = data.get('prompts', [])
        prompt_map = data.get('prompt_map', {})
        
        update_yolo_prompts(prompts, prompt_map, yolo_model)
        print(f"YOLO prompts updated via topic: {prompts}")
        
    except Exception as e:
        print(f"Failed to update YOLO prompts from topic: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run merged ArUco, YOLOE, and drop pose tracker.")
    parser.add_argument("--camera-topic", type=str, default="/camera/image_rgb",
                        help="ROS2 topic to subscribe for camera images (default: /camera/image_rgb)")
    parser.add_argument("--depth-topic", type=str, default=None,
                        help="ROS2 topic to subscribe for depth images (optional, uses config distance if not provided)")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--aruco", action='store_true',
                        help="Enable ArUco marker detection")
    parser.add_argument("--yoloe", action='store_true',
                        help="Enable YOLOE object detection")
    parser.add_argument("--drop", action='store_true',
                        help="Publish drop poses instead of ArUco poses. Drop poses are calculated from position offsets in aruco_config.json transformed to world frame.")
    parser.add_argument("--yolo-mode", type=str, default="prompt-set",
                        help="YOLO mode: 'prompt-set' for prompted detection (default: prompt-set)")
    parser.add_argument("--yolo-model", type=str, default="aruco_camera_localizer/yoloe-11s-seg.pt",
                        help="YOLO model path (default: aruco_camera_localizer/yoloe-11s-seg.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.4,
                        help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--yolo-prompts", type=str, nargs='+', 
                        default=["hand"],
                        help="YOLO detection prompts (default: hand)")
    parser.add_argument("--yolo-prompt-map", type=str, nargs='+',
                        help="Custom prompt mapping for prompts (format: prompt1:color1 prompt2:color2)")
    parser.add_argument("--headless", action='store_true',
                        help="Run without displaying OpenCV window")
    # Use parse_known_args to avoid conflicts with ROS args
    args, unknown = parser.parse_known_args()
    return args, unknown

def convert_2d_orientation_to_quaternion(orientation_angle, cam_quat, opencv_to_camera_quat):
    """
    Convert 2D orientation angle from PCA to 3D quaternion in world frame.
    
    Transformation chain:
    1. 2D angle → rotation in OpenCV frame (Z-axis rotation)
    2. OpenCV frame → Camera frame (opencv_to_camera)
    3. Camera frame → World/Base frame (camera quaternion)
    
    Args:
        orientation_angle: 2D orientation angle in radians from PCA
        cam_quat: Camera quaternion in world frame [x, y, z, w]
        opencv_to_camera_quat: OpenCV to camera frame quaternion [x, y, z, w]
    
    Returns:
        quaternion: 3D quaternion in world frame [x, y, z, w]
    """
    # Step 1: Create rotation around Z-axis in OpenCV frame
    z_rotation = R.from_euler('z', orientation_angle)
    opencv_quat = z_rotation.as_quat()
    
    # Step 2: Transform from OpenCV frame to camera frame
    opencv_to_cam = R.from_quat(opencv_to_camera_quat)
    camera_orientation = opencv_to_cam * R.from_quat(opencv_quat)
    
    # Step 3: Transform from camera frame to world frame
    cam_rotation = R.from_quat(cam_quat)
    world_orientation = cam_rotation * camera_orientation
    
    return world_orientation.as_quat()

def extract_object_roi(image, box):
    """Extract the region of interest for the detected object"""
    x1, y1, x2, y2 = map(int, box)
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1)

def find_orientation_pca(roi):
    """Find object orientation using Principal Component Analysis"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get contour points
        points = largest_contour.reshape(-1, 2)
        
        if len(points) < 3:
            return None
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # Get the first principal component (direction of maximum variance)
        principal_axis = pca.components_[0]
        
        # Calculate angle from the principal axis
        angle = np.arctan2(principal_axis[1], principal_axis[0])
        
        return angle, pca.explained_variance_ratio_[0]
        
    except Exception as e:
        return None

def find_orientation_contour(roi):
    """Find object orientation using contour analysis (minAreaRect)"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]  # Angle in degrees
        
        # Convert to radians and normalize
        angle_rad = np.radians(angle)
        
        return angle_rad, rect[1]  # Return angle and dimensions
        
    except Exception as e:
        return None

def draw_axis_aligned_line(image, box, orientation_angle=None):
    """Draw a line through the leading axis aligned with object orientation"""
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    if orientation_angle is not None:
        # Use the calculated orientation
        # Calculate line endpoints based on orientation
        line_length = max(x2 - x1, y2 - y1) // 2
        
        # Calculate endpoints of the axis line
        end_x1 = int(center_x + line_length * np.cos(orientation_angle))
        end_y1 = int(center_y + line_length * np.sin(orientation_angle))
        end_x2 = int(center_x - line_length * np.cos(orientation_angle))
        end_y2 = int(center_y - line_length * np.sin(orientation_angle))
        
        # Draw the oriented axis line (thick cyan line)
        cv2.line(image, (end_x1, end_y1), (end_x2, end_y2), (255, 255, 0), 4)  # Cyan line
        
        # Draw arrow to show direction
        arrow_length = 30
        arrow_x = int(end_x1 + arrow_length * np.cos(orientation_angle))
        arrow_y = int(end_y1 + arrow_length * np.sin(orientation_angle))
        cv2.arrowedLine(image, (end_x1, end_y1), (arrow_x, arrow_y), (0, 255, 255), 3, tipLength=0.3)  # Yellow arrow
        
    else:
        # Fallback to simple axis-aligned line (original behavior)
        width = x2 - x1
        height = y2 - y1
        
        if width > height:
            # Horizontal line
            cv2.line(image, (x1, center_y), (x2, center_y), (255, 255, 0), 3)
        else:
            # Vertical line
            cv2.line(image, (center_x, y1), (center_x, y2), (255, 255, 0), 3)

def detect_yolo_blobs(frame, yolo_model, camera_matrix, cam_pos, cam_quat, yolo_prompts, yolo_prompt_map, opencv_to_camera_quat, distance=0.132, depth_image=None, bridge_node=None, conf_threshold=0.4, nms_threshold=0.3):
    """Detect objects using YOLO and convert to world points, grouped by color"""
    detected_color_points = {}
    detection_metadata = []  # Store boxes, orientations, and other metadata
    
    # Run YOLO detection
    results = yolo_model.predict(frame, verbose=False, conf=conf_threshold)
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_raw = results[0].boxes.xyxy.cpu().numpy()
        scores_raw = results[0].boxes.conf.cpu().numpy()
        class_ids_raw = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Apply NMS
        boxes_nms = []
        for box in boxes_raw:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            boxes_nms.append([x1, y1, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_raw.tolist(), conf_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            
            # Process each detection
            for idx in indices:
                box = boxes_raw[idx]
                score = scores_raw[idx]
                class_id = int(class_ids_raw[idx])
                
                # Get class name and map to color
                class_name = yolo_prompts[class_id] if class_id < len(yolo_prompts) else f"class_{class_id}"
                color_name = yolo_prompt_map.get(class_name, class_name)
                # Replace spaces with underscores in color_name for object naming
                color_name = color_name.replace(' ', '_')
                
                # Calculate center of bounding box
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Extract depth value at center if depth image is available
                actual_distance = distance  # Default to config distance
                if depth_image is not None:
                    center_x_int = int(center_x)
                    center_y_int = int(center_y)
                    
                    if (0 <= center_x_int < depth_image.shape[1] and 
                        0 <= center_y_int < depth_image.shape[0]):
                        depth_raw = depth_image[center_y_int, center_x_int]
                        
                        # Handle different depth formats
                        if depth_raw > 0:
                            # Isaac Sim typically uses meters (float32), regular depth uses mm (uint16)
                            if depth_image.dtype == np.float32:
                                actual_distance = depth_raw  # Already in meters
                            else:
                                actual_distance = depth_raw / 1000.0  # Convert mm to meters
                        else:
                            # Invalid depth value, use config distance
                            pass
                
                # Step 1: Ray in OpenCV frame
                pixel = np.array([center_x, center_y, 1.0])
                ray_opencv = np.linalg.inv(camera_matrix) @ pixel
                
                # Step 2: Transform from OpenCV frame to camera frame
                R_opencv_to_cam = R.from_quat(opencv_to_camera_quat)
                ray_cam = R_opencv_to_cam.apply(ray_opencv)
                
                # Step 3: Transform ray to world frame
                R_wc = R.from_quat(cam_quat).as_matrix()
                ray_world = R_wc @ ray_cam
                cam_origin_world = np.array(cam_pos)
                
                # Step 4: Place object at actual distance along ray
                ray_normalized = ray_world / np.linalg.norm(ray_world)
                point_world = cam_origin_world + ray_normalized * actual_distance
                
                # Step 5: Apply calibration offset to object position
                if bridge_node is not None:
                    point_world = bridge_node.apply_calibration_offset(point_world)
                
                
                # Extract ROI for orientation analysis
                roi, roi_offset = extract_object_roi(frame, box)
                
                # Try to find object orientation
                orientation_angle = None
                
                # Method 1: Try PCA analysis
                pca_result = find_orientation_pca(roi)
                if pca_result is not None:
                    orientation_angle, variance_ratio = pca_result
                
                # Method 2: Try contour analysis if PCA failed
                if orientation_angle is None:
                    contour_result = find_orientation_contour(roi)
                    if contour_result is not None:
                        orientation_angle, dimensions = contour_result
                
                # Store metadata for visualization
                detection_index = len(detection_metadata)  # Track original detection order
                detection_metadata.append({
                    'box': box,
                    'score': score,
                    'class_name': class_name,
                    'color_name': color_name,
                    'orientation_angle': orientation_angle,
                    'world_point': point_world,  # Store world point for matching
                    'detection_index': detection_index  # Track original order
                })
                
                # Store by color with detection index for matching
                if color_name not in detected_color_points:
                    detected_color_points[color_name] = []
                detected_color_points[color_name].append({
                    'point': point_world,
                    'detection_index': detection_index
                })
    
    return detected_color_points, detection_metadata

def load_aruco_config():
    """Load aruco_config.json and create a mapping from marker_id to row and offset"""
    # Try to get package share directory, fall back to relative path (same pattern as config_loader)
    try:
        pkg_dir = get_package_share_directory("aruco_camera_localizer")
        config_path = os.path.join(pkg_dir, "config", "aruco_config.json")
    except:
        # Fallback for development/testing
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(pkg_dir, "config", "aruco_config.json")
    
    with open(config_path, 'r') as f:
        aruco_config = json.load(f)
    
    # Create mapping: marker_id -> (row_name, offset_dict)
    marker_to_row = {}
    for row_name, row_data in aruco_config["marker_rows"].items():
        marker_ids = row_data["marker_ids"]
        offset = row_data["position_offset"]
        for marker_id in marker_ids:
            marker_to_row[marker_id] = (row_name, offset)
    
    return marker_to_row

def transform_offset_marker_to_world(offset_marker, marker_world_pos, marker_world_quat):
    """
    Transform an offset from ArUco marker frame to world frame.
    
    Args:
        offset_marker: numpy array [X, Y, Z] in marker frame
        marker_world_pos: numpy array [x, y, z] marker position in world frame
        marker_world_quat: numpy array [x, y, z, w] marker orientation in world frame
    
    Returns:
        numpy array [x, y, z] drop position in world frame
    """
    # Create rotation from marker quaternion
    r_marker = R.from_quat(marker_world_quat)
    
    # Transform offset from marker frame to world frame
    offset_world = r_marker.apply(offset_marker)
    
    # Add to marker position to get drop position in world frame
    drop_pos_world = marker_world_pos + offset_world
    
    return drop_pos_world

def main():
    # Parse args first, before initializing ROS
    args, unknown_args = parse_args()
    
    # Validate that at least one detection method is enabled
    if not args.aruco and not args.yoloe:
        print("Error: At least one of --aruco or --yoloe must be enabled")
        return
    
    # Load aruco config if --drop mode is enabled
    marker_to_row = None
    if args.drop:
        if not args.aruco:
            print("Warning: --drop requires --aruco to be enabled. Ignoring --drop flag.")
            args.drop = False
        else:
            marker_to_row = load_aruco_config()
            print(f"Loaded ArUco config for drop mode. Found {len(marker_to_row)} marker mappings.")
    
    # Start ROS node with remaining args
    bridge_node = start_ros_node(camera_topic=args.camera_topic, depth_topic=args.depth_topic)
    
    # Initialize YOLO model if YOLOE is enabled
    yolo_model = None
    if args.yoloe:
        # Set up YOLO prompt topics
        from std_msgs.msg import String

        # Topic subscription for real-time prompt updates
        prompts_subscription = bridge_node.create_subscription(
            String,
            '/yolo_prompts_update',
            yolo_prompts_callback,
            10
        )
        
        # YOLO prompts publisher for external monitoring
        yolo_prompts_pub = bridge_node.create_publisher(String, '/yolo_prompts', 10)
        
        # Timer to publish current prompts periodically
        def publish_current_prompts():
            """Publish current YOLO prompts for external monitoring"""
            try:
                prompts, prompt_map = get_yolo_prompts()
                # Replace spaces with underscores in prompts for topic publishing
                prompts_normalized = [p.replace(' ', '_') for p in prompts]
                # Also normalize prompt_map keys and values
                prompt_map_normalized = {k.replace(' ', '_'): v.replace(' ', '_') if isinstance(v, str) else v 
                                         for k, v in prompt_map.items()}
                prompts_data = {
                    'prompts': prompts_normalized,
                    'prompt_map': prompt_map_normalized
                }
                
                msg = String()
                msg.data = json.dumps(prompts_data)
                yolo_prompts_pub.publish(msg)
                
            except Exception as e:
                print(f"Failed to publish current prompts: {e}")
        
        prompts_timer = bridge_node.create_timer(1.0, publish_current_prompts)

        # Parse prompt mapping from command line if provided
        yolo_prompt_map = YOLO_PROMPT_MAP.copy()
        if args.yolo_prompt_map:
            for mapping in args.yolo_prompt_map:
                if ':' in mapping:
                    prompt, color = mapping.split(':', 1)
                    yolo_prompt_map[prompt] = color.strip()
        
        # Update global variables with command line arguments
        update_yolo_prompts(args.yolo_prompts, yolo_prompt_map)

        # Initialize YOLO model with dynamic prompts using improved loading
        print(f"YOLO mode: {args.yolo_mode}")
        print(f"Loading YOLO model: {args.yolo_model}")
        
        # Get script directory and construct absolute model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, args.yolo_model.split('/')[-1])  # Get just the filename
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}, trying relative path: {args.yolo_model}")
            model_path = args.yolo_model
        
        # Change to script directory for model loading
        original_cwd = os.getcwd()
        os.chdir(script_dir)
        
        try:
            yolo_model = YOLOE(model_path)
            print(f"✅ Loaded YOLOE model from {model_path}")
            
            # Pre-compute text embeddings for all prompts
            print("Pre-computing text embeddings...")
            text_embeddings = yolo_model.get_text_pe(args.yolo_prompts)
            
            # Set prompts using the combined embeddings
            yolo_model.set_classes(args.yolo_prompts, text_embeddings)
            print(f"✅ YOLO model loaded with prompts: {args.yolo_prompts}")
            print(f"✅ YOLO prompt mapping: {yolo_prompt_map}")
        finally:
            # Restore working directory
            os.chdir(original_cwd)
        
        # Update the topic callback to use the yolo_model
        def topic_callback_wrapper(msg):
            return yolo_prompts_callback(msg, yolo_model)

        # Recreate the topic subscription with the wrapped callback
        bridge_node.destroy_subscription(prompts_subscription)

        prompts_subscription = bridge_node.create_subscription(
            String,
            '/yolo_prompts_update',
            topic_callback_wrapper,
            10
        )

    # Initialize ArUco tracking if ArUco is enabled
    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    prev_rvecs = {}
    parameters = None
    if args.aruco:
        parameters = aruco.DetectorParameters()

    # Load filter tuning from robot_config.yaml
    z_range_min, z_range_max = config.get_z_range()
    max_movement, hold_required, ghost_timeout = config.get_stability_params()
    q_params = config.get_kalman_process_noise()
    r_params = config.get_kalman_measurement_noise()
    blend_factor = config.get_blend_factor()

    frame_idx = 0
    talk = not args.suppress_prints
    
    print("\n" + "="*60)
    print("Merged Camera Localizer")
    print("="*60)
    if args.aruco:
        print("✓ ArUco detection: ENABLED")
    if args.yoloe:
        print("✓ YOLOE detection: ENABLED")
    if args.drop:
        print("✓ Drop poses: ENABLED")
    print(f"Waiting for camera frames on {args.camera_topic}...")
    if not args.headless:
        print("Press 'q' in the OpenCV window to quit.")
    else:
        print("Running in headless mode (no OpenCV window).")
    print("="*60 + "\n")

    detected_objects = []
    
    try:
        while True:
            # Get the latest frame from the camera topic
            frame = bridge_node.get_latest_frame()
            
            # If no frame available yet, wait a bit and continue
            if frame is None:
                import time
                time.sleep(0.01)  # 10ms
                continue

            frame_idx += 1
            ee_pos, ee_quat = bridge_node.get_ee_pose()
            cam_pos, cam_quat = bridge_node.get_camera_pose()

            # Initialize lists for detected objects
            identified_aruco = []
            yolo_detected_objects = []
            drop_poses = []

            # ArUco Detection Section
            if args.aruco:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
                estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                            kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, OPENCV_TO_CAMERA_QUAT, talk,
                            prev_rvecs=prev_rvecs, z_range_min=z_range_min, z_range_max=z_range_max,
                            max_movement=max_movement, hold_required=hold_required, ghost_timeout=ghost_timeout,
                            blend_factor=blend_factor, q_params=q_params, r_params=r_params)

                # After estimating pose, collect marker world positions
                for marker_id in kalman_filters:
                    if marker_stabilities[marker_id]["confirmed"]:
                        tvec, rvec = kalman_filters[marker_id].predict()
                        rquat = rvec_to_quat(rvec)
                        world_pos = transform_point_cam_to_world(tvec, cam_pos, cam_quat)
                        world_rot = transform_orientation_cam_to_world(rquat, cam_quat)
                        identified_aruco.append({
                                            "name": f"aruco_{marker_id}",
                                            "points": [world_pos],
                                            "position": world_pos,
                                            "quaternion": world_rot,
                                            'inferred': False,
                                        })

            # YOLO Detection Section
            if args.yoloe:
                # Check for dynamic YOLO prompt updates
                try:
                    updated_prompts, updated_prompt_map = get_yolo_prompts()
                except Exception as e:
                    print(f"Error checking YOLO prompts: {e}")

                current_prompts, current_prompt_map = get_yolo_prompts()
                
                # Get latest depth image if available
                depth_image = bridge_node.get_latest_depth()
                
                detected_color_points, detection_metadata = detect_yolo_blobs(
                    frame, yolo_model, CAMERA_MATRIX, cam_pos, cam_quat,
                    current_prompts, current_prompt_map,
                    OPENCV_TO_CAMERA_QUAT, 
                    distance=DETECTION_DISTANCE, depth_image=depth_image, bridge_node=bridge_node, conf_threshold=args.yolo_conf, nms_threshold=0.3
                )
                
                # Convert YOLO detections to object format for objects_poses topic
                # Use global variable for previous frame's objects
                global previous_yolo_objects

                # Create a mapping from detection metadata to world points
                # Group metadata by color for easier lookup
                metadata_by_color = {}
                for metadata in detection_metadata:
                    color_name = metadata['color_name']
                    if color_name not in metadata_by_color:
                        metadata_by_color[color_name] = []
                    metadata_by_color[color_name].append(metadata)

                # Convert YOLO detections to objects
                # Use position-based matching with previous frame to prevent ID flipping
                for color_name, world_points_data in detected_color_points.items():
                    # Get previous objects for this color
                    prev_objects = previous_yolo_objects.get(color_name, [])

                    # Get metadata for this color
                    color_metadata = metadata_by_color.get(color_name, [])

                    # Match new detections to previous objects by position
                    # Create list of (point_data, metadata) pairs
                    detection_pairs = []
                    for point_data in world_points_data:
                        point = point_data['point']
                        detection_idx = point_data['detection_index']

                        # Find corresponding metadata
                        matching_metadata = None
                        for metadata in color_metadata:
                            if metadata.get('detection_index') == detection_idx:
                                matching_metadata = metadata
                                break

                        if matching_metadata is not None:
                            detection_pairs.append((point_data, matching_metadata))

                    # Match detections to previous objects by closest position
                    matched_indices = set()
                    object_assignments = {}  # detection_idx -> object_index

                    if prev_objects and len(prev_objects) == len(detection_pairs):
                        for prev_idx, prev_obj in enumerate(prev_objects):
                            prev_pos = prev_obj['position']
                            min_dist = float('inf')
                            best_detection_idx = None

                            for point_data, metadata in detection_pairs:
                                detection_idx = point_data['detection_index']
                                if detection_idx in matched_indices:
                                    continue

                                point = point_data['point']
                                dist = np.linalg.norm(point - prev_pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_detection_idx = detection_idx

                            if best_detection_idx is not None and min_dist < 0.1:  # 10cm threshold
                                object_assignments[best_detection_idx] = prev_idx
                                matched_indices.add(best_detection_idx)

                    # Assign remaining detections to new indices
                    next_index = len(prev_objects) if prev_objects else 0
                    for point_data, metadata in detection_pairs:
                        detection_idx = point_data['detection_index']
                        if detection_idx not in object_assignments:
                            object_assignments[detection_idx] = next_index
                            next_index += 1

                    # Create objects in order of their assigned indices
                    sorted_pairs = sorted(detection_pairs, key=lambda x: object_assignments.get(x[0]['detection_index'], 999))

                    for i, (point_data, metadata) in enumerate(sorted_pairs):
                        point = point_data['point']
                        detection_idx = point_data['detection_index']
                        object_index = object_assignments.get(detection_idx, i)

                        # Get orientation from detection metadata
                        orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default identity quaternion
                        if metadata['orientation_angle'] is not None:
                            orientation_quat = convert_2d_orientation_to_quaternion(
                                metadata['orientation_angle'], cam_quat, OPENCV_TO_CAMERA_QUAT
                            )

                        # Store object name in metadata for label display
                        object_name = f"{color_name}_{object_index}"
                        metadata['object_name'] = object_name

                        yolo_detected_objects.append({
                            "name": object_name,
                            "points": [point],
                            "position": point,
                            "quaternion": orientation_quat,
                            'inferred': False,
                        })

                # Update previous objects for next frame
                previous_yolo_objects = {}
                for obj in yolo_detected_objects:
                    name_parts = obj["name"].rsplit("_", 1)
                    if len(name_parts) == 2:
                        color_name = name_parts[0]
                        if color_name not in previous_yolo_objects:
                            previous_yolo_objects[color_name] = []
                        previous_yolo_objects[color_name].append({
                            "name": obj["name"],
                            "position": obj["position"]
                        })

            # Combine all detected objects for drawing (aruco + yoloe)
            all_objects = identified_aruco + yolo_detected_objects

            # Drop pose calculation (if enabled and ArUco is enabled)
            if args.drop and args.aruco and marker_to_row is not None:
                for marker_id in kalman_filters:
                    if marker_stabilities[marker_id]["confirmed"]:
                        # Check if this marker has a drop pose configuration
                        if marker_id in marker_to_row:
                            row_name, offset_dict = marker_to_row[marker_id]
                            # Get marker world pose
                            tvec, rvec = kalman_filters[marker_id].predict()
                            rquat = rvec_to_quat(rvec)
                            marker_world_pos = transform_point_cam_to_world(tvec, cam_pos, cam_quat)
                            marker_world_rot = transform_orientation_cam_to_world(rquat, cam_quat)
                            
                            # Convert offset from dict to numpy array [X, Y, Z]
                            offset_marker = np.array([offset_dict["X"], offset_dict["Y"], offset_dict["Z"]])
                            
                            # Transform offset from marker frame to world frame
                            drop_pos_world = transform_offset_marker_to_world(
                                offset_marker, marker_world_pos, marker_world_rot
                            )
                            
                            # Use marker orientation for drop pose (or identity if you prefer)
                            drop_quat = marker_world_rot
                            
                            drop_poses.append({
                                "name": f"drop_{marker_id}",
                                "points": [drop_pos_world],
                                "position": drop_pos_world,
                                "quaternion": drop_quat,
                                'inferred': False,
                            })
                
                # Publish drop poses
                bridge_node.publish_drop_poses(drop_poses)
            
            # Publish object poses 
            bridge_node.publish_object_poses(yolo_detected_objects)
            
            # Always publish aruco poses separately (if aruco is enabled)
            if args.aruco:
                bridge_node.publish_aruco_poses(identified_aruco)

            # Draw all objects
            draw_text(frame, cam_pos, cam_quat, all_objects, frame_idx, ee_pos, ee_quat)
            draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, all_objects, [])

            # Draw YOLO detections with bounding boxes and axis lines (AFTER all other drawing)
            if args.yoloe:
                for detection in detection_metadata:
                    box = detection['box']
                    score = detection['score']
                    class_name = detection['class_name']
                    orientation_angle = detection['orientation_angle']
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw center dot
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Draw axis-aligned line with orientation
                    draw_axis_aligned_line(frame, box, orientation_angle)
                    
                    # Draw label with confidence and ID (replace spaces with underscores for display)
                    # Use object_name if available (includes ID), otherwise use class_name
                    if 'object_name' in detection:
                        label = f"{detection['object_name']}: {score:.2f}"
                    else:
                        class_name_display = class_name.replace(' ', '_')
                        label = f"{class_name_display}: {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 4), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Publish the annotated frame (same as what's displayed in OpenCV window)
            bridge_node.publish_annotated_image(frame)
            
            if not args.headless:
                cv2.imshow("Merged Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

