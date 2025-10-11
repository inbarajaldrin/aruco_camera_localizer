import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.config_loader import get_config
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from aruco_camera_localizer.detection_functions import detect_markers, estimate_pose, \
    identify_objects_from_blobs, attempt_recovery_for_missing_objects
from aruco_camera_localizer.object_frame_definitions import define_jenga_contacts, define_jenga_contour, hard_define_contour
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_color_dot_poses
import threading
import rclpy
import argparse
import json
from ultralytics import YOLOE
from max_camera_msgs.srv import UpdateYoloPrompts

# Load configuration from YAML
config = get_config()

c_width = config.get_camera_width()
c_height = config.get_camera_height()
c_hfov = config.get_camera_hfov()
c_vfov = config.get_camera_vfov()

fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

def convert_2d_orientation_to_quaternion(orientation_angle, cam_quat):
    """
    Convert 2D orientation angle from PCA to 3D quaternion in world frame.
    
    Args:
        orientation_angle: 2D orientation angle in radians from PCA
        cam_quat: Camera quaternion in world frame [x, y, z, w]
    
    Returns:
        quaternion: 3D quaternion in world frame [x, y, z, w]
    """
    # Create rotation around Z-axis (vertical) based on 2D orientation
    # The 2D angle represents the object's orientation in the image plane
    z_rotation = R.from_euler('z', orientation_angle)
    
    # Convert to quaternion
    z_quat = z_rotation.as_quat()  # Returns [x, y, z, w]
    
    # Transform the orientation from camera frame to world frame
    # The camera's orientation affects how the 2D orientation maps to 3D
    cam_rotation = R.from_quat(cam_quat)
    world_orientation = cam_rotation * R.from_quat(z_quat)
    
    return world_orientation.as_quat()

CAMERA_MATRIX = config.get_camera_matrix()
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
MARKER_SIZE = 0.019  # meters
BLOCK_LENGTH = 0.072 # meters
BLOCK_WIDTH = 0.024 # meters
BLOCK_THICKNESS = 0.014 # meters
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    # "DICT_5X5_250": aruco.DICT_5X5_250
}
OBJECT_DICTS = { # mm
    "allen_key": [38.8, 102.6, 129.5],
    "wrench": [37, 70, 70]
}
TARGET_POSES = {
    # position mm and orientation degrees
    "jenga": ([40, -600, 10], [0, 0, 0]),
    "wrench": ([40, -600, 10], [0, 0, 0]),
    "allen_key": ([40, -600, 10], [0, 0, 0]),
}

# YOLO detection settings - only hand detection
YOLO_PROMPTS = ["hand"]
YOLO_COLOR_MAP = {
    "hand": "hand"
}

# Global variables for dynamic YOLO prompt management
yolo_prompts_lock = threading.Lock()
current_yolo_prompts = YOLO_PROMPTS.copy()
current_yolo_color_map = YOLO_COLOR_MAP.copy()

# Generic color for all YOLO detections (cyan in BGR)
GENERIC_COLOR = (255, 255, 0)

pusher_distance_max = 0.030

trackers = {}

def start_ros_node():
    rclpy.init()
    node = LocalizerBridge()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def get_yolo_prompts():
    """Get current YOLO prompts (thread-safe)"""
    with yolo_prompts_lock:
        return current_yolo_prompts.copy(), current_yolo_color_map.copy()

def update_yolo_prompts(prompts, color_map, yolo_model=None):
    """Update YOLO prompts and color mapping (thread-safe)"""
    with yolo_prompts_lock:
        global current_yolo_prompts, current_yolo_color_map
        current_yolo_prompts = prompts.copy()
        current_yolo_color_map = color_map.copy()
        print(f"Updated YOLO prompts: {current_yolo_prompts}")
        print(f"Updated color mapping: {current_yolo_color_map}")
        
        # Update YOLO model if provided
        if yolo_model is not None:
            try:
                yolo_model.set_classes(current_yolo_prompts, yolo_model.get_text_pe(current_yolo_prompts))
                print(f"YOLO model updated with new prompts")
            except Exception as e:
                print(f"Failed to update YOLO model: {e}")

def update_yolo_prompts_callback(request, response, yolo_model=None):
    """Service callback for updating YOLO prompts"""
    try:
        prompts = json.loads(request.prompts_json)
        color_map = json.loads(request.color_map_json) if request.color_map_json else {}
        
        update_yolo_prompts(prompts, color_map, yolo_model)
        
        response.success = True
        response.message = f"Updated YOLO prompts to: {prompts}"
        print(f"YOLO prompts updated via service: {prompts}")
        
    except Exception as e:
        response.success = False
        response.message = f"Failed to update YOLO prompts: {str(e)}"
        print(f"Failed to update YOLO prompts: {e}")
        
    return response

def yolo_prompts_callback(msg, yolo_model=None):
    """Topic callback for real-time YOLO prompt updates"""
    try:
        data = json.loads(msg.data)
        prompts = data.get('prompts', [])
        color_map = data.get('color_map', {})
        
        update_yolo_prompts(prompts, color_map, yolo_model)
        print(f"YOLO prompts updated via topic: {prompts}")
        
    except Exception as e:
        print(f"Failed to update YOLO prompts from topic: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with YOLO detection.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--no-pushers", action='store_true',
                        help="Stops detecting yellow and green pushers")
    parser.add_argument("--recommend-push", action='store_true',
                        help="For each object, recommend where to push")
    parser.add_argument("--yolo-model", type=str, default="max_camera_localizer/yoloe-11s-seg.pt",
                        help="YOLO model path (default: max_camera_localizer/yoloe-11s-seg.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.4,
                        help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--yolo-prompts", type=str, nargs='+', 
                        default=["hand"],
                        help="YOLO detection prompts (default: hand)")
    parser.add_argument("--yolo-color-map", type=str, nargs='+',
                        help="Custom color mapping for prompts (format: prompt1:color1 prompt2:color2)")
    return parser.parse_args()

def pick_closest_blob(blobs, last_position):
    if not blobs:
        return None
    if last_position is None:
        return blobs[0]
    blobs_np = np.array(blobs)
    distances = np.linalg.norm(blobs_np - last_position, axis=1)
    closest_idx = np.argmin(distances)
    return blobs[closest_idx]

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

def detect_yolo_blobs(frame, yolo_model, camera_matrix, cam_pos, cam_quat, yolo_prompts, yolo_color_map, height=0.01, conf_threshold=0.4, nms_threshold=0.3):
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
                color_name = yolo_color_map.get(class_name, class_name)
                
                # Calculate center of bounding box
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Ray in camera frame
                pixel = np.array([center_x, center_y, 1.0])
                ray_cam = np.linalg.inv(camera_matrix) @ pixel
                
                # Transform ray to world frame
                R_wc = R.from_quat(cam_quat).as_matrix()
                ray_world = R_wc @ ray_cam
                cam_origin_world = np.array(cam_pos)
                
                # Ray-plane intersection with z = height over table
                t = (height - cam_origin_world[2]) / ray_world[2]
                point_world = cam_origin_world + t * ray_world
                
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
                detection_metadata.append({
                    'box': box,
                    'score': score,
                    'class_name': class_name,
                    'color_name': color_name,
                    'orientation_angle': orientation_angle
                })
                
                # Store by color
                if color_name not in detected_color_points:
                    detected_color_points[color_name] = []
                detected_color_points[color_name].append(point_world)
    
    return detected_color_points, detection_metadata

def main():
    args = parse_args()
    bridge_node = start_ros_node()
    
    # Set up YOLO prompt services and topics
    from std_msgs.msg import String
    
    # Service for updating YOLO prompts
    update_prompts_service = bridge_node.create_service(
        UpdateYoloPrompts,
        '/update_yolo_prompts',
        update_yolo_prompts_callback
    )
    
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
            prompts, color_map = get_yolo_prompts()
            prompts_data = {
                'prompts': prompts,
                'color_map': color_map
            }
            
            msg = String()
            msg.data = json.dumps(prompts_data)
            yolo_prompts_pub.publish(msg)
            
        except Exception as e:
            print(f"Failed to publish current prompts: {e}")
    
    prompts_timer = bridge_node.create_timer(1.0, publish_current_prompts)

    # Parse color mapping from command line if provided
    yolo_color_map = YOLO_COLOR_MAP.copy()
    if args.yolo_color_map:
        for mapping in args.yolo_color_map:
            if ':' in mapping:
                prompt, color = mapping.split(':', 1)
                yolo_color_map[prompt] = color.strip()
    
    # Update global variables with command line arguments
    update_yolo_prompts(args.yolo_prompts, yolo_color_map)

    # Initialize YOLO model with dynamic prompts
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo_model = YOLOE(args.yolo_model)
    yolo_model.set_classes(args.yolo_prompts, yolo_model.get_text_pe(args.yolo_prompts))
    print(f"YOLO model loaded with prompts: {args.yolo_prompts}")
    print(f"YOLO color mapping: {yolo_color_map}")
    
    # Update the service and topic callbacks to use the yolo_model
    def service_callback_wrapper(request, response):
        return update_yolo_prompts_callback(request, response, yolo_model)
    
    def topic_callback_wrapper(msg):
        return yolo_prompts_callback(msg, yolo_model)
    
    # Recreate the service and topic with the wrapped callbacks
    bridge_node.destroy_service(update_prompts_service)
    bridge_node.destroy_subscription(prompts_subscription)
    
    update_prompts_service = bridge_node.create_service(
        UpdateYoloPrompts,
        '/update_yolo_prompts',
        service_callback_wrapper
    )
    
    prompts_subscription = bridge_node.create_subscription(
        String,
        '/yolo_prompts_update',
        topic_callback_wrapper,
        10
    )

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0

    # Current YOLO prompts (will be updated dynamically)
    # These are now managed by the global functions

    if args.camera_id is not None:
        cam_id = args.camera_id
    else:        
        available = detect_available_cameras()
        if not available:
            return
        cam_id = select_camera(available)
        if cam_id is None:
            return

    talk = not args.suppress_prints
    if args.recommend_push:
        from max_camera_localizer.data_predict import predict_pusher_outputs

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return

    parameters = aruco.DetectorParameters()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    print("Press 'q' to quit.")

    detected_objects = []
    last_pushers = {"green": None, "yellow": None}
    unconfirmed_blobs = {"green": None, "yellow": None}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check for dynamic YOLO prompt updates
        try:
            updated_prompts, updated_color_map = get_yolo_prompts()
            # Check if prompts have changed (this will be handled by the global update functions)
            # The YOLO model will be updated when the prompts actually change
        except Exception as e:
            print(f"Error checking YOLO prompts: {e}")

        # Publish raw camera image
        bridge_node.publish_image(frame)

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        identified_jenga = []
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

        # Aruco Section
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, talk)

        # After estimating pose, collect marker world positions
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"]:
                tvec, rvec = kalman_filters[marker_id].predict()
                rquat = rvec_to_quat(rvec)
                world_pos = transform_point_cam_to_world(tvec, cam_pos, cam_quat)
                world_rot = transform_orientation_cam_to_world(rquat, cam_quat)
                identified_jenga.append({
                                    "name": f"jenga_{marker_id}",
                                    "points": [world_pos],
                                    "position": world_pos,
                                    "quaternion": world_rot,
                                    'inferred': False,
                                })

        objects = identified_jenga + detected_objects

        # YOLO Detection Section (replaces color blob detection)
        current_prompts, current_color_map = get_yolo_prompts()
        detected_color_points, detection_metadata = detect_yolo_blobs(
            frame, yolo_model, CAMERA_MATRIX, cam_pos, cam_quat, 
            current_prompts, current_color_map,
            height=0.01, conf_threshold=args.yolo_conf, nms_threshold=0.3
        )
        
        # Convert YOLO detections to object format for objects_poses topic
        yolo_detected_objects = []
        
        # Create a mapping from detection metadata to world points
        # Group metadata by color for easier lookup
        metadata_by_color = {}
        for metadata in detection_metadata:
            color_name = metadata['color_name']
            if color_name not in metadata_by_color:
                metadata_by_color[color_name] = []
            metadata_by_color[color_name].append(metadata)
        
        for color_name, world_points in detected_color_points.items():
            # Skip pusher colors as they're handled separately
            if color_name in ["green", "yellow"]:
                continue
            
            # Get metadata for this color
            color_metadata = metadata_by_color.get(color_name, [])
            
            # Add each detected point as an object
            for i, point in enumerate(world_points):
                # Get orientation from detection metadata
                orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default identity quaternion
                
                # Find corresponding detection metadata for this point
                if i < len(color_metadata):
                    metadata = color_metadata[i]
                    if metadata['orientation_angle'] is not None:
                        # Convert 2D orientation angle to 3D quaternion
                        orientation_quat = convert_2d_orientation_to_quaternion(
                            metadata['orientation_angle'], cam_quat
                        )
                
                yolo_detected_objects.append({
                    "name": f"{color_name}_dot_{i}",
                    "points": [point],
                    "position": point,
                    "quaternion": orientation_quat,
                    'inferred': False,
                })
        
        # Object identification (only for blue points for now)
        if "blue" in detected_color_points:
            identified_objects = identify_objects_from_blobs(detected_color_points["blue"], OBJECT_DICTS)
        else:
            identified_objects = []

        # Pusher section
        pushers = {}
        nearest_pushers = []
        if not args.no_pushers:
            # Process pusher colors from YOLO detections
            for color_name in ["green", "yellow"]:
                if color_name in detected_color_points:
                    world_points = detected_color_points[color_name]
                    if world_points:
                        best_point = pick_closest_blob(world_points, last_pushers.get(color_name))
                        pushers[color_name] = (best_point, GENERIC_COLOR)
                        last_pushers[color_name] = best_point

            # Working block for pusher-object interaction detection
            all_xyz = []
            all_kappa = []
            all_meta = []
            if objects:
                for obj_idx, obj in enumerate(objects):
                    if 'contour' not in obj or obj['contour'] is None:
                        continue
                    xyz = obj['contour']['xyz']
                    kappa = obj['contour']['kappa']
                    all_xyz.extend(xyz)
                    all_kappa.extend(kappa)
                    all_meta.extend([(obj_idx, i) for i in range(len(xyz))])

                if all_xyz:
                    all_xyz = np.array(all_xyz)
                    all_kappa = np.array(all_kappa)

                    tree = cKDTree(all_xyz)

                    for color, pusher in pushers.items():
                        if pusher is not None:
                            pusher_pos, col = pusher
                            distance, contour_idx = tree.query(pusher_pos)
                            if distance > pusher_distance_max:
                                continue
                            nearest_point = all_xyz[contour_idx]
                            kappa_value = all_kappa[contour_idx]
                            obj_index, local_contour_index = all_meta[contour_idx]
                            nearest_pushers.append({
                                'pusher_name': color,
                                'frame_number': frame_idx,
                                'color': col,
                                'pusher_location': pusher_pos,
                                'nearest_point': nearest_point,
                                'kappa': kappa_value,
                                'object_index': obj_index,
                                'local_contour_index': local_contour_index
                            })

        # Check for disappeared objects
        missing = False
        for det in detected_objects:
            if not any(obj["name"] == det["name"] for obj in identified_objects):
                missing = True
        
        # Attempt recovery if any objects are missing
        if missing: 
            blue_points = detected_color_points.get("blue", [])
            recovered_objects = attempt_recovery_for_missing_objects(detected_objects, blue_points, known_triangles=OBJECT_DICTS)
        else:
            recovered_objects = None

        # Avoid duplicating recovered ones already present
        if recovered_objects:
            for rec in recovered_objects:
                if not any(obj["name"] == rec["name"] for obj in identified_objects):
                    identified_objects.append(rec)

        # Bonus: For the ML test run, predict where the pushers should go
        for obj in identified_objects+identified_jenga:
            color = (255, 255, 0)
            name = obj["name"]
            if "jenga" in name:
                name = "jenga"
            if name in ["allen_key", "wrench", "jenga"] and 'contour' in obj and obj['contour'] is not None:
                if args.recommend_push:
                    posex = obj["position"][0]
                    posey = obj["position"][1]
                    objquat = obj["quaternion"]
                    objeuler = R.from_quat(objquat).as_euler('xyz')
                    oriy = objeuler[2]
                    prediction = predict_pusher_outputs(name, posex, posey, oriy, TARGET_POSES[name])
                    index = prediction['predicted_index']

                    # Draw predicted points
                    recommended = []
                    for ind in index:
                        label = f"pusher recommended @ contour {ind}"
                        pusher_point_world = obj['contour']['xyz'][ind]
                        pusher_point_img = transform_points_world_to_img([pusher_point_world], cam_pos, cam_quat, CAMERA_MATRIX)
                        pusher_point_normal = obj['contour']['normals'][ind]

                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - h - 20 - 5), (pusher_point_img[0][0] + w - 20, pusher_point_img[0][1] - 20 + 5), (0, 0, 0), -1)
                        cv2.putText(frame, label, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        cv2.circle(frame, pusher_point_img[0], 5, color)

                        recommended.append([pusher_point_world, pusher_point_normal])
                    if len(recommended) == 1:
                        recommended.append(recommended[0])
                    
                    bridge_node.publish_recommended_contacts(recommended)

                # draw target
                target_contour = hard_define_contour(TARGET_POSES[name][0], TARGET_POSES[name][1], name)
                contour_xyz = target_contour["xyz"]
                contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, CAMERA_MATRIX)
                contour_img = np.array(contour_img)
                contour_img.reshape((-1, 1, 2))
                contour_img = contour_img[::20]
                cv2.polylines(frame,[contour_img],False,color)

        detected_objects = identified_objects.copy()
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        # Publish all objects including YOLO detections to objects_poses topic
        bridge_node.publish_object_poses(identified_objects+identified_jenga+yolo_detected_objects)
        bridge_node.publish_contacts(nearest_pushers)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, nearest_pushers)
        # Create a simple color map using the generic color for all detected colors
        color_map = {color: GENERIC_COLOR for color in detected_color_points.keys()}
        draw_color_dot_poses(frame, CAMERA_MATRIX, cam_pos, cam_quat, detected_color_points, color_map)

        # Draw YOLO detections with bounding boxes and axis lines (AFTER all other drawing)
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
            
            # Draw label with confidence
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Publish the annotated frame (same as what's displayed in OpenCV window)
        bridge_node.publish_annotated_image(frame)
        
        cv2.imshow("YOLO-based Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
