import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam, transform_orientation_world_to_cam, slerp_quat
from aruco_camera_localizer.detection_functions import detect_markers, estimate_pose
from aruco_camera_localizer.kalman_functions import QuaternionKalman
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_grasp_points
from aruco_camera_localizer.filter_config import FilterConfig
import threading
import rclpy
import argparse

c_width = 1280 # pix
c_hfov = 69.4 # deg
fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

c_height = 720 # pix
c_vfov = 42.5 # deg
fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

# Sim mode camera parameters (used when image topic is provided)
fx_sim = 731.78
fy_sim = 731.78

def create_camera_matrix(use_sim_mode=False):
    """Create camera matrix based on mode (sim or real camera)"""
    if use_sim_mode:
        # Sim mode: use fx = fy = 731.78
        return np.array([[fx_sim, 0, c_width / 2],
                         [0, fy_sim, c_height / 2],
                         [0, 0, 1]], dtype=np.float32)
    else:
        # Real camera mode: use calculated fx and fy
        return np.array([[fx, 0, c_width / 2],
                         [0, fy, c_height / 2],
                         [0, 0, 1]], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
# Marker dimensions
marker_size_mm = 21  # total marker size in mm
border_width_percent = 5  # white border percentage

# Calculate actual ArUco pattern size
MARKER_SIZE = marker_size_mm / 1000.0  # Convert to meters (21mm = 0.021m)
white_border_mm = marker_size_mm * (border_width_percent / 100.0)  # 5% of 21mm = 1.05mm
BORDER_WIDTH = white_border_mm / 1000.0  # Convert to meters (1.05mm = 0.00105m)
TOTAL_MARKER_SIZE = MARKER_SIZE - 2 * BORDER_WIDTH  # Actual ArUco pattern size
ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    # "DICT_5X5_250": aruco.DICT_5X5_250
}


trackers = {}

# =============================================================================
# ARUCO LOCALIZER FUNCTIONS
# =============================================================================

def load_aruco_annotations(json_file):
    """Load ArUco marker annotations from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['markers'], data.get('aruco_dictionary', 'DICT_4X4_50')

def get_available_models(data_dir):
    """Get list of available models from the data directory"""
    aruco_dir = Path(data_dir) / "aruco"
    
    if not aruco_dir.exists():
        return []
    
    # Get all aruco files
    aruco_files = list(aruco_dir.glob("*_aruco.json"))
    
    # Extract model names (remove _aruco.json suffix)
    available_models = {f.stem.replace("_aruco", "") for f in aruco_files}
    return sorted(list(available_models))

def estimate_object_pose_from_marker(marker_pose, aruco_annotation, cam_pos=None, cam_quat=None):
    """
    Estimate the 6D pose of the object center from ArUco marker pose.
    Uses homogeneous transformation matrices to compute position and orientation together.
    Returns position (tvec) and orientation (rvec) as rotation vector.
    
    Note: OpenCV's solvePnP returns marker pose in camera frame, but the marker's
    coordinate frame convention may differ from the CAD model. The JSON format uses
    T_object_to_marker which is inverted to get T_marker_to_object for pose estimation.
    
    Args:
        marker_pose: Tuple of (marker_tvec, marker_rvec) in camera frame
        aruco_annotation: Dictionary with marker annotation data from JSON (must contain T_object_to_marker)
        cam_pos: Optional camera position in world frame (for surface_normal verification)
        cam_quat: Optional camera orientation in world frame (for surface_normal verification)
    """
    # Get marker position and rotation
    marker_tvec, marker_rvec = marker_pose
    
    # Convert marker rotation vector to rotation matrix
    # Use the detected marker pose directly (no in-plane rotation removal)
    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    marker_tvec = marker_tvec.flatten()
    
    # Get the marker's transformation to object center from annotation
    # JSON format: T_object_to_marker (from object to marker)
    # Need to invert to get T_marker_to_object
    if 'T_object_to_marker' not in aruco_annotation:
        raise ValueError(
            f"Invalid JSON format! Marker ID {aruco_annotation.get('aruco_id', 'unknown')} missing required 'T_object_to_marker' field. "
            f"Available keys: {list(aruco_annotation.keys())}"
        )
    
    obj_to_marker_data = aruco_annotation['T_object_to_marker']
    
    # Get position and rotation from object to marker
    t_obj_to_marker = np.array([
        obj_to_marker_data['position']['x'],
        obj_to_marker_data['position']['y'], 
        obj_to_marker_data['position']['z']
    ])
    
    obj_to_marker_rot = obj_to_marker_data['rotation']
    
    # Use quaternion from JSON (quaternions are always available)
    quat = obj_to_marker_rot['quaternion']
    quat_array = np.array([quat['x'], quat['y'], quat['z'], quat['w']])  # scipy uses x, y, z, w
    R_obj_to_marker = R.from_quat(quat_array).as_matrix()
    
    # Invert to get T_marker_to_object
    R_marker_to_obj = R_obj_to_marker.T
    t_marker_to_obj = -R_marker_to_obj @ t_obj_to_marker
    
    # Build homogeneous transformation matrices
    # T_camera_to_marker: Marker pose in camera frame (4x4)
    # Using detected marker pose directly (no in-plane rotation removal)
    R_cm = marker_rotation_matrix  # R_camera_to_marker
    t_cm = marker_tvec
    
    # T_marker_to_object: Transformation from marker frame to object frame
    # The annotator stores R_marker_to_world (from marker to world/object frame)
    # and position of object in marker frame, so we can directly compose:
    # T_camera_to_object = T_camera_to_marker @ T_marker_to_object
    
    # Build homogeneous transformation matrices
    T_camera_to_marker = np.eye(4)
    T_camera_to_marker[:3, :3] = R_cm
    T_camera_to_marker[:3, 3] = t_cm
    
    T_marker_to_object = np.eye(4)
    T_marker_to_object[:3, :3] = R_marker_to_obj
    T_marker_to_object[:3, 3] = t_marker_to_obj
    
    # Compose transformations: T_camera_to_object = T_camera_to_marker @ T_marker_to_object
    # This is the chain: camera → marker → object
    T_camera_to_object = T_camera_to_marker @ T_marker_to_object
    
    # Extract position and orientation from combined transformation
    object_rotation_matrix = T_camera_to_object[:3, :3]
    object_tvec = T_camera_to_object[:3, 3]
    
    # Convert rotation matrix to rotation vector (standard OpenCV format)
    object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)
    
    return object_tvec, object_rvec

def quat_to_rvec(quat):
    """Convert quaternion to rotation vector"""
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    return rvec

def load_wireframe_data(json_file):
    """Load wireframe data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['vertices'], data['edges']

def load_grasp_points_data(json_file):
    """Load grasp points data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['grasp_points']


def validate_position_movement(object_pos_world, object_quat_world, prev_pos_world, prev_quat_world, 
                                frames_since_last, robot_moving, filter_config=None):
    """
    Validate object movement between frames to filter outliers.
    Rejects detections if the object has moved more than physically possible.
    
    Args:
        object_pos_world: Current object position in world frame (3D vector)
        object_quat_world: Current object orientation in world frame (quaternion)
        prev_pos_world: Previous object position in world frame (3D vector) or None
        prev_quat_world: Previous object orientation in world frame (quaternion) or None
        frames_since_last: Number of frames since last detection (for multi-frame gaps)
        robot_moving: Whether the robot is currently moving
        filter_config: FilterConfig instance (uses defaults if None)
    
    Returns:
        bool: True if movement is reasonable, False if outlier (moved too much)
    """
    # Use default config if not provided
    if filter_config is None:
        filter_config = FilterConfig()
    
    # If no previous pose, accept this detection (first detection)
    if prev_pos_world is None or prev_quat_world is None:
        return True
    
    # Calculate time elapsed (accounting for frame gaps)
    dt = frames_since_last / filter_config.movement_fps  # Time in seconds
    
    # Calculate position movement
    position_diff = np.linalg.norm(object_pos_world - prev_pos_world)
    
    # Calculate rotation difference (quaternion angular distance)
    quat_dot = np.abs(np.dot(object_quat_world, prev_quat_world))
    quat_dot = np.clip(quat_dot, -1.0, 1.0)
    rotation_diff = 2 * np.arccos(quat_dot)  # Angular distance in radians
    
    # Maximum allowed velocities (adjust based on your use case)
    # These represent maximum physically possible movement speeds
    if robot_moving:
        # When robot is moving, objects can move faster (robot might be moving them)
        MAX_VELOCITY = filter_config.movement_max_velocity_moving
        MAX_ANGULAR_VELOCITY = filter_config.movement_max_angular_velocity_moving
    else:
        # When robot is stationary, objects should move very slowly (if at all)
        MAX_VELOCITY = filter_config.movement_max_velocity_stationary
        MAX_ANGULAR_VELOCITY = filter_config.movement_max_angular_velocity_stationary
    
    # Calculate velocities
    if dt > 0:
        velocity = position_diff / dt
        angular_velocity = rotation_diff / dt
    else:
        # If dt is 0 or very small, check absolute movement
        velocity = position_diff * filter_config.movement_fps  # Approximate velocity
        angular_velocity = rotation_diff * filter_config.movement_fps  # Approximate angular velocity
    
    # Reject if movement exceeds maximum possible velocity
    if velocity > MAX_VELOCITY:
        return False  # Moved too fast (outlier)
    
    if angular_velocity > MAX_ANGULAR_VELOCITY:
        return False  # Rotated too fast (outlier)
    
    # Also check absolute movement thresholds (safety check for very large jumps)
    if position_diff > filter_config.movement_max_absolute_movement:
        return False  # Absolute movement too large (likely outlier)
    
    if rotation_diff > filter_config.movement_max_absolute_rotation:
        return False  # Absolute rotation too large (likely outlier)
    
    return True  # Movement is reasonable

def calculate_scale_factor_from_aruco(corners, marker_size):
    """Calculate scale factor from ArUco marker pixel size"""
    if not corners:
        return 1.0
    
    # Calculate pixel size of the first detected marker
    corner = corners[0][0]  # First marker, first corner set
    # Calculate side length in pixels
    side1 = np.linalg.norm(corner[0] - corner[1])  # Top side
    side2 = np.linalg.norm(corner[1] - corner[2])  # Right side
    side3 = np.linalg.norm(corner[2] - corner[3])  # Bottom side
    side4 = np.linalg.norm(corner[3] - corner[0])  # Left side
    
    # Average side length in pixels
    avg_pixel_size = (side1 + side2 + side3 + side4) / 4.0
    
    # Scale factor = physical_size / pixel_size
    scale_factor = marker_size / avg_pixel_size
    
    return scale_factor

def project_vertices_to_image(vertices, camera_matrix, dist_coeffs, scale_factor=1.0):
    """Project 3D vertices to 2D image coordinates with scale factor"""
    if len(vertices) == 0:
        return np.array([])
    
    # Apply scale factor to vertices
    scaled_vertices = vertices * scale_factor
    
    # Project points to image plane
    projected_points, _ = cv2.projectPoints(
        scaled_vertices.astype(np.float32), 
        np.zeros((3, 1)),  # No rotation (already in camera frame)
        np.zeros((3, 1)),  # No translation (already in camera frame)
        camera_matrix, 
        dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(np.int32)

def create_wireframe_mask(projected_vertices, edges, image_shape):
    """Create a binary mask of the wireframe boundary with smart clipping"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(projected_vertices) == 0:
        return mask
    
    height, width = image_shape[:2]
    
    # Check if too many vertices are outside the frame (object too close)
    vertices_in_frame = 0
    for vertex in projected_vertices:
        x, y = vertex
        if 0 <= x < width and 0 <= y < height:
            vertices_in_frame += 1
    
    # If less than 25% of vertices are in frame, don't render wireframe (object too close)
    if vertices_in_frame < len(projected_vertices) * 0.25:
        return mask
    
    # Apply smart clipping: scale down vertices that are outside the frame
    clipped_vertices = []
    for vertex in projected_vertices:
        x, y = vertex
        
        # If vertex is outside frame, clip it to the nearest edge
        if x < 0:
            x = 0
        elif x >= width:
            x = width - 1
            
        if y < 0:
            y = 0
        elif y >= height:
            y = height - 1
            
        clipped_vertices.append([x, y])
    
    # Draw wireframe edges on mask using clipped vertices with thicker lines
    for edge in edges:
        if len(edge) >= 2:
            start_idx, end_idx = edge[0], edge[1]
            if start_idx < len(clipped_vertices) and end_idx < len(clipped_vertices):
                start_point = tuple(clipped_vertices[start_idx])
                end_point = tuple(clipped_vertices[end_idx])
                cv2.line(mask, start_point, end_point, 255, 4)  # Increased thickness for stability
    
    # Apply strong morphological operations to stabilize the mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    # Apply Gaussian blur to smooth edges and reduce flickering
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = (mask > 127).astype(np.uint8) * 255  # Threshold back to binary
    
    # Find contours and fill them with much more conservative approach
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sort contours by area and fill only the largest, most stable one
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Only fill if the largest contour is substantial and stable
        if len(sorted_contours) > 0 and cv2.contourArea(sorted_contours[0]) > 1000:  # Much higher threshold
            cv2.fillPoly(mask, [sorted_contours[0]], 255)
    
    return mask




def start_ros_node(image_topic=None):
    rclpy.init()
    node = LocalizerBridge(image_topic)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with optional camera ID or ROS image topic.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--image-topic", type=str, default=None,
                        help="ROS2 image topic to subscribe to (e.g., '/camera/image_raw'). If provided, camera input is disabled.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--headless", action='store_true',
                        help="Run in headless mode: no OpenCV window, no logging, but annotated stream still published.")
    return parser.parse_args()



def main():
    args = parse_args()
    # Set headless mode early to suppress all logging
    headless_mode = args.headless
    
    # Create filter configuration
    filter_config = FilterConfig()

    # NOTE: Don't start ROS2 yet - it must be started AFTER camera selection
    # because select_camera() uses cv2.imshow() which conflicts with ROS2 threading

    # Create camera matrix based on mode (sim mode uses image topic)
    use_sim_mode = args.image_topic is not None
    CAMERA_MATRIX = create_camera_matrix(use_sim_mode)
    if not headless_mode:
        if use_sim_mode:
            print(f"Sim mode: Using fx = fy = {fx_sim}")
        else:
            print(f"Real camera mode: Using fx = {fx}, fy = {fy}")

    # Load aruco_localizer data
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    
    if not data_dir.exists():
        # Fallback to absolute path
        data_dir = Path("/home/aaugus11/Desktop/ros2_ws/src/aruco_camera_localizer/data")
    
    if not data_dir.exists():
        if not headless_mode:
            print(f"Could not find data directory at {data_dir}")
        return
    
    # Load all model data
    available_models = get_available_models(data_dir)
    if not available_models:
        if not headless_mode:
            print(f"No models found in data directory: {data_dir}")
        return
    
    if not headless_mode:
        print(f"Available models: {available_models}")
    
    model_data = {}
    marker_annotations = {}
    
    
    for model_name in available_models:
        aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
        wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
        grasp_file = data_dir / "grasp" / f"{model_name}_grasp_points_all_markers.json"
        
        try:
            aruco_annotations, aruco_dictionary = load_aruco_annotations(aruco_annotations_file)
            
            # Load wireframe data if available
            wireframe_vertices = None
            wireframe_edges = None
            if wireframe_file.exists():
                try:
                    wireframe_vertices, wireframe_edges = load_wireframe_data(wireframe_file)
                    if not headless_mode:
                        print(f"Loaded wireframe for {model_name}: {len(wireframe_vertices)} vertices, {len(wireframe_edges)} edges")
                except Exception as e:
                    if not headless_mode:
                        print(f"Warning: Could not load wireframe for {model_name}: {e}")
            
            # Load grasp points data if available
            grasp_points = None
            if grasp_file.exists():
                try:
                    grasp_points = load_grasp_points_data(grasp_file)
                    if not headless_mode:
                        print(f"Loaded grasp points for {model_name}: {len(grasp_points)} points")
                except Exception as e:
                    if not headless_mode:
                        print(f"Warning: Could not load grasp points for {model_name}: {e}")
            
            # Create a dictionary mapping (marker_id, dictionary) tuples to their annotations
            # This allows the same marker ID to exist in multiple models with different dictionaries
            for annotation in aruco_annotations:
                marker_id = annotation['aruco_id']
                key = (marker_id, aruco_dictionary)
                marker_annotations[key] = {
                    'annotation': annotation,
                    'model_name': model_name,
                    'aruco_dictionary': aruco_dictionary
                }
            
            model_data[model_name] = {
                'aruco_annotations': aruco_annotations,
                'wireframe_vertices': wireframe_vertices,
                'wireframe_edges': wireframe_edges,
                'grasp_points': grasp_points
            }
            
        except Exception as e:
            if not headless_mode:
                print(f"Error loading model {model_name}: {e}")
            continue
    
    if not model_data:
        if not headless_mode:
            print("No model data loaded successfully")
        return
    

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0
    
    # Track which marker is currently active for each object (to prevent switching)
    active_markers = {}  # {model_name: marker_id}
    last_object_poses = {}  # {model_name: (object_tvec, object_rvec, frame_idx)} - for pose consistency checking
    last_object_poses_world = {}  # {model_name: (object_pos_world, object_quat_world, frame_idx)} - for holding pose when out of scene
    previous_object_quaternions = {}  # {model_name: quaternion} - for SLERP smoothing
    prev_object_poses_world = {}  # {model_name: (object_pos_world, object_quat_world, frame_idx)} - for movement-based outlier rejection
    
    

    # Determine input source
    use_ros_topic = args.image_topic is not None
    cap = None

    if use_ros_topic:
        # Start ROS node before waiting for images
        bridge_node = start_ros_node(args.image_topic)

        if not headless_mode:
            print(f"Using ROS image topic: {args.image_topic}")
            print("Waiting for images from ROS topic...")
        # Wait for first frame
        while True:
            frame, frame_available = bridge_node.get_latest_frame()
            if frame_available:
                if not headless_mode:
                    print("Received first frame from ROS topic")
                break
            time.sleep(0.1)
    else:
        # Camera mode - select camera BEFORE starting ROS2
        if args.camera_id is not None:
            cam_id = args.camera_id
        else:
            available = detect_available_cameras()
            if not available:
                return
            cam_id = select_camera(available)
            if cam_id is None:
                return

        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            return

        # Now start ROS node AFTER camera selection is complete
        bridge_node = start_ros_node(None)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)

    # In headless mode, suppress all prints and disable OpenCV window
    talk = not args.suppress_prints and not headless_mode
    parameters = aruco.DetectorParameters()
    if not headless_mode:
        print("Press 'q' to quit.")

    detected_objects = []
    # Robot movement tracking for minimum z selection
    prev_ee_pos = None
    robot_slow_movement_threshold = filter_config.robot_slow_movement_threshold
    # Track how long the arm has been stationary (to handle motion blur after movement stops)
    frames_stationary = 0  # Number of consecutive frames the arm has been stationary
    stationary_settle_time = filter_config.stationary_settle_time
    
    while True:
        if use_ros_topic:
            frame, frame_available = bridge_node.get_latest_frame()
            if not frame_available:
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Publish raw camera image
        bridge_node.publish_image(frame)

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        identified_jenga = []
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

        # Detect robot movement - determine if robot is stationary or moving slowly
        robot_moving = True  # Default to moving
        if prev_ee_pos is not None:
            # Calculate robot movement distance
            robot_movement = np.linalg.norm(ee_pos - prev_ee_pos)
            # Robot is considered stationary/slow if movement is below threshold
            robot_moving = robot_movement > robot_slow_movement_threshold
            
            # Track how long the arm has been stationary
            if robot_moving:
                frames_stationary = 0  # Reset counter when arm is moving
            else:
                frames_stationary += 1  # Increment when arm is stationary
        else:
            frames_stationary = 0  # Initialize if no previous position
        
        prev_ee_pos = ee_pos.copy() if ee_pos is not None else None
        
        # Determine if arm has just stopped moving (motion blur period)
        # After movement stops, prefer fresh detections for a short period to handle motion blur
        just_stopped_moving = not robot_moving and frames_stationary < stationary_settle_time

        # Aruco Section - Now using aruco_localizer objects
        corners, ids, dict_names = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, dict_names, CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, 
                    filter_config=filter_config, talk=talk, robot_moving=robot_moving)
        
        
        # Calculate scale factor from ArUco marker detection
        scale_factor = calculate_scale_factor_from_aruco(corners, TOTAL_MARKER_SIZE)

        # Process confirmed markers and convert to object poses (one marker per object)
        # Use marker persistence: stick with current marker unless it's lost or pose differs too much
        object_detections = {}  # {model_name: (object_tvec, object_rvec, distance, marker_id)}
        
        # Only process markers that are actually detected in the current frame (no ghost tracking)
        detected_marker_ids = set(int(id_val) for id_val in ids) if ids is not None else set()
        
        # First, collect all candidate markers for each object
        candidate_markers = {}  # {model_name: [(marker_id, object_tvec, object_rvec, quality, distance), ...]}
        
        for marker_id in detected_marker_ids:
            if marker_id in marker_stabilities and marker_stabilities[marker_id]["confirmed"]:
                # Get the dictionary this marker was detected from
                detected_dict = marker_stabilities[marker_id].get("aruco_dictionary")
                
                if detected_dict is None:
                    continue
                
                # Look up annotation using (marker_id, dictionary) key
                annotation_key = (marker_id, detected_dict)
                
                if annotation_key not in marker_annotations:
                    # No annotation for this marker_id + dictionary combination
                    continue
                
                model_name = marker_annotations[annotation_key]['model_name']
                
                stability = marker_stabilities[marker_id]
                
                # Use Kalman filter's smoothed state if enabled, otherwise use confirmed pose
                # Use confirmed pose if available (don't require last_frame == frame_idx for first detection)
                if stability.get("confirmed_tvec") is not None and stability.get("confirmed_rvec") is not None:
                    if filter_config.enable_kalman_filter and marker_id in kalman_filters:
                        # Get smoothed pose from Kalman filter (after correction)
                        kalman = kalman_filters[marker_id]
                        tvec, rvec = kalman.get_state()
                    else:
                        # Use confirmed pose directly when Kalman is disabled
                        tvec = stability.get("confirmed_tvec")
                        rvec = stability.get("confirmed_rvec")
                else:
                    # Skip if no confirmed pose available
                    continue
                
                # Get object pose from marker pose
                marker_annotation = marker_annotations[annotation_key]['annotation']
                try:
                    object_tvec, object_rvec = estimate_object_pose_from_marker(
                        (tvec, rvec), marker_annotation, cam_pos=cam_pos, cam_quat=cam_quat
                    )
                except Exception as e:
                    if not headless_mode:
                        print(f"[{marker_id}] Error estimating object pose: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
                except ValueError as e:
                    # Skip markers with old/invalid JSON format
                    continue
                
                distance = np.linalg.norm(object_tvec)
                quality = stability.get("measurement_quality", 0.5)
                
                # Quality threshold filter: reject low-quality detections
                if filter_config.enable_quality_threshold:
                    if quality < filter_config.min_quality_threshold:
                        # Skip low-quality marker detection
                        continue
                
                # Store candidate marker
                if model_name not in candidate_markers:
                    candidate_markers[model_name] = []
                candidate_markers[model_name].append((marker_id, object_tvec, object_rvec, quality, distance))
        
        # Now select the best marker for each object (with persistence)
        for model_name, candidates in candidate_markers.items():
            if not candidates:
                continue
            
            # Check if we have an active marker for this object
            current_marker_id = active_markers.get(model_name)
            
            if current_marker_id is not None:
                # Check if current marker is still in candidates
                current_candidate = None
                for marker_id, object_tvec, object_rvec, quality, distance in candidates:
                    if marker_id == current_marker_id:
                        current_candidate = (marker_id, object_tvec, object_rvec, quality, distance)
                        break
                
                if current_candidate is not None:
                    # Current marker is still detected - check pose consistency
                    marker_id, object_tvec, object_rvec, quality, distance = current_candidate
                    
                    # Check if pose differs too much from last known pose
                    should_switch = False
                    if model_name in last_object_poses:
                        prev_tvec, prev_rvec, prev_frame = last_object_poses[model_name]
                        
                        # Calculate pose difference
                        position_diff = np.linalg.norm(object_tvec - prev_tvec)
                        
                        # Calculate rotation difference (quaternion angular distance)
                        object_quat = rvec_to_quat(object_rvec)
                        prev_quat = rvec_to_quat(prev_rvec)
                        quat_dot = np.abs(np.dot(object_quat, prev_quat))
                        quat_dot = np.clip(quat_dot, -1.0, 1.0)
                        rotation_diff = 2 * np.arccos(quat_dot)
                        
                        # Adaptive thresholds based on robot movement
                        if robot_moving:
                            MAX_POSITION_DIFF = filter_config.pose_consistency_max_position_diff_moving
                            MAX_ROTATION_DIFF = filter_config.pose_consistency_max_rotation_diff_moving
                        else:
                            MAX_POSITION_DIFF = filter_config.pose_consistency_max_position_diff_stationary
                            MAX_ROTATION_DIFF = filter_config.pose_consistency_max_rotation_diff_stationary
                        
                        # If pose differs too much, current marker might be lost/occluded - switch
                        if position_diff > MAX_POSITION_DIFF or rotation_diff > MAX_ROTATION_DIFF:
                            should_switch = True
                    
                    if not should_switch:
                        # Keep using current marker
                        object_detections[model_name] = (object_tvec, object_rvec, distance, marker_id)
                        last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)
                        continue
                    # else: pose differs too much, fall through to select new marker
                # else: current marker not detected, fall through to select new marker
            
            # Need to select a new marker (either no active marker, or current one lost/inconsistent)
            # Select best candidate (highest quality, or closest if quality similar)
            if len(candidates) == 1:
                # Only one candidate
                marker_id, object_tvec, object_rvec, quality, distance = candidates[0]
                object_detections[model_name] = (object_tvec, object_rvec, distance, marker_id)
                active_markers[model_name] = marker_id
                last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)
            else:
                # Multiple candidates - prefer higher quality, then closer distance
                candidates.sort(key=lambda x: (-x[3], x[4]))  # Sort by quality (desc), then distance (asc)
                marker_id, object_tvec, object_rvec, quality, distance = candidates[0]
                object_detections[model_name] = (object_tvec, object_rvec, distance, marker_id)
                active_markers[model_name] = marker_id
                last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)
        
        # Clean up active markers for objects not seen this frame
        objects_seen_this_frame = set(candidate_markers.keys())
        for model_name in list(active_markers.keys()):
            if model_name not in objects_seen_this_frame:
                # Object not detected - clear active marker after some time
                if model_name in last_object_poses:
                    prev_tvec, prev_rvec, prev_frame = last_object_poses[model_name]
                    if frame_idx - prev_frame > filter_config.pose_consistency_clear_timeout:
                        del active_markers[model_name]
                        del last_object_poses[model_name]
                        if model_name in last_object_poses_world:
                            del last_object_poses_world[model_name]
                        if model_name in prev_object_poses_world:
                            del prev_object_poses_world[model_name]
                        if model_name in previous_object_quaternions:
                            del previous_object_quaternions[model_name]
        
        # Process each object detection
        for model_name, (object_tvec, object_rvec, distance, marker_id) in object_detections.items():
            # Check if pose is valid
            if (object_tvec is not None and object_rvec is not None and 
                not np.any(np.isnan(object_tvec)) and not np.any(np.isnan(object_rvec))):
                
                # Use Kalman-filtered pose (temporally smoothed)
                # Convert to world frame
                object_quat = rvec_to_quat(object_rvec)
                object_pos_world = transform_point_cam_to_world(object_tvec, cam_pos, cam_quat)
                object_quat_world = transform_orientation_cam_to_world(object_quat, cam_quat)
                
                # Movement validation filter: check if moved too much
                if filter_config.enable_movement_validation:
                    prev_pos_world = None
                    prev_quat_world = None
                    frames_since_last = 1  # Default to 1 frame if no previous pose
                    if model_name in prev_object_poses_world:
                        prev_pos_world, prev_quat_world, prev_frame = prev_object_poses_world[model_name]
                        frames_since_last = frame_idx - prev_frame
                    
                    if not validate_position_movement(object_pos_world, object_quat_world, 
                                                        prev_pos_world, prev_quat_world, 
                                                        frames_since_last, robot_moving, filter_config=filter_config):
                        # Skip this detection - moved too much (outlier)
                        continue
                
                # SLERP smoothing filter: smooth quaternion interpolation (reduces jitter)
                if filter_config.enable_slerp_smoothing:
                    if model_name in previous_object_quaternions:
                        prev_quat_world = previous_object_quaternions[model_name]
                        # SLERP between previous and current quaternion for smooth transition
                        object_quat_world = slerp_quat(prev_quat_world, object_quat_world, blend=filter_config.slerp_blend_factor)
                
                # Update previous quaternion for next frame (always update for consistency)
                previous_object_quaternions[model_name] = object_quat_world.copy()
                
                # Update previous quaternion for next frame
                previous_object_quaternions[model_name] = object_quat_world.copy()
                
                # Store last known world frame pose for ghost tracking (pose holding)
                last_object_poses_world[model_name] = (object_pos_world.copy(), object_quat_world.copy(), frame_idx)
                
                # Store previous pose for movement-based outlier rejection
                prev_object_poses_world[model_name] = (object_pos_world.copy(), object_quat_world.copy(), frame_idx)
                
                # Create final object
                final_object = {
                    "name": model_name,
                    "points": [object_pos_world],
                    "position": object_pos_world,
                    "quaternion": object_quat_world,
                    'inferred': False,  # Always fresh detection (no ghost tracking)
                    'ghost_tracked': False,  # Not ghost-tracked (fresh detection)
                    'no_display': False,  # Always display (no timeout)
                    "object_tvec": object_tvec,
                    "object_rvec": object_rvec
                }
                identified_jenga.append(final_object)
        
        # Ghost tracking filter: hold pose when markers are not detected (use confirmed pose from marker stability)
        # Check for objects that were seen recently but not detected in current frame
        if filter_config.enable_ghost_tracking:
            for model_name in list(active_markers.keys()):
                if model_name not in objects_seen_this_frame and model_name in last_object_poses_world:
                    prev_pos_world, prev_quat_world, prev_frame = last_object_poses_world[model_name]
                    frames_since_seen = frame_idx - prev_frame
                    
                    # Only ghost track if object was seen recently (within timeout)
                    if frames_since_seen <= filter_config.ghost_tracking_timeout:
                        # Try to use confirmed pose from marker stability if available
                        marker_id = active_markers.get(model_name)
                        use_confirmed_pose = False
                        
                        if marker_id is not None and marker_id in marker_stabilities:
                            stability = marker_stabilities[marker_id]
                            # Use confirmed pose if marker is still confirmed (even if not detected this frame)
                            if stability.get("confirmed", False) and stability.get("confirmed_tvec") is not None:
                                confirmed_tvec = stability["confirmed_tvec"]
                                confirmed_rvec = stability["confirmed_rvec"]
                                
                                # Get object pose from confirmed marker pose
                                detected_dict = stability.get("aruco_dictionary")
                                
                                if detected_dict is None:
                                    continue
                                
                                # Look up annotation using (marker_id, dictionary) key
                                annotation_key = (marker_id, detected_dict)
                                
                                if annotation_key not in marker_annotations:
                                    # No annotation for this marker_id + dictionary combination
                                    continue
                                
                                marker_annotation = marker_annotations[annotation_key]['annotation']
                                try:
                                    object_tvec_cam, object_rvec_cam = estimate_object_pose_from_marker(
                                        (confirmed_tvec, confirmed_rvec), marker_annotation, cam_pos=cam_pos, cam_quat=cam_quat
                                    )
                                    
                                    # Convert to world frame
                                    object_quat_cam = rvec_to_quat(object_rvec_cam)
                                    object_pos_world = transform_point_cam_to_world(object_tvec_cam, cam_pos, cam_quat)
                                    object_quat_world = transform_orientation_cam_to_world(object_quat_cam, cam_quat)
                                    
                                    use_confirmed_pose = True
                                except ValueError:
                                    # Fall back to last known pose if annotation is invalid
                                    pass
                        
                        if not use_confirmed_pose:
                            # Fall back to last known world pose (no prediction, just freeze it)
                            object_pos_world = prev_pos_world.copy()
                            object_quat_world = prev_quat_world.copy()
                        
                        # Convert held world pose back to camera frame
                        object_pos_cam = transform_point_world_to_cam(object_pos_world, cam_pos, cam_quat)
                        object_quat_cam = transform_orientation_world_to_cam(object_quat_world, cam_quat)
                        object_rvec = quat_to_rvec(object_quat_cam)
                        object_tvec = object_pos_cam
                        
                        # Create ghost-tracked object with held pose (no validation needed - we're holding the last known good pose)
                        ghost_object = {
                            "name": model_name,
                            "points": [object_pos_world],
                            "position": object_pos_world,
                            "quaternion": object_quat_world,
                            'inferred': True,  # Mark as inferred (ghost-tracked)
                            'ghost_tracked': True,  # Mark as ghost-tracked
                            'no_display': False,  # Still display text and publish topics
                            "object_tvec": object_tvec,
                            "object_rvec": object_rvec
                        }
                        identified_jenga.append(ghost_object)
                
        objects = identified_jenga + detected_objects

        # Wireframe Mask Visualization for ArUco Objects (always enabled)
        # Skip wireframe drawing for ghost-tracked objects (they're out of scene)
        for obj in identified_jenga:
                # Skip wireframe for ghost-tracked objects
                if obj.get('ghost_tracked', False):
                    continue
                
                model_name = obj["name"]  # Now the name is just the model name
                
                if model_name in model_data and model_data[model_name]['wireframe_vertices'] is not None:
                    # Use the same object pose as published (world frame)
                    object_pos_world = obj["position"]
                    object_quat_world = obj["quaternion"]
                    
                    # Convert world frame pose to camera frame for validation
                    object_pos_cam = transform_point_world_to_cam(object_pos_world, cam_pos, cam_quat)
                    
                    # Validate wireframe position before drawing
                    # Check if object is in front of camera and within reasonable distance
                    is_valid_position = True
                    
                    # Check depth (z should be positive and reasonable)
                    if object_pos_cam[2] <= 0.01:  # Behind camera or too close
                        is_valid_position = False
                    elif object_pos_cam[2] > 2.0:  # Too far away (unlikely to be visible)
                        is_valid_position = False
                    elif object_pos_cam[2] < 0.05:  # Very close (might be invalid)
                        is_valid_position = False
                    
                    # Check if object is within reasonable bounds (not too far from camera center)
                    distance_from_camera = np.linalg.norm(object_pos_cam)
                    if distance_from_camera > 2.5:  # Too far
                        is_valid_position = False
                    
                    if not is_valid_position:
                        # Skip wireframe drawing if position is invalid
                        continue
                    
                    # Transform wireframe using world frame pose (same as published)
                    wireframe_vertices = model_data[model_name]['wireframe_vertices']
                    wireframe_edges = model_data[model_name]['wireframe_edges']
                    
                    # Transform vertices from object frame to world frame, then to image
                    # This uses the same pose as published
                    rot_matrix = R.from_quat(object_quat_world).as_matrix()
                    world_vertices = []
                    for vertex in wireframe_vertices:
                        # Transform from object frame to world frame
                        vertex_world = rot_matrix @ np.array(vertex) + object_pos_world
                        world_vertices.append(vertex_world)
                    
                    # Transform world vertices to image coordinates (same function used for grasp points)
                    projected_vertices = transform_points_world_to_img(world_vertices, cam_pos, cam_quat, CAMERA_MATRIX)
                    
                    # Filter out None values (points behind camera) and convert to numpy array format
                    valid_projected = []
                    for v in projected_vertices:
                        if v is not None:
                            valid_projected.append(v)
                    
                    # Additional validation: check if at least some vertices are within image bounds
                    if len(valid_projected) > 0:
                        vertices_in_bounds = 0
                        for v in valid_projected:
                            if 0 <= v[0] < frame.shape[1] and 0 <= v[1] < frame.shape[0]:
                                vertices_in_bounds += 1
                        
                        # If less than 10% of vertices are in bounds, object is likely out of view
                        if vertices_in_bounds < len(valid_projected) * 0.1:
                            continue  # Skip drawing if object is mostly out of view
                    
                    # Draw wireframe lines directly on the frame (no mask needed)
                    # Map original vertex indices to valid projected indices
                    vertex_map = {}
                    valid_idx = 0
                    for orig_idx, v in enumerate(projected_vertices):
                        if v is not None:
                            vertex_map[orig_idx] = valid_idx
                            valid_idx += 1
                    
                    for edge in wireframe_edges:
                        if len(edge) >= 2:
                            start_idx, end_idx = edge[0], edge[1]
                            # Only draw if both vertices are valid (not None)
                            if (start_idx in vertex_map and end_idx in vertex_map and
                                start_idx < len(projected_vertices) and end_idx < len(projected_vertices)):
                                start_point = tuple(valid_projected[vertex_map[start_idx]])
                                end_point = tuple(valid_projected[vertex_map[end_idx]])
                                # Draw green wireframe lines directly
                                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Blue blob detection removed - only using ArUco markers now
        identified_objects = []
        detected_objects = []
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects+identified_jenga)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, [])
        
        # Draw grasp points (always enabled with wireframe)
        draw_grasp_points(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, model_data)

        # Publish the annotated frame (what shows up in OpenCV window)
        bridge_node.publish_annotated_stream(frame)

        # Only show OpenCV window if not in headless mode
        if not headless_mode:
            cv2.imshow("Merged Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Cleanup: release resources and shut down ROS2
    if cap is not None:
        cap.release()
    if not headless_mode:
        cv2.destroyAllWindows()
    
    # Shut down ROS2 properly
    try:
        bridge_node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        if not headless_mode:
            print(f"Warning during ROS2 shutdown: {e}")

if __name__ == "__main__":
    main()