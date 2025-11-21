import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import time
from datetime import datetime
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam, transform_orientation_world_to_cam, slerp_quat
from aruco_camera_localizer.detection_functions import detect_markers, estimate_pose
from aruco_camera_localizer.kalman_functions import QuaternionKalman
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_grasp_points
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
    return data['markers']

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

def estimate_object_pose_from_marker(marker_pose, aruco_annotation):
    """
    Estimate the 6D pose of the object center from ArUco marker pose.
    This is the same function from object_pose_estimator_kalman.py
    """
    # Get marker position and rotation
    marker_tvec, marker_rvec = marker_pose
    
    # Convert marker rotation vector to rotation matrix
    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    
    # Get the marker's pose relative to CAD center from annotation
    marker_relative_pose = aruco_annotation['pose_relative_to_cad_center']
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Get marker position relative to object center (in object frame)
    marker_pos_in_object = np.array([
        marker_relative_pose['position']['x'],
        marker_relative_pose['position']['y'], 
        marker_relative_pose['position']['z']
    ])
    
    # Apply coordinate transformation
    marker_pos_in_object = coord_transform @ marker_pos_in_object
    
    # Get marker orientation relative to object center
    marker_rot = marker_relative_pose['rotation']
    marker_rotation_in_object = euler_to_rotation_matrix(
        marker_rot['roll'], marker_rot['pitch'], marker_rot['yaw']
    )
    
    # Apply coordinate system transformation to the rotation matrix
    marker_rotation_in_object = coord_transform @ marker_rotation_in_object @ coord_transform.T
    
    # Calculate object center position in camera frame
    object_origin_in_marker_frame = marker_rotation_in_object.T @ (-marker_pos_in_object)
    object_tvec = marker_tvec.flatten() + marker_rotation_matrix @ object_origin_in_marker_frame
    
    # Calculate object center orientation in camera frame
    object_rotation_matrix = marker_rotation_matrix @ marker_rotation_in_object.T
    
    # Convert back to rotation vector
    object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)
    
    return object_tvec, object_rvec

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix"""
    r, p, y = roll, pitch, yaw
    
    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    
    # Combine rotations (order: Rz * Ry * Rx)
    return Rz @ Ry @ Rx

def quat_to_rvec(quat):
    """Convert quaternion to rotation vector"""
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    return rvec

def estimate_pose_covariance(measurement_quality, distance, is_ghost, base_covariance=None):
    """
    Estimate covariance matrix for a marker pose based on measurement quality, distance, and ghost status.
    
    Args:
        measurement_quality: Quality score (0.0-1.0)
        distance: Distance from camera to marker
        is_ghost: Whether this is a ghost (predicted) measurement
        base_covariance: Base covariance matrix (3x3 for position, 3x3 for orientation)
    
    Returns:
        position_cov: 3x3 covariance matrix for position
        orientation_cov: 3x3 covariance matrix for orientation (as quaternion)
    """
    if base_covariance is None:
        # Default base covariance
        base_pos_cov = np.eye(3) * 0.01  # 1cm base uncertainty
        base_rot_cov = np.eye(3) * 0.01  # ~0.01 rad base uncertainty
    else:
        base_pos_cov, base_rot_cov = base_covariance
    
    # Scale by inverse quality (lower quality = higher covariance)
    quality_factor = 1.0 / (measurement_quality + 0.1)  # Avoid division by zero
    
    # Scale by distance (farther = higher uncertainty)
    distance_factor = 1.0 + distance * 2.0  # Increase uncertainty with distance
    
    # Scale by ghost status (ghost = higher uncertainty)
    ghost_factor = 2.0 if is_ghost else 1.0
    
    # Combine factors
    total_factor = quality_factor * distance_factor * ghost_factor
    
    position_cov = base_pos_cov * total_factor
    orientation_cov = base_rot_cov * total_factor
    
    return position_cov, orientation_cov

def fuse_object_poses(object_poses, weights=None, measurement_qualities=None, distances=None, is_ghost_list=None):
    """
    Fuse multiple object poses from different markers into a single stable pose using covariance-based fusion.
    
    Args:
        object_poses: List of (tvec, rvec) tuples
        weights: Optional pre-calculated weights (if None, will use covariance-based weights)
        measurement_qualities: List of quality scores (0.0-1.0) for each pose
        distances: List of distances for each pose
        is_ghost_list: List of boolean indicating if each pose is ghost
    
    Returns:
        fused_tvec, fused_rvec: Fused pose
    """
    if not object_poses:
        return None, None
    
    if len(object_poses) == 1:
        return object_poses[0]
    
    # If covariance information is available, use covariance-based weighting
    if measurement_qualities is not None and distances is not None and is_ghost_list is not None:
        # Estimate covariance for each pose
        position_covs = []
        orientation_covs = []
        
        for quality, dist, is_ghost in zip(measurement_qualities, distances, is_ghost_list):
            pos_cov, rot_cov = estimate_pose_covariance(quality, dist, is_ghost)
            position_covs.append(pos_cov)
            orientation_covs.append(rot_cov)
        
        # Use inverse covariance weighting: weight_i = 1 / trace(covariance_i)
        # Lower covariance (more certain) = higher weight
        pos_weights = [1.0 / (np.trace(cov) + 1e-6) for cov in position_covs]
        rot_weights = [1.0 / (np.trace(cov) + 1e-6) for cov in orientation_covs]
        
        # Normalize weights
        pos_weight_sum = sum(pos_weights)
        rot_weight_sum = sum(rot_weights)
        if pos_weight_sum > 1e-6:
            pos_weights = [w / pos_weight_sum for w in pos_weights]
        if rot_weight_sum > 1e-6:
            rot_weights = [w / rot_weight_sum for w in rot_weights]
        
        # Fuse positions using covariance-weighted average
        fused_tvec = np.zeros(3)
        for (tvec, _), weight in zip(object_poses, pos_weights):
            fused_tvec += tvec * weight
        
        # Fuse rotations using weighted quaternion averaging
        quaternions = []
        for (_, rvec) in object_poses:
            quat = rvec_to_quat(rvec)
            quaternions.append(quat)
        
        # Weighted quaternion averaging
        fused_quat = np.zeros(4)
        for quat, weight in zip(quaternions, rot_weights):
            # Ensure quaternion is in the same hemisphere
            if np.dot(fused_quat, quat) < 0:
                quat = -quat
            fused_quat += quat * weight
        
        # Normalize the fused quaternion
        quat_norm = np.linalg.norm(fused_quat)
        if quat_norm > 1e-8:  # Avoid division by zero
            fused_quat = fused_quat / quat_norm
        else:
            # Fallback to identity quaternion if norm is too small
            fused_quat = np.array([0, 0, 0, 1])
        
        # Convert back to rotation vector
        fused_rvec = quat_to_rvec(fused_quat)
        
        return fused_tvec, fused_rvec
    
    # Fallback to simple weighted average if no covariance info
    if weights is None:
        weights = [1.0] * len(object_poses)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 1e-6:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(object_poses)] * len(object_poses)
    
    # Fuse positions (weighted average)
    fused_tvec = np.zeros(3)
    for (tvec, _), weight in zip(object_poses, weights):
        fused_tvec += tvec * weight
    
    # Fuse rotations using quaternion averaging
    quaternions = []
    for (_, rvec) in object_poses:
        quat = rvec_to_quat(rvec)
        quaternions.append(quat)
    
    # Weighted quaternion averaging
    fused_quat = np.zeros(4)
    for quat, weight in zip(quaternions, weights):
        # Ensure quaternion is in the same hemisphere
        if np.dot(fused_quat, quat) < 0:
            quat = -quat
        fused_quat += quat * weight
    
    # Normalize the fused quaternion
    quat_norm = np.linalg.norm(fused_quat)
    if quat_norm > 1e-8:  # Avoid division by zero
        fused_quat = fused_quat / quat_norm
    else:
        # Fallback to identity quaternion if norm is too small
        fused_quat = np.array([0, 0, 0, 1])
    
    # Convert back to rotation vector
    fused_rvec = quat_to_rvec(fused_quat)
    
    return fused_tvec, fused_rvec

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


def transform_mesh_to_camera_frame(vertices, object_pose):
    """Transform mesh vertices from object center frame to camera frame"""
    object_tvec, object_rvec = object_pose
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(object_rvec)
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Transform vertices from object center frame to camera frame
    transformed_vertices = []
    for vertex in vertices:
        # Apply coordinate system transformation
        vertex_transformed = coord_transform @ np.array(vertex)
        
        # Transform from object frame to camera frame
        vertex_cam = rotation_matrix @ vertex_transformed + object_tvec
        transformed_vertices.append(vertex_cam)
    
    return np.array(transformed_vertices)

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
    parser.add_argument("--debug-record", action='store_true',
                        help="Record debug data to JSON file for analysis of prediction fluctuations.")
    return parser.parse_args()



def main():
    args = parse_args()
    # Set headless mode early to suppress all logging
    headless_mode = args.headless
    bridge_node = start_ros_node(args.image_topic)
    
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
    
    if not headless_mode:
        print(f"DEBUG: About to load models: {available_models}")
    
    for model_name in available_models:
        aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
        wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
        grasp_file = data_dir / "grasp" / f"{model_name}_grasp_points_all_markers.json"
        
        try:
            aruco_annotations = load_aruco_annotations(aruco_annotations_file)
            
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
            
            # Create a dictionary mapping marker IDs to their annotations
            for annotation in aruco_annotations:
                marker_id = annotation['aruco_id']
                marker_annotations[marker_id] = {
                    'annotation': annotation,
                    'model_name': model_name
                }
            
            model_data[model_name] = {
                'aruco_annotations': aruco_annotations,
                'wireframe_vertices': wireframe_vertices,
                'wireframe_edges': wireframe_edges,
                'grasp_points': grasp_points
            }
            
            if not headless_mode:
                print(f"Loaded {model_name}: {len(aruco_annotations)} markers")
                print(f"DEBUG: Added {model_name} to model_data")
        except Exception as e:
            if not headless_mode:
                print(f"Error loading model {model_name}: {e}")
            continue
    
    if not model_data:
        if not headless_mode:
            print("No model data loaded successfully")
        return
    
    if not headless_mode:
        print(f"Total markers to track: {len(marker_annotations)}")
        print(f"DEBUG: Final model_data keys: {list(model_data.keys())}")
        print(f"Marker IDs: {sorted(marker_annotations.keys())}")

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0
    
    # Temporal smoothing for fused object poses
    object_pose_history = {}  # {model_name: {'tvec': prev_tvec, 'rvec': prev_rvec, 'tvec_world': prev_tvec_world, 'rvec_world': prev_rvec_world, 'last_fresh_frame': frame_idx}}
    # Track when objects were last seen with fresh detections (not all ghost)
    object_last_fresh_frame = {}  # {model_name: last_frame_with_fresh_detections}
    
    # Smoothing factor - adjust based on whether we have fresh detections or ghost data
    # When we have fresh detections: use high alpha (very responsive, real-time feedback)
    # When we have ghost data: use lower alpha (more stable, prevent flickering)
    smoothing_alpha_fresh = 0.8  # When all detections are fresh (80% new, 20% old - very responsive)
    smoothing_alpha_ghost = 0.2  # When using ghost data (20% new, 80% old - more stable)
    smoothing_alpha_stationary_ghost = 0.05  # When stationary and using ghost (5% new, 95% old - very stable)
    
    # Timeout for ghost tracking - after this time, stop displaying wireframe but continue tracking/publishing
    # Set to 2 seconds for both stationary and moving cases
    # After timeout: objects are still tracked and pose is published, but wireframe is not displayed
    ghost_tracking_timeout_stationary = 60  # frames - 2 seconds at 30fps
    ghost_tracking_timeout_moving = 60  # frames - 2 seconds at 30fps

    # Determine input source
    use_ros_topic = args.image_topic is not None
    cap = None
    
    if use_ros_topic:
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
        # Camera mode
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
    robot_slow_movement_threshold = 0.01  # meters - 10mm threshold for slow movement detection
    # Track how long the arm has been stationary (to handle motion blur after movement stops)
    frames_stationary = 0  # Number of consecutive frames the arm has been stationary
    stationary_settle_time = 30  # frames - ~1 second at 30fps - time to wait after movement stops before trusting previous values
    
    # Debug recording setup
    debug_data = [] if args.debug_record else None
    debug_start_time = time.time() if args.debug_record else None
    if args.debug_record:
        debug_file = Path(__file__).parent.parent / f"debug_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if not headless_mode:
            print(f"Debug recording enabled. Data will be saved to: {debug_file}")
    
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
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, talk, robot_moving=robot_moving)
        
        
        # Calculate scale factor from ArUco marker detection
        scale_factor = calculate_scale_factor_from_aruco(corners, TOTAL_MARKER_SIZE)

        # After estimating pose, collect ALL confirmed markers and fuse them by object
        # Include both detected markers and markers in ghost tracking mode
        object_detections = {}  # {model_name: [(object_tvec, object_rvec, distance, marker_id, is_ghost)]}
        
        # Collect all confirmed markers (including those in ghost tracking) and convert to object poses
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"] and marker_id in marker_annotations:
                stability = marker_stabilities[marker_id]
                frames_missing = stability.get("frames_missing", 0)
                is_ghost = frames_missing > 0  # True if marker is in ghost tracking mode
                
                # For ghost tracking, use last known world pose and convert back to camera frame
                if is_ghost and stability.get("last_known_tvec_world") is not None:
                    # Use last known world pose (objects don't move)
                    marker_pos_world = stability["last_known_tvec_world"]
                    marker_quat_world = stability["last_known_rvec_world"]
                    
                    # Convert back to camera frame for object pose estimation
                    marker_tvec_cam = transform_point_world_to_cam(marker_pos_world, cam_pos, cam_quat)
                    marker_quat_cam = transform_orientation_world_to_cam(marker_quat_world, cam_quat)
                    marker_rvec_cam = quat_to_rvec(marker_quat_cam)
                    
                    tvec, rvec = marker_tvec_cam, marker_rvec_cam
                else:
                    # Use normal measurement (detected or Kalman prediction)
                    use_kalman_filter = False  # Set to True to use Kalman filter predictions
                    
                    if use_kalman_filter:
                        tvec, rvec = kalman_filters[marker_id].predict()
                    else:
                        tvec, rvec = kalman_filters[marker_id].get_raw_measurement()
                
                # Get object pose from marker pose
                marker_annotation = marker_annotations[marker_id]['annotation']
                object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), marker_annotation)
                
                model_name = marker_annotations[marker_id]['model_name']
                distance = np.linalg.norm(object_tvec)
                
                # Get measurement quality from marker stability
                measurement_quality = marker_stabilities[marker_id].get("measurement_quality", 1.0)
                
                # Group by object type
                if model_name not in object_detections:
                    object_detections[model_name] = []
                object_detections[model_name].append((object_tvec, object_rvec, distance, marker_id, is_ghost, measurement_quality))
        
        # Fuse poses for each object using weighted averaging
        for model_name, detections in object_detections.items():
            if not detections:
                continue
                
            # Extract poses and calculate weights (closer markers get higher weight, ghost markers get lower weight)
            object_poses = [(tvec, rvec) for tvec, rvec, _, _, _, _ in detections]
            distances = [dist for _, _, dist, _, _, _ in detections]
            marker_ids = [mid for _, _, _, mid, _, _ in detections]
            is_ghost_list = [is_ghost for _, _, _, _, is_ghost, _ in detections]
            measurement_qualities = [quality for _, _, _, _, _, quality in detections]
            
            # Calculate weights (inverse distance - closer markers get higher weight)
            # Ghost markers get reduced weight (trust them less)
            # Measurement quality also affects weight (higher quality = higher weight)
            base_weights = [1.0 / (dist + 0.1) for dist in distances]  # Add small epsilon to avoid division by zero
            weights = []
            for i, (base_weight, is_ghost, quality) in enumerate(zip(base_weights, is_ghost_list, measurement_qualities)):
                # Start with base weight
                weight = base_weight
                
                # Apply ghost weight factor
                if is_ghost:
                    # Reduce weight for ghost markers (trust them less)
                    # When stationary, trust more; when moving, trust less
                    ghost_weight_factor = 0.5 if robot_moving else 0.8
                    weight *= ghost_weight_factor
                
                # Apply measurement quality factor (higher quality = higher weight)
                # Quality is 0.0-1.0, so we scale it appropriately
                quality_factor = 0.5 + 0.5 * quality  # Scale to 0.5-1.0 range (never completely ignore)
                weight *= quality_factor
                
                weights.append(weight)
            
            # Fuse poses using covariance-based fusion
            fused_tvec, fused_rvec = fuse_object_poses(
                object_poses, 
                weights=weights,  # Pass pre-calculated weights as fallback
                measurement_qualities=measurement_qualities,
                distances=distances,
                is_ghost_list=is_ghost_list
            )
            
            # Check if fused pose is valid
            if (fused_tvec is not None and fused_rvec is not None and 
                not np.any(np.isnan(fused_tvec)) and not np.any(np.isnan(fused_rvec))):
                
                # Adjust smoothing based on whether we have fresh detections or ghost data
                # When we have fresh detections: use high alpha for real-time feedback
                # When we have ghost data: use lower alpha to prevent flickering
                # Special handling: when arm just stopped moving, prefer fresh detections (motion blur)
                has_ghost = any(is_ghost_list)
                all_ghost = all(is_ghost_list)  # All markers are ghost
                
                # Calculate average measurement quality for this object
                avg_quality = np.mean(measurement_qualities) if measurement_qualities else 1.0
                
                # Check timeout if all detections are ghost
                # After timeout, mark as no_display (no wireframe) but continue tracking/publishing
                timeout_exceeded = False
                if all_ghost and model_name in object_last_fresh_frame:
                    last_fresh = object_last_fresh_frame[model_name]
                    frames_since_fresh = frame_idx - last_fresh
                    
                    # Determine timeout based on robot movement state
                    if robot_moving:
                        timeout = ghost_tracking_timeout_moving
                    else:
                        timeout = ghost_tracking_timeout_stationary
                    
                    # If object has been all ghost for too long, mark as no_display
                    if frames_since_fresh > timeout:
                        timeout_exceeded = True
                        if talk and frame_idx % 30 == 0 and not headless_mode:
                            print(f"[{model_name}] Timeout: all markers ghost for {frames_since_fresh} frames (timeout={timeout}) - stopping wireframe display, continuing pose tracking")
                
                if has_ghost:
                    # Using ghost data - be more conservative to prevent flickering
                    if just_stopped_moving:
                        # Arm just stopped - prefer fresh detections even if using ghost (motion blur period)
                        # Use higher alpha to quickly update pose after movement stops
                        smoothing_alpha = smoothing_alpha_ghost * 2.0  # Double the ghost alpha (40% new, 60% old)
                        smoothing_alpha = min(smoothing_alpha, 0.5)  # Cap at 50% to avoid too much noise
                    elif not robot_moving:
                        # Arm has been stationary for a while - be very conservative
                        smoothing_alpha = smoothing_alpha_stationary_ghost
                    else:
                        # When moving and using ghost, still conservative but less so
                        smoothing_alpha = smoothing_alpha_ghost
                else:
                    # All detections are fresh
                    if just_stopped_moving:
                        # Arm just stopped - prefer fresh detections strongly (motion blur period)
                        # Use very high alpha to quickly get accurate pose after movement stops
                        smoothing_alpha = 0.9  # 90% new, 10% old - very responsive
                    else:
                        # Normal fresh detection - use high alpha for real-time feedback
                        # Adjust based on measurement quality: higher quality = more trust in new measurement
                        smoothing_alpha = smoothing_alpha_fresh * (0.7 + 0.3 * avg_quality)  # Scale between 70%-100% of base alpha
                
                # Apply temporal smoothing to fused pose
                if model_name in object_pose_history:
                    prev_tvec = object_pose_history[model_name]['tvec']
                    prev_rvec = object_pose_history[model_name]['rvec']
                    
                    # Smooth position (linear interpolation)
                    smoothed_tvec = smoothing_alpha * fused_tvec + (1 - smoothing_alpha) * prev_tvec
                    
                    # Smooth rotation (quaternion slerp)
                    fused_quat = rvec_to_quat(fused_rvec)
                    prev_quat = rvec_to_quat(prev_rvec)
                    smoothed_quat = slerp_quat(prev_quat, fused_quat, smoothing_alpha)
                    smoothed_rvec = quat_to_rvec(smoothed_quat)
                else:
                    # First detection - use fused pose directly (no smoothing)
                    smoothed_tvec, smoothed_rvec = fused_tvec, fused_rvec
                
                smoothed_quat = rvec_to_quat(smoothed_rvec)
                
                # Convert to world frame
                object_pos_world = transform_point_cam_to_world(smoothed_tvec, cam_pos, cam_quat)
                object_quat_world = transform_orientation_cam_to_world(smoothed_quat, cam_quat)
                
                # Update pose history for next frame (store both camera and world frames)
                object_pose_history[model_name] = {
                    'tvec': smoothed_tvec.copy(),
                    'rvec': smoothed_rvec.copy(),
                    'tvec_world': object_pos_world.copy(),
                    'rvec_world': object_quat_world.copy(),
                    'last_fresh_frame': frame_idx
                }
                
                # Update last fresh frame if we have fresh detections (not all ghost)
                if not has_ghost:
                    object_last_fresh_frame[model_name] = frame_idx
                elif model_name not in object_last_fresh_frame:
                    # First time seeing this object, even if ghost, record it
                    object_last_fresh_frame[model_name] = frame_idx
                
                # Create final object
                # Mark as no_display if timeout exceeded (no wireframe/grasp points, but still publish pose)
                final_object = {
                    "name": model_name,
                    "points": [object_pos_world],
                    "position": object_pos_world,
                    "quaternion": object_quat_world,
                    'inferred': any(is_ghost_list),  # Mark as inferred if any markers are ghost
                    'no_display': timeout_exceeded,  # Mark as no_display if timeout exceeded (no wireframe/grasp)
                    "object_tvec": smoothed_tvec,
                    "object_rvec": smoothed_rvec
                }
                identified_jenga.append(final_object)
                
                if talk and frame_idx % 30 == 0:  # Only print every 30 frames
                    avg_distance = np.mean(distances)
                    ghost_count = sum(is_ghost_list)
                    if not headless_mode:
                        status = f"({ghost_count} ghost)" if ghost_count > 0 else ""
                        print(f"[{model_name}] Fused from {len(detections)} markers {status} - Avg distance: {avg_distance:.3f}m")
                        print(f"  Pos: {object_pos_world}")
                        print(f"  Quat: {object_quat_world}")
                        print(f"  Markers: {marker_ids}")
        
        # Handle objects that were previously detected but are now missing
        # Use previous known poses when detections are completely missing
        # After timeout, mark as no_display but continue tracking/publishing
        for model_name in list(object_pose_history.keys()):
            if model_name not in object_detections:
                # Object was detected before but not in current frame
                prev_pose = object_pose_history[model_name]
                last_fresh = object_last_fresh_frame.get(model_name, frame_idx)
                frames_since_fresh = frame_idx - last_fresh
                
                # Determine timeout based on robot movement state
                if robot_moving:
                    timeout = ghost_tracking_timeout_moving
                else:
                    timeout = ghost_tracking_timeout_stationary
                
                # Check if timeout exceeded - mark as no_display but continue tracking
                timeout_exceeded = frames_since_fresh > timeout
                
                # Always continue tracking and publishing, even after timeout
                if prev_pose.get('tvec_world') is not None:
                    # Use previous world pose directly (objects don't move)
                    object_pos_world = prev_pose['tvec_world'].copy()
                    object_quat_world = prev_pose['rvec_world'].copy()
                    
                    # Convert back to camera frame for consistency
                    object_tvec_cam = transform_point_world_to_cam(object_pos_world, cam_pos, cam_quat)
                    object_quat_cam = transform_orientation_world_to_cam(object_quat_world, cam_quat)
                    object_rvec_cam = quat_to_rvec(object_quat_cam)
                    
                    # Create object from previous pose
                    # Mark as no_display if timeout exceeded (no wireframe/grasp points, but still publish pose)
                    final_object = {
                        "name": model_name,
                        "points": [object_pos_world],
                        "position": object_pos_world,
                        "quaternion": object_quat_world,
                        'inferred': True,  # Always inferred when using previous value (no current detections)
                        'no_display': timeout_exceeded,  # Mark as no_display if timeout exceeded
                        "object_tvec": object_tvec_cam,
                        "object_rvec": object_rvec_cam
                    }
                    identified_jenga.append(final_object)
                    
                    if talk and frame_idx % 30 == 0 and not headless_mode:
                        status = f"(timeout exceeded, no wireframe/grasp)" if timeout_exceeded else ""
                        print(f"[{model_name}] Using previous pose (no current detections, missing for {frames_since_fresh}/{timeout} frames) {status}")

        objects = identified_jenga + detected_objects

        # Wireframe Mask Visualization for ArUco Objects (only for best detections)
        # Skip wireframe drawing if timeout exceeded (no_display flag)
        for obj in identified_jenga:
            model_name = obj["name"]  # Now the name is just the model name
            
            # Skip wireframe drawing if no_display flag is set (timeout exceeded)
            if obj.get('no_display', False):
                continue  # Don't draw wireframe for objects that exceeded timeout
            
            if model_name in model_data and model_data[model_name]['wireframe_vertices'] is not None:
                # Get object pose in camera frame
                object_pos_world = obj["position"]
                object_quat_world = obj["quaternion"]
                
                # Transform to camera frame
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
                    # This prevents displaying wireframe when ghost tracking has drifted
                    continue
                
                # For quaternion, we need to transform from world to camera frame
                cam_rotation_matrix = R.from_quat(cam_quat).as_matrix()
                object_rotation_matrix = R.from_quat(object_quat_world).as_matrix()
                object_rotation_cam = cam_rotation_matrix.T @ object_rotation_matrix
                object_quat_cam = R.from_matrix(object_rotation_cam).as_quat()
                
                # Convert quaternion to rotation vector
                object_rotation_matrix = R.from_quat(object_quat_cam).as_matrix()
                object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)
                
                # Transform wireframe to camera frame
                wireframe_vertices = model_data[model_name]['wireframe_vertices']
                wireframe_edges = model_data[model_name]['wireframe_edges']
                
                transformed_vertices = transform_mesh_to_camera_frame(wireframe_vertices, (object_pos_cam, object_rvec))
                projected_vertices = project_vertices_to_image(transformed_vertices, CAMERA_MATRIX, DIST_COEFFS, scale_factor)
                
                # Additional validation: check if at least some vertices are within image bounds
                if len(projected_vertices) > 0:
                    vertices_in_bounds = 0
                    for v in projected_vertices:
                        if 0 <= v[0] < frame.shape[1] and 0 <= v[1] < frame.shape[0]:
                            vertices_in_bounds += 1
                    
                    # If less than 10% of vertices are in bounds, object is likely out of view
                    if vertices_in_bounds < len(projected_vertices) * 0.1:
                        continue  # Skip drawing if object is mostly out of view
                
                # Draw wireframe lines directly on the frame (no mask needed)
                for edge in wireframe_edges:
                    if len(edge) >= 2:
                        start_idx, end_idx = edge[0], edge[1]
                        if start_idx < len(projected_vertices) and end_idx < len(projected_vertices):
                            start_point = tuple(projected_vertices[start_idx])
                            end_point = tuple(projected_vertices[end_idx])
                            # Draw green wireframe lines directly
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Blue blob detection removed - only using ArUco markers now
        identified_objects = []
        detected_objects = []
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects+identified_jenga)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, [])
        draw_grasp_points(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, model_data)

        # Publish the annotated frame (what shows up in OpenCV window)
        bridge_node.publish_annotated_stream(frame)
        
        # Record debug data if enabled
        if args.debug_record and debug_data is not None:
            frame_time = time.time() - debug_start_time
            
            # Collect marker data
            marker_data = {}
            for marker_id in kalman_filters:
                if marker_id in marker_stabilities:
                    stability = marker_stabilities[marker_id]
                    kalman = kalman_filters[marker_id]
                    
                    # Get Kalman state
                    velocity = kalman.get_velocity() if hasattr(kalman, 'get_velocity') else None
                    acceleration = kalman.get_acceleration() if hasattr(kalman, 'get_acceleration') else None
                    
                    # Get predictions
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    pred_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                    pred_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
                    
                    # Get raw measurement if available
                    raw_tvec, raw_rvec = kalman.get_raw_measurement() if hasattr(kalman, 'get_raw_measurement') else (None, None)
                    raw_quat = rvec_to_quat(raw_rvec) if raw_rvec is not None else None
                    raw_pos_world = transform_point_cam_to_world(raw_tvec, cam_pos, cam_quat) if raw_tvec is not None else None
                    raw_quat_world = transform_orientation_cam_to_world(raw_quat, cam_quat) if raw_quat is not None else None
                    
                    # Get Kalman corrected state (statePost)
                    kalman_state = kalman.kf.statePost.flatten() if hasattr(kalman, 'kf') else None
                    corrected_tvec = kalman_state[0:3] if kalman_state is not None else None
                    corrected_quat = kalman_state[3:7] if kalman_state is not None else None
                    corrected_pos_world = transform_point_cam_to_world(corrected_tvec, cam_pos, cam_quat) if corrected_tvec is not None else None
                    corrected_quat_world = transform_orientation_cam_to_world(corrected_quat, cam_quat) if corrected_quat is not None else None
                    
                    marker_data[marker_id] = {
                        "confirmed": bool(stability.get("confirmed", False)),
                        "frames_missing": int(stability.get("frames_missing", 0)),
                        "measurement_quality": float(stability.get("measurement_quality", 1.0)),
                        "prediction_mode": stability.get("prediction_mode", None),
                        "last_known_pos_world": stability.get("last_known_tvec_world", None).tolist() if stability.get("last_known_tvec_world") is not None else None,
                        "last_known_quat_world": stability.get("last_known_rvec_world", None).tolist() if stability.get("last_known_rvec_world") is not None else None,
                        "kalman_pred_pos_cam": pred_tvec.tolist(),
                        "kalman_pred_quat_cam": pred_quat.tolist(),
                        "kalman_pred_pos_world": pred_pos_world.tolist(),
                        "kalman_pred_quat_world": pred_quat_world.tolist(),
                        "kalman_corrected_pos_cam": corrected_tvec.tolist() if corrected_tvec is not None else None,
                        "kalman_corrected_quat_cam": corrected_quat.tolist() if corrected_quat is not None else None,
                        "kalman_corrected_pos_world": corrected_pos_world.tolist() if corrected_pos_world is not None else None,
                        "kalman_corrected_quat_world": corrected_quat_world.tolist() if corrected_quat_world is not None else None,
                        "raw_measurement_pos_cam": raw_tvec.tolist() if raw_tvec is not None else None,
                        "raw_measurement_quat_cam": raw_quat.tolist() if raw_quat is not None else None,
                        "raw_measurement_pos_world": raw_pos_world.tolist() if raw_pos_world is not None else None,
                        "raw_measurement_quat_world": raw_quat_world.tolist() if raw_quat_world is not None else None,
                        "velocity": velocity.tolist() if velocity is not None else None,
                        "acceleration": acceleration.tolist() if acceleration is not None else None,
                        "confirmed_tvec": stability.get("confirmed_tvec", None).tolist() if stability.get("confirmed_tvec") is not None else None,
                        "confirmed_rvec": stability.get("confirmed_rvec", None).tolist() if stability.get("confirmed_rvec") is not None else None
                    }
            
            # Collect object data
            object_data = {}
            for obj in identified_jenga:
                model_name = obj["name"]
                object_data[model_name] = {
                    "position_world": obj["position"].tolist(),
                    "quaternion_world": obj["quaternion"].tolist(),
                    "inferred": bool(obj.get("inferred", False)),
                    "no_display": bool(obj.get("no_display", False))
                }
            
            # Record frame data
            frame_record = {
                "frame": int(frame_idx),
                "timestamp": float(frame_time),
                "robot_moving": bool(robot_moving),
                "frames_stationary": int(frames_stationary),
                "just_stopped_moving": bool(just_stopped_moving),
                "ee_position": ee_pos.tolist() if ee_pos is not None else None,
                "ee_quaternion": ee_quat.tolist() if ee_quat is not None else None,
                "camera_position": cam_pos.tolist() if cam_pos is not None else None,
                "camera_quaternion": cam_quat.tolist() if cam_quat is not None else None,
                "robot_movement_distance": float(np.linalg.norm(ee_pos - prev_ee_pos)) if prev_ee_pos is not None and ee_pos is not None else 0.0,
                "markers": marker_data,
                "objects": object_data,
                "num_detected_markers": int(len(corners) if corners else 0),
                "detected_marker_ids": [int(id_val) for id_val in ids] if ids else []
            }
            debug_data.append(frame_record)

        # Only show OpenCV window if not in headless mode
        if not headless_mode:
            cv2.imshow("Merged Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup: release resources and shut down ROS2 before saving debug data
    if cap is not None:
        cap.release()
    if not headless_mode:
        cv2.destroyAllWindows()
    
    # Shut down ROS2 properly before saving debug data
    # This prevents threading conflicts during JSON serialization
    try:
        bridge_node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        if not headless_mode:
            print(f"Warning during ROS2 shutdown: {e}")
    
    # Save debug data if recording was enabled
    # Do this after ROS2 shutdown to avoid threading conflicts
    if args.debug_record and debug_data is not None:
        try:
            if not headless_mode:
                print(f"\nSaving debug data to: {debug_file}")
                print(f"Recorded {len(debug_data)} frames")
            with open(debug_file, 'w') as f:
                json.dump({
                    "recording_info": {
                        "start_time": datetime.fromtimestamp(debug_start_time).isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "total_frames": len(debug_data),
                        "duration_seconds": time.time() - debug_start_time
                    },
                    "frames": debug_data
                }, f, indent=2)
            if not headless_mode:
                print(f"Debug data saved successfully")
        except KeyboardInterrupt:
            # If user interrupts during save, just exit
            if not headless_mode:
                print(f"\nDebug data save interrupted")
        except Exception as e:
            if not headless_mode:
                print(f"Error saving debug data: {e}")

if __name__ == "__main__":
    main()