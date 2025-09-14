import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from max_camera_localizer.camera_selection import detect_available_cameras, select_camera
from max_camera_localizer.localizer_bridge import LocalizerBridge
from max_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from max_camera_localizer.detection_functions import detect_markers, detect_color_blobs, detect_color_blobs_in_mask, estimate_pose, \
    identify_objects_from_blobs, attempt_recovery_for_missing_objects
from max_camera_localizer.object_frame_definitions import define_jenga_contacts, define_jenga_contour, hard_define_contour
from max_camera_localizer.drawing_functions import draw_text, draw_object_lines
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

CAMERA_MATRIX = np.array([[fx, 0, c_width / 2],
                          [0, fy, c_height / 2],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
MARKER_SIZE = 0.021  # meters - from aruco_localizer
BLOCK_LENGTH = 0.072 # meters
BLOCK_WIDTH = 0.024 # meters
BLOCK_THICKNESS = 0.014 # meters
ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
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

blue_range = [np.array([100, 80, 80]), np.array([140, 255, 255])]
green_range = [np.array([35, 80, 100]), np.array([75, 255, 255])]
yellow_range = [np.array([15, 80, 60]), np.array([35, 255, 255])]

pusher_distance_max = 0.030

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
    
    # Apply scaling and coordinate transformation
    marker_pos_in_object = coord_transform @ (marker_pos_in_object * 1.25)
    
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

def load_wireframe_data(json_file):
    """Load wireframe data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['vertices'], data['edges']

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
        # Apply coordinate system transformation and scaling
        vertex_transformed = coord_transform @ (np.array(vertex) * 1.25)
        
        # Transform from object frame to camera frame
        vertex_cam = rotation_matrix @ vertex_transformed + object_tvec
        transformed_vertices.append(vertex_cam)
    
    return np.array(transformed_vertices)

def project_vertices_to_image(vertices, camera_matrix, dist_coeffs):
    """Project 3D vertices to 2D image coordinates"""
    if len(vertices) == 0:
        return np.array([])
    
    # Project points to image plane
    projected_points, _ = cv2.projectPoints(
        vertices.astype(np.float32), 
        np.zeros((3, 1)),  # No rotation (already in camera frame)
        np.zeros((3, 1)),  # No translation (already in camera frame)
        camera_matrix, 
        dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(np.int32)

def create_wireframe_mask(projected_vertices, edges, image_shape):
    """Create a binary mask of the wireframe boundary"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(projected_vertices) == 0:
        return mask
    
    # Filter out vertices that are outside the image bounds
    height, width = image_shape[:2]
    valid_vertices = []
    valid_indices = []
    
    for i, vertex in enumerate(projected_vertices):
        x, y = vertex
        if 0 <= x < width and 0 <= y < height:
            valid_vertices.append(vertex)
            valid_indices.append(i)
    
    if len(valid_vertices) == 0:
        return mask
    
    # Create mapping from original indices to valid indices
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
    
    # Draw wireframe edges on mask
    for edge in edges:
        if len(edge) >= 2:
            start_idx, end_idx = edge[0], edge[1]
            if start_idx in index_map and end_idx in index_map:
                start_point = tuple(valid_vertices[index_map[start_idx]])
                end_point = tuple(valid_vertices[index_map[end_idx]])
                cv2.line(mask, start_point, end_point, 255, 2)
    
    # Fill the wireframe boundary to create a solid mask
    # Find contours and fill them
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Fill the largest contour (main object boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    return mask

def extract_color_range_from_mask(frame, mask, min_samples=100):
    """Extract color range from pixels within the mask"""
    # Convert frame to HSV for better color analysis
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get pixels within the mask
    masked_pixels = hsv_frame[mask > 0]
    
    print(f"DEBUG: Mask has {np.sum(mask > 0)} pixels, need {min_samples} minimum")
    
    if len(masked_pixels) < min_samples:
        print(f"DEBUG: Not enough pixels in mask ({len(masked_pixels)} < {min_samples})")
        return None, None
    
    # Calculate color statistics
    h_mean, h_std = np.mean(masked_pixels[:, 0]), np.std(masked_pixels[:, 0])
    s_mean, s_std = np.mean(masked_pixels[:, 1]), np.std(masked_pixels[:, 1])
    v_mean, v_std = np.mean(masked_pixels[:, 2]), np.std(masked_pixels[:, 2])
    
    print(f"DEBUG: Color stats - H: {h_mean:.1f}±{h_std:.1f}, S: {s_mean:.1f}±{s_std:.1f}, V: {v_mean:.1f}±{v_std:.1f}")
    
    # Create color range with some tolerance
    h_tolerance = max(10, h_std * 2)  # At least 10 degrees tolerance
    s_tolerance = max(30, s_std * 2)  # At least 30 saturation tolerance
    v_tolerance = max(30, v_std * 2)  # At least 30 value tolerance
    
    lower_bound = np.array([
        max(0, h_mean - h_tolerance),
        max(0, s_mean - s_tolerance),
        max(0, v_mean - v_tolerance)
    ], dtype=np.uint8)
    
    upper_bound = np.array([
        min(179, h_mean + h_tolerance),
        min(255, s_mean + s_tolerance),
        min(255, v_mean + v_tolerance)
    ], dtype=np.uint8)
    
    print(f"DEBUG: Generated HSV range - Lower: {lower_bound}, Upper: {upper_bound}")
    
    return lower_bound, upper_bound



def start_ros_node():
    rclpy.init()
    node = LocalizerBridge()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with optional camera ID.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--no-pushers", action='store_true',
                        help="Stops detecting yellow and green pushers")
    parser.add_argument("--recommend-push", action='store_true',
                        help="For each object, recommend where to push")
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

def match_points(new_blobs, unconfirmed_blobs, confirmed_blobs):
    pass

def main():
    args = parse_args()
    bridge_node = start_ros_node()

    # Load aruco_localizer data
    current_dir = Path(__file__).parent
    source_dir = current_dir.parent.parent.parent / "src" / "max_camera_localizer"
    data_dir = source_dir / "aruco-grasp-annotator" / "data"
    
    if not data_dir.exists():
        data_dir = Path("/home/aaugus11/Desktop/ros2_ws/src/max_camera_localizer/aruco-grasp-annotator/data")
    
    if not data_dir.exists():
        print(f"Could not find aruco-grasp-annotator data directory at {data_dir}")
        return
    
    # Load all model data
    available_models = get_available_models(data_dir)
    if not available_models:
        print(f"No models found in data directory: {data_dir}")
        return
    
    print(f"Available models: {available_models}")
    
    model_data = {}
    marker_annotations = {}
    
    print(f"DEBUG: About to load models: {available_models}")
    
    for model_name in available_models:
        aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
        wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
        
        try:
            aruco_annotations = load_aruco_annotations(aruco_annotations_file)
            
            # Load wireframe data if available
            wireframe_vertices = None
            wireframe_edges = None
            if wireframe_file.exists():
                try:
                    wireframe_vertices, wireframe_edges = load_wireframe_data(wireframe_file)
                    print(f"Loaded wireframe for {model_name}: {len(wireframe_vertices)} vertices, {len(wireframe_edges)} edges")
                except Exception as e:
                    print(f"Warning: Could not load wireframe for {model_name}: {e}")
            
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
                'wireframe_edges': wireframe_edges
            }
            
            print(f"Loaded {model_name}: {len(aruco_annotations)} markers")
            print(f"DEBUG: Added {model_name} to model_data")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    if not model_data:
        print("No model data loaded successfully")
        return
    
    print(f"Total markers to track: {len(marker_annotations)}")
    print(f"DEBUG: Final model_data keys: {list(model_data.keys())}")
    print(f"Marker IDs: {sorted(marker_annotations.keys())}")

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0

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
    unconfirmed_blobs = {"green": None, "yellow": None}
    while True:
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

        # Aruco Section - Now using aruco_localizer objects
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, talk)

        # After estimating pose, collect marker world positions and convert to object poses
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"] and marker_id in marker_annotations:
                tvec, rvec = kalman_filters[marker_id].predict()
                rquat = rvec_to_quat(rvec)
                
                # Get object pose from marker pose using correct estimation
                marker_annotation = marker_annotations[marker_id]['annotation']
                object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), marker_annotation)
                object_quat = rvec_to_quat(object_rvec)
                
                # Convert object pose to world frame
                object_pos_world = transform_point_cam_to_world(object_tvec, cam_pos, cam_quat)
                object_quat_world = transform_orientation_cam_to_world(object_quat, cam_quat)
                
                model_name = marker_annotations[marker_id]['model_name']
                
                identified_jenga.append({
                                    "name": f"{model_name}_{marker_id}",
                                    "points": [object_pos_world],
                                    "position": object_pos_world,
                                    "quaternion": object_quat_world,
                                    'inferred': False,
                                })
                
                # The existing draw_object_lines function will handle visualization
                # No need for additional wireframe drawing here
                
                if talk:
                    print(f"[{model_name}_{marker_id}] Object WORLD Pose:")
                    print(f"  Pos: {object_pos_world}")
                    print(f"  Quat: {object_quat_world}")

        objects = identified_jenga + detected_objects

        # Blue Blob Section
        world_points, _ = detect_color_blobs(frame, blue_range, (255,0,0), CAMERA_MATRIX, cam_pos, cam_quat)
        identified_objects = identify_objects_from_blobs(world_points, OBJECT_DICTS)

        # Dynamic HSV Range Extraction and Pusher Detection
        pushers = {"green": None, "yellow": None}
        nearest_pushers = []
        
        if not args.no_pushers:
            # Extract dynamic HSV ranges from wireframe boundaries
            dynamic_green_range = None
            dynamic_yellow_range = None
            
            print(f"DEBUG: Processing {len(identified_jenga)} detected objects for HSV extraction")
            
            # Collect color ranges from all objects
            all_color_ranges = []
            
            # For each detected object with wireframe data, extract color ranges
            for obj in identified_jenga:
                obj_name = obj["name"]
                print(f"DEBUG: Processing object: {obj_name}")
                if "_" in obj_name:
                    # Extract model name by removing the marker ID (last part after the last underscore)
                    # e.g., "line_brown_scaled70_23" -> "line_brown_scaled70"
                    parts = obj_name.split("_")
                    model_name = "_".join(parts[:-1])  # Remove the last part (marker ID)
                    print(f"DEBUG: Model name: {model_name}")
                else:
                    print(f"DEBUG: Skipping object {obj_name} - no underscore found")
                    continue
                
                if model_name in model_data:
                    print(f"DEBUG: Found model data for {model_name}")
                    if model_data[model_name]['wireframe_vertices'] is not None:
                        print(f"DEBUG: Found wireframe data for {model_name}")
                    else:
                        print(f"DEBUG: No wireframe data for {model_name}")
                        continue
                else:
                    print(f"DEBUG: Model {model_name} not found in model_data")
                    continue
                
                # Get object pose in camera frame
                object_pos_world = obj["position"]
                object_quat_world = obj["quaternion"]
                
                # Transform to camera frame
                object_pos_cam = transform_point_world_to_cam(object_pos_world, cam_pos, cam_quat)
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
                projected_vertices = project_vertices_to_image(transformed_vertices, CAMERA_MATRIX, DIST_COEFFS)
                
                # Create wireframe mask
                wireframe_mask = create_wireframe_mask(projected_vertices, wireframe_edges, frame.shape)
                
                # Extract color range from within the wireframe
                color_range = extract_color_range_from_mask(frame, wireframe_mask)
                print(f"DEBUG: Color range extracted: {color_range is not None}")
                
                if color_range is not None:
                    print(f"DEBUG: Color range: {color_range}")
                    all_color_ranges.append(color_range)
                    print(f"Dynamic HSV range extracted for {obj_name}: {color_range}")
                else:
                    print(f"DEBUG: No color range extracted for {obj_name}")
                
                # Debug: Show wireframe mask (optional)
                if talk and wireframe_mask is not None:
                    # Create a colored overlay to show the wireframe mask
                    mask_overlay = np.zeros_like(frame)
                    mask_overlay[wireframe_mask > 0] = [0, 255, 0]  # Green overlay
                    frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)
            
            # Combine color ranges from all objects
            if all_color_ranges:
                # Calculate the union of all color ranges (expand the range to include all objects)
                all_lower = np.array([cr[0] for cr in all_color_ranges])
                all_upper = np.array([cr[1] for cr in all_color_ranges])
                
                # Take the minimum lower bounds and maximum upper bounds
                combined_lower = np.min(all_lower, axis=0)
                combined_upper = np.max(all_upper, axis=0)
                
                dynamic_green_range = (combined_lower, combined_upper)
                dynamic_yellow_range = (combined_lower, combined_upper)
                
                print(f"DEBUG: Combined color range from {len(all_color_ranges)} objects:")
                print(f"  Lower: {combined_lower}")
                print(f"  Upper: {combined_upper}")
            else:
                print("DEBUG: No valid color ranges extracted from any objects")
            
            # Use dynamic ranges if available, otherwise fall back to fixed ranges
            current_green_range = dynamic_green_range if dynamic_green_range is not None else green_range
            current_yellow_range = dynamic_yellow_range if dynamic_yellow_range is not None else yellow_range
            
            print(f"DEBUG: Using green range: {current_green_range}")
            print(f"DEBUG: Using yellow range: {current_yellow_range}")
            
            # Create combined mask for all detected objects
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Add each object's wireframe mask to the combined mask
            for obj in identified_jenga:
                obj_name = obj["name"]
                if "_" in obj_name:
                    parts = obj_name.split("_")
                    model_name = "_".join(parts[:-1])  # Remove the last part (marker ID)
                    
                    if model_name in model_data and model_data[model_name]['wireframe_vertices'] is not None:
                        # Get object pose
                        object_pos = obj["position"]
                        object_quat = obj["quaternion"]
                        
                        # Transform object pose to camera frame
                        object_rotation_matrix = R.from_quat(object_quat).as_matrix()
                        cam_rotation_matrix = R.from_quat(cam_quat).as_matrix()
                        object_quat_cam = R.from_matrix(cam_rotation_matrix.T @ object_rotation_matrix).as_quat()
                        
                        # Convert quaternion to rotation vector for the function
                        object_rvec_cam = R.from_quat(object_quat_cam).as_rotvec()
                        
                        # Transform mesh vertices to camera frame
                        vertices_cam = transform_mesh_to_camera_frame(
                            model_data[model_name]['wireframe_vertices'], 
                            (object_pos, object_rvec_cam)
                        )
                        
                        # Project vertices to image coordinates
                        projected_vertices = project_vertices_to_image(vertices_cam, CAMERA_MATRIX, DIST_COEFFS)
                        
                        # Create wireframe mask for this object
                        wireframe_mask = create_wireframe_mask(projected_vertices, model_data[model_name]['wireframe_edges'], frame.shape[:2])
                        
                        # Add to combined mask
                        combined_mask = cv2.bitwise_or(combined_mask, wireframe_mask)
            
            # Now detect pushers only within the combined wireframe mask
            # Check if we have valid color ranges before using them
            if current_green_range is not None and current_green_range[0] is not None and current_green_range[1] is not None:
                world_points_green, _ = detect_color_blobs_in_mask(frame, current_green_range, (0, 255, 0), CAMERA_MATRIX, cam_pos, cam_quat, combined_mask, min_area=150, merge_threshold=0)
            else:
                world_points_green = []
                print("DEBUG: Skipping green pusher detection - invalid color range")
                
            if current_yellow_range is not None and current_yellow_range[0] is not None and current_yellow_range[1] is not None:
                world_points_yellow, _ = detect_color_blobs_in_mask(frame, current_yellow_range, (0, 255, 255), CAMERA_MATRIX, cam_pos, cam_quat, combined_mask, min_area=150, merge_threshold=0)
            else:
                world_points_yellow = []
                print("DEBUG: Skipping yellow pusher detection - invalid color range")
            
            print(f"DEBUG: Found {len(world_points_green)} green pushers, {len(world_points_yellow)} yellow pushers within wireframe boundaries")

            if world_points_green:
                best_green = pick_closest_blob(world_points_green, last_pushers["green"])
                pushers["green"] = (best_green, (0, 255, 0))
                last_pushers["green"] = best_green

            if world_points_yellow:
                best_yellow = pick_closest_blob(world_points_yellow, last_pushers["yellow"])
                pushers["yellow"] = (best_yellow, (0, 255, 255))
                last_pushers["yellow"] = best_yellow
            

            # Working block for pusher-object interaction detection
            # For now, gets nearest contour point to each pusher
            all_xyz = []
            all_kappa = []
            all_meta = []  # to keep track of which object and index a point came from
            if objects:  # At least one pusher detected
                for obj_idx, obj in enumerate(objects):
                    # Skip objects that don't have contour data (like Jenga blocks)
                    if 'contour' not in obj or obj['contour'] is None:
                        continue
                    xyz = obj['contour']['xyz']
                    kappa = obj['contour']['kappa']
                    all_xyz.extend(xyz)
                    all_kappa.extend(kappa)
                    all_meta.extend([(obj_idx, i) for i in range(len(xyz))])

                if all_xyz:  # Only process if we have contour data
                    all_xyz = np.array(all_xyz)
                    all_kappa = np.array(all_kappa)

                    tree = cKDTree(all_xyz)

                    for color, pusher in pushers.items():
                        if pusher is not None:
                            pusher_pos, col = pusher
                            distance, contour_idx = tree.query(pusher_pos)
                            if distance > pusher_distance_max: # Must be within 30mm (accounts for differences in z)
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
            recovered_objects = attempt_recovery_for_missing_objects(detected_objects, world_points, known_triangles=OBJECT_DICTS)
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
            # Skip objects that don't have contour data for pusher recommendations
            if name in ["allen_key", "wrench", "jenga"] and 'contour' in obj and obj['contour'] is not None:
                if args.recommend_push:
                    posex = obj["position"][0]
                    posey = obj["position"][1]
                    objquat = obj["quaternion"]
                    objeuler = R.from_quat(objquat).as_euler('xyz')
                    oriy = objeuler[2]
                    prediction = predict_pusher_outputs(name, posex, posey, oriy, TARGET_POSES[name])
                    index = prediction['predicted_index']

                    # Draw predicted points (of the one or two given)
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
                        # duplicate single pusher
                        recommended.append(recommended[0])
                    
                    bridge_node.publish_recommended_contacts(recommended)

                # draw target
                target_contour = hard_define_contour(TARGET_POSES[name][0], TARGET_POSES[name][1], name)
                # Draw low-res Contour
                contour_xyz = target_contour["xyz"]
                contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, CAMERA_MATRIX)
                contour_img = np.array(contour_img)
                contour_img.reshape((-1, 1, 2))
                contour_img = contour_img[::20]
                cv2.polylines(frame,[contour_img],False,color)

        detected_objects = identified_objects.copy()
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects+identified_jenga)
        bridge_node.publish_contacts(nearest_pushers)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, nearest_pushers)

        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()