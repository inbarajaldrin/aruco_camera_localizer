import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
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
    
    # Apply coordinate transformation only (scaling handled in wireframe transformation)
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

def fuse_object_poses(object_poses, weights=None):
    """
    Fuse multiple object poses from different markers into a single stable pose.
    """
    if not object_poses:
        return None, None
    
    if len(object_poses) == 1:
        return object_poses[0]
    
    if weights is None:
        weights = [1.0] * len(object_poses)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
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
    return parser.parse_args()



def main():
    args = parse_args()
    bridge_node = start_ros_node(args.image_topic)

    # Load aruco_localizer data
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    
    if not data_dir.exists():
        # Fallback to absolute path
        data_dir = Path("/home/aaugus11/Desktop/ros2_ws/src/aruco_camera_localizer/data")
    
    if not data_dir.exists():
        print(f"Could not find data directory at {data_dir}")
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
        grasp_file = data_dir / "grasp" / f"{model_name}_grasp_points_all_markers.json"
        
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
            
            # Load grasp points data if available
            grasp_points = None
            if grasp_file.exists():
                try:
                    grasp_points = load_grasp_points_data(grasp_file)
                    print(f"Loaded grasp points for {model_name}: {len(grasp_points)} points")
                except Exception as e:
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

    # Determine input source
    use_ros_topic = args.image_topic is not None
    cap = None
    
    if use_ros_topic:
        print(f"Using ROS image topic: {args.image_topic}")
        print("Waiting for images from ROS topic...")
        # Wait for first frame
        import time
        while True:
            frame, frame_available = bridge_node.get_latest_frame()
            if frame_available:
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

    talk = not args.suppress_prints
    parameters = aruco.DetectorParameters()
    print("Press 'q' to quit.")

    detected_objects = []
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

        # Aruco Section - Now using aruco_localizer objects
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, talk)

        # After estimating pose, collect ALL confirmed markers and fuse them by object
        object_detections = {}  # {model_name: [(object_tvec, object_rvec, distance, marker_id)]}
        
        # Collect all confirmed markers and convert to object poses
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"] and marker_id in marker_annotations:
                # Toggle between raw measurements (default) and Kalman predictions
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
                
                # Group by object type
                if model_name not in object_detections:
                    object_detections[model_name] = []
                object_detections[model_name].append((object_tvec, object_rvec, distance, marker_id))
        
        # Fuse poses for each object using weighted averaging
        for model_name, detections in object_detections.items():
            if not detections:
                continue
                
            # Extract poses and calculate weights (closer markers get higher weight)
            object_poses = [(tvec, rvec) for tvec, rvec, _, _ in detections]
            distances = [dist for _, _, dist, _ in detections]
            marker_ids = [mid for _, _, _, mid in detections]
            
            # Calculate weights (inverse distance - closer markers get higher weight)
            weights = [1.0 / (dist + 0.1) for dist in distances]  # Add small epsilon to avoid division by zero
            
            # Fuse poses
            fused_tvec, fused_rvec = fuse_object_poses(object_poses, weights)
            
            # Check if fused pose is valid
            if (fused_tvec is not None and fused_rvec is not None and 
                not np.any(np.isnan(fused_tvec)) and not np.any(np.isnan(fused_rvec))):
                # Use fused pose directly (simplified approach without object-level Kalman)
                smoothed_tvec, smoothed_rvec = fused_tvec, fused_rvec
                smoothed_quat = rvec_to_quat(smoothed_rvec)
                
                # Convert to world frame
                object_pos_world = transform_point_cam_to_world(smoothed_tvec, cam_pos, cam_quat)
                object_quat_world = transform_orientation_cam_to_world(smoothed_quat, cam_quat)
                
                # Create final object
                final_object = {
                    "name": model_name,
                    "points": [object_pos_world],
                    "position": object_pos_world,
                    "quaternion": object_quat_world,
                    'inferred': False,
                    "object_tvec": smoothed_tvec,
                    "object_rvec": smoothed_rvec
                }
                identified_jenga.append(final_object)
                
                if talk:
                    avg_distance = np.mean(distances)
                    print(f"[{model_name}] Fused from {len(detections)} markers - Avg distance: {avg_distance:.3f}m")
                    print(f"  Pos: {object_pos_world}")
                    print(f"  Quat: {object_quat_world}")
                    print(f"  Markers: {marker_ids}")

        objects = identified_jenga + detected_objects

        # Wireframe Mask Visualization for ArUco Objects (only for best detections)
        for obj in identified_jenga:
            model_name = obj["name"]  # Now the name is just the model name
            
            if model_name in model_data and model_data[model_name]['wireframe_vertices'] is not None:
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
                
                # Debug: Show pose values for line_red_scaled70
                if model_name == "line_red_scaled70" and talk:
                    print(f"DEBUG {model_name}: Object pose in camera frame:")
                    print(f"  Position: {object_pos_cam}")
                    print(f"  Rotation: {object_rvec.flatten()}")
                
                transformed_vertices = transform_mesh_to_camera_frame(wireframe_vertices, (object_pos_cam, object_rvec))
                projected_vertices = project_vertices_to_image(transformed_vertices, CAMERA_MATRIX, DIST_COEFFS)
                
                # Debug: Show projected vertices for line_red_scaled70
                if model_name == "line_red_scaled70" and talk:
                    print(f"  Projected vertices range: X=[{projected_vertices[:, 0].min():.1f}, {projected_vertices[:, 0].max():.1f}], Y=[{projected_vertices[:, 1].min():.1f}, {projected_vertices[:, 1].max():.1f}]")
                
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
        bridge_node.publish_grasp_points(identified_objects+identified_jenga, model_data)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, [])
        draw_grasp_points(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, model_data)

        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()