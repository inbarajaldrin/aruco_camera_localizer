import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.camera_selection import detect_available_cameras, select_camera
from max_camera_localizer.localizer_bridge import LocalizerBridge
from max_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from max_camera_localizer.detection_functions import detect_markers, detect_color_blobs, estimate_pose, \
    identify_objects_from_blobs, attempt_recovery_for_missing_objects
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
    return parser.parse_args()



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

        # Wireframe Mask Visualization for ArUco Objects
        for obj in identified_jenga:
            obj_name = obj["name"]
            if "_" in obj_name:
                # Extract model name by removing the marker ID (last part after the last underscore)
                # e.g., "line_brown_scaled70_23" -> "line_brown_scaled70"
                parts = obj_name.split("_")
                model_name = "_".join(parts[:-1])  # Remove the last part (marker ID)
            else:
                continue
            
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
                
                transformed_vertices = transform_mesh_to_camera_frame(wireframe_vertices, (object_pos_cam, object_rvec))
                projected_vertices = project_vertices_to_image(transformed_vertices, CAMERA_MATRIX, DIST_COEFFS)
                
                # Create wireframe mask
                wireframe_mask = create_wireframe_mask(projected_vertices, wireframe_edges, frame.shape)
                
                # Show wireframe mask overlay
                if wireframe_mask is not None:
                    # Create a colored overlay to show the wireframe mask
                    mask_overlay = np.zeros_like(frame)
                    mask_overlay[wireframe_mask > 0] = [0, 255, 0]  # Green overlay
                    frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)

        # Blue Blob Section
        world_points, _ = detect_color_blobs(frame, blue_range, (255,0,0), CAMERA_MATRIX, cam_pos, cam_quat)
        identified_objects = identify_objects_from_blobs(world_points, OBJECT_DICTS)


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


        detected_objects = identified_objects.copy()
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(identified_objects+identified_jenga)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_jenga, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_jenga, [])

        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()