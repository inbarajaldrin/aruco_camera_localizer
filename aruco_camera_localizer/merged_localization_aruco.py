import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.config_loader import get_config
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
from aruco_camera_localizer.detection_functions import detect_markers, estimate_pose
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines
import threading
import rclpy
import argparse
import json
import os
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

fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

CAMERA_MATRIX = config.get_camera_matrix()
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
MARKER_SIZE = 0.032  # meters
BLOCK_LENGTH = 0.072 # meters
BLOCK_WIDTH = 0.024 # meters
BLOCK_THICKNESS = 0.014 # meters
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    # "DICT_5X5_250": aruco.DICT_5X5_250
}

trackers = {}

def start_ros_node(camera_topic='/camera/image_raw'):
    rclpy.init()
    node = LocalizerBridge(camera_topic=camera_topic, depth_topic=None)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with optional camera ID.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--camera-topic", type=str, default=None,
                        help="ROS2 topic to subscribe for camera images (e.g., /camera/image_raw). If provided, uses ROS topic instead of camera ID.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--drop", action='store_true',
                        help="Publish drop poses instead of ArUco poses. Drop poses are calculated from position offsets in aruco_config.json transformed to world frame.")
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
    args = parse_args()
    
    # Load aruco config if --drop mode is enabled
    marker_to_row = None
    if args.drop:
        marker_to_row = load_aruco_config()
        print(f"Loaded ArUco config for drop mode. Found {len(marker_to_row)} marker mappings.")
    
    # Determine if using ROS topic or camera ID
    use_ros_topic = args.camera_topic is not None
    camera_topic = args.camera_topic if args.camera_topic else '/camera/image_raw'
    
    # Start ROS node (always needed for pose subscriptions)
    # If using direct camera capture, still pass a topic (won't be used, but LocalizerBridge needs it)
    bridge_node = start_ros_node(camera_topic=camera_topic)

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    prev_rvecs = {}
    frame_idx = 0

    # Load filter tuning from robot_config.yaml
    z_range_min, z_range_max = config.get_z_range()
    max_movement, hold_required, ghost_timeout = config.get_stability_params()
    q_params = config.get_kalman_process_noise()
    r_params = config.get_kalman_measurement_noise()
    blend_factor = config.get_blend_factor()

    talk = not args.suppress_prints

    # Set up camera source (either ROS topic or direct camera capture)
    cap = None
    parameters = aruco.DetectorParameters()
    
    if not use_ros_topic:
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
    else:
        print(f"Using ROS topic: {camera_topic}")
        print("Waiting for camera frames...")
    
    print("Press 'q' to quit.")

    detected_objects = []
    while True:
        # Get frame from either ROS topic or camera capture
        if use_ros_topic:
            frame = bridge_node.get_latest_frame()
            if frame is None:
                import time
                time.sleep(0.01)  # 10ms
                continue
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        identified_aruco = []
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

        # Aruco Section
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

        objects = identified_aruco + detected_objects

        identified_objects = []

        detected_objects = identified_objects.copy()
        # Camera pose is published by external package, we only subscribe to it
        
        # If --drop mode, calculate and publish drop poses instead of aruco poses
        if args.drop and marker_to_row is not None:
            drop_poses = []
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
            # Also publish ArUco poses when in drop mode
            bridge_node.publish_aruco_poses(identified_objects+identified_aruco)
        else:
            # Normal mode: publish ArUco poses
            bridge_node.publish_object_poses(identified_objects+identified_aruco)
            bridge_node.publish_aruco_poses(identified_objects+identified_aruco)
        draw_text(frame, cam_pos, cam_quat, identified_objects+identified_aruco, frame_idx, ee_pos, ee_quat)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects+identified_aruco, [])

        # Publish the annotated frame
        bridge_node.publish_annotated_image(frame)
        
        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()