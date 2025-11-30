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

# Dynamic Color Range Configuration
# Add or remove color ranges here - the rest of the code will automatically adjust
COLOR_RANGES = {
    "blue": [np.array([100, 80, 80]), np.array([140, 255, 255])],
    "red": [np.array([170, 80, 80]), np.array([180, 255, 255])],
    "green": [np.array([35, 80, 100]), np.array([75, 255, 255])],
    "yellow": [np.array([15, 80, 60]), np.array([35, 255, 255])],
    # Add more colors here as needed:
    # "purple": [np.array([130, 80, 80]), np.array([160, 255, 255])],
    # "orange": [np.array([10, 80, 80]), np.array([25, 255, 255])],
    # "pink": [np.array([160, 80, 80]), np.array([180, 255, 255])],
}

# Color visualization settings (BGR format for OpenCV)
COLOR_VISUALIZATION = {
    "blue": (255, 0, 0),      # Blue in BGR
    "red": (0, 0, 255),       # Red in BGR  
    "green": (0, 255, 0),     # Green in BGR
    "yellow": (0, 255, 255),  # Yellow in BGR
    # Add corresponding visualization colors for new ranges
}


trackers = {}

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

        identified_objects = []

        detected_objects = identified_objects.copy()
        # Camera pose is published by external package, we only subscribe to it
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