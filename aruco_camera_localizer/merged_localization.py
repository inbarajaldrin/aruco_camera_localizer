import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.geometric_functions import (
    rvec_to_quat, quat_to_rvec, transform_orientation_cam_to_world,
    transform_orientation_world_to_cam, transform_point_cam_to_world,
    transform_points_world_to_img, transform_point_world_to_cam, slerp_quat
)
from aruco_camera_localizer.detection_functions import build_detectors, detect_markers, estimate_poses
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_grasp_points
from aruco_camera_localizer.filter_config import FilterConfig
from aruco_camera_localizer.robot_config import RobotConfig
from aruco_camera_localizer.data_path_finder import (
    find_aruco_data_dir, get_models_by_type, get_model_subtypes, load_symmetry_data
)
from aruco_camera_localizer.tuning_panel import (
    TuningPanel, RobotTuningPanel, handle_key, draw_help_overlay
)
import threading
import rclpy
import argparse

_models_by_type = get_models_by_type()
BOARD_MODELS = _models_by_type.get('board', set())
OBJECT_MODELS = _models_by_type.get('object', set())


def build_camera_intrinsics(robot_config):
    """Build camera matrix and distortion coefficients from robot config."""
    if robot_config.camera_matrix is not None:
        fx, fy, cx, cy = robot_config.camera_matrix
        camera_matrix = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]], dtype=np.float32)
    else:
        w, h = robot_config.camera_width, robot_config.camera_height
        fx = w / (2 * np.tan(np.deg2rad(robot_config.camera_hfov / 2)))
        fy = h / (2 * np.tan(np.deg2rad(robot_config.camera_vfov / 2)))
        camera_matrix = np.array([[fx, 0, w / 2],
                                   [0, fy, h / 2],
                                   [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(robot_config.distortion_coeffs, dtype=np.float32).reshape(-1, 1)
    return camera_matrix, dist_coeffs

# Marker dimensions
marker_size_mm = 21  # total marker size in mm (including white border)
border_width_percent = 5  # white border percentage
MARKER_SIZE = marker_size_mm / 1000.0  # Convert to meters
white_border_mm = marker_size_mm * (border_width_percent / 100.0)
BORDER_WIDTH = white_border_mm / 1000.0
TOTAL_MARKER_SIZE = MARKER_SIZE - 2 * BORDER_WIDTH  # Detectable black square size

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_50": aruco.DICT_5X5_50,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_aruco_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Dimensions: (length=X, width=Y, height=Z) in meters
    object_dims = None
    if 'cad_object_info' in data and 'dimensions' in data['cad_object_info']:
        d = data['cad_object_info']['dimensions']
        object_dims = (d.get('length'), d.get('width'), d.get('height'))
    return data['markers'], data.get('aruco_dictionary', 'DICT_4X4_50'), object_dims


def get_available_models(data_dir):
    aruco_dir = Path(data_dir) / "aruco"
    if not aruco_dir.exists():
        return []
    aruco_files = list(aruco_dir.glob("*_aruco.json"))
    available_models = {f.stem.replace("_aruco", "") for f in aruco_files}
    return sorted(list(available_models))


def load_wireframe_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['vertices'], data['edges']


def load_grasp_points_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['grasp_points']


# =============================================================================
# POSE ESTIMATION
# =============================================================================

def estimate_object_pose_from_marker(marker_pose, aruco_annotation):
    """
    Estimate 6D pose of object center from a single ArUco marker pose.

    Uses T_object_to_marker from the JSON annotation, inverts it, and chains
    with the detected marker pose: T_cam_to_object = T_cam_to_marker @ T_marker_to_object

    Args:
        marker_pose: Tuple of (marker_tvec, marker_rvec) in camera frame
        aruco_annotation: Dictionary with 'T_object_to_marker' field from JSON

    Returns:
        (object_tvec, object_rvec) in camera frame
    """
    marker_tvec, marker_rvec = marker_pose

    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    marker_tvec = marker_tvec.flatten()

    if 'T_object_to_marker' not in aruco_annotation:
        raise ValueError(
            f"Marker ID {aruco_annotation.get('aruco_id', '?')} missing 'T_object_to_marker'"
        )

    obj_to_marker_data = aruco_annotation['T_object_to_marker']

    t_obj_to_marker = np.array([
        obj_to_marker_data['position']['x'],
        obj_to_marker_data['position']['y'],
        obj_to_marker_data['position']['z']
    ])

    quat = obj_to_marker_data['rotation']['quaternion']
    quat_array = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
    R_obj_to_marker = R.from_quat(quat_array).as_matrix()

    # Invert: T_marker_to_object
    R_marker_to_obj = R_obj_to_marker.T
    t_marker_to_obj = -R_marker_to_obj @ t_obj_to_marker

    # Chain: T_cam_to_object = T_cam_to_marker @ T_marker_to_object
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = marker_rotation_matrix
    T_cam_marker[:3, 3] = marker_tvec

    T_marker_obj = np.eye(4)
    T_marker_obj[:3, :3] = R_marker_to_obj
    T_marker_obj[:3, 3] = t_marker_to_obj

    T_cam_obj = T_cam_marker @ T_marker_obj

    object_tvec = T_cam_obj[:3, 3]
    object_rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])

    return object_tvec, object_rvec


def estimate_board_pose_combined(board_corners, board_marker_keys, marker_annotations,
                                 camera_matrix, dist_coeffs, total_marker_size):
    """Estimate board/object pose from all visible markers via solvePnP.

    Uses solvePnPGeneric with IPPE to get both planar ambiguity solutions.
    Returns all valid solutions so the caller can pick the physically correct one.

    Returns:
        List of (object_tvec, object_rvec, reproj_error) tuples, or None if failed.
        Solutions are sorted by reprojection error (best first).
    """
    half = total_marker_size / 2.0
    local_corners = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    obj_pts = []
    img_pts = []

    for corners_2d, mkey in zip(board_corners, board_marker_keys):
        ann = marker_annotations[mkey]['annotation']
        t = np.array([
            ann['T_object_to_marker']['position']['x'],
            ann['T_object_to_marker']['position']['y'],
            ann['T_object_to_marker']['position']['z']
        ])
        q = ann['T_object_to_marker']['rotation']['quaternion']
        R_mat = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()

        for lc in local_corners:
            obj_pts.append(R_mat @ lc + t)
        for pt in corners_2d.reshape(-1, 2):
            img_pts.append(pt)

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    solutions = []
    try:
        num_solutions, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        if num_solutions == 0:
            return None
    except Exception:
        # IPPE may fail for non-planar configs; fall back to SQPNP
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
            if not success:
                return None
            rvecs, tvecs = [rvec], [tvec]
        except Exception:
            return None

    for rvec, tvec in zip(rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        rms = np.sqrt(np.mean(np.sum((img_pts - projected) ** 2, axis=1)))
        solutions.append((tvec.flatten(), rvec, rms))

    solutions.sort(key=lambda s: s[2])
    return solutions


def compute_tilt(quat_world):
    """Return the tilt angle (degrees) — how far the best-aligned local axis is from world Z."""
    R_obj = R.from_quat(quat_world).as_matrix()
    tilt_cos = np.max(np.abs(R_obj[2, :]))
    return np.degrees(np.arccos(np.clip(tilt_cos, 0, 1)))


def pick_best_solution(solutions_cam, cam_pos, cam_quat, prev_quat_world=None,
                       reproj_ratio_threshold=2.0):
    """Pick the IPPE solution with the smallest tilt (best table alignment).

    When the two IPPE solutions have similar reprojection errors (ratio < threshold),
    the pose is ambiguous (marker nearly fronto-parallel to camera). In that case,
    skip tilt-based disambiguation and just return the best-reproj solution, since
    tilt scoring is unreliable when both solutions are geometrically near-equivalent.

    When one solution is clearly worse (ratio >= threshold), tilt-based disambiguation
    is reliable and used to pick the best table-aligned pose.

    Args:
        solutions_cam: list of (tvec, rvec, reproj_error) in camera frame
        cam_pos, cam_quat: camera pose in world frame
        prev_quat_world: previous frame's world quaternion for this object (optional)
        reproj_ratio_threshold: min ratio of worst/best reproj error to consider
            disambiguation reliable (default 2.0)

    Returns:
        (tvec, rvec, reproj_error) — best solution in camera frame
    """
    if len(solutions_cam) == 1:
        return solutions_cam[0]

    # Sort by reproj error (best first)
    by_reproj = sorted(solutions_cam, key=lambda s: s[2])
    best_reproj = by_reproj[0][2]
    worst_reproj = by_reproj[-1][2]

    # If reproj errors are similar, the case is ambiguous — just use lowest reproj
    if best_reproj <= 0 or worst_reproj / best_reproj < reproj_ratio_threshold:
        return by_reproj[0]

    # Reproj errors are clearly different — tilt disambiguation is reliable
    scored = []
    for tvec, rvec, reproj in solutions_cam:
        quat_cam = rvec_to_quat(rvec)
        quat_world = transform_orientation_cam_to_world(quat_cam, cam_quat)
        tilt = compute_tilt(quat_world)
        scored.append((tilt, quat_world, tvec, rvec, reproj))

    scored.sort(key=lambda x: x[0])
    best_tilt = scored[0][0]

    # Find all solutions within 5° of best tilt
    tied = [s for s in scored if s[0] - best_tilt <= 5.0]

    if len(tied) > 1 and prev_quat_world is not None:
        # Break tie using temporal consistency (closest to previous orientation)
        def _quat_dist(s):
            return 1.0 - abs(np.dot(s[1], prev_quat_world))
        tied.sort(key=_quat_dist)

    winner = tied[0]
    return (winner[2], winner[3], winner[4])


def snap_orientation_to_cardinal(quat_world):
    """Snap constrained axes by aligning the free axis exactly with world Z.

    Finds which local object axis is most aligned with world Z (table normal),
    then applies the smallest rotation to make it point exactly along ±world Z.
    This preserves the yaw (free rotation around table normal) and avoids
    Euler-angle gimbal lock entirely.

    Returns:
        (snapped_quat, free_axis_idx) where free_axis_idx is 0=X, 1=Y, 2=Z
    """
    R_obj = R.from_quat(quat_world).as_matrix()

    # Which local axis is most aligned with world Z?
    free_axis_idx = np.argmax(np.abs(R_obj[2, :]))
    v = R_obj[:, free_axis_idx]
    target = np.array([0.0, 0.0, np.sign(v[2]) if abs(v[2]) > 1e-6 else 1.0])

    # Minimum rotation from v to target (preserves yaw)
    cross = np.cross(v, target)
    sin_a = np.linalg.norm(cross)
    cos_a = np.dot(v, target)

    if sin_a < 1e-10:
        if cos_a > 0:
            return quat_world, free_axis_idx  # already aligned
        perp = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, v) * v
        perp /= np.linalg.norm(perp)
        R_corr = R.from_rotvec(np.pi * perp).as_matrix()
    else:
        axis = cross / sin_a
        angle = np.arctan2(sin_a, cos_a)
        R_corr = R.from_rotvec(angle * axis).as_matrix()

    return R.from_matrix(R_corr @ R_obj).as_quat(), free_axis_idx


# =============================================================================
# ARUCO DETECTOR PARAMETERS
# =============================================================================

def create_detector_parameters(filter_config):
    """Create ArUco detector parameters from filter_config overrides.

    Starts with OpenCV defaults, then applies any detector params from config.
    """
    params = aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMax = filter_config.detector_adaptive_thresh_win_max
    params.minMarkerPerimeterRate = filter_config.detector_min_perim_rate
    params.maxMarkerPerimeterRate = filter_config.detector_max_perim_rate
    params.polygonalApproxAccuracyRate = filter_config.detector_poly_approx_rate
    params.cornerRefinementMethod = filter_config.detector_corner_refine_method

    return params


# =============================================================================
# ROS2 SETUP
# =============================================================================

def start_ros_node(image_topic=None, robot_config=None):
    rclpy.init()
    node = LocalizerBridge(image_topic, robot_config=robot_config)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node


def parse_args():
    parser = argparse.ArgumentParser(description="ArUco marker localizer.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--image-topic", type=str, default=None,
                        help="ROS2 image topic (e.g., '/camera/image_raw'). Enables sim mode.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Suppress console output of detected poses.")
    parser.add_argument("--headless", action='store_true',
                        help="No OpenCV window, but annotated stream is still published.")
    parser.add_argument("--filter-tune", action='store_true',
                        help="Show interactive tuning panel for real-time filter config adjustment.")
    parser.add_argument("--robot-tune", action='store_true',
                        help="Show interactive tuning panel for robot/camera config adjustment.")
    return parser.parse_args()


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    args = parse_args()
    headless_mode = args.headless
    robot_config = RobotConfig()
    filter_config = FilterConfig()
    CAMERA_MATRIX, DIST_COEFFS = build_camera_intrinsics(robot_config)
    c_width, c_height = robot_config.camera_width, robot_config.camera_height

    # Load data
    data_dir = find_aruco_data_dir()
    if data_dir is None:
        if not headless_mode:
            print("Could not find aruco-grasp-annotator data directory in Documents folder")
        return

    available_models = get_available_models(data_dir)
    if not available_models:
        if not headless_mode:
            print(f"No models found in data directory: {data_dir}")
        return

    if not headless_mode:
        print(f"Loaded {len(available_models)} models")

    model_data = {}
    marker_annotations = {}  # (marker_id, dict_name) -> {annotation, model_name}

    for model_name in available_models:
        aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
        wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
        grasp_file = data_dir / "grasp" / f"{model_name}_grasp_points_all_markers.json"

        try:
            aruco_annotations, aruco_dictionary, object_dims = load_aruco_annotations(aruco_annotations_file)

            wireframe_vertices, wireframe_edges = None, None
            if wireframe_file.exists():
                try:
                    wireframe_vertices, wireframe_edges = load_wireframe_data(wireframe_file)
                except Exception:
                    pass

            grasp_points = None
            if grasp_file.exists():
                try:
                    grasp_points = load_grasp_points_data(grasp_file)
                except Exception:
                    pass

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
                'grasp_points': grasp_points,
                'object_dims': object_dims
            }

        except Exception as e:
            if not headless_mode:
                print(f"Error loading model {model_name}: {e}")
            continue

    if not model_data:
        if not headless_mode:
            print("No model data loaded successfully")
        return

    # Load fold symmetry and subtype data for orientation snapping
    model_subtypes = get_model_subtypes(data_dir)
    symmetry_data = load_symmetry_data(data_dir)

    # State tracking
    frame_idx = 0
    last_object_poses = {}    # {model_name: (object_tvec, object_rvec, frame_idx)}
    prev_poses_world = {}     # {model_name: (pos_world, quat_world)} for EMA smoothing
    motion_pause_holdover = 0  # frames to stay paused after speed drops
    prev_marker_rvecs = {}    # {(marker_id, dict_name): rvec} for pose ambiguity resolution

    # Camera / input source setup
    use_ros_topic = args.image_topic is not None
    cap = None

    auto_settle = False
    auto_settle_exposure = False
    auto_settle_wb = False
    settle_frames_remaining = 0
    _last_exposure = robot_config.exposure
    _last_wb_temp = robot_config.white_balance_temp

    if use_ros_topic:
        bridge_node = start_ros_node(args.image_topic, robot_config=robot_config)
        if not headless_mode:
            print(f"Using ROS image topic: {args.image_topic}")
            print("Waiting for images from ROS topic...")
        while True:
            frame, frame_available = bridge_node.get_latest_frame()
            if frame_available:
                if not headless_mode:
                    print("Received first frame from ROS topic")
                break
            time.sleep(0.1)
    else:
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

        bridge_node = start_ros_node(None, robot_config=robot_config)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)

        auto_settle_exposure = robot_config.exposure_mode == 'auto'
        auto_settle_wb = robot_config.white_balance_mode == 'auto'
        auto_settle = auto_settle_exposure or auto_settle_wb
        if auto_settle:
            # Camera auto is already running by default.
            # Wait a few frames for values to stabilize, then lock.
            settle_frames_remaining = 30
            if not headless_mode:
                print("Will lock camera settings after 30 frames...")
        else:
            settle_frames_remaining = 0
        # Apply manual values immediately for non-auto modes
        if not auto_settle_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, robot_config.exposure)
        if not auto_settle_wb:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, robot_config.white_balance_temp)
        # Track last-applied values so we know when panel changes them
        _last_exposure = robot_config.exposure
        _last_wb_temp = robot_config.white_balance_temp

    talk = not args.suppress_prints and not headless_mode
    parameters = create_detector_parameters(filter_config)
    detectors = build_detectors(ARUCO_DICTS, parameters)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    paused = False
    last_frame = None

    tuning = (args.filter_tune or args.robot_tune) and not headless_mode

    tuning_panel = None
    robot_panel = None
    if not headless_mode:
        if tuning or args.robot_tune:
            print("Press 'q' to quit, 'p' pause, 's' save config, 'd' toggle rejected.")
            if args.filter_tune:
                tuning_panel = TuningPanel(filter_config, parameters)
            if args.robot_tune:
                robot_panel = RobotTuningPanel(robot_config)
        else:
            print("Press 'q' to quit. Ctrl+C to exit.")

    try:
     while True:
        # --- Capture frame ---
        if use_ros_topic:
            frame, frame_available = bridge_node.get_latest_frame()
            if not frame_available:
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # --- Auto-exposure/WB settle period ---
        if auto_settle and settle_frames_remaining > 0:
            settle_frames_remaining -= 1
            if not headless_mode:
                label = f"Auto-settling... {settle_frames_remaining}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Merged Detection", frame)
                cv2.waitKey(1)
            if settle_frames_remaining == 0:
                # Lock values and update config with settled values
                if auto_settle_exposure:
                    settled_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    cap.set(cv2.CAP_PROP_EXPOSURE, settled_exposure)
                    robot_config.exposure = settled_exposure
                    _last_exposure = settled_exposure
                    if not headless_mode:
                        print(f"  Locked exposure: {settled_exposure}")
                if auto_settle_wb:
                    settled_wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settled_wb)
                    robot_config.white_balance_temp = settled_wb
                    _last_wb_temp = settled_wb
                    if not headless_mode:
                        print(f"  Locked white balance: {settled_wb}")
                # Update tuning panel to show locked values
                if robot_panel and robot_panel.alive:
                    robot_panel.refresh_from_config()
                auto_settle = False
            continue

        # --- Check for auto-settle re-trigger from tuning panel ---
        if robot_panel and robot_panel.alive and robot_panel.auto_settle_requested:
            if cap is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2 aperture priority (auto)
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                auto_settle = True
                auto_settle_exposure = True
                auto_settle_wb = True
                settle_frames_remaining = 30
                if not headless_mode:
                    print("Re-triggering auto-settle...")
                continue

        # --- Sync tuning panel values ---
        if tuning_panel and tuning_panel.alive:
            tuning_panel.sync_to_config()
            detectors = build_detectors(ARUCO_DICTS, parameters)
        if robot_panel and robot_panel.alive:
            robot_panel.sync_to_config()
            # Apply camera changes live if values changed
            if cap is not None:
                if robot_config.exposure != _last_exposure:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 manual mode
                    cap.set(cv2.CAP_PROP_EXPOSURE, robot_config.exposure)
                    _last_exposure = robot_config.exposure
                if robot_config.white_balance_temp != _last_wb_temp:
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, robot_config.white_balance_temp)
                    _last_wb_temp = robot_config.white_balance_temp

        # --- Pause mode: reuse frozen frame instead of live capture ---
        if tuning and paused and isinstance(last_frame, np.ndarray):
            frame = last_frame.copy()
        elif isinstance(frame, np.ndarray):
            last_frame = frame.copy()

        bridge_node.publish_image(frame)
        frame_idx += 1

        # --- Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if filter_config.enable_clahe:
            gray = clahe.apply(gray)

        # --- Get camera pose ---
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

        # --- Check if robot is moving fast ---
        motion_paused = False
        if filter_config.enable_motion_pause and filter_config.motion_speed_threshold > 0:
            ee_speed = bridge_node.get_ee_speed()
            if ee_speed > filter_config.motion_speed_threshold:
                motion_paused = True
                motion_pause_holdover = filter_config.motion_pause_holdover
            elif motion_pause_holdover > 0:
                motion_paused = True
                motion_pause_holdover -= 1
                if motion_pause_holdover == 0:
                    prev_poses_world.clear()

        # --- Skip detection when robot is moving ---
        if motion_paused:
            bridge_node.publish_annotated_stream(frame)
            if not headless_mode:
                cv2.imshow("Merged Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # --- Detect markers ---
        corners, ids, dict_names = detect_markers(frame, gray, detectors)

        # Build raw corners map for board combined solvePnP
        detected_corners = {}
        for corner, mid, dname in zip(corners, ids, dict_names):
            detected_corners[(mid, dname)] = corner

        # --- Estimate per-marker poses (used for non-board objects) ---
        marker_poses, rejected_markers, all_raw_markers = estimate_poses(
            corners, ids, dict_names, CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE,
            z_range_min=filter_config.z_range_min,
            z_range_max=filter_config.z_range_max,
            max_reproj_error=filter_config.max_reproj_error,
            talk=talk,
            prev_rvecs=prev_marker_rvecs
        )

        # --- Board models: combined multi-marker solvePnP ---
        board_results = {}
        if filter_config.board_pose_mode == 'combined':
            for model_name in BOARD_MODELS:
                b_corners = []
                b_keys = []
                for mkey, corner in detected_corners.items():
                    if mkey in marker_annotations and marker_annotations[mkey]['model_name'] == model_name:
                        b_corners.append(corner)
                        b_keys.append(mkey)
                if not b_corners:
                    continue
                solutions = estimate_board_pose_combined(
                    b_corners, b_keys, marker_annotations,
                    CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE
                )
                if solutions is not None:
                    # Filter by z-range
                    valid = [(t, r, e) for t, r, e in solutions
                             if filter_config.z_range_min <= t[2] <= filter_config.z_range_max]
                    if valid:
                        if filter_config.enable_ippe_disambiguation and cam_pos is not None and cam_quat is not None:
                            prev_q = prev_poses_world[model_name][1] if model_name in prev_poses_world else None
                            best = pick_best_solution(valid, cam_pos, cam_quat, prev_q,
                                                    filter_config.ippe_reproj_ratio)
                        else:
                            best = valid[0]
                        board_results[model_name] = best

        # --- Non-board object pose ---
        combined_object_results = {}
        candidates = {}

        if filter_config.object_pose_mode == 'combined':
            # Combined multi-marker solvePnP for objects with 2+ visible markers
            for model_name in OBJECT_MODELS:
                if model_name in BOARD_MODELS:
                    continue
                o_corners = []
                o_keys = []
                for mkey, corner in detected_corners.items():
                    if mkey in marker_annotations and marker_annotations[mkey]['model_name'] == model_name:
                        o_corners.append(corner)
                        o_keys.append(mkey)
                if len(o_corners) >= 2:
                    solutions = estimate_board_pose_combined(
                        o_corners, o_keys, marker_annotations,
                        CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE
                    )
                    if solutions is not None:
                        valid = [(t, r, e) for t, r, e in solutions
                                 if filter_config.z_range_min <= t[2] <= filter_config.z_range_max
                                 and e <= filter_config.max_reproj_error]
                        if valid:
                            if filter_config.enable_ippe_disambiguation and cam_pos is not None and cam_quat is not None:
                                prev_q = prev_poses_world[model_name][1] if model_name in prev_poses_world else None
                                best = pick_best_solution(valid, cam_pos, cam_quat, prev_q,
                                                    filter_config.ippe_reproj_ratio)
                            else:
                                best = valid[0]
                            combined_object_results[model_name] = best

        # Single-marker fallback (or primary when mode='single')
        wireframe_candidates = {}  # {model_name: [(marker_key, tvec, rvec, reproj, z)]}
        for marker_key, pose_data in marker_poses.items():
            if marker_key not in marker_annotations:
                continue
            model_name = marker_annotations[marker_key]['model_name']
            if model_name in BOARD_MODELS and filter_config.board_pose_mode == 'combined':
                continue
            if model_name in board_results:
                continue
            annotation = marker_annotations[marker_key]['annotation']

            # Try all IPPE solutions (primary + alternates)
            all_marker_solutions = [pose_data] + pose_data.get('alt_solutions', [])
            for sol in all_marker_solutions:
                try:
                    object_tvec, object_rvec = estimate_object_pose_from_marker(
                        (sol['tvec'], sol['rvec']), annotation
                    )
                except ValueError:
                    continue
                entry = (marker_key, object_tvec, object_rvec, sol['reproj_error'],
                         sol['tvec'][2])
                # Always collect for wireframe side-marker selection
                wireframe_candidates.setdefault(model_name, []).append(entry)
                # Only add to pose candidates if not already solved via combined
                if model_name not in combined_object_results:
                    candidates.setdefault(model_name, []).append(entry)

        # --- Select object pose ---
        detected_objects = []
        objects_seen = set()

        # Add board + combined object results
        all_combined = {**board_results, **combined_object_results}
        for model_name, (object_tvec, object_rvec, _) in all_combined.items():
            objects_seen.add(model_name)
            last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)
            raw_tvec, raw_rvec = object_tvec.copy(), object_rvec.copy()

            if cam_pos is None or cam_quat is None:
                continue
            if np.any(np.isnan(cam_pos)) or np.any(np.isnan(cam_quat)):
                continue

            object_quat = rvec_to_quat(object_rvec)
            object_pos_world = transform_point_cam_to_world(object_tvec, cam_pos, cam_quat)
            object_quat_world = transform_orientation_cam_to_world(object_quat, cam_quat)

            # Save pre-snap world pose for wireframe (IPPE-disambiguated, no snap)
            presnap_pos_world = object_pos_world.copy()
            presnap_quat_world = object_quat_world.copy()

            pose_modified = False
            if model_name in BOARD_MODELS:
                if filter_config.board_yaw_only:
                    object_quat_world, free_axis_idx = snap_orientation_to_cardinal(
                        object_quat_world)
                    pose_modified = True
                if filter_config.board_snap_z:
                    obj_dims = model_data.get(model_name, {}).get('object_dims')
                    if obj_dims is not None and obj_dims[2] is not None:
                        object_pos_world[2] = robot_config.table_z + obj_dims[2] / 2.0
                        pose_modified = True

            # Tilt check for non-board objects
            if model_name not in BOARD_MODELS:
                R_check = R.from_quat(object_quat_world).as_matrix()
                tilt_cos = np.max(np.abs(R_check[2, :]))
                tilt_deg = np.degrees(np.arccos(np.clip(tilt_cos, 0, 1)))
                within_tilt = tilt_deg <= filter_config.snap_tilt_threshold

                # Fold symmetry snapping
                if filter_config.enable_fold_snap and within_tilt:
                    subtype = model_subtypes.get(model_name)
                    if subtype in filter_config.fold_snap_subtypes:
                        object_quat_world, free_axis_idx = snap_orientation_to_cardinal(
                            object_quat_world)
                        pose_modified = True

                # Object Z snap
                if filter_config.object_snap_z and within_tilt:
                    R_obj = R.from_quat(object_quat_world).as_matrix()
                    free_axis_idx = np.argmax(np.abs(R_obj[2, :]))
                    obj_dims = model_data.get(model_name, {}).get('object_dims')
                    if obj_dims is not None and obj_dims[free_axis_idx] is not None:
                        expected_z = robot_config.table_z + obj_dims[free_axis_idx] / 2.0
                        if abs(object_pos_world[2] - expected_z) <= filter_config.object_snap_z_tolerance:
                            object_pos_world[2] = expected_z
                            pose_modified = True

            if filter_config.enable_ema_smoothing and model_name in prev_poses_world:
                prev_pos, prev_quat = prev_poses_world[model_name]
                alpha = filter_config.ema_alpha
                object_pos_world = (1.0 - alpha) * prev_pos + alpha * object_pos_world
                object_quat_world = slerp_quat(prev_quat, object_quat_world, blend=alpha)
                pose_modified = True

            prev_poses_world[model_name] = (object_pos_world.copy(), object_quat_world.copy())

            # Reproject snapped world pose back to camera frame
            object_tvec = transform_point_world_to_cam(
                object_pos_world, cam_pos, cam_quat)
            object_rvec = quat_to_rvec(
                transform_orientation_world_to_cam(object_quat_world, cam_quat))

            # Reproject pre-snap world pose (IPPE-disambiguated, no snap/EMA)
            presnap_tvec = transform_point_world_to_cam(
                presnap_pos_world, cam_pos, cam_quat)
            presnap_rvec = quat_to_rvec(
                transform_orientation_world_to_cam(presnap_quat_world, cam_quat))

            detected_objects.append({
                "name": model_name, "points": [object_pos_world],
                "position": object_pos_world, "quaternion": object_quat_world,
                "inferred": False, "ghost_tracked": False, "no_display": False,
                "object_tvec": object_tvec, "object_rvec": object_rvec,
                "raw_tvec": raw_tvec, "raw_rvec": raw_rvec,
                "presnap_tvec": presnap_tvec, "presnap_rvec": presnap_rvec
            })

        # Single-marker results (fallback for objects, or primary for boards in single mode)
        for model_name, marker_list in candidates.items():
            objects_seen.add(model_name)

            # Pick the solution with best table alignment (lowest tilt) or lowest reproj error
            if filter_config.enable_ippe_disambiguation and cam_pos is not None and cam_quat is not None:
                solutions_for_pick = [(e[1], e[2], e[3]) for e in marker_list]
                prev_q = prev_poses_world[model_name][1] if model_name in prev_poses_world else None
                _, object_tvec, object_rvec, _, _ = marker_list[0]  # fallback
                best_sol = pick_best_solution(solutions_for_pick, cam_pos, cam_quat, prev_q,
                                                             filter_config.ippe_reproj_ratio)
                object_tvec, object_rvec = best_sol[0], best_sol[1]
            else:
                chosen = min(marker_list, key=lambda e: e[3])
                _, object_tvec, object_rvec, _, _ = chosen

            last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)
            raw_tvec, raw_rvec = object_tvec.copy(), object_rvec.copy()

            # --- Transform to world frame ---
            if cam_pos is None or cam_quat is None:
                continue
            if np.any(np.isnan(cam_pos)) or np.any(np.isnan(cam_quat)):
                continue

            object_quat = rvec_to_quat(object_rvec)
            object_pos_world = transform_point_cam_to_world(object_tvec, cam_pos, cam_quat)
            object_quat_world = transform_orientation_cam_to_world(object_quat, cam_quat)

            # Save pre-snap world pose for wireframe
            presnap_pos_world = object_pos_world.copy()
            presnap_quat_world = object_quat_world.copy()

            pose_modified = False

            if model_name in BOARD_MODELS:
                if filter_config.board_yaw_only:
                    object_quat_world, free_axis_idx = snap_orientation_to_cardinal(
                        object_quat_world)
                    pose_modified = True
                if filter_config.board_snap_z:
                    obj_dims = model_data.get(model_name, {}).get('object_dims')
                    if obj_dims is not None and obj_dims[2] is not None:
                        object_pos_world[2] = robot_config.table_z + obj_dims[2] / 2.0
                        pose_modified = True
            else:
                # Tilt check for non-board objects
                R_check = R.from_quat(object_quat_world).as_matrix()
                tilt_cos = np.max(np.abs(R_check[2, :]))
                tilt_deg = np.degrees(np.arccos(np.clip(tilt_cos, 0, 1)))
                within_tilt = tilt_deg <= filter_config.snap_tilt_threshold

                # Fold symmetry snapping
                if filter_config.enable_fold_snap and within_tilt:
                    subtype = model_subtypes.get(model_name)
                    if subtype in filter_config.fold_snap_subtypes:
                        object_quat_world, free_axis_idx = snap_orientation_to_cardinal(
                            object_quat_world)
                        pose_modified = True

                # Object Z snap
                if filter_config.object_snap_z and within_tilt:
                    R_obj = R.from_quat(object_quat_world).as_matrix()
                    free_axis_idx = np.argmax(np.abs(R_obj[2, :]))
                    obj_dims = model_data.get(model_name, {}).get('object_dims')
                    if obj_dims is not None and obj_dims[free_axis_idx] is not None:
                        expected_z = robot_config.table_z + obj_dims[free_axis_idx] / 2.0
                        if abs(object_pos_world[2] - expected_z) <= filter_config.object_snap_z_tolerance:
                            object_pos_world[2] = expected_z
                        pose_modified = True

            # --- Optional EMA smoothing ---
            if filter_config.enable_ema_smoothing and model_name in prev_poses_world:
                prev_pos, prev_quat = prev_poses_world[model_name]
                alpha = filter_config.ema_alpha
                object_pos_world = (1.0 - alpha) * prev_pos + alpha * object_pos_world
                object_quat_world = slerp_quat(prev_quat, object_quat_world, blend=alpha)
                pose_modified = True

            prev_poses_world[model_name] = (object_pos_world.copy(), object_quat_world.copy())

            # Reproject snapped world pose back to camera frame
            object_tvec = transform_point_world_to_cam(
                object_pos_world, cam_pos, cam_quat)
            object_rvec = quat_to_rvec(
                transform_orientation_world_to_cam(object_quat_world, cam_quat))

            # Reproject pre-snap world pose (IPPE-disambiguated, no snap/EMA)
            presnap_tvec = transform_point_world_to_cam(
                presnap_pos_world, cam_pos, cam_quat)
            presnap_rvec = quat_to_rvec(
                transform_orientation_world_to_cam(presnap_quat_world, cam_quat))

            detected_objects.append({
                "name": model_name,
                "points": [object_pos_world],
                "position": object_pos_world,
                "quaternion": object_quat_world,
                "inferred": False,
                "ghost_tracked": False,
                "no_display": False,
                "object_tvec": object_tvec,
                "object_rvec": object_rvec,
                "raw_tvec": raw_tvec,
                "raw_rvec": raw_rvec,
                "presnap_tvec": presnap_tvec,
                "presnap_rvec": presnap_rvec
            })

        # --- Ghost tracking & active marker cleanup ---
        for model_name in list(last_object_poses.keys()):
            if model_name not in objects_seen:
                object_tvec, object_rvec, last_frame = last_object_poses[model_name]
                age = frame_idx - last_frame

                if not filter_config.enable_active_marker_tracking or age > filter_config.active_marker_timeout:
                    del last_object_poses[model_name]
                    prev_poses_world.pop(model_name, None)
                elif filter_config.enable_ghost_tracking and age <= filter_config.ghost_timeout and model_name in prev_poses_world:
                    pos_world, quat_world = prev_poses_world[model_name]
                    detected_objects.append({
                        "name": model_name,
                        "points": [pos_world],
                        "position": pos_world,
                        "quaternion": quat_world,
                        "inferred": False,
                        "ghost_tracked": True,
                        "no_display": False,
                        "object_tvec": object_tvec,
                        "object_rvec": object_rvec,
                        "raw_tvec": object_tvec,
                        "raw_rvec": object_rvec
                    })

        # --- Pick best side-marker pose per object for wireframe ---
        side_marker_poses = {}  # {model_name: (tvec, rvec)}
        if filter_config.wireframe_prefer_side_markers and cam_pos is not None and cam_quat is not None:
            for model_name, wf_list in wireframe_candidates.items():
                if not wf_list:
                    continue
                # Group IPPE solutions by marker key, pick best per marker
                per_marker = {}
                for entry in wf_list:
                    mkey = entry[0]
                    per_marker.setdefault(mkey, []).append(entry)

                # Disambiguate: for each marker, pick IPPE solution closest to
                # the published object pose (if available)
                pub_quat = None
                for obj in detected_objects:
                    if obj["name"] == model_name and "quaternion" in obj:
                        pub_quat = obj["quaternion"]
                        break

                best_per_marker = []
                for mkey, solutions in per_marker.items():
                    if pub_quat is not None and len(solutions) > 1:
                        # Pick solution whose world orientation is closest to published
                        def _quat_dist(entry):
                            _, _, rvec, _, _ = entry
                            qw = transform_orientation_cam_to_world(
                                rvec_to_quat(rvec), cam_quat)
                            return 1.0 - abs(np.dot(qw, pub_quat))
                        best_per_marker.append(min(solutions, key=_quat_dist))
                    else:
                        # Single solution or no published pose — pick lowest reproj
                        best_per_marker.append(min(solutions, key=lambda e: e[3]))

                # Among disambiguated markers, pick most side-facing
                def _side_score(entry):
                    _, tvec, rvec, _, _ = entry
                    quat_w = transform_orientation_cam_to_world(
                        rvec_to_quat(rvec), cam_quat)
                    return compute_tilt(quat_w)
                best = max(best_per_marker, key=_side_score)
                side_marker_poses[model_name] = (best[1], best[2])

        # --- Wireframe visualization (direct camera-frame projection) ---
        for obj in ([] if motion_paused else detected_objects):
            model_name = obj["name"]
            if model_name not in model_data or model_data[model_name]['wireframe_vertices'] is None:
                continue

            wireframe_vertices = model_data[model_name]['wireframe_vertices']
            wireframe_edges = model_data[model_name]['wireframe_edges']
            pts_3d = np.array(wireframe_vertices, dtype=np.float32)

            # Choose wireframe pose source
            if filter_config.wireframe_prefer_side_markers and model_name in side_marker_poses:
                # Use side marker orientation but snapped translation for stable size
                _, side_rvec = side_marker_poses[model_name]
                obj_tvec = obj["object_tvec"]  # snapped + reprojected
                obj_rvec = side_rvec
                if obj_tvec[2] <= 0.01 or obj_tvec[2] > 2.0:
                    continue
                projected, _ = cv2.projectPoints(pts_3d, obj_rvec, obj_tvec, CAMERA_MATRIX, DIST_COEFFS)
                projected = projected.reshape(-1, 2).astype(int)
            elif (model_name in BOARD_MODELS and filter_config.wireframe_raw_pose_boards) or \
                 (model_name not in BOARD_MODELS and filter_config.wireframe_raw_pose_objects):
                # Use raw camera-frame pose (post-IPPE, pre-world-transform)
                obj_tvec = obj.get("raw_tvec", obj["object_tvec"])
                obj_rvec = obj.get("raw_rvec", obj["object_rvec"])
                if obj_tvec[2] <= 0.01 or obj_tvec[2] > 2.0:
                    continue
                projected, _ = cv2.projectPoints(pts_3d, obj_rvec, obj_tvec, CAMERA_MATRIX, DIST_COEFFS)
                projected = projected.reshape(-1, 2).astype(int)
            else:
                # Use world-frame pose (same projection path as RGB axes)
                obj_pos = obj["position"]
                obj_rot = R.from_quat(obj["quaternion"]).as_matrix()
                world_pts = [obj_pos + obj_rot @ v for v in pts_3d]
                img_pts = transform_points_world_to_img(world_pts, cam_pos, cam_quat, CAMERA_MATRIX)
                if len(img_pts) != len(pts_3d):
                    continue
                projected = np.array(img_pts, dtype=int)

            for edge in wireframe_edges:
                if len(edge) >= 2:
                    si, ei = edge[0], edge[1]
                    if si < len(projected) and ei < len(projected):
                        p1 = tuple(projected[si])
                        p2 = tuple(projected[ei])
                        if all(-2000 < c < 4000 for c in p1 + p2):
                            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Update rejection log on tuning panel
        if tuning_panel is not None and tuning_panel.alive:
            tuning_panel.update_rejection_log(marker_poses, rejected_markers, all_raw_markers,
                                              detected_objects)

        # --- Debug: draw all raw unfiltered detections ---
        if filter_config.debug_show_rejected and not motion_paused:
            for (marker_id, dict_name), raw_data in all_raw_markers.items():
                raw_tvec = raw_data['tvec']
                raw_rvec = raw_data['rvec']
                if raw_tvec[2] <= 0.01 or raw_tvec[2] > 5.0:
                    continue

                # Draw raw unfiltered wireframe in red for all markers
                if (marker_id, dict_name) in marker_annotations:
                    m_model = marker_annotations[(marker_id, dict_name)]['model_name']
                    m_annot = marker_annotations[(marker_id, dict_name)]['annotation']
                    if m_model in model_data and model_data[m_model]['wireframe_vertices'] is not None:
                        try:
                            obj_tvec_rej, obj_rvec_rej = estimate_object_pose_from_marker(
                                (raw_tvec, raw_rvec), m_annot)
                            wf_verts = model_data[m_model]['wireframe_vertices']
                            wf_edges = model_data[m_model]['wireframe_edges']
                            pts = np.array(wf_verts, dtype=np.float32)
                            proj, _ = cv2.projectPoints(pts, obj_rvec_rej, obj_tvec_rej, CAMERA_MATRIX, DIST_COEFFS)
                            proj = proj.reshape(-1, 2).astype(int)
                            for edge in wf_edges:
                                if len(edge) >= 2:
                                    si, ei = edge[0], edge[1]
                                    if si < len(proj) and ei < len(proj):
                                        p1, p2 = tuple(proj[si]), tuple(proj[ei])
                                        if all(-2000 < c < 4000 for c in p1 + p2):
                                            cv2.line(frame, p1, p2, (0, 0, 255), 1)
                        except (ValueError, Exception):
                            pass

        # --- Publish and draw ---
        if cam_pos is not None and cam_quat is not None:
            bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(detected_objects)
        if not motion_paused:
            draw_text(frame, cam_pos, cam_quat, detected_objects, frame_idx, ee_pos, ee_quat, euler_convention=filter_config.euler_convention)
            draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, detected_objects, [])
            draw_grasp_points(frame, CAMERA_MATRIX, cam_pos, cam_quat, detected_objects, model_data)

        bridge_node.publish_annotated_stream(frame)

        if not headless_mode:
            if tuning:
                draw_help_overlay(frame, paused)
            cv2.imshow("Merged Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if tuning:
                should_quit, paused = handle_key(key, filter_config, paused, tuning_panel, robot_panel)
                if should_quit:
                    break
            else:
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    # Cleanup
    if cap is not None:
        cap.release()
    if not headless_mode:
        cv2.destroyAllWindows()

    try:
        bridge_node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
