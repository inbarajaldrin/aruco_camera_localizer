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
from aruco_camera_localizer.detection_functions import detect_markers, estimate_poses
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines, draw_grasp_points
from aruco_camera_localizer.filter_config import FilterConfig
from aruco_camera_localizer.data_path_finder import (
    find_aruco_data_dir, get_models_by_type, get_model_subtypes, load_symmetry_data
)
import threading
import rclpy
import argparse

# Camera parameters (real camera)
c_width = 1280  # pixels
c_hfov = 69.4   # degrees
fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))

c_height = 720   # pixels
c_vfov = 42.5    # degrees
fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))

# Table height in robot base frame (meters)
TABLE_Z = -0.11
_models_by_type = get_models_by_type()
BOARD_MODELS = _models_by_type.get('board', set())
OBJECT_MODELS = _models_by_type.get('object', set())

# Sim mode camera parameters
fx_sim = 731.78
fy_sim = 731.78


def create_camera_matrix(use_sim_mode=False):
    if use_sim_mode:
        return np.array([[fx_sim, 0, c_width / 2],
                         [0, fy_sim, c_height / 2],
                         [0, 0, 1]], dtype=np.float32)
    else:
        return np.array([[fx, 0, c_width / 2],
                         [0, fy, c_height / 2],
                         [0, 0, 1]], dtype=np.float32)


DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)  # datasheet says <= 1.5%

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
    object_height = None
    if 'cad_object_info' in data and 'dimensions' in data['cad_object_info']:
        object_height = data['cad_object_info']['dimensions'].get('height')
    return data['markers'], data.get('aruco_dictionary', 'DICT_4X4_50'), object_height


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
                                 camera_matrix, dist_coeffs, marker_size):
    """Estimate board pose from all visible markers via a single solvePnP call.

    Instead of solving each marker independently (which suffers from IPPE
    ambiguity for coplanar top-face markers), this combines all detected
    marker corners into one solve with their known 3D positions in the
    object frame. More points spread over a larger area = fully constrained.

    Returns:
        (object_tvec, object_rvec, reproj_error) or None if failed
    """
    half = marker_size / 2.0
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

    try:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            return None
    except Exception:
        return None

    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    rms = np.sqrt(np.mean(np.sum((img_pts - projected) ** 2, axis=1)))

    return tvec.flatten(), rvec, rms


def snap_orientation_to_cardinal(quat_world, snap_angle_deg=90.0, fold_counts=None):
    """Snap constrained axes by aligning the free axis exactly with world Z.

    Finds which local object axis is most aligned with world Z (table normal),
    then applies the smallest rotation to make it point exactly along ±world Z.
    This preserves the yaw (free rotation around table normal) and avoids
    Euler-angle gimbal lock entirely.

    Args:
        quat_world: World-frame quaternion [x,y,z,w]
        snap_angle_deg: (unused in matrix approach, kept for API compat)
        fold_counts: (unused in matrix approach, kept for API compat)
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
            return quat_world  # already aligned
        perp = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, v) * v
        perp /= np.linalg.norm(perp)
        R_corr = R.from_rotvec(np.pi * perp).as_matrix()
    else:
        axis = cross / sin_a
        angle = np.arctan2(sin_a, cos_a)
        R_corr = R.from_rotvec(angle * axis).as_matrix()

    return R.from_matrix(R_corr @ R_obj).as_quat()


# =============================================================================
# ARUCO DETECTOR PARAMETERS
# =============================================================================

def create_detector_parameters():
    """Create tuned ArUco detector parameters for small (21mm) markers."""
    params = aruco.DetectorParameters()

    # Adaptive thresholding for small markers
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7

    # Marker perimeter constraints
    # At 0.6m with fx~928: marker ~32px -> perimeter ~128px
    # Image perimeter = 2*(1280+720) = 4000px
    params.minMarkerPerimeterRate = 0.008   # ~32px / 4000px
    params.maxMarkerPerimeterRate = 0.4     # allows close-range detection

    # Tighter corner accuracy (default is 0.05)
    params.polygonalApproxAccuracyRate = 0.03

    # Sub-pixel corner refinement (critical for small marker pose accuracy)
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1

    return params


# =============================================================================
# ROS2 SETUP
# =============================================================================

def start_ros_node(image_topic=None):
    rclpy.init()
    node = LocalizerBridge(image_topic)
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
    return parser.parse_args()


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    args = parse_args()
    headless_mode = args.headless
    filter_config = FilterConfig()

    # Camera matrix
    use_sim_mode = args.image_topic is not None
    CAMERA_MATRIX = create_camera_matrix(use_sim_mode)
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
            aruco_annotations, aruco_dictionary, object_height = load_aruco_annotations(aruco_annotations_file)

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
                'object_height': object_height
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
    board_active_markers = {} # {model_name: marker_key} sticky marker for boards
    prev_marker_rvecs = {}    # {(marker_id, dict_name): rvec} for pose ambiguity resolution

    # Camera / input source setup
    use_ros_topic = args.image_topic is not None
    cap = None

    if use_ros_topic:
        bridge_node = start_ros_node(args.image_topic)
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

        bridge_node = start_ros_node(None)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)

    talk = not args.suppress_prints and not headless_mode
    parameters = create_detector_parameters()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if not headless_mode:
        print("Press 'q' to quit.")

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

        bridge_node.publish_image(frame)
        frame_idx += 1

        # --- Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # --- Get camera pose ---
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()

        # --- Detect markers ---
        corners, ids, dict_names = detect_markers(frame, gray, ARUCO_DICTS, parameters)

        # Build raw corners map for board combined solvePnP
        detected_corners = {}
        for corner, mid, dname in zip(corners, ids, dict_names):
            detected_corners[(mid, dname)] = corner

        # --- Estimate per-marker poses (used for non-board objects) ---
        marker_poses = estimate_poses(
            corners, ids, dict_names, CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE,
            z_range_min=filter_config.z_range_min,
            z_range_max=filter_config.z_range_max,
            talk=talk,
            prev_rvecs=prev_marker_rvecs
        )

        # --- Board models: combined multi-marker solvePnP ---
        board_results = {}
        for model_name in BOARD_MODELS:
            b_corners = []
            b_keys = []
            for mkey, corner in detected_corners.items():
                if mkey in marker_annotations and marker_annotations[mkey]['model_name'] == model_name:
                    b_corners.append(corner)
                    b_keys.append(mkey)
            if not b_corners:
                continue
            result = estimate_board_pose_combined(
                b_corners, b_keys, marker_annotations,
                CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE
            )
            if result is not None:
                obj_tvec, obj_rvec, reproj = result
                if filter_config.z_range_min <= obj_tvec[2] <= filter_config.z_range_max:
                    board_results[model_name] = (obj_tvec, obj_rvec, reproj)

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
                    result = estimate_board_pose_combined(
                        o_corners, o_keys, marker_annotations,
                        CAMERA_MATRIX, DIST_COEFFS, TOTAL_MARKER_SIZE
                    )
                    if result is not None:
                        obj_tvec, obj_rvec, reproj = result
                        if filter_config.z_range_min <= obj_tvec[2] <= filter_config.z_range_max:
                            combined_object_results[model_name] = (obj_tvec, obj_rvec, reproj)

        # Single-marker fallback (or primary when mode='single')
        for marker_key, pose_data in marker_poses.items():
            if marker_key not in marker_annotations:
                continue
            model_name = marker_annotations[marker_key]['model_name']
            if model_name in BOARD_MODELS:
                continue
            if model_name in combined_object_results:
                continue  # already solved via combined
            annotation = marker_annotations[marker_key]['annotation']
            try:
                object_tvec, object_rvec = estimate_object_pose_from_marker(
                    (pose_data['tvec'], pose_data['rvec']), annotation
                )
            except ValueError:
                continue
            candidates.setdefault(model_name, []).append((
                marker_key, object_tvec, object_rvec, pose_data['reproj_error'],
                pose_data['tvec'][2]
            ))

        # --- Select object pose ---
        detected_objects = []
        objects_seen = set()

        # Add board + combined object results
        all_combined = {**board_results, **combined_object_results}
        for model_name, (object_tvec, object_rvec, _) in all_combined.items():
            objects_seen.add(model_name)
            last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)

            if cam_pos is None or cam_quat is None:
                continue
            if np.any(np.isnan(cam_pos)) or np.any(np.isnan(cam_quat)):
                continue

            object_quat = rvec_to_quat(object_rvec)
            object_pos_world = transform_point_cam_to_world(object_tvec, cam_pos, cam_quat)
            object_quat_world = transform_orientation_cam_to_world(object_quat, cam_quat)

            pose_modified = False
            if model_name in BOARD_MODELS:
                if filter_config.board_snap_z:
                    obj_height = model_data.get(model_name, {}).get('object_height')
                    if obj_height is not None:
                        object_pos_world[2] = TABLE_Z + obj_height / 2.0
                        pose_modified = True
                if filter_config.board_yaw_only:
                    yaw = R.from_quat(object_quat_world).as_euler('xyz')[2]
                    object_quat_world = R.from_euler('xyz', [0.0, 0.0, yaw]).as_quat()
                    pose_modified = True

            # Fold symmetry snapping for non-board objects
            if model_name not in BOARD_MODELS and filter_config.enable_fold_snap:
                subtype = model_subtypes.get(model_name)
                if subtype in filter_config.fold_snap_subtypes:
                    if subtype == 'peg':
                        fold_counts = symmetry_data.get(model_name)
                        object_quat_world = snap_orientation_to_cardinal(
                            object_quat_world, fold_counts=fold_counts)
                    else:
                        object_quat_world = snap_orientation_to_cardinal(
                            object_quat_world, snap_angle_deg=filter_config.block_snap_angle)
                    pose_modified = True

            if filter_config.enable_ema_smoothing and model_name in prev_poses_world:
                prev_pos, prev_quat = prev_poses_world[model_name]
                alpha = filter_config.ema_alpha
                object_pos_world = (1.0 - alpha) * prev_pos + alpha * object_pos_world
                object_quat_world = slerp_quat(prev_quat, object_quat_world, blend=alpha)
                pose_modified = True

            prev_poses_world[model_name] = (object_pos_world.copy(), object_quat_world.copy())

            # Reproject to camera frame only if constraints/smoothing changed the pose
            if pose_modified:
                object_tvec = transform_point_world_to_cam(
                    object_pos_world, cam_pos, cam_quat)
                object_rvec = quat_to_rvec(
                    transform_orientation_world_to_cam(object_quat_world, cam_quat))

            detected_objects.append({
                "name": model_name, "points": [object_pos_world],
                "position": object_pos_world, "quaternion": object_quat_world,
                "inferred": False, "ghost_tracked": False, "no_display": False,
                "object_tvec": object_tvec, "object_rvec": object_rvec
            })

        # Add non-board results (single-marker fallback)
        for model_name, marker_list in candidates.items():
            objects_seen.add(model_name)

            # Pick the marker closest to the camera (lowest z)
            chosen = min(marker_list, key=lambda e: e[4])
            _, object_tvec, object_rvec, _, _ = chosen

            last_object_poses[model_name] = (object_tvec.copy(), object_rvec.copy(), frame_idx)

            # --- Transform to world frame ---
            if cam_pos is None or cam_quat is None:
                continue
            if np.any(np.isnan(cam_pos)) or np.any(np.isnan(cam_quat)):
                continue

            object_quat = rvec_to_quat(object_rvec)
            object_pos_world = transform_point_cam_to_world(object_tvec, cam_pos, cam_quat)
            object_quat_world = transform_orientation_cam_to_world(object_quat, cam_quat)

            pose_modified = False

            # Fold symmetry snapping for non-board objects
            if filter_config.enable_fold_snap:
                subtype = model_subtypes.get(model_name)
                if subtype in filter_config.fold_snap_subtypes:
                    if subtype == 'peg':
                        fold_counts = symmetry_data.get(model_name)
                        object_quat_world = snap_orientation_to_cardinal(
                            object_quat_world, fold_counts=fold_counts)
                    else:
                        object_quat_world = snap_orientation_to_cardinal(
                            object_quat_world, snap_angle_deg=filter_config.block_snap_angle)
                    pose_modified = True

            # --- Optional EMA smoothing ---
            if filter_config.enable_ema_smoothing and model_name in prev_poses_world:
                prev_pos, prev_quat = prev_poses_world[model_name]
                alpha = filter_config.ema_alpha
                object_pos_world = (1.0 - alpha) * prev_pos + alpha * object_pos_world
                object_quat_world = slerp_quat(prev_quat, object_quat_world, blend=alpha)
                pose_modified = True

            prev_poses_world[model_name] = (object_pos_world.copy(), object_quat_world.copy())

            # Reproject to camera frame only if constraints/smoothing changed the pose
            if pose_modified:
                object_tvec = transform_point_world_to_cam(
                    object_pos_world, cam_pos, cam_quat)
                object_rvec = quat_to_rvec(
                    transform_orientation_world_to_cam(object_quat_world, cam_quat))

            detected_objects.append({
                "name": model_name,
                "points": [object_pos_world],
                "position": object_pos_world,
                "quaternion": object_quat_world,
                "inferred": False,
                "ghost_tracked": False,
                "no_display": False,
                "object_tvec": object_tvec,
                "object_rvec": object_rvec
            })

        # --- Clean up stale objects ---
        for model_name in list(last_object_poses.keys()):
            if model_name not in objects_seen:
                _, _, last_frame = last_object_poses[model_name]

                if frame_idx - last_frame > filter_config.active_marker_timeout:
                    del last_object_poses[model_name]
                    prev_poses_world.pop(model_name, None)
                    board_active_markers.pop(model_name, None)

        # --- Wireframe visualization (direct camera-frame projection) ---
        for obj in detected_objects:
            model_name = obj["name"]
            if model_name not in model_data or model_data[model_name]['wireframe_vertices'] is None:
                continue

            obj_tvec = obj["object_tvec"]
            obj_rvec = obj["object_rvec"]

            if obj_tvec[2] <= 0.01 or obj_tvec[2] > 2.0:
                continue

            wireframe_vertices = model_data[model_name]['wireframe_vertices']
            wireframe_edges = model_data[model_name]['wireframe_edges']

            pts_3d = np.array(wireframe_vertices, dtype=np.float32)
            projected, _ = cv2.projectPoints(pts_3d, obj_rvec, obj_tvec, CAMERA_MATRIX, DIST_COEFFS)
            projected = projected.reshape(-1, 2).astype(int)

            for edge in wireframe_edges:
                if len(edge) >= 2:
                    si, ei = edge[0], edge[1]
                    if si < len(projected) and ei < len(projected):
                        p1 = tuple(projected[si])
                        p2 = tuple(projected[ei])
                        if all(-2000 < c < 4000 for c in p1 + p2):
                            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # --- Publish and draw ---
        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_object_poses(detected_objects)
        draw_text(frame, cam_pos, cam_quat, detected_objects, frame_idx, ee_pos, ee_quat, euler_convention=filter_config.euler_convention)
        draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, detected_objects, [])
        draw_grasp_points(frame, CAMERA_MATRIX, cam_pos, cam_quat, detected_objects, model_data)

        bridge_node.publish_annotated_stream(frame)

        if not headless_mode:
            cv2.imshow("Merged Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Cleanup
    if cap is not None:
        cap.release()
    if not headless_mode:
        cv2.destroyAllWindows()

    try:
        bridge_node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        if not headless_mode:
            print(f"Warning during ROS2 shutdown: {e}")


if __name__ == "__main__":
    main()
