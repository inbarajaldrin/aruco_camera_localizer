import cv2
import numpy as np


def _rotation_angle(rvec1, rvec2):
    """Angle in radians between two rotation vectors."""
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_diff = R1.T @ R2
    cos_angle = (np.trace(R_diff) - 1.0) / 2.0
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def build_detectors(aruco_dicts, parameters):
    """Pre-build ArucoDetector objects for each dictionary (call once at startup).

    Returns:
        List of (dict_name, ArucoDetector) tuples.
    """
    detectors = []
    for dict_name, dict_id in aruco_dicts.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        detectors.append((dict_name, detector))
    return detectors


def detect_markers(frame, gray, detectors):
    """Detect ArUco markers across multiple dictionaries.

    Args:
        frame: BGR frame (unused, kept for API compat)
        gray: Grayscale image to detect in
        detectors: List of (dict_name, ArucoDetector) from build_detectors()

    Returns:
        all_corners: List of detected corner arrays
        all_ids: List of marker IDs (int)
        all_dict_names: List of dictionary names corresponding to each detection
    """
    all_corners, all_ids, all_dict_names = [], [], []
    for dict_name, detector in detectors:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                all_corners.append(corners[i])
                all_ids.append(int(marker_id))
                all_dict_names.append(dict_name)
    return all_corners, all_ids, all_dict_names


def estimate_poses(corners, ids, dict_names, camera_matrix, dist_coeffs, total_marker_size,
                   z_range_min=0.05, z_range_max=2.0, max_reproj_error=None, talk=True, prev_rvecs=None):
    """Estimate pose for each detected marker using solvePnPGeneric.

    Uses solvePnPGeneric with IPPE_SQUARE to get both pose solutions, then
    picks the one rotationally consistent with the previous frame to avoid
    pose ambiguity flipping.

    Args:
        corners: List of detected corner arrays from detect_markers
        ids: List of marker IDs from detect_markers
        dict_names: List of dictionary names from detect_markers
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        total_marker_size: Physical marker size in meters (the detectable black square)
        z_range_min: Minimum valid depth in meters
        z_range_max: Maximum valid depth in meters
        talk: Whether to print debug messages
        prev_rvecs: Optional dict of {(marker_id, dict_name): rvec} from
            previous frame. Updated in-place with chosen rvecs. Pass the same
            dict across frames to enable temporal consistency.

    Returns:
        dict mapping (marker_id, dict_name) -> {
            'tvec': translation vector (3,),
            'rvec': rotation vector (3,1),
            'reproj_error': RMS reprojection error in pixels,
            'dict_name': dictionary name string
        }
    """
    results = {}
    rejected = {}  # markers that failed z-range or reproj error
    all_raw = {}   # all detections unfiltered (best solution per marker)

    if not corners or not ids or not dict_names:
        return results, {}, {}

    half_size = total_marker_size / 2.0
    object_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    for corner, marker_id, dict_name in zip(corners, ids, dict_names):
        image_points = corner[0].reshape(-1, 2)

        try:
            num_solutions, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if num_solutions == 0:
                continue
        except Exception as e:
            if talk:
                print(f"[{marker_id}] solvePnPGeneric exception: {e}")
            continue

        key = (marker_id, dict_name)

        # Collect all solutions
        all_solutions = []
        valid_solutions = []
        rejected_solutions = []
        for sol_rvec, sol_tvec in zip(rvecs, tvecs):
            sol_tvec_flat = sol_tvec.flatten()
            projected_points, _ = cv2.projectPoints(object_points, sol_rvec, sol_tvec, camera_matrix, dist_coeffs)
            projected_points = projected_points.reshape(-1, 2)
            rms_error = np.sqrt(np.mean(np.sum((image_points - projected_points) ** 2, axis=1)))
            sol_entry = {
                'tvec': sol_tvec_flat,
                'rvec': sol_rvec,
                'reproj_error': rms_error,
                'dict_name': dict_name
            }
            all_solutions.append(sol_entry)
            if sol_tvec_flat[2] < z_range_min or sol_tvec_flat[2] > z_range_max:
                sol_entry['reject_reason'] = f'z={sol_tvec_flat[2]:.3f} out of [{z_range_min},{z_range_max}]'
                rejected_solutions.append(sol_entry)
                continue
            if max_reproj_error is not None and rms_error > max_reproj_error:
                sol_entry['reject_reason'] = f'reproj={rms_error:.2f} > {max_reproj_error}'
                rejected_solutions.append(sol_entry)
                continue
            valid_solutions.append(sol_entry)

        # Store best raw detection (unfiltered)
        if all_solutions:
            all_solutions.sort(key=lambda s: s['reproj_error'])
            all_raw[key] = all_solutions[0]

        if not valid_solutions:
            if rejected_solutions:
                rejected_solutions.sort(key=lambda s: s['reproj_error'])
                rejected[key] = rejected_solutions[0]
            continue

        # Primary solution (lowest reproj error)
        valid_solutions.sort(key=lambda s: s['reproj_error'])
        primary = valid_solutions[0]

        if prev_rvecs is not None:
            prev_rvecs[key] = primary['rvec'].copy()

        # Store primary + alt solutions
        primary['alt_solutions'] = valid_solutions[1:] if len(valid_solutions) > 1 else []
        results[key] = primary

    return results, rejected, all_raw
