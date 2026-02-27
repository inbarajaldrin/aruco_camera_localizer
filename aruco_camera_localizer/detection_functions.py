import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, slerp_quat, quat_to_rvec
from aruco_camera_localizer.kalman_functions import QuaternionKalman


def _rotation_angle(rvec1, rvec2):
    """Angle in radians between two rotation vectors."""
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_diff = R1.T @ R2
    cos_angle = (np.trace(R_diff) - 1.0) / 2.0
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def detect_markers(frame, gray, aruco_dicts, parameters):
    all_corners, all_ids = [], []
    for dict_id in aruco_dicts.values():
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids.flatten())
            # Draw detected markers on the frame
            aruco.drawDetectedMarkers(frame, corners, ids)
    return all_corners, all_ids

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame,
                  cam_pos, cam_quat, opencv_to_camera_quat, talk=True,
                  prev_rvecs=None, z_range_min=0.05, z_range_max=2.0,
                  max_movement=0.05, hold_required=5, ghost_timeout=15,
                  blend_factor=0.99, q_params=None, r_params=None):
    """Estimate marker poses using IPPE disambiguation + Kalman filtering.

    Uses solvePnPGeneric with IPPE_SQUARE to obtain both pose solutions for
    each marker, then selects the best one via z-range filtering, reprojection
    error, and temporal rotational consistency. The selected pose is fed through
    the existing stability-check / Kalman pipeline.

    All filter parameters have sensible defaults and can be overridden from
    robot_config.yaml via the config_loader.
    """
    half_size = marker_size / 2

    object_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman(q_params=q_params, r_params=r_params)
                marker_stabilities[marker_id] = {
                    "last_tvec": None,
                    "last_frame": -1,
                    "confirmed": False,
                    "hold_counter": 0
                }
                last_seen_frames[marker_id] = 0

            kalman = kalman_filters[marker_id]
            stability = marker_stabilities[marker_id]

            image_points = corner[0].reshape(-1, 2)

            # --- IPPE: get both pose solutions ---
            try:
                num_solutions, rvecs_sol, tvecs_sol, _ = cv2.solvePnPGeneric(
                    object_points, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if num_solutions == 0:
                    continue
            except Exception as e:
                if talk:
                    print(f"[marker {marker_id}] solvePnPGeneric failed: {e}")
                continue

            # Score each solution: reprojection error + z-range filter
            valid_solutions = []
            for sol_rvec, sol_tvec in zip(rvecs_sol, tvecs_sol):
                z = sol_tvec.flatten()[2]
                if z < z_range_min or z > z_range_max:
                    continue
                projected, _ = cv2.projectPoints(object_points, sol_rvec, sol_tvec,
                                                 camera_matrix, dist_coeffs)
                reproj_err = np.sqrt(np.mean(np.sum(
                    (image_points - projected.reshape(-1, 2)) ** 2, axis=1)))
                valid_solutions.append((sol_rvec, sol_tvec, reproj_err))

            if not valid_solutions:
                continue

            # Sort by reprojection error (best first)
            valid_solutions.sort(key=lambda s: s[2])

            # Pick solution: temporal consistency if prev_rvec exists, else lowest reproj
            if prev_rvecs is not None and marker_id in prev_rvecs:
                # Choose solution rotationally closest to previous frame
                best_sol = min(valid_solutions,
                               key=lambda s: _rotation_angle(s[0], prev_rvecs[marker_id]))
            else:
                best_sol = valid_solutions[0]

            rvec, tvec, reproj_err = best_sol

            # Store chosen rvec for next frame's temporal consistency
            if prev_rvecs is not None:
                prev_rvecs[marker_id] = rvec.copy()

            # --- Transform from OpenCV frame to camera frame ---
            tvec_opencv = tvec.flatten()
            R_opencv_to_cam = R.from_quat(opencv_to_camera_quat)
            tvec_cam = R_opencv_to_cam.apply(tvec_opencv)

            quat_opencv = rvec_to_quat(rvec)
            quat_cam = (R_opencv_to_cam * R.from_quat(quat_opencv)).as_quat()

            # --- Stability check + Kalman pipeline (unchanged logic) ---
            tvec_flat = tvec_cam
            distance = np.linalg.norm(tvec_flat - stability["last_tvec"]) if stability["last_tvec"] is not None else 0
            movement_ok = distance < max_movement

            if movement_ok:
                stability["hold_counter"] += 1
            else:
                stability["hold_counter"] = 0

            stability["last_tvec"] = tvec_flat
            stability["last_frame"] = current_frame

            if stability["hold_counter"] >= hold_required:
                stability["confirmed"] = True

                measured_quat = quat_cam
                pred_tvec, pred_rvec = kalman.predict()
                pred_quat = rvec_to_quat(pred_rvec)
                blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
                blended_rvec = quat_to_rvec(blended_quat)
                blended_tvec = blend_factor * tvec_flat + (1 - blend_factor) * pred_tvec
                kalman.correct(blended_tvec, blended_rvec)

                # Convert back to OpenCV frame for visualization
                R_cam_to_opencv = R.from_quat(opencv_to_camera_quat).inv()
                tvec_opencv_viz = R_cam_to_opencv.apply(blended_tvec)
                quat_opencv_viz = (R_cam_to_opencv * R.from_quat(blended_quat)).as_quat()
                rvec_opencv_viz = quat_to_rvec(quat_opencv_viz)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_opencv_viz, tvec_opencv_viz, marker_size * 0.5)
                last_seen_frames[marker_id] = current_frame

    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        if current_frame - last_seen < ghost_timeout:
            pred_tvec, pred_rvec = kalman.predict()
            # Ghost predictions are used internally but not visualized
        else:
            stability["confirmed"] = False

