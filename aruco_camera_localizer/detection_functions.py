import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, slerp_quat, quat_to_rvec
from aruco_camera_localizer.kalman_functions import QuaternionKalman

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
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, opencv_to_camera_quat, talk=True):
    max_movement = 0.05  # meters
    hold_required = 5    # frames it must persist
    half_size = marker_size / 2

    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman()
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
            object_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if success:
                # solvePnP returns pose in OpenCV frame, need to transform to camera frame first
                tvec_opencv = tvec.flatten()
                
                # Step 1: Transform tvec from OpenCV frame to camera frame
                R_opencv_to_cam = R.from_quat(opencv_to_camera_quat)
                tvec_cam = R_opencv_to_cam.apply(tvec_opencv)
                
                # Step 2: Transform rvec (as quaternion) from OpenCV frame to camera frame
                quat_opencv = rvec_to_quat(rvec)
                quat_cam = R_opencv_to_cam * R.from_quat(quat_opencv)
                quat_cam = quat_cam.as_quat()
                
                # Use camera frame values for tracking
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
                    # Only print when marker is first confirmed (transitions from unconfirmed to confirmed)
                    was_confirmed = stability["confirmed"]
                    stability["confirmed"] = True

                    measured_quat = quat_cam  # Already in camera frame
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    blend_factor = 0.99
                    blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
                    blended_rvec = quat_to_rvec(blended_quat)
                    blended_tvec = blend_factor * tvec_flat + (1 - blend_factor) * pred_tvec
                    kalman.correct(blended_tvec, blended_rvec)
                    
                    # Convert back to OpenCV frame for visualization (drawFrameAxes expects OpenCV frame)
                    R_cam_to_opencv = R.from_quat(opencv_to_camera_quat).inv()
                    tvec_opencv_viz = R_cam_to_opencv.apply(blended_tvec)
                    quat_opencv_viz = (R_cam_to_opencv * R.from_quat(blended_quat)).as_quat()
                    rvec_opencv_viz = quat_to_rvec(quat_opencv_viz)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_opencv_viz, tvec_opencv_viz, marker_size * 0.5)
                    last_seen_frames[marker_id] = current_frame
                    # Convert from camera frame to world frame
                    marker_pos_world = transform_point_cam_to_world(blended_tvec, cam_pos, cam_quat)
                    marker_quat_world = transform_orientation_cam_to_world(blended_quat, cam_quat)


    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        if current_frame - last_seen < 15:
            pred_tvec, pred_rvec = kalman.predict()
            # Ghost predictions are used internally but not visualized to reduce clutter
        else:
            stability["confirmed"] = False

