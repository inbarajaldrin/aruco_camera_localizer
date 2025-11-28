import cv2
import cv2.aruco as aruco
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world
from aruco_camera_localizer.kalman_functions import QuaternionKalman
from aruco_camera_localizer.geometric_functions import transform_points_world_to_img, quat_to_rvec, complete_triangle, pick_best_candidate
from aruco_camera_localizer.object_frame_definitions import define_body_frame_allen_key, define_body_frame_wrench

def detect_markers(frame, gray, aruco_dicts, parameters):
    all_corners, all_ids = [], []
    for dict_id in aruco_dicts.values():
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids.flatten())
            # aruco.drawDetectedMarkers(frame, corners, ids)
    return all_corners, all_ids

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, talk=True, robot_moving=True):
    half_size = marker_size / 2
    
    # Validate camera pose (if it's wrong, all transformations will be wrong)
    if cam_pos is None or cam_quat is None:
        return  # Skip processing if camera pose is not available
    if np.any(np.isnan(cam_pos)) or np.any(np.isinf(cam_pos)) or np.any(np.isnan(cam_quat)) or np.any(np.isinf(cam_quat)):
        if talk and estimate_pose.debug_counter % 30 == 0:
            print(f"WARNING: Invalid camera pose - pos: {cam_pos}, quat: {cam_quat}")
        return  # Skip processing if camera pose is invalid
    
    # Static counter to reduce debug output frequency
    if not hasattr(estimate_pose, 'debug_counter'):
        estimate_pose.debug_counter = 0
    estimate_pose.debug_counter += 1

    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman()
                marker_stabilities[marker_id] = {
                    "last_tvec": None,
                    "confirmed_tvec": None,  # Only updated after confirmation
                    "confirmed_rvec": None,  # Only updated after confirmation
                    "last_known_tvec": None,  # Last known pose before reset (for outlier checking)
                    "last_known_rvec": None,  # Last known pose before reset (for outlier checking)
                    "last_frame": -1,
                    "confirmed": False,
                    "rejection_count": 0,  # Track consecutive rejections
                    "measurement_quality": 1.0,  # Measurement quality (0.0-1.0, 1.0 = perfect)
                    "quality_history": [],  # Rolling history of measurement quality
                    "max_quality_history": 20,  # Maximum number of quality values to track
                    "measurement_history": [],  # Recent measurements for adaptive outlier rejection
                    "max_measurement_history": 20  # Maximum number of measurements to track
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
                tvec_flat = tvec.flatten()
                
                # Calculate reprojection error for measurement quality
                projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                reprojection_errors = np.linalg.norm(image_points - projected_points, axis=1)
                rms_error = np.sqrt(np.mean(reprojection_errors**2))
                
                # Convert RMS error to quality score (0.0-1.0)
                # Lower error = higher quality
                # Typical good error: < 1 pixel, bad error: > 5 pixels
                max_acceptable_error = 5.0  # pixels
                measurement_quality = max(0.0, min(1.0, 1.0 - (rms_error / max_acceptable_error)))
                
                # Update quality history
                stability = marker_stabilities[marker_id]
                quality_history = stability.get("quality_history", [])
                quality_history.append(measurement_quality)
                max_history = stability.get("max_quality_history", 20)
                if len(quality_history) > max_history:
                    quality_history.pop(0)
                stability["quality_history"] = quality_history
                
                # Use rolling average for current quality
                avg_quality = np.mean(quality_history) if quality_history else measurement_quality
                stability["measurement_quality"] = avg_quality
                
                # NOTE: Outlier rejection filters removed - all measurements are accepted
                # To re-enable outlier rejection, uncomment one or both of the following filters:
                #
                # 1. EUCLIDEAN DISTANCE FILTER (fast pre-filter using cdist from scipy):
                #    - Check Euclidean distance from reference position (confirmed_tvec or last_known_tvec)
                #    - Thresholds: 0.10m (10cm) when moving, 0.05m (5cm) when stationary
                #    - Rejects measurements that exceed threshold
                #    - Clears reference after 5 consecutive rejections
                #
                # 2. MAHALANOBIS DISTANCE FILTER (statistical outlier rejection):
                #    - Track measurement history (last 20 measurements)
                #    - Compute mean and covariance from last 10 measurements
                #    - Calculate Mahalanobis distance for position and rotation
                #    - Adaptive thresholds: 3.0 sigma (moving) or 2.0 sigma (stationary) for position
                #    - Adaptive rotation thresholds: 0.35 rad (moving) or 0.20 rad (stationary)
                #    - Rejects outliers based on statistical distance
                #    - Clears reference after 10 consecutive rejections
                #
                # Both filters use measurement_history and rejection_count from stability dictionary.
                # Original implementation can be found in git history.
                
                # Update last tvec and confirm immediately
                stability["last_tvec"] = tvec_flat
                stability["last_frame"] = current_frame
                
                # Confirm immediately (no hold counter)
                stability["confirmed"] = True
                # Update confirmed baseline
                stability["confirmed_tvec"] = tvec_flat.copy()
                stability["confirmed_rvec"] = rvec.copy()
                # Also update last known pose for outlier checking after reset
                stability["last_known_tvec"] = tvec_flat.copy()
                stability["last_known_rvec"] = rvec.copy()

                measured_quat = rvec_to_quat(rvec)
                pred_tvec, pred_rvec = kalman.predict()
                pred_quat = rvec_to_quat(pred_rvec)
                
                # Use Kalman filter properly - let it handle the blending internally
                # Only use manual blending for very noisy measurements
                # Pass robot_moving flag to use minimum z when robot is stationary
                kalman.correct(tvec_flat, rvec, robot_moving=robot_moving)
                last_seen_frames[marker_id] = current_frame

    # Reset confirmation for markers not detected in current frame
    detected_marker_ids = set(int(id_val) for id_val in ids) if ids is not None else set()
    for marker_id in list(kalman_filters.keys()):
        if marker_id not in detected_marker_ids:
            # Marker not detected in current frame - reset confirmation
            if marker_id in marker_stabilities:
                marker_stabilities[marker_id]["confirmed"] = False
                marker_stabilities[marker_id]["confirmed_tvec"] = None
                marker_stabilities[marker_id]["confirmed_rvec"] = None
                    

def detect_object(p1, p2, p3, name, inferred):
    if name == "allen_key":
        pos, quat, contacts, contour = define_body_frame_allen_key(p1, p2, p3)
    elif name == "wrench":
        pos, quat, contacts, contour = define_body_frame_wrench(p1, p2, p3)
    obj = {
        "name": name,
        "points": (p1, p2, p3),
        "position": pos,
        "quaternion": quat,
        'inferred': inferred,
        "contacts": contacts,
        "contour": contour
    }
    return obj

def identify_objects_from_blobs(world_points, object_dicts, tolerance=10.0):
    identified_objects = []

    for tri_pts in combinations(world_points, 3):
        p1, p2, p3 = np.array(tri_pts[0]), np.array(tri_pts[1]), np.array(tri_pts[2])
        sides = sorted([
            1000 * np.linalg.norm(p1 - p2),
            1000 * np.linalg.norm(p2 - p3),
            1000 * np.linalg.norm(p3 - p1)
        ])

        for name, template in object_dicts.items():
            expected = sorted(template)
            diffs = [abs(a - b) for a, b in zip(sides, expected)]
            if all(d < tolerance for d in diffs):
                identified_objects.append(detect_object(p1, p2, p3, name, False))
                break  # One match per triangle

    return identified_objects

def attempt_recovery_for_missing_objects(last_objects, current_points, known_triangles, merge_threshold=0.03):
    recovered = []

    for prev in last_objects:
        name = prev["name"]
        prev_pts = prev["points"]
        matched_pts = []
        unmatched_prev_pt = None

        # Find current points close to previous ones
        for prev_pt in prev_pts:
            found = False
            for cur_pt in current_points:
                if np.linalg.norm(prev_pt - cur_pt) < merge_threshold:
                    matched_pts.append((prev_pt, cur_pt))
                    found = True
                    break
            if not found:
                unmatched_prev_pt = prev_pt
        if len(matched_pts) < 2:
            continue  # not enough info to infer

        # If more than two points matched, pick best two (closest to original positions)
        if len(matched_pts) > 2:
            matched_pts.sort(key=lambda pair: np.linalg.norm(pair[0] - pair[1]))
            unmatched_prev_pt = matched_pts[2][0]
            matched_pts = matched_pts[:2]

        if len(matched_pts) == 2:
            cur_pts = [pair[1] for pair in matched_pts]

            side_lengths = known_triangles[name]
            candidates = complete_triangle(cur_pts[0], cur_pts[1], side_lengths)
            if candidates:
                inferred_p3 = pick_best_candidate(candidates, unmatched_prev_pt)
            else:
                inferred_p3 = None
            # print("INFERRED", inferred_p3)

            # for inferred_p3 in candidates:
            try:
                recovered.append(detect_object(cur_pts[0], cur_pts[1], inferred_p3, name, True))
            except:
                continue
    return recovered