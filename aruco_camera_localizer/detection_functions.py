import cv2
import cv2.aruco as aruco
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world
from aruco_camera_localizer.kalman_functions import QuaternionKalman
from aruco_camera_localizer.geometric_functions import transform_points_world_to_img, quat_to_rvec, complete_triangle, pick_best_candidate, slerp_quat
from aruco_camera_localizer.object_frame_definitions import define_body_frame_allen_key, define_body_frame_wrench
from aruco_camera_localizer.filter_config import FilterConfig

def detect_markers(frame, gray, aruco_dicts, parameters):
    all_corners, all_ids, all_dict_names = [], [], []
    for dict_name, dict_id in aruco_dicts.items():
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                all_corners.append(corners[i])
                all_ids.append(marker_id)
                all_dict_names.append(dict_name)
            # aruco.drawDetectedMarkers(frame, corners, ids)
    return all_corners, all_ids, all_dict_names

def cleanup_old_markers(kalman_filters, marker_stabilities, last_seen_frames, current_frame, 
                         detected_marker_ids, cleanup_threshold=300, talk=True):
    """
    Remove markers that haven't been seen for a long time to prevent unbounded dictionary growth.
    
    Args:
        kalman_filters: Dictionary of Kalman filters by marker ID
        marker_stabilities: Dictionary of marker stability data by marker ID
        last_seen_frames: Dictionary of last seen frame numbers by marker ID
        current_frame: Current frame number
        detected_marker_ids: Set of marker IDs detected in current frame (don't remove these)
        cleanup_threshold: Number of frames since last seen before removal (default: 300 = ~10s at 30fps)
        talk: Whether to print debug messages
    """
    markers_to_remove = []
    
    for marker_id in list(kalman_filters.keys()):
        # Don't remove markers that are currently detected
        if marker_id in detected_marker_ids:
            continue
        
        # Check how long since this marker was last seen
        last_seen = last_seen_frames.get(marker_id, 0)
        frames_since_last_seen = current_frame - last_seen
        
        # Remove if not seen for longer than threshold
        if frames_since_last_seen > cleanup_threshold:
            markers_to_remove.append(marker_id)
    
    # Remove old markers from all dictionaries
    for marker_id in markers_to_remove:
        last_seen = last_seen_frames.get(marker_id, 0)
        frames_since_last_seen = current_frame - last_seen
        
        if marker_id in kalman_filters:
            del kalman_filters[marker_id]
        if marker_id in marker_stabilities:
            del marker_stabilities[marker_id]
        if marker_id in last_seen_frames:
            del last_seen_frames[marker_id]
        
        if talk:
            print(f"[Cleanup] Removed marker {marker_id} (not seen for {frames_since_last_seen} frames)")
    
    return len(markers_to_remove)

def estimate_pose(frame, corners, ids, dict_names, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, 
                  filter_config=None, talk=True, robot_moving=True):
    # Use default config if not provided
    if filter_config is None:
        filter_config = FilterConfig()
    
    half_size = marker_size / 2
    
    # Validate camera pose (if it's wrong, all transformations will be wrong)
    # But don't block marker detection - just warn and skip world frame transformations
    camera_pose_valid = True
    if cam_pos is None or cam_quat is None:
        camera_pose_valid = False
        if talk and estimate_pose.debug_counter % 30 == 0:
            print(f"WARNING: Camera pose not available - will detect markers but skip world frame transforms")
    elif np.any(np.isnan(cam_pos)) or np.any(np.isinf(cam_pos)) or np.any(np.isnan(cam_quat)) or np.any(np.isinf(cam_quat)):
        camera_pose_valid = False
        if talk and estimate_pose.debug_counter % 30 == 0:
            print(f"WARNING: Invalid camera pose - pos: {cam_pos}, quat: {cam_quat} - will detect markers but skip world frame transforms")
    
    # Static counter to reduce debug output frequency
    if not hasattr(estimate_pose, 'debug_counter'):
        estimate_pose.debug_counter = 0
    estimate_pose.debug_counter += 1

    if corners and ids and dict_names:
        
        for corner, marker_id, dict_name in zip(corners, ids, dict_names):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                if filter_config.enable_kalman_filter:
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
                    "max_quality_history": filter_config.quality_history_size,  # Maximum number of quality values to track
                    "measurement_history": [],  # Recent measurements for adaptive outlier rejection
                    "max_measurement_history": filter_config.mahalanobis_measurement_history_size,  # Maximum number of measurements to track
                    "missed_frames": 0,  # Track consecutive missed detections
                    "aruco_dictionary": dict_name  # Store which dictionary this marker was detected from
                }
                last_seen_frames[marker_id] = 0
            else:
                # Update dictionary if it changed (shouldn't happen, but validate)
                if marker_stabilities[marker_id].get("aruco_dictionary") != dict_name:
                    marker_stabilities[marker_id]["aruco_dictionary"] = dict_name

            kalman = kalman_filters.get(marker_id) if filter_config.enable_kalman_filter else None
            stability = marker_stabilities[marker_id]

            image_points = corner[0].reshape(-1, 2)
            object_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            # Use IPPE_SQUARE flag like the simple detection for better accuracy
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not success:
                    if talk and estimate_pose.debug_counter % 30 == 0:
                        print(f"[{marker_id}] solvePnP failed")
                    continue
            except Exception as e:
                if talk:
                    print(f"[{marker_id}] solvePnP exception: {e}")
                continue
            
            # Process successful pose estimation
            try:
                tvec_flat = tvec.flatten()
            except Exception as e:
                if talk:
                    print(f"[{marker_id}] Error flattening tvec: {e}")
                continue
            
            # Calculate reprojection error for measurement quality
            projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
            projected_points = projected_points.reshape(-1, 2)
            reprojection_errors = np.linalg.norm(image_points - projected_points, axis=1)
            rms_error = np.sqrt(np.mean(reprojection_errors**2))
            
            # Convert RMS error to quality score (0.0-1.0)
            # Lower error = higher quality
            # Typical good error: < 1 pixel, bad error: > 5 pixels
            max_acceptable_error = filter_config.max_acceptable_error
            measurement_quality = max(0.0, min(1.0, 1.0 - (rms_error / max_acceptable_error)))
            
            # Update quality history
            stability = marker_stabilities[marker_id]
            quality_history = stability.get("quality_history", [])
            quality_history.append(measurement_quality)
            max_history = filter_config.quality_history_size
            if len(quality_history) > max_history:
                quality_history.pop(0)
            stability["quality_history"] = quality_history
            
            # Use rolling average for current quality
            avg_quality = np.mean(quality_history) if quality_history else measurement_quality
            stability["measurement_quality"] = avg_quality
            
            # Z-range validation filter
            if filter_config.enable_z_range_validation:
                min_z = filter_config.z_range_min
                max_z = filter_config.z_range_max
                if tvec_flat[2] < min_z or tvec_flat[2] > max_z:
                    # Outlier: Z out of reasonable range
                    if talk and estimate_pose.debug_counter % 30 == 0:
                        print(f"[{marker_id}] Outlier: Z={tvec_flat[2]:.3f}m out of range [{min_z:.3f}, {max_z:.3f}]")
                    continue
            
            # Check if marker hasn't been seen for a long time - if so, clear stale pose data
            # This prevents comparing new detections against very old positions
            # Only clear if marker was previously seen (last_seen > 0), not on first detection
            last_seen = last_seen_frames.get(marker_id, 0)
            if last_seen > 0:  # Only check if marker was previously seen
                frames_since_last_seen = current_frame - last_seen
                stale_pose_threshold = 60  # frames - ~2 seconds at 30fps
                
                if frames_since_last_seen > stale_pose_threshold:
                    # Marker hasn't been seen for a while - clear stale pose data to allow fresh start
                    stability["confirmed_tvec"] = None
                    stability["confirmed_rvec"] = None
                    stability["last_known_tvec"] = None
                    stability["last_known_rvec"] = None
                    stability["confirmed"] = False
                    stability["measurement_history"] = []  # Clear history too
                    stability["rejection_count"] = 0  # Reset rejection count
                    if talk and estimate_pose.debug_counter % 30 == 0:
                        print(f"[{marker_id}] Cleared stale pose data after {frames_since_last_seen} frames - allowing fresh detection")
            
            # Adaptive outlier rejection using Mahalanobis distance
            # Track measurement history and compute adaptive thresholds
            measurement_history = stability.get("measurement_history", [])
            max_history = filter_config.mahalanobis_measurement_history_size
            
            # Check against confirmed pose if available, otherwise check against last known pose
            check_tvec = stability.get("confirmed_tvec") if stability.get("confirmed_tvec") is not None else stability.get("last_known_tvec")
            check_rvec = stability.get("confirmed_rvec") if stability.get("confirmed_rvec") is not None else stability.get("last_known_rvec")
            
            # Track if this measurement should be rejected
            is_outlier = False
            
            # Mahalanobis distance outlier rejection filter
            if filter_config.enable_mahalanobis_outlier_rejection and check_tvec is not None and check_rvec is not None and len(measurement_history) >= 3:
                    # Calculate measurement statistics from history
                    recent_tvecs = [m["tvec"] for m in measurement_history[-10:]]  # Last 10 measurements
                    recent_rvecs = [m["rvec"] for m in measurement_history[-10:]]
                    
                    # Compute mean and covariance of recent measurements
                    mean_tvec = np.mean(recent_tvecs, axis=0)
                    cov_tvec = np.cov(np.array(recent_tvecs).T)
                    
                    # Ensure covariance is positive definite
                    cov_tvec += np.eye(3) * 1e-6
                    
                    # Compute Mahalanobis distance for position
                    diff_tvec = tvec_flat - mean_tvec
                    try:
                        mahal_distance_pos = np.sqrt(diff_tvec @ np.linalg.inv(cov_tvec) @ diff_tvec)
                    except:
                        # Fallback to Euclidean distance if covariance is singular
                        mahal_distance_pos = np.linalg.norm(diff_tvec)
                    
                    # Compute rotation statistics
                    rotation_angles = []
                    for r in recent_rvecs:
                        R_hist, _ = cv2.Rodrigues(r)
                        R_check, _ = cv2.Rodrigues(check_rvec)
                        R_rel = R_hist @ R_check.T
                        rvec_rel, _ = cv2.Rodrigues(R_rel)
                        rotation_angles.append(np.linalg.norm(rvec_rel))
                    
                    mean_rot_angle = np.mean(rotation_angles)
                    std_rot_angle = np.std(rotation_angles) if len(rotation_angles) > 1 else 0.1
                    
                    # Current rotation angle
                    R_current, _ = cv2.Rodrigues(rvec)
                    R_check, _ = cv2.Rodrigues(check_rvec)
                    R_relative = R_current @ R_check.T
                    rvec_relative, _ = cv2.Rodrigues(R_relative)
                    rotation_angle = np.linalg.norm(rvec_relative)
                    
                    # Mahalanobis distance for rotation (normalized)
                    if std_rot_angle > 1e-6:
                        mahal_distance_rot = abs(rotation_angle - mean_rot_angle) / std_rot_angle
                    else:
                        mahal_distance_rot = abs(rotation_angle - mean_rot_angle) / 0.1
                    
                    # Adaptive thresholds based on measurement variance
                    # Base thresholds - loosened to reduce false rejections
                    if robot_moving:
                        base_mahal_threshold = filter_config.mahalanobis_base_threshold_moving
                        base_rot_threshold = filter_config.mahalanobis_rot_threshold_moving
                    else:
                        base_mahal_threshold = filter_config.mahalanobis_base_threshold_stationary
                        base_rot_threshold = filter_config.mahalanobis_rot_threshold_stationary
                    
                    # Adjust threshold based on variance (higher variance = more lenient, but cap it)
                    variance_factor = 1.0 + min(np.trace(cov_tvec) * 5.0, filter_config.mahalanobis_variance_factor_cap)
                    adaptive_mahal_threshold = base_mahal_threshold * variance_factor
                    adaptive_rot_threshold = base_rot_threshold * (1.0 + min(std_rot_angle, filter_config.mahalanobis_rot_factor_cap))
                    
                    # Reject if Mahalanobis distance exceeds threshold
                    if mahal_distance_pos > adaptive_mahal_threshold or mahal_distance_rot > adaptive_mahal_threshold:
                        # Outlier detected using Mahalanobis distance
                        is_outlier = True
                        stability["rejection_count"] = stability.get("rejection_count", 0) + 1
                        if stability["rejection_count"] > filter_config.mahalanobis_rejection_count_threshold:
                            rejection_count_before_clear = stability["rejection_count"]
                            stability["last_known_tvec"] = None
                            stability["last_known_rvec"] = None
                            stability["rejection_count"] = 0
                            stability["measurement_history"] = []  # Clear history
                            if talk and estimate_pose.debug_counter % 30 == 0:
                                print(f"[{marker_id}] Cleared last_known pose after {rejection_count_before_clear} rejections - allowing recovery")
                        else:
                            if talk and estimate_pose.debug_counter % 30 == 0:
                                print(f"[{marker_id}] Outlier (Mahal): pos={mahal_distance_pos:.2f} (thresh={adaptive_mahal_threshold:.2f}), rot={mahal_distance_rot:.2f}")
            
            # Fallback to simple distance check if not enough history
            elif filter_config.enable_simple_outlier_rejection and check_tvec is not None and check_rvec is not None:
                # Use simple distance-based rejection as fallback
                distance = np.linalg.norm(tvec_flat - check_tvec)
                if robot_moving:
                    outlier_rejection_movement_threshold = filter_config.simple_outlier_movement_threshold_moving
                    outlier_rejection_rotation_threshold = filter_config.simple_outlier_rotation_threshold_moving
                else:
                    outlier_rejection_movement_threshold = filter_config.simple_outlier_movement_threshold_stationary
                    outlier_rejection_rotation_threshold = filter_config.simple_outlier_rotation_threshold_stationary
                
                # Reject if distance exceeds threshold (fixed: was checking wrong condition)
                if distance > outlier_rejection_movement_threshold:
                    is_outlier = True
                    stability["rejection_count"] = stability.get("rejection_count", 0) + 1
                    if stability["rejection_count"] > filter_config.simple_outlier_rejection_count_threshold:
                        stability["last_known_tvec"] = None
                        stability["last_known_rvec"] = None
                        stability["rejection_count"] = 0
                    if talk and estimate_pose.debug_counter % 30 == 0:
                        print(f"[{marker_id}] Outlier (distance): {distance*1000:.1f}mm > {outlier_rejection_movement_threshold*1000:.1f}mm")
                
                # Check rotation (only if distance check passed)
                if not is_outlier:
                    R_current, _ = cv2.Rodrigues(rvec)
                    R_check, _ = cv2.Rodrigues(check_rvec)
                    R_relative = R_current @ R_check.T
                    rvec_relative, _ = cv2.Rodrigues(R_relative)
                    rotation_angle = np.linalg.norm(rvec_relative)
                    
                    if rotation_angle > outlier_rejection_rotation_threshold:
                        is_outlier = True
                        stability["rejection_count"] = stability.get("rejection_count", 0) + 1
                        if stability["rejection_count"] > filter_config.simple_outlier_rejection_count_threshold:
                            stability["last_known_tvec"] = None
                            stability["last_known_rvec"] = None
                            stability["rejection_count"] = 0
                        if talk and estimate_pose.debug_counter % 30 == 0:
                            print(f"[{marker_id}] Outlier (rotation): {np.degrees(rotation_angle):.1f}° > {np.degrees(outlier_rejection_rotation_threshold):.1f}°")
            
            # Skip this measurement if it's an outlier
            if is_outlier:
                if talk and estimate_pose.debug_counter % 30 == 0:
                    print(f"[{marker_id}] Rejected as outlier")
                continue
            
            # Only add measurement to history after it passes all filters
            measurement_history.append({
                "tvec": tvec_flat.copy(),
                "rvec": rvec.copy(),
                "frame": current_frame
            })
            if len(measurement_history) > max_history:
                measurement_history.pop(0)
            stability["measurement_history"] = measurement_history
            
            # Reset rejection count if measurement passes
            stability["rejection_count"] = 0
            stability["missed_frames"] = 0  # Reset missed frames counter
            
            # Update last tvec
            stability["last_tvec"] = tvec_flat
            stability["last_frame"] = current_frame
            
            # Confirm immediately (no hold counter)
            stability["confirmed"] = True
            
            # Marker stability confirmation smoothing filter
            if filter_config.enable_marker_stability_smoothing:
                # Update confirmed baseline with temporal smoothing to reduce flickering
                # When stationary, use more aggressive smoothing (smaller alpha = more smoothing)
                # When moving, use less smoothing to track movement better
                if robot_moving:
                    smoothing_alpha = filter_config.stability_smoothing_alpha_moving
                else:
                    smoothing_alpha = filter_config.stability_smoothing_alpha_stationary
                
                if stability["confirmed_tvec"] is not None and stability["confirmed_rvec"] is not None:
                    # Blend position (linear interpolation)
                    prev_tvec = stability["confirmed_tvec"]
                    smoothed_tvec = (1.0 - smoothing_alpha) * prev_tvec + smoothing_alpha * tvec_flat
                    
                    # Blend orientation (SLERP for quaternions)
                    prev_rvec = stability["confirmed_rvec"]
                    prev_quat = rvec_to_quat(prev_rvec)
                    current_quat = rvec_to_quat(rvec)
                    smoothed_quat = slerp_quat(prev_quat, current_quat, blend=smoothing_alpha)
                    smoothed_rvec = quat_to_rvec(smoothed_quat)
                    
                    # Update confirmed baseline with smoothed values
                    stability["confirmed_tvec"] = smoothed_tvec.copy()
                    stability["confirmed_rvec"] = smoothed_rvec.copy()
                else:
                    # First confirmation - use measurement directly
                    stability["confirmed_tvec"] = tvec_flat.copy()
                    stability["confirmed_rvec"] = rvec.copy()
            else:
                # No smoothing - use measurement directly
                stability["confirmed_tvec"] = tvec_flat.copy()
                stability["confirmed_rvec"] = rvec.copy()
            
            # Also update last known pose for outlier checking after reset
            stability["last_known_tvec"] = tvec_flat.copy()
            stability["last_known_rvec"] = rvec.copy()

            # Kalman filter correction
            if filter_config.enable_kalman_filter:
                measured_quat = rvec_to_quat(rvec)
                pred_tvec, pred_rvec = kalman.predict()
                pred_quat = rvec_to_quat(pred_rvec)
                
                # Use Kalman filter properly - let it handle the blending internally
                # Only use manual blending for very noisy measurements
                # Pass robot_moving flag to use minimum z when robot is stationary
                kalman.correct(tvec_flat, rvec, robot_moving=robot_moving)
                last_seen_frames[marker_id] = current_frame

    # Get set of currently detected marker IDs for cleanup and confirmation reset
    detected_marker_ids = set(int(id_val) for id_val in ids) if ids is not None else set()
    
    # Cleanup old markers periodically to prevent unbounded dictionary growth
    # Run cleanup every 60 frames (~2 seconds at 30fps) to balance performance and memory
    cleanup_frequency = 60
    if current_frame % cleanup_frequency == 0:
        cleanup_old_markers(kalman_filters, marker_stabilities, last_seen_frames, current_frame,
                           detected_marker_ids, cleanup_threshold=300, talk=talk)
    
    # Reset confirmation for markers not detected in current frame
    # Only reset after multiple consecutive missed frames to reduce flickering
    if filter_config.enable_marker_confirmation_reset:
        missed_frames_threshold = filter_config.marker_confirmation_missed_frames_threshold
        
        for marker_id in list(kalman_filters.keys()):
            if marker_id not in detected_marker_ids:
                # Marker not detected in current frame - increment missed frames counter
                if marker_id in marker_stabilities:
                    stability = marker_stabilities[marker_id]
                    stability["missed_frames"] = stability.get("missed_frames", 0) + 1
                    
                    # Only reset confirmation after multiple consecutive misses
                    if stability["missed_frames"] >= missed_frames_threshold:
                        stability["confirmed"] = False
                        # Keep confirmed_tvec/rvec for outlier checking, but mark as unconfirmed
                        # Don't clear them immediately - they're still useful for comparison
                        # They'll be cleared when a new marker is confirmed
                    

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