import cv2
import cv2.aruco as aruco
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world
from aruco_camera_localizer.kalman_functions import QuaternionKalman
from aruco_camera_localizer.geometric_functions import transform_points_world_to_img, slerp_quat, quat_to_rvec, complete_triangle, pick_best_candidate
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

def detect_color_blobs(frame, color_range, color, camera_matrix, cam_pos, cam_quat, height=0.01, min_area=120, merge_threshold=0.02):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue range in HSV
    mask = cv2.inRange(hsv, color_range[0], color_range[1])
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    world_points = []
    image_points = []

    if contours:
        for cnt in contours:
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)            
            if area < min_area:
                continue  # skip tiny blobs
            if M["m00"] > 0:
                cv2.drawContours(frame, [cnt], 0, (255, 255, 255), 1)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Step 1: Ray in camera frame
                pixel = np.array([cx, cy, 1.0])
                ray_cam = np.linalg.inv(camera_matrix) @ pixel

                # Step 2: Transform ray to world frame
                R_wc = R.from_quat(cam_quat).as_matrix()
                ray_world = R_wc @ ray_cam
                cam_origin_world = np.array(cam_pos)

                # Step 3: Ray-plane intersection with z = height over table
                t = (height - cam_origin_world[2]) / ray_world[2]
                point_world = cam_origin_world + t * ray_world
                world_points.append(point_world)

    # Step 4: Merge nearby points in world frame
    merged_world_points = []
    used = set()
    for i, pt in enumerate(world_points):
        if i in used:
            continue
        cluster = [pt]
        used.add(i)
        for j in range(i + 1, len(world_points)):
            if j in used:
                continue
            if np.linalg.norm(world_points[j] - pt) < merge_threshold:
                cluster.append(world_points[j])
                used.add(j)
        cluster_avg = np.mean(cluster, axis=0)
        merged_world_points.append(cluster_avg)
    image_points = transform_points_world_to_img(merged_world_points, cam_pos, cam_quat, camera_matrix)
    for (u,v) in image_points:
        cv2.circle(frame, (u, v), 5, color, -1)
    return merged_world_points, image_points

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, talk=True, robot_moving=True):
    hold_required = 1    # frames it must persist - reduced for faster confirmation
    half_size = marker_size / 2
    
    # Validate camera pose (if it's wrong, all transformations will be wrong)
    if cam_pos is None or cam_quat is None:
        return  # Skip processing if camera pose is not available
    if np.any(np.isnan(cam_pos)) or np.any(np.isinf(cam_pos)) or np.any(np.isnan(cam_quat)) or np.any(np.isinf(cam_quat)):
        if talk and estimate_pose.debug_counter % 30 == 0:
            print(f"WARNING: Invalid camera pose - pos: {cam_pos}, quat: {cam_quat}")
        return  # Skip processing if camera pose is invalid
    
    # Check if camera Z is reasonable (should be around 0-1m for tabletop setup)
    if abs(cam_pos[2]) > 2.0:  # Camera Z should be reasonable
        if talk and estimate_pose.debug_counter % 30 == 0:
            print(f"WARNING: Camera Z out of range: {cam_pos[2]:.3f}m - pos: {cam_pos}, quat: {cam_quat}")
        return  # Skip processing if camera pose is unreasonable
    
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
                    "last_known_tvec_world": None,  # Last known pose in world frame (for backtracking)
                    "last_known_rvec_world": None,  # Last known rotation in world frame (for backtracking)
                    "last_frame": -1,
                    "confirmed": False,
                    "hold_counter": 0,
                    "rejection_count": 0,  # Track consecutive rejections
                    "frames_missing": 0,  # Track consecutive frames without detection
                    "measurement_quality": 1.0,  # Measurement quality (0.0-1.0, 1.0 = perfect)
                    "quality_history": [],  # Rolling history of measurement quality
                    "max_quality_history": 20,  # Maximum number of quality values to track
                    "prev_cam_pos": None,  # Previous camera position for motion compensation
                    "prev_cam_quat": None,  # Previous camera quaternion for motion compensation
                    "measurement_history": [],  # Recent measurements for adaptive outlier rejection
                    "max_measurement_history": 20,  # Maximum number of measurements to track
                    "prediction_mode": None,  # Track current prediction mode ("kalman" or "last_known")
                    "prev_prediction_mode": None,  # Previous prediction mode (for smooth transitions)
                    "mode_transition_frames": 0  # Frames since mode switch (for smooth transitions)
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
                
                # Always validate Z-range (not just first measurement)
                min_z = 0.05  # Minimum depth: 5cm
                max_z = 2.0   # Maximum depth: 2m
                if tvec_flat[2] < min_z or tvec_flat[2] > max_z:
                    # Outlier: Z out of reasonable range
                    if talk and estimate_pose.debug_counter % 30 == 0:
                        print(f"[{marker_id}] Outlier: Z={tvec_flat[2]:.3f}m out of range [{min_z:.3f}, {max_z:.3f}]")
                    continue
                
                # Adaptive outlier rejection using Mahalanobis distance
                # Track measurement history and compute adaptive thresholds
                measurement_history = stability.get("measurement_history", [])
                max_history = stability.get("max_measurement_history", 20)
                
                # Add current measurement to history (before checking if it's an outlier)
                measurement_history.append({
                    "tvec": tvec_flat.copy(),
                    "rvec": rvec.copy(),
                    "frame": current_frame
                })
                if len(measurement_history) > max_history:
                    measurement_history.pop(0)
                stability["measurement_history"] = measurement_history
                
                # Check against confirmed pose if available, otherwise check against last known pose
                check_tvec = stability.get("confirmed_tvec") if stability.get("confirmed_tvec") is not None else stability.get("last_known_tvec")
                check_rvec = stability.get("confirmed_rvec") if stability.get("confirmed_rvec") is not None else stability.get("last_known_rvec")
                
                if check_tvec is not None and check_rvec is not None and len(measurement_history) >= 3:
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
                    # Base thresholds
                    if robot_moving:
                        base_mahal_threshold = 3.0  # 3 sigma for moving
                        base_rot_threshold = 0.35  # radians
                    else:
                        base_mahal_threshold = 2.0  # 2 sigma for stationary (stricter)
                        base_rot_threshold = 0.20  # radians
                    
                    # Adjust threshold based on variance (higher variance = more lenient)
                    variance_factor = 1.0 + np.trace(cov_tvec) * 10.0  # Scale variance contribution
                    adaptive_mahal_threshold = base_mahal_threshold * variance_factor
                    adaptive_rot_threshold = base_rot_threshold * (1.0 + std_rot_angle)
                    
                    # Reject if Mahalanobis distance exceeds threshold
                    if mahal_distance_pos > adaptive_mahal_threshold or mahal_distance_rot > adaptive_mahal_threshold:
                        # Outlier detected using Mahalanobis distance
                        stability["rejection_count"] = stability.get("rejection_count", 0) + 1
                        if stability["rejection_count"] > 10:
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
                        continue
                
                # Fallback to simple distance check if not enough history
                elif check_tvec is not None and check_rvec is not None:
                    # Use simple distance-based rejection as fallback
                    distance = np.linalg.norm(tvec_flat - check_tvec)
                    if robot_moving:
                        outlier_rejection_movement_min = 0.030
                        outlier_rejection_movement_max = 0.085
                        outlier_rejection_rotation_threshold = 0.35
                    else:
                        outlier_rejection_movement_min = 0.015
                        outlier_rejection_movement_max = 0.050
                        outlier_rejection_rotation_threshold = 0.20
                    
                    if outlier_rejection_movement_min <= distance <= outlier_rejection_movement_max:
                        stability["rejection_count"] = stability.get("rejection_count", 0) + 1
                        if stability["rejection_count"] > 10:
                            stability["last_known_tvec"] = None
                            stability["last_known_rvec"] = None
                            stability["rejection_count"] = 0
                        continue
                    
                    # Check rotation
                    R_current, _ = cv2.Rodrigues(rvec)
                    R_check, _ = cv2.Rodrigues(check_rvec)
                    R_relative = R_current @ R_check.T
                    rvec_relative, _ = cv2.Rodrigues(R_relative)
                    rotation_angle = np.linalg.norm(rvec_relative)
                    
                    if rotation_angle > outlier_rejection_rotation_threshold:
                        stability["rejection_count"] = stability.get("rejection_count", 0) + 1
                        if stability["rejection_count"] > 10:
                            stability["last_known_tvec"] = None
                            stability["last_known_rvec"] = None
                            stability["rejection_count"] = 0
                        continue
                
                # Reset rejection count if measurement passes
                stability["rejection_count"] = 0
                
                # Update last tvec and increment hold counter
                stability["last_tvec"] = tvec_flat
                stability["last_frame"] = current_frame
                stability["hold_counter"] += 1

                if stability["hold_counter"] >= hold_required:
                    # Before confirming, test the transformation to catch bad camera poses early
                    test_quat = rvec_to_quat(rvec)
                    test_pos_world = transform_point_cam_to_world(tvec_flat, cam_pos, cam_quat)
                    
                    # Validate world frame Z (should be reasonable for tabletop objects)
                    world_z_min = -0.5  # Minimum world Z (0.5m below camera)
                    world_z_max = 1.5   # Maximum world Z (1.5m above camera)
                    if test_pos_world[2] < world_z_min or test_pos_world[2] > world_z_max:
                        # Outlier: World Z is unreasonable, reject confirmation
                        if talk and estimate_pose.debug_counter % 30 == 0:
                            print(f"[{marker_id}] Outlier: World Z={test_pos_world[2]:.3f}m out of range [{world_z_min:.3f}, {world_z_max:.3f}] - rejecting confirmation")
                        # Reset hold counter to prevent confirmation with bad transformation
                        stability["hold_counter"] = 0
                        continue
                    
                    # Transformation looks good, proceed with confirmation
                    stability["confirmed"] = True
                    # Only update confirmed baseline after confirmation
                    stability["confirmed_tvec"] = tvec_flat.copy()
                    stability["confirmed_rvec"] = rvec.copy()
                    # Also update last known pose for outlier checking after reset
                    stability["last_known_tvec"] = tvec_flat.copy()
                    stability["last_known_rvec"] = rvec.copy()
                    stability["frames_missing"] = 0  # Reset missing counter on detection

                    measured_quat = rvec_to_quat(rvec)
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    
                    # Use Kalman filter properly - let it handle the blending internally
                    # Only use manual blending for very noisy measurements
                    # Pass robot_moving flag to use minimum z when robot is stationary
                    kalman.correct(tvec_flat, rvec, robot_moving=robot_moving)
                    last_seen_frames[marker_id] = current_frame
                    
                    # Get the corrected state from Kalman filter
                    corrected_tvec, corrected_rvec = kalman.predict()
                    corrected_quat = rvec_to_quat(corrected_rvec)
                    
                    # Convert to world frame
                    marker_pos_world = transform_point_cam_to_world(corrected_tvec, cam_pos, cam_quat)
                    
                    marker_quat_world = transform_orientation_cam_to_world(corrected_quat, cam_quat)
                    
                    # Store last known pose in world frame for backtracking
                    stability["last_known_tvec_world"] = marker_pos_world.copy()
                    stability["last_known_rvec_world"] = marker_quat_world.copy()
                    
                    # Update camera pose tracking for motion compensation
                    stability["prev_cam_pos"] = cam_pos.copy() if cam_pos is not None else None
                    stability["prev_cam_quat"] = cam_quat.copy() if cam_quat is not None else None
                    
                    if talk and estimate_pose.debug_counter % 30 == 0:  # Only print every 30 calls
                        quality = stability.get("measurement_quality", 1.0)
                        print(f"[{marker_id}] Confirmed: t={tvec_flat}, r={rvec.flatten()}, quality={quality:.3f}")
                        print(f"[{marker_id}] WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
                elif talk and estimate_pose.debug_counter % 30 == 0:  # Only print every 30 calls
                    print(f"[{marker_id}] Holding: t={tvec_flat}, hold={stability['hold_counter']}")


    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        frames_since_last_seen = current_frame - last_seen
        stability["frames_missing"] = frames_since_last_seen
        
        # Ghost tracking - use longer duration when robot is stationary
        # When stationary: trust previous values more, extend ghost tracking significantly
        # When moving: use shorter duration but still allow backtracking
        if robot_moving:
            max_ghost_frames = 10  # When moving, allow 10 frames of ghost tracking
        else:
            max_ghost_frames = 50  # When stationary, allow 50 frames (much more trust)
        
        if frames_since_last_seen < max_ghost_frames:
            # Determine prediction mode
            new_mode = None
            if frames_since_last_seen < 5:
                new_mode = "kalman"
            else:
                new_mode = "last_known"
            
            # Track mode transitions for smooth blending
            current_mode = stability.get("prediction_mode")
            if current_mode != new_mode:
                stability["prev_prediction_mode"] = current_mode  # Store previous mode
                stability["prediction_mode"] = new_mode
                stability["mode_transition_frames"] = 0
            else:
                stability["mode_transition_frames"] += 1
            
            # Use Kalman prediction for short-term missing detections
            if frames_since_last_seen < 5:
                # Use Kalman prediction for very recent misses
                pred_tvec, pred_rvec = kalman.predict()
                pred_quat = rvec_to_quat(pred_rvec)
                marker_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                marker_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
                
                # Temporal consistency check: validate velocity and acceleration
                velocity = kalman.get_velocity()
                acceleration = kalman.get_acceleration()
                vel_magnitude = np.linalg.norm(velocity)
                acc_magnitude = np.linalg.norm(acceleration)
                
                # Physical constraints: objects on table don't move fast
                max_velocity = 0.5  # m/s - unrealistic for stationary objects
                max_acceleration = 2.0  # m/s^2 - unrealistic acceleration
                
                if vel_magnitude > max_velocity or acc_magnitude > max_acceleration:
                    # Prediction violates physical constraints - use last known pose instead
                    if stability.get("last_known_tvec_world") is not None:
                        marker_pos_world = stability["last_known_tvec_world"].copy()
                        marker_quat_world = stability["last_known_rvec_world"].copy()
                        if talk and estimate_pose.debug_counter % 30 == 0:
                            print(f"[{marker_id}] Kalman prediction violates constraints (vel={vel_magnitude:.3f}, acc={acc_magnitude:.3f}) - using last known")
            else:
                # For longer misses, use last known world pose (backtracking)
                # When arm is moving, we need to account for camera movement
                if stability.get("last_known_tvec_world") is not None:
                    # Objects don't move in world frame, so use last known world pose
                    marker_pos_world = stability["last_known_tvec_world"].copy()
                    marker_quat_world = stability["last_known_rvec_world"].copy()
                    
                    # Camera motion compensation: if camera moved, the object's position in camera frame changes
                    # but its world position stays constant. We already have world position, so we're good.
                    # However, if we want to improve Kalman predictions, we could transform the prediction
                    # from previous camera frame to current camera frame
                    if robot_moving and stability.get("prev_cam_pos") is not None and stability.get("prev_cam_quat") is not None:
                        # Camera has moved - we can optionally refine the prediction
                        # For now, we trust the world pose (objects don't move)
                        # The world pose is already correct, no compensation needed
                        pass
                else:
                    # Fallback to Kalman prediction if no world pose stored
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    marker_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                    marker_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
            
            # Smooth mode transitions: blend between modes during transition
            transition_frames = stability.get("mode_transition_frames", 0)
            prev_mode = stability.get("prev_prediction_mode")
            if transition_frames < 5 and prev_mode is not None and prev_mode != new_mode:
                # Blend between old and new predictions over 5 frames
                blend_alpha = min(1.0, transition_frames / 5.0)  # 0.0 to 1.0
                
                # Get prediction from previous mode
                if prev_mode == "kalman":
                    # Get Kalman prediction
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    old_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                    old_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
                else:
                    # Use last known
                    old_pos_world = stability.get("last_known_tvec_world", marker_pos_world)
                    old_quat_world = stability.get("last_known_rvec_world", marker_quat_world)
                
                # Blend positions (exponential smoothing during transition)
                marker_pos_world = (1.0 - blend_alpha) * old_pos_world + blend_alpha * marker_pos_world
                
                # Blend orientations using quaternion slerp
                from aruco_camera_localizer.geometric_functions import slerp_quat
                marker_quat_world = slerp_quat(old_quat_world, marker_quat_world, blend_alpha)
            
            # Orientation continuity check: ensure quaternion changes are smooth
            if stability.get("last_known_rvec_world") is not None:
                prev_quat = stability["last_known_rvec_world"]
                quat_diff = np.abs(np.dot(prev_quat, marker_quat_world))
                # Quaternion dot product close to 1 means similar orientation
                if quat_diff < 0.5:  # Large orientation jump detected
                    # Smooth the orientation change
                    from aruco_camera_localizer.geometric_functions import slerp_quat
                    marker_quat_world = slerp_quat(prev_quat, marker_quat_world, 0.3)  # 30% toward new, 70% old
            
            # DO NOT update last known world pose during ghost tracking
            # The world pose should only be updated when we have fresh detections
            # This prevents drift accumulation during ghost tracking
            # The stored world pose remains frozen at the last known good measurement
            
            if talk and estimate_pose.debug_counter % 30 == 0:  # Only print every 30 calls
                method = "Kalman" if frames_since_last_seen < 5 else ("Last known (moving)" if robot_moving else "Last known (stationary)")
                print(f"[{marker_id}] Ghost ({method}): missing={frames_since_last_seen} frames")
                print(f"[{marker_id}] GHOST WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
        else:
            # Reset confirmation after too many missed frames
            # Only reset if we've exceeded the maximum ghost tracking duration
            stability["confirmed"] = False
            # Keep last known pose for outlier checking even after reset
            # Only clear confirmed_tvec/rvec, but keep last_known_tvec/rvec
            if stability.get("confirmed_tvec") is not None:
                stability["last_known_tvec"] = stability["confirmed_tvec"].copy()
            if stability.get("confirmed_rvec") is not None:
                stability["last_known_rvec"] = stability["confirmed_rvec"].copy()
            stability["confirmed_tvec"] = None  # Reset confirmed baseline
            stability["confirmed_rvec"] = None  # Reset confirmed baseline
            stability["rejection_count"] = 0  # Reset rejection count
            stability["frames_missing"] = 0  # Reset missing counter
            kalman.reset()  # Reset the Kalman filter
            if talk:
                print(f"[{marker_id}] Lost tracking after {frames_since_last_seen} frames - resetting confirmation and Kalman filter")

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