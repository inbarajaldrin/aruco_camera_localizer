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
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, talk=True):
    max_movement = 0.023  # meters - 30mm threshold for overall movement
    max_rotation = 0.1   # radians - ~11 degrees threshold for rotation changes
    hold_required = 2    # frames it must persist - reduced for faster confirmation
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
                tvec_flat = tvec.flatten()
                
                # Always validate Z-range (not just first measurement)
                min_z = 0.05  # Minimum depth: 5cm
                max_z = 2.0   # Maximum depth: 2m
                if tvec_flat[2] < min_z or tvec_flat[2] > max_z:
                    # Outlier: Z out of reasonable range
                    if talk and estimate_pose.debug_counter % 30 == 0:
                        print(f"[{marker_id}] Outlier: Z={tvec_flat[2]:.3f}m out of range [{min_z:.3f}, {max_z:.3f}]")
                    continue
                
                # Reject outliers if movement or rotation is too large (only compare against confirmed measurements)
                if stability.get("confirmed_tvec") is not None and stability.get("confirmed_rvec") is not None:
                    # Check position change
                    distance = np.linalg.norm(tvec_flat - stability["confirmed_tvec"])
                    if distance > max_movement:
                        # Outlier: movement too large, skip this measurement
                        if talk and estimate_pose.debug_counter % 30 == 0:
                            print(f"[{marker_id}] Outlier: movement={distance*1000:.1f}mm > {max_movement*1000:.1f}mm")
                        continue
                    
                    # Check rotation change
                    # Convert rotation vectors to rotation matrices
                    R_current, _ = cv2.Rodrigues(rvec)
                    R_confirmed, _ = cv2.Rodrigues(stability["confirmed_rvec"])
                    # Compute relative rotation: R_relative = R_current @ R_confirmed.T
                    R_relative = R_current @ R_confirmed.T
                    # Convert back to rotation vector
                    rvec_relative, _ = cv2.Rodrigues(R_relative)
                    rotation_angle = np.linalg.norm(rvec_relative)
                    
                    if rotation_angle > max_rotation:
                        # Outlier: rotation too large, skip this measurement
                        if talk and estimate_pose.debug_counter % 30 == 0:
                            print(f"[{marker_id}] Outlier: rotation={np.degrees(rotation_angle):.1f}° > {np.degrees(max_rotation):.1f}°")
                        continue
                
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

                    measured_quat = rvec_to_quat(rvec)
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    
                    # Use Kalman filter properly - let it handle the blending internally
                    # Only use manual blending for very noisy measurements
                    kalman.correct(tvec_flat, rvec)
                    last_seen_frames[marker_id] = current_frame
                    
                    # Get the corrected state from Kalman filter
                    corrected_tvec, corrected_rvec = kalman.predict()
                    corrected_quat = rvec_to_quat(corrected_rvec)
                    
                    # Convert to world frame
                    marker_pos_world = transform_point_cam_to_world(corrected_tvec, cam_pos, cam_quat)
                    
                    marker_quat_world = transform_orientation_cam_to_world(corrected_quat, cam_quat)
                    
                    if talk and estimate_pose.debug_counter % 30 == 0:  # Only print every 30 calls
                        print(f"[{marker_id}] Confirmed: t={tvec_flat}, r={rvec.flatten()}")
                        print(f"[{marker_id}] WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
                elif talk and estimate_pose.debug_counter % 30 == 0:  # Only print every 30 calls
                    print(f"[{marker_id}] Holding: t={tvec_flat}, hold={stability['hold_counter']}")


    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        # Ghost tracking - only predict for a few frames after last detection
        if current_frame - last_seen < 5:  # Reduced from 15 to 5 frames
            pred_tvec, pred_rvec = kalman.predict()
            # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, pred_rvec, pred_tvec, marker_size * 0.5)
            if not current_frame == last_seen:
                # Convert to world frame
                pred_quat = rvec_to_quat(pred_rvec)
                marker_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                marker_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
                if talk and estimate_pose.debug_counter % 30 == 0:  # Only print every 30 calls
                    print(f"[{marker_id}] Ghost: t={pred_tvec}, r={pred_rvec}")
                    print(f"[{marker_id}] GHOST WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
        else:
            # Reset confirmation after too many missed frames
            stability["confirmed"] = False
            stability["confirmed_tvec"] = None  # Reset confirmed baseline
            stability["confirmed_rvec"] = None  # Reset confirmed baseline
            kalman.reset()  # Reset the Kalman filter
            if talk:
                print(f"[{marker_id}] Lost tracking - resetting confirmation and Kalman filter")

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