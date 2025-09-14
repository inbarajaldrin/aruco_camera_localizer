import cv2
import cv2.aruco as aruco
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world
from max_camera_localizer.kalman_functions import QuaternionKalman
from max_camera_localizer.geometric_functions import transform_points_world_to_img, slerp_quat, quat_to_rvec, complete_triangle, pick_best_candidate
from max_camera_localizer.object_frame_definitions import define_body_frame_allen_key, define_body_frame_wrench

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
    max_movement = 0.10  # meters - increased for more tolerance
    hold_required = 3    # frames it must persist - reduced for faster confirmation
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
                tvec_flat = tvec.flatten()
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
                    if talk:
                        print(f"[{marker_id}] Confirmed: t={tvec_flat}, r={rvec.flatten()}")
                        print(f"[{marker_id}] WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
                elif talk:
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
                if talk:
                    print(f"[{marker_id}] Ghost: t={pred_tvec}, r={pred_rvec}")
                    print(f"[{marker_id}] GHOST WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
        else:
            # Reset confirmation after too many missed frames
            stability["confirmed"] = False
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