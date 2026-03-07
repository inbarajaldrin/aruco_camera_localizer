import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from aruco_camera_localizer.localizer_bridge import LocalizerBridge
from aruco_camera_localizer.config_loader import get_config
from aruco_camera_localizer.geometric_functions import transform_points_world_to_img
from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines
import threading
import rclpy
import argparse
import json
import os
from ultralytics import YOLOE

# Load configuration from YAML
config = get_config()

c_width = config.get_camera_width()
c_height = config.get_camera_height()
c_hfov = config.get_camera_hfov()
c_vfov = config.get_camera_vfov()

# OpenCV to camera frame transformation
OPENCV_TO_CAMERA_QUAT = config.get_opencv_to_camera_quaternion()
print(f"OpenCV-to-Camera quaternion: {OPENCV_TO_CAMERA_QUAT}")

# Ground plane Z offset from config (may be None if not configured for this robot)
_GROUND_PLANE_Z_CONFIG = config.get_ground_plane_z_offset()

fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

# Previous cuboid params for temporal seeding: {color_name: [(x, y, yaw, w, l), ...]}
_prev_cuboid_params = {}

def _lookup_prev_cuboid(color_name, world_point, threshold=0.02):
    """Find previous frame's cuboid params for the closest object of the same color.

    Returns [x, y, yaw, w, l] or None. Threshold in meters (default 20mm).
    """
    prev_list = _prev_cuboid_params.get(color_name)
    if not prev_list:
        return None
    best_dist = threshold
    best_params = None
    for entry in prev_list:
        dist = np.linalg.norm(world_point[:2] - np.array([entry[0], entry[1]]))
        if dist < best_dist:
            best_dist = dist
            best_params = entry
    return list(best_params) if best_params is not None else None

def convert_2d_orientation_to_quaternion(orientation_angle, cam_quat, opencv_to_camera_quat):
    """
    Convert 2D orientation angle from PCA to 3D quaternion in world frame.
    
    Transformation chain:
    1. 2D angle → rotation in OpenCV frame (Z-axis rotation)
    2. OpenCV frame → Camera frame (opencv_to_camera)
    3. Camera frame → World/Base frame (camera quaternion)
    
    Args:
        orientation_angle: 2D orientation angle in radians from PCA
        cam_quat: Camera quaternion in world frame [x, y, z, w]
        opencv_to_camera_quat: OpenCV to camera frame quaternion [x, y, z, w]
    
    Returns:
        quaternion: 3D quaternion in world frame [x, y, z, w]
    """
    # Step 1: Create rotation around Z-axis in OpenCV frame
    z_rotation = R.from_euler('z', orientation_angle)
    opencv_quat = z_rotation.as_quat()
    
    # Step 2: Transform from OpenCV frame to camera frame
    opencv_to_cam = R.from_quat(opencv_to_camera_quat)
    camera_orientation = opencv_to_cam * R.from_quat(opencv_quat)
    
    # Step 3: Transform from camera frame to world frame
    cam_rotation = R.from_quat(cam_quat)
    world_orientation = cam_rotation * camera_orientation
    
    return world_orientation.as_quat()

CAMERA_MATRIX = config.get_camera_matrix()

# YOLO detection settings - only hand detection
YOLO_PROMPTS = ["hand"]
YOLO_PROMPT_MAP = {
    "hand": "hand"
}

# Global variables for dynamic YOLO prompt management
yolo_prompts_lock = threading.Lock()
current_yolo_prompts = YOLO_PROMPTS.copy()
current_yolo_prompt_map = YOLO_PROMPT_MAP.copy()

# Generic color for all YOLO detections (cyan in BGR)
GENERIC_COLOR = (255, 255, 0)


trackers = {}
# Store previous frame's objects for position-based matching
previous_yolo_objects = {}  # color_name -> list of {name, position}

def start_ros_node(camera_topic='/camera/image_raw', depth_topic=None):
    rclpy.init()
    node = LocalizerBridge(camera_topic=camera_topic, depth_topic=depth_topic)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def get_yolo_prompts():
    """Get current YOLO prompts (thread-safe)"""
    with yolo_prompts_lock:
        return current_yolo_prompts.copy(), current_yolo_prompt_map.copy()

def update_yolo_prompts(prompts, prompt_map, yolo_model=None):
    """Update YOLO prompts and prompt mapping (thread-safe)"""
    with yolo_prompts_lock:
        global current_yolo_prompts, current_yolo_prompt_map
        current_yolo_prompts = prompts.copy()
        current_yolo_prompt_map = prompt_map.copy()
        print(f"Updated YOLO prompts: {current_yolo_prompts}")
        print(f"Updated prompt mapping: {current_yolo_prompt_map}")
        
        # Update YOLO model if provided
        if yolo_model is not None:
            try:
                yolo_model.set_classes(current_yolo_prompts, yolo_model.get_text_pe(current_yolo_prompts))
                print(f"YOLO model updated with new prompts")
            except Exception as e:
                print(f"Failed to update YOLO model: {e}")

def yolo_prompts_callback(msg, yolo_model=None):
    """Topic callback for real-time YOLO prompt updates"""
    try:
        data = json.loads(msg.data)
        prompts = data.get('prompts', [])
        prompt_map = data.get('prompt_map', {})
        
        update_yolo_prompts(prompts, prompt_map, yolo_model)
        print(f"YOLO prompts updated via topic: {prompts}")
        
    except Exception as e:
        print(f"Failed to update YOLO prompts from topic: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with YOLO detection.")
    parser.add_argument("--camera-topic", type=str, default="/camera/image_raw",
                        help="ROS2 topic to subscribe for camera images (default: /camera/image_raw)")
    parser.add_argument("--depth-topic", type=str, default=None,
                        help="ROS2 topic to subscribe for depth images (optional, uses config distance if not provided)")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    parser.add_argument("--yolo-mode", type=str, default="prompt-set",
                        help="YOLO mode: 'prompt-set' for prompted detection (default: prompt-set)")
    parser.add_argument("--yolo-model", type=str, default="aruco_camera_localizer/yoloe-11s-seg.pt",
                        help="YOLO model path (default: aruco_camera_localizer/yoloe-11s-seg.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.4,
                        help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--yolo-prompts", type=str, nargs='+', 
                        default=["hand"],
                        help="YOLO detection prompts (default: hand)")
    parser.add_argument("--yolo-prompt-map", type=str, nargs='+',
                        help="Custom prompt mapping for prompts (format: prompt1:color1 prompt2:color2)")
    parser.add_argument("--headless", action='store_true',
                        help="Run without GUI window (no cv2.imshow). Annotated frames still published to /annotated_image.")
    parser.add_argument("--additional-z-offset", type=float, default=0.0,
                        help="Additional Z offset added to ground_plane_z_offset from config (meters). For debug/tuning.")
    # Use parse_known_args to avoid conflicts with ROS args
    args, unknown = parser.parse_known_args()
    return args, unknown

def pick_closest_blob(blobs, last_position):
    if not blobs:
        return None
    if last_position is None:
        return blobs[0]
    blobs_np = np.array(blobs)
    distances = np.linalg.norm(blobs_np - last_position, axis=1)
    closest_idx = np.argmin(distances)
    return blobs[closest_idx]

def extract_object_roi(image, box):
    """Extract the region of interest for the detected object"""
    x1, y1, x2, y2 = map(int, box)
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1)

def find_centroid_and_orientation_moments(roi, mask_roi=None):
    """Find object centroid and orientation using cv2.moments on filled mask.

    Uses image moments (area-weighted) which is mathematically equivalent to
    PCA on all filled pixels, but more robust than PCA on contour boundary
    points (which suffers from non-uniform boundary sampling bias).

    Args:
        roi: BGR image crop of the bounding box
        mask_roi: Optional binary mask from segmentation model (same size as roi)

    Returns:
        (centroid_x, centroid_y, angle, elongation) in ROI coordinates, or None
    """
    try:
        if mask_roi is not None and mask_roi.any():
            binary = mask_roi.astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use largest connected component only (avoids centroid landing in
        # empty space between disconnected mask fragments)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        component_mask = np.zeros_like(binary)
        cv2.drawContours(component_mask, [largest], -1, 255, -1)

        M = cv2.moments(component_mask, binaryImage=True)
        if M['m00'] < 1.0:
            return None

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # Orientation from central moments (equivalent to PCA on filled pixels)
        mu20 = M['mu20'] / M['m00']
        mu11 = M['mu11'] / M['m00']
        mu02 = M['mu02'] / M['m00']
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        # Elongation ratio (eigenvalue ratio of covariance matrix)
        common = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
        lambda1 = 0.5 * (mu20 + mu02 + common)
        lambda2 = max(0.5 * (mu20 + mu02 - common), 1e-10)
        elongation = lambda1 / lambda2

        return cx, cy, theta, elongation

    except Exception:
        return None



def _backproject_rect_to_table(mask_roi, bx1, by1, camera_matrix,
                               opencv_to_camera_quat, cam_quat, cam_pos, ground_plane_z):
    """Back-project minAreaRect corners onto the table plane for perspective-correct centroid.

    The moments centroid on a 2D mask is biased toward the camera due to perspective
    (the near side of an object occupies more pixels). By projecting the oriented
    rectangle corners onto the known table plane and averaging in 3D, this bias
    is eliminated.

    Returns:
        (centroid_3d, orientation_angle_world) or None on failure.
        centroid_3d: np.array([x, y, z]) on the table plane.
        orientation_angle_world: yaw of the object's long axis in world XY (radians).
    """
    try:
        binary = (mask_roi > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            return None

        rect = cv2.minAreaRect(largest)
        corners_roi = cv2.boxPoints(rect).astype(np.float64)  # (4, 2)

        # Offset to full-image pixel coordinates
        corners_img = corners_roi + np.array([bx1, by1], dtype=np.float64)

        # Precompute transforms
        K_inv = np.linalg.inv(camera_matrix)
        R_o2c = R.from_quat(opencv_to_camera_quat)
        R_wc = R.from_quat(cam_quat).as_matrix()
        cam_origin = np.array(cam_pos, dtype=np.float64)

        # Ray-table intersect for each of the 4 corners
        table_points = []
        for u, v in corners_img:
            ray_opencv = K_inv @ np.array([u, v, 1.0])
            ray_cam = R_o2c.apply(ray_opencv)
            ray_world = R_wc @ ray_cam

            if abs(ray_world[2]) < 1e-6:
                return None  # ray nearly parallel to table
            t = (ground_plane_z - cam_origin[2]) / ray_world[2]
            if t <= 0:
                return None  # ray points away from table
            table_points.append(cam_origin + ray_world * t)

        table_points = np.array(table_points)  # (4, 3)
        centroid_3d = table_points.mean(axis=0)

        # Orientation: atan2 of the longest edge projected onto XY
        best_len, best_edge = 0.0, None
        for i in range(4):
            edge = table_points[(i + 1) % 4] - table_points[i]
            elen = np.linalg.norm(edge[:2])
            if elen > best_len:
                best_len, best_edge = elen, edge
        orientation_angle_world = np.arctan2(best_edge[1], best_edge[0])

        return centroid_3d, orientation_angle_world
    except Exception:
        return None


def _backproject_mask_to_pointcloud(mask_roi, depth_roi, bx1, by1, camera_matrix,
                                     opencv_to_camera_quat, cam_quat, cam_pos):
    """Back-project all masked depth pixels to a 3D point cloud in world frame.

    Args:
        mask_roi: Boolean mask (H, W) for the object within the bounding box.
        depth_roi: Depth image crop (H, W) aligned with mask_roi.
        bx1, by1: Top-left corner of the bounding box in full image coords.
        camera_matrix: 3x3 intrinsic matrix K.
        opencv_to_camera_quat: [x,y,z,w] quaternion from OpenCV to camera frame.
        cam_quat: [x,y,z,w] camera quaternion in world frame.
        cam_pos: [x,y,z] camera position in world frame.

    Returns:
        np.ndarray (N, 3) of world-frame 3D points, or None if too few valid pixels.
    """
    # Erode mask to strip noisy boundary pixels (depth edges bleed)
    mask_u8 = mask_roi.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    # Fall back to original mask if erosion kills everything
    if eroded.any():
        mask_roi = eroded > 0

    # Get local pixel coords where mask is True
    v_local, u_local = np.where(mask_roi)
    if len(v_local) == 0:
        return None

    # Sample depth at mask pixels
    depths = depth_roi[v_local, u_local].astype(np.float64)

    # Handle uint16 encoding (millimeters)
    if depth_roi.dtype == np.uint16:
        depths = depths / 1000.0

    # Keep only finite & positive depths
    valid = np.isfinite(depths) & (depths > 0)
    if valid.sum() < 10:
        return None

    v_local = v_local[valid]
    u_local = u_local[valid]
    depths = depths[valid]

    # Convert to full-image pixel coords
    u = u_local.astype(np.float64) + bx1
    v = v_local.astype(np.float64) + by1

    # Build homogeneous pixel coords (3, N)
    ones = np.ones_like(u)
    pixels = np.stack([u, v, ones], axis=0)  # (3, N)

    # Rays in OpenCV frame: K_inv @ pixels → (3, N)
    K_inv = np.linalg.inv(camera_matrix)
    rays_opencv = K_inv @ pixels  # (3, N)

    # Transform to camera frame (vectorized)
    R_o2c = R.from_quat(opencv_to_camera_quat)
    rays_cam = R_o2c.apply(rays_opencv.T)  # (N, 3)

    # Transform to world frame
    R_wc = R.from_quat(cam_quat).as_matrix()  # (3, 3)
    rays_world = (R_wc @ rays_cam.T).T  # (N, 3)

    # Scale rays by depth (Z-depth interpretation: depth is along camera Z-axis)
    cam_origin = np.array(cam_pos, dtype=np.float64)
    points_world = cam_origin + rays_world * depths[:, np.newaxis]

    return points_world


def _fit_cuboid_obb(points_3d):
    """Fit an oriented bounding box (OBB) to a 3D point cloud via PCA.

    Applies IQR-based outlier removal along each PCA axis so that noisy
    mask edges / table bleed pixels don't inflate the cuboid.

    Args:
        points_3d: np.ndarray (N, 3) of world-frame 3D points.

    Returns:
        (centroid, quaternion, dimensions) or None.
        centroid: np.array([x, y, z])
        quaternion: np.array([x, y, z, w]) orientation of the OBB axes
        dimensions: np.array([w, h, d]) extents along each PCA axis (meters)
    """
    if points_3d is None or len(points_3d) < 10:
        return None

    try:
        # --- IQR outlier removal using PCA axes (always) ---
        pca = PCA(n_components=3)
        pca.fit(points_3d)
        pca_axes = pca.components_  # (3, 3)

        centroid_raw = points_3d.mean(axis=0)
        centered = points_3d - centroid_raw
        projections = centered @ pca_axes.T  # (N, 3)

        inlier_mask = np.ones(len(points_3d), dtype=bool)
        for ax in range(3):
            q1, q3 = np.percentile(projections[:, ax], [25, 75])
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            inlier_mask &= (projections[:, ax] >= lo) & (projections[:, ax] <= hi)

        inlier_pts = points_3d[inlier_mask]
        if len(inlier_pts) < 10:
            inlier_pts = points_3d

        centroid = inlier_pts.mean(axis=0)

        # --- OBB axes from PCA on inliers ---
        pca.fit(inlier_pts)
        axes = pca.components_
        rot_matrix = axes.T
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[:, 2] = -rot_matrix[:, 2]
            axes = rot_matrix.T

        # Dimensions from inlier projections
        centered = inlier_pts - centroid
        projections = centered @ axes.T
        dimensions = projections.ptp(axis=0)

        quaternion = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

        return centroid, quaternion, dimensions

    except Exception:
        return None


def _project_cuboid_corners(center, quat, dims, cam_pos, cam_quat,
                             camera_matrix, opencv_to_camera_quat):
    """Project 8 cuboid corners to 2D image pixels.

    Projection chain: world → camera frame → OpenCV frame → image (K).

    Returns:
        (8, 2) float64 array of (u, v) pixel coords, or None if any corner is behind camera.
    """
    half = np.asarray(dims, dtype=np.float64) / 2.0
    signs = np.array([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=np.float64)
    local_corners = signs * half  # (8, 3)

    rot = R.from_quat(quat).as_matrix()
    world_corners = (rot @ local_corners.T).T + np.asarray(center, dtype=np.float64)

    R_wc_inv = R.from_quat(cam_quat).inv().as_matrix()
    cam_origin = np.asarray(cam_pos, dtype=np.float64)
    cam_corners = (R_wc_inv @ (world_corners - cam_origin).T).T

    R_c2o = R.from_quat(opencv_to_camera_quat).inv().as_matrix()
    opencv_corners = (R_c2o @ cam_corners.T).T

    z_vals = opencv_corners[:, 2]
    if np.any(z_vals <= 0.01):
        return None

    us = camera_matrix[0, 0] * opencv_corners[:, 0] / z_vals + camera_matrix[0, 2]
    vs = camera_matrix[1, 1] * opencv_corners[:, 1] / z_vals + camera_matrix[1, 2]
    return np.column_stack([us, vs])  # (8, 2) float64


def draw_cuboid_wireframe(image, cuboid_center, cuboid_quaternion, cuboid_dimensions,
                          cam_pos, cam_quat, camera_matrix, opencv_to_camera_quat,
                          color=(0, 255, 255), thickness=2):
    """Draw a 3D oriented bounding box wireframe projected onto the image."""
    pts_2d = _project_cuboid_corners(cuboid_center, cuboid_quaternion, cuboid_dimensions,
                                     cam_pos, cam_quat, camera_matrix, opencv_to_camera_quat)
    if pts_2d is None:
        return

    pts_int = pts_2d.astype(int)
    pts_list = list(map(tuple, pts_int))

    # 12 edges of a cuboid
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # along axis 2
        (0, 2), (1, 3), (4, 6), (5, 7),  # along axis 1
        (0, 4), (1, 5), (2, 6), (3, 7),  # along axis 0
    ]
    for i, j in edges:
        cv2.line(image, pts_list[i], pts_list[j], color, thickness)


def _fit_cuboid_from_silhouette(mask_roi, bx1, by1, camera_matrix,
                                 opencv_to_camera_quat, cam_quat, cam_pos,
                                 ground_plane_z, known_height=None,
                                 bbox=None, prev_params=None):
    """Fit 3D cuboid by maximizing IoU between projected silhouette and seg mask.

    Instead of relying on depth data and PCA (which misaligns with object edges),
    this optimizes the 3D cuboid pose so its 2D projection best matches the
    segmentation mask — an analysis-by-synthesis approach.

    Args:
        mask_roi: Binary mask (H, W) of the detected object within its bbox.
        bx1, by1: Top-left corner of bbox in full image coords.
        camera_matrix: 3x3 intrinsic matrix K.
        opencv_to_camera_quat: [x,y,z,w] quaternion from OpenCV to camera frame.
        cam_quat: [x,y,z,w] camera quaternion in world frame.
        cam_pos: [x,y,z] camera position in world frame.
        ground_plane_z: Known Z height of the ground plane in arm base frame.
        known_height: Object height in meters (default 0.011 for lego bricks).
        bbox: (x1, y1, x2, y2) YOLOE detection bbox for containment constraint.
        prev_params: Optional [x, y, yaw, w, l] from previous frame for temporal seeding.

    Returns:
        (center, quaternion, dimensions) or None on failure.
    """
    try:
        h = known_height if known_height is not None else 0.011

        # --- Initialize from back-projected minAreaRect ---
        rect_result = _backproject_rect_to_table(
            mask_roi, bx1, by1, camera_matrix,
            opencv_to_camera_quat, cam_quat, cam_pos, ground_plane_z)
        if rect_result is None:
            return None

        centroid_3d, yaw0 = rect_result

        # Get initial width/length from the back-projected corners
        binary = (mask_roi > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        corners_roi = cv2.boxPoints(rect).astype(np.float64)

        # Back-project corners to table to get real-world edge lengths
        corners_img = corners_roi + np.array([bx1, by1], dtype=np.float64)
        K_inv = np.linalg.inv(camera_matrix)
        R_o2c = R.from_quat(opencv_to_camera_quat)
        R_wc = R.from_quat(cam_quat).as_matrix()
        cam_origin = np.array(cam_pos, dtype=np.float64)

        table_pts = []
        for u, v in corners_img:
            ray_opencv = K_inv @ np.array([u, v, 1.0])
            ray_cam = R_o2c.apply(ray_opencv)
            ray_world = R_wc @ ray_cam
            if abs(ray_world[2]) < 1e-6:
                return None
            t = (ground_plane_z - cam_origin[2]) / ray_world[2]
            if t <= 0:
                return None
            table_pts.append(cam_origin + ray_world * t)
        table_pts = np.array(table_pts)

        # Edge lengths from consecutive corners
        edge_lens = [np.linalg.norm(table_pts[(i+1)%4] - table_pts[i])
                     for i in range(4)]
        w0 = max(edge_lens[0], edge_lens[2])  # opposing edges ~equal
        l0 = max(edge_lens[1], edge_lens[3])

        x0, y0 = centroid_3d[0], centroid_3d[1]

        # --- Build reference mask in a padded ROI for IoU comparison ---
        pad = 20
        mh, mw = mask_roi.shape[:2]
        roi_x1 = max(bx1 - pad, 0)
        roi_y1 = max(by1 - pad, 0)
        roi_x2 = bx1 + mw + pad
        roi_y2 = by1 + mh + pad
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        ref_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        # Place mask_roi into the padded ROI
        ox = bx1 - roi_x1
        oy = by1 - roi_y1
        ref_binary = (mask_roi > 0.5).astype(np.uint8)
        ref_mask[oy:oy+mh, ox:ox+mw] = ref_binary
        ref_area = float(ref_mask.sum())
        if ref_area < 5:
            return None

        # --- Precompute constant transforms for the objective ---
        R_wc_inv = R.from_quat(cam_quat).inv().as_matrix()
        R_c2o = R.from_quat(opencv_to_camera_quat).inv().as_matrix()
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        _signs = np.array([
            [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
            [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
        ], dtype=np.float64)

        # Bbox containment bounds (full image coords)
        if bbox is not None:
            bb_x1, bb_y1, bb_x2, bb_y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            bb_diag = max(np.hypot(bb_x2 - bb_x1, bb_y2 - bb_y1), 1.0)
        else:
            bb_x1 = bb_y1 = bb_x2 = bb_y2 = bb_diag = None

        def _objective(params):
            """Negative IoU with bbox overflow penalty."""
            x, y, yaw, w, l = params
            if w <= 0.001 or l <= 0.001:
                return 0.0

            center = np.array([x, y, ground_plane_z + h / 2.0])
            rot = R.from_euler('z', yaw).as_matrix()
            half = np.array([w, l, h]) / 2.0
            local = _signs * half
            world_corners = (rot @ local.T).T + center

            cam_corners = (R_wc_inv @ (world_corners - cam_origin).T).T
            opencv_corners = (R_c2o @ cam_corners.T).T

            z_vals = opencv_corners[:, 2]
            if np.any(z_vals <= 0.01):
                return 0.0

            us = fx * opencv_corners[:, 0] / z_vals + cx
            vs = fy * opencv_corners[:, 1] / z_vals + cy

            # Shift to ROI coordinates
            pts = np.column_stack([us - roi_x1, vs - roi_y1]).astype(np.int32)

            hull = cv2.convexHull(pts)
            rendered = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.fillConvexPoly(rendered, hull, 1)

            intersection = float(np.logical_and(rendered, ref_mask).sum())
            union = float(np.logical_or(rendered, ref_mask).sum())
            if union < 1:
                return 0.0
            iou = intersection / union

            # Bbox containment penalty: penalize corners outside the YOLOE bbox
            if bb_diag is not None:
                overflow = 0.0
                overflow += np.sum(np.maximum(bb_x1 - us, 0.0))  # left overflow
                overflow += np.sum(np.maximum(us - bb_x2, 0.0))  # right overflow
                overflow += np.sum(np.maximum(bb_y1 - vs, 0.0))  # top overflow
                overflow += np.sum(np.maximum(vs - bb_y2, 0.0))  # bottom overflow
                # Normalize by bbox diagonal so penalty is ~0-1 scale
                penalty = overflow / bb_diag
                iou = iou - 0.5 * penalty

            return -iou

        # --- Optimize with Nelder-Mead ---
        # Try previous frame's params first (temporal seeding for stability)
        seeds = []
        if prev_params is not None:
            seeds.append(list(prev_params))
        seeds.append([x0, y0, yaw0, w0, l0])

        best_result = None
        best_iou = -1.0
        for seed in seeds:
            result = minimize(_objective, seed, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            iou = -result.fun
            if iou > best_iou:
                best_iou = iou
                best_result = result

        # Try extra yaw seeds if IoU is still poor
        if best_iou < 0.3:
            for yaw_offset in [np.pi/4, np.pi/2]:
                alt_params = [x0, y0, yaw0 + yaw_offset, w0, l0]
                alt_result = minimize(_objective, alt_params, method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                alt_iou = -alt_result.fun
                if alt_iou > best_iou:
                    best_iou = alt_iou
                    best_result = alt_result

        if best_iou < 0.1:
            return None  # fit too poor to use

        # --- Build output in same format as _fit_cuboid_obb ---
        xf, yf, yaw_f, wf, lf = best_result.x
        center_out = np.array([xf, yf, ground_plane_z + h / 2.0])
        quat_out = R.from_euler('z', yaw_f).as_quat()  # [x, y, z, w]
        dims_out = np.array([wf, lf, h])

        return center_out, quat_out, dims_out

    except Exception:
        return None


def draw_cuboid_orientation_axes(image, cuboid_center, cuboid_quaternion, cuboid_dimensions,
                                 cam_pos, cam_quat, camera_matrix, opencv_to_camera_quat,
                                 show_major=True, show_minor=True):
    """Draw orientation axes from the cuboid fit, projected into the image.

    - Major axis (larger horizontal dim): LIGHT BLUE (cyan) line + yellow arrowhead
    - Minor axis (smaller horizontal dim): MAGENTA line + yellow arrowhead
    """
    dims = np.asarray(cuboid_dimensions, dtype=np.float64)
    rot = R.from_quat(cuboid_quaternion).as_matrix()
    center = np.asarray(cuboid_center, dtype=np.float64)

    # dims[0] = w (axis 0), dims[1] = l (axis 1), dims[2] = h (axis 2, vertical)
    if dims[0] >= dims[1]:
        major_idx, minor_idx = 0, 1
    else:
        major_idx, minor_idx = 1, 0

    # Build world-space axis endpoints for enabled axes
    axes_to_draw = []  # (index_into_colors, p_pos, p_neg)
    for i, (idx, enabled) in enumerate([(major_idx, show_major), (minor_idx, show_minor)]):
        if not enabled:
            continue
        direction = rot[:, idx]
        half_len = dims[idx] / 2.0
        p_pos = center + direction * half_len * 1.5
        p_neg = center - direction * half_len * 1.5
        axes_to_draw.append((i, p_pos, p_neg))

    if not axes_to_draw:
        return

    # Project to image
    R_wc_inv = R.from_quat(cam_quat).inv().as_matrix()
    R_c2o = R.from_quat(opencv_to_camera_quat).inv().as_matrix()
    cam_origin = np.asarray(cam_pos, dtype=np.float64)

    def _project_point(pt_world):
        cam_pt = R_wc_inv @ (pt_world - cam_origin)
        opencv_pt = R_c2o @ cam_pt
        if opencv_pt[2] <= 0.01:
            return None
        u = camera_matrix[0, 0] * opencv_pt[0] / opencv_pt[2] + camera_matrix[0, 2]
        v = camera_matrix[1, 1] * opencv_pt[1] / opencv_pt[2] + camera_matrix[1, 2]
        return int(u), int(v)

    center_px = _project_point(center)
    if center_px is None:
        return

    line_colors = [
        (255, 255, 0),  # LIGHT BLUE (cyan in BGR) = major axis
        (255, 0, 255),  # MAGENTA = minor axis
    ]
    arrow_color = (0, 255, 255)  # YELLOW arrowhead

    for color_idx, p_pos, p_neg in axes_to_draw:
        px_pos = _project_point(p_pos)
        px_neg = _project_point(p_neg)
        if px_pos is None or px_neg is None:
            continue
        cv2.line(image, px_neg, px_pos, line_colors[color_idx], 3)
        # Fixed-size yellow arrowhead (30px) past the line end
        dx = px_pos[0] - center_px[0]
        dy = px_pos[1] - center_px[1]
        length = max((dx**2 + dy**2)**0.5, 1e-3)
        arrow_end = (int(px_pos[0] + 30 * dx / length),
                     int(px_pos[1] + 30 * dy / length))
        cv2.arrowedLine(image, px_pos, arrow_end, arrow_color, 3, tipLength=0.5)


def detect_yolo_blobs(frame, yolo_model, camera_matrix, cam_pos, cam_quat, yolo_prompts, yolo_prompt_map, opencv_to_camera_quat, depth_image=None, bridge_node=None, conf_threshold=0.4, nms_threshold=0.3, ground_plane_z=None):
    """Detect objects using YOLO and convert to world points, grouped by color"""
    detected_color_points = {}
    detection_metadata = []  # Store boxes, orientations, and other metadata
    
    # Run YOLO detection
    results = yolo_model.predict(frame, verbose=False, conf=conf_threshold)
    
    # Extract segmentation masks if available
    seg_masks = None
    if results[0].masks is not None:
        seg_masks = results[0].masks.data.cpu().numpy()  # (N, H, W) binary masks

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_raw = results[0].boxes.xyxy.cpu().numpy()
        scores_raw = results[0].boxes.conf.cpu().numpy()
        class_ids_raw = results[0].boxes.cls.cpu().numpy().astype(int)

        # Apply NMS
        boxes_nms = []
        for box in boxes_raw:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            boxes_nms.append([x1, y1, w, h])

        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_raw.tolist(), conf_threshold, nms_threshold)

        if len(indices) > 0:
            indices = indices.flatten()

            # Process each detection
            for idx in indices:
                box = boxes_raw[idx]
                score = scores_raw[idx]
                class_id = int(class_ids_raw[idx])

                # Get class name and map to color
                class_name = yolo_prompts[class_id] if class_id < len(yolo_prompts) else f"class_{class_id}"
                color_name = yolo_prompt_map.get(class_name, class_name)
                # Replace spaces with underscores in color_name for object naming
                color_name = color_name.replace(' ', '_')

                x1, y1, x2, y2 = box
                bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)

                # Extract seg mask for this detection (if available)
                mask_roi = None
                if seg_masks is not None and idx < len(seg_masks):
                    full_mask = seg_masks[idx]
                    if full_mask.shape != (frame.shape[0], frame.shape[1]):
                        full_mask = cv2.resize(full_mask, (frame.shape[1], frame.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                    mask_roi = full_mask[by1:by2, bx1:bx2] > 0.5

                # Centroid + orientation from image moments on filled mask
                roi, roi_offset = extract_object_roi(frame, box)
                moments_result = find_centroid_and_orientation_moments(roi, mask_roi)
                if moments_result is not None:
                    cx_roi, cy_roi, orientation_angle, elongation = moments_result
                    # Convert ROI-local centroid to full-image pixel coordinates
                    center_x = bx1 + cx_roi
                    center_y = by1 + cy_roi
                else:
                    # Fallback to bbox center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    orientation_angle = None

                # Extract depth from bounding box region (not just center pixel)
                actual_distance = None
                if depth_image is not None:
                    if mask_roi is not None and mask_roi.any():
                        # Sample depth only at segmentation mask pixels (actual object surface)
                        depth_bbox = depth_image[by1:by2, bx1:bx2]
                        masked_depth = depth_bbox[mask_roi].flatten()
                        valid = masked_depth[np.isfinite(masked_depth) & (masked_depth > 0)]
                    else:
                        # Fallback: inner 60% of bbox (no seg mask available)
                        bw = bx2 - bx1
                        bh = by2 - by1
                        margin_x = int(bw * 0.2)
                        margin_y = int(bh * 0.2)
                        roi_x1 = max(0, bx1 + margin_x)
                        roi_y1 = max(0, by1 + margin_y)
                        roi_x2 = min(depth_image.shape[1], bx2 - margin_x)
                        roi_y2 = min(depth_image.shape[0], by2 - margin_y)
                        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                            depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2].flatten()
                            valid = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
                        else:
                            valid = np.array([])

                    if len(valid) > 0:
                        if depth_image.dtype == np.float32:
                            actual_distance = float(np.median(valid))
                        else:
                            actual_distance = float(np.median(valid)) / 1000.0

                # Step 1: Ray direction in OpenCV frame (Z=1 at unit depth)
                pixel = np.array([center_x, center_y, 1.0])
                ray_opencv = np.linalg.inv(camera_matrix) @ pixel

                # Step 2: Transform from OpenCV frame to camera frame
                R_opencv_to_cam = R.from_quat(opencv_to_camera_quat)
                ray_cam = R_opencv_to_cam.apply(ray_opencv)

                # Step 3: Transform ray to world frame
                R_wc = R.from_quat(cam_quat).as_matrix()
                ray_world = R_wc @ ray_cam
                cam_origin_world = np.array(cam_pos)

                # Step 4: Place object along ray
                cuboid_result = None

                # Look up previous cuboid params for temporal seeding
                _prev_seed = None
                if ground_plane_z is not None and abs(ray_world[2]) > 1e-6:
                    t_approx = (ground_plane_z - cam_origin_world[2]) / ray_world[2]
                    if t_approx > 0:
                        approx_world = cam_origin_world + ray_world * t_approx
                        _prev_seed = _lookup_prev_cuboid(color_name, approx_world)

                if actual_distance is not None or ground_plane_z is None:
                    # Have depth data (or no ground_plane_z configured)
                    # Try 3D cuboid fitting via depth point cloud + PCA
                    if mask_roi is not None and depth_image is not None:
                        depth_roi_full = depth_image[by1:by2, bx1:bx2]
                        pts_3d = _backproject_mask_to_pointcloud(
                            mask_roi, depth_roi_full, bx1, by1, camera_matrix,
                            opencv_to_camera_quat, cam_quat, cam_pos)
                        if pts_3d is not None and len(pts_3d) >= 10:
                            cuboid_result = _fit_cuboid_obb(pts_3d)

                    # When ground_plane_z available, try silhouette fitting for tighter wireframes
                    if ground_plane_z is not None and mask_roi is not None:
                        sil_result = _fit_cuboid_from_silhouette(
                            mask_roi, bx1, by1, camera_matrix,
                            opencv_to_camera_quat, cam_quat, cam_pos, ground_plane_z,
                            bbox=(bx1, by1, bx2, by2),
                            prev_params=_prev_seed)
                        if sil_result is not None:
                            cuboid_result = sil_result

                    if cuboid_result is not None:
                        point_world = cuboid_result[0]  # cuboid centroid
                    elif actual_distance is not None:
                        # Fallback: single ray + median depth (Z-depth interpretation)
                        point_world = cam_origin_world + ray_world * actual_distance
                    else:
                        # No depth, no ground plane — place along ray at 0.1m
                        ray_normalized = ray_world / np.linalg.norm(ray_world)
                        point_world = cam_origin_world + ray_normalized * 0.1
                else:
                    # No depth data — try silhouette-based cuboid fitting
                    if mask_roi is not None and ground_plane_z is not None:
                        cuboid_result = _fit_cuboid_from_silhouette(
                            mask_roi, bx1, by1, camera_matrix,
                            opencv_to_camera_quat, cam_quat, cam_pos, ground_plane_z,
                            bbox=(bx1, by1, bx2, by2),
                            prev_params=_prev_seed)

                    if cuboid_result is not None:
                        point_world = cuboid_result[0]
                    else:
                        # Fallback: back-projected rect centroid or ray-table
                        rect_result = None
                        if mask_roi is not None:
                            rect_result = _backproject_rect_to_table(
                                mask_roi, bx1, by1, camera_matrix,
                                opencv_to_camera_quat, cam_quat, cam_pos, ground_plane_z)

                        if rect_result is not None:
                            point_world = rect_result[0]
                        else:
                            # Last resort: intersect ray with ground plane
                            if abs(ray_world[2]) > 1e-6:
                                t = (ground_plane_z - cam_origin_world[2]) / ray_world[2]
                                if t > 0:
                                    point_world = cam_origin_world + ray_world * t
                                else:
                                    ray_normalized = ray_world / np.linalg.norm(ray_world)
                                    point_world = cam_origin_world + ray_normalized * 0.1
                            else:
                                ray_normalized = ray_world / np.linalg.norm(ray_world)
                                point_world = cam_origin_world + ray_normalized * 0.1

                # Step 5: Apply calibration offset to object position
                if bridge_node is not None:
                    point_world = bridge_node.apply_calibration_offset(point_world)

                # Store metadata for visualization
                detection_index = len(detection_metadata)  # Track original detection order
                meta = {
                    'box': box,
                    'score': score,
                    'class_name': class_name,
                    'color_name': color_name,
                    'orientation_angle': orientation_angle,
                    'world_point': point_world,  # Store world point for matching
                    'detection_index': detection_index  # Track original order
                }
                if mask_roi is not None:
                    meta['mask_roi'] = mask_roi
                    meta['mask_bbox'] = (bx1, by1, bx2, by2)
                if cuboid_result is not None:
                    meta['cuboid_center'] = cuboid_result[0]
                    meta['cuboid_quaternion'] = cuboid_result[1]
                    meta['cuboid_dimensions'] = cuboid_result[2]
                detection_metadata.append(meta)

                # Store by color with detection index for matching
                if color_name not in detected_color_points:
                    detected_color_points[color_name] = []
                detected_color_points[color_name].append({
                    'point': point_world,
                    'detection_index': detection_index
                })
    
    return detected_color_points, detection_metadata

def main():
    # Parse args first, before initializing ROS
    args, unknown_args = parse_args()

    # Compute effective ground_plane_z from config + CLI offset
    if _GROUND_PLANE_Z_CONFIG is not None:
        ground_plane_z = _GROUND_PLANE_Z_CONFIG + args.additional_z_offset
        print(f"Ground plane Z: {_GROUND_PLANE_Z_CONFIG} (config) + {args.additional_z_offset} (CLI) = {ground_plane_z}")
    else:
        ground_plane_z = args.additional_z_offset if args.additional_z_offset != 0.0 else None
        if ground_plane_z is not None:
            print(f"Ground plane Z: {ground_plane_z} (CLI only, no config value)")
        else:
            print("Ground plane Z: not configured (cuboid table-snapping disabled)")

    # Start ROS node with remaining args
    bridge_node = start_ros_node(camera_topic=args.camera_topic, depth_topic=args.depth_topic)
    
    # Set up YOLO prompt topics
    from std_msgs.msg import String

    # Topic subscription for real-time prompt updates
    prompts_subscription = bridge_node.create_subscription(
        String,
        '/yolo_prompts_update',
        yolo_prompts_callback,
        10
    )
    
    # YOLO prompts publisher for external monitoring
    yolo_prompts_pub = bridge_node.create_publisher(String, '/yolo_prompts', 10)
    
    # Timer to publish current prompts periodically
    def publish_current_prompts():
        """Publish current YOLO prompts for external monitoring"""
        try:
            prompts, prompt_map = get_yolo_prompts()
            # Replace spaces with underscores in prompts for topic publishing
            prompts_normalized = [p.replace(' ', '_') for p in prompts]
            # Also normalize prompt_map keys and values
            prompt_map_normalized = {k.replace(' ', '_'): v.replace(' ', '_') if isinstance(v, str) else v 
                                     for k, v in prompt_map.items()}
            prompts_data = {
                'prompts': prompts_normalized,
                'prompt_map': prompt_map_normalized
            }
            
            msg = String()
            msg.data = json.dumps(prompts_data)
            yolo_prompts_pub.publish(msg)
            
        except Exception as e:
            print(f"Failed to publish current prompts: {e}")
    
    prompts_timer = bridge_node.create_timer(1.0, publish_current_prompts)

    # Parse prompt mapping from command line if provided
    yolo_prompt_map = YOLO_PROMPT_MAP.copy()
    if args.yolo_prompt_map:
        for mapping in args.yolo_prompt_map:
            if ':' in mapping:
                prompt, color = mapping.split(':', 1)
                yolo_prompt_map[prompt] = color.strip()
    
    # Update global variables with command line arguments
    update_yolo_prompts(args.yolo_prompts, yolo_prompt_map)

    # Initialize YOLO model with dynamic prompts using improved loading
    print(f"YOLO mode: {args.yolo_mode}")
    print(f"Loading YOLO model: {args.yolo_model}")
    
    # Get script directory and construct absolute model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.yolo_model.split('/')[-1])  # Get just the filename
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}, trying relative path: {args.yolo_model}")
        model_path = args.yolo_model
    
    # Change to script directory for model loading
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    
    try:
        import hashlib
        import torch

        yolo_model = YOLOE(model_path)
        print(f"Loaded YOLOE model from {model_path}")

        # Cache text embeddings to disk keyed by prompt hash
        prompt_key = hashlib.md5(",".join(sorted(args.yolo_prompts)).encode()).hexdigest()[:12]
        cache_path = os.path.join(script_dir, f".text_embeddings_{prompt_key}.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached text embeddings from {cache_path}")
            cache = torch.load(cache_path, weights_only=True)
            text_embeddings = cache["embeddings"]
            print(f"Loaded cached embeddings for prompts: {cache['prompts']}")
        else:
            print("Computing text embeddings (first time, ~90s for CLIP model download)...")
            text_embeddings = yolo_model.get_text_pe(args.yolo_prompts)
            torch.save({"prompts": args.yolo_prompts, "embeddings": text_embeddings}, cache_path)
            print(f"Cached text embeddings to {cache_path}")

        yolo_model.set_classes(args.yolo_prompts, text_embeddings)
        print(f"YOLO model ready with prompts: {args.yolo_prompts}")
        print(f"YOLO prompt mapping: {yolo_prompt_map}")
    finally:
        # Restore working directory
        os.chdir(original_cwd)
    
    # Update the topic callback to use the yolo_model
    def topic_callback_wrapper(msg):
        return yolo_prompts_callback(msg, yolo_model)

    # Recreate the topic subscription with the wrapped callback
    bridge_node.destroy_subscription(prompts_subscription)

    prompts_subscription = bridge_node.create_subscription(
        String,
        '/yolo_prompts_update',
        topic_callback_wrapper,
        10
    )

    frame_idx = 0

    # Current YOLO prompts (will be updated dynamically)
    # These are now managed by the global functions

    talk = not args.suppress_prints

    # Wait for first camera frame before entering main loop
    # (model loading blocks CPU for ~20s, starving the spin thread)
    import time as _time
    print("Waiting for first camera frame...")
    for _i in range(300):  # up to 30s
        if bridge_node.get_latest_frame() is not None:
            print(f"First frame received after {(_i+1)*0.1:.1f}s")
            break
        _time.sleep(0.1)

    print("\n" + "="*60)
    print("YOLO Camera Localizer")
    print("="*60)
    print(f"Waiting for camera frames on {args.camera_topic}...")
    print("Make sure the camera_publisher node is running!")
    if not args.headless:
        print("Press 'q' in the OpenCV window to quit.")
    print("="*60 + "\n")

    detected_objects = []
    
    try:
        while True:
            # Get the latest frame from the camera topic
            frame = bridge_node.get_latest_frame()
            
            # If no frame available yet, wait for next frame
            if frame is None:
                import time
                time.sleep(0.1)  # 100ms — give spin thread time for DDS callbacks
                continue

            # Check for dynamic YOLO prompt updates
            try:
                updated_prompts, updated_prompt_map = get_yolo_prompts()
                # Check if prompts have changed (this will be handled by the global update functions)
                # The YOLO model will be updated when the prompts actually change
            except Exception as e:
                print(f"Error checking YOLO prompts: {e}")

            frame_idx += 1
            ee_pos, ee_quat = bridge_node.get_ee_pose()
            cam_pos, cam_quat = bridge_node.get_camera_pose()

            # YOLO Detection Section
            current_prompts, current_prompt_map = get_yolo_prompts()
            
            # Get latest depth image if available
            depth_image = bridge_node.get_latest_depth()
            
            detected_color_points, detection_metadata = detect_yolo_blobs(
                frame, yolo_model, CAMERA_MATRIX, cam_pos, cam_quat,
                current_prompts, current_prompt_map,
                OPENCV_TO_CAMERA_QUAT, 
                depth_image=depth_image, bridge_node=bridge_node, conf_threshold=args.yolo_conf, nms_threshold=0.3, ground_plane_z=ground_plane_z
            )
            
            # Convert YOLO detections to object format for objects_poses topic
            yolo_detected_objects = []
            
            # Use global variable for previous frame's objects
            global previous_yolo_objects
            
            # Create a mapping from detection metadata to world points
            # Group metadata by color for easier lookup
            metadata_by_color = {}
            for metadata in detection_metadata:
                color_name = metadata['color_name']
                if color_name not in metadata_by_color:
                    metadata_by_color[color_name] = []
                metadata_by_color[color_name].append(metadata)
            
            # Convert YOLO detections to objects (skip pusher colors)
            # Use position-based matching with previous frame to prevent ID flipping
            for color_name, world_points_data in detected_color_points.items():
                # Get previous objects for this color
                prev_objects = previous_yolo_objects.get(color_name, [])
                
                # Get metadata for this color
                color_metadata = metadata_by_color.get(color_name, [])
                
                # Match new detections to previous objects by position
                # Create list of (point_data, metadata) pairs
                detection_pairs = []
                for point_data in world_points_data:
                    point = point_data['point']
                    detection_idx = point_data['detection_index']
                    
                    # Find corresponding metadata
                    matching_metadata = None
                    for metadata in color_metadata:
                        if metadata.get('detection_index') == detection_idx:
                            matching_metadata = metadata
                            break
                    
                    if matching_metadata is not None:
                        detection_pairs.append((point_data, matching_metadata))
                
                # Match detections to previous objects by closest position
                matched_indices = set()
                object_assignments = {}  # detection_idx -> object_index
                
                if prev_objects and len(prev_objects) == len(detection_pairs):
                    # Match each previous object to closest new detection
                    for prev_idx, prev_obj in enumerate(prev_objects):
                        prev_pos = prev_obj['position']
                        min_dist = float('inf')
                        best_detection_idx = None
                        
                        for point_data, metadata in detection_pairs:
                            detection_idx = point_data['detection_index']
                            if detection_idx in matched_indices:
                                continue
                            
                            point = point_data['point']
                            dist = np.linalg.norm(point - prev_pos)
                            if dist < min_dist:
                                min_dist = dist
                                best_detection_idx = detection_idx
                        
                        if best_detection_idx is not None and min_dist < 0.1:  # 10cm threshold
                            object_assignments[best_detection_idx] = prev_idx
                            matched_indices.add(best_detection_idx)
                
                # Assign remaining detections to new indices
                next_index = len(prev_objects) if prev_objects else 0
                for point_data, metadata in detection_pairs:
                    detection_idx = point_data['detection_index']
                    if detection_idx not in object_assignments:
                        object_assignments[detection_idx] = next_index
                        next_index += 1
                
                # Create objects in order of their assigned indices
                sorted_pairs = sorted(detection_pairs, key=lambda x: object_assignments.get(x[0]['detection_index'], 999))
                
                for i, (point_data, metadata) in enumerate(sorted_pairs):
                    point = point_data['point']
                    detection_idx = point_data['detection_index']
                    object_index = object_assignments.get(detection_idx, i)
                    
                    # Get orientation: cuboid yaw from major or minor axis, fall back to moments
                    if 'cuboid_quaternion' in metadata:
                        dims = metadata['cuboid_dimensions']
                        rot = R.from_quat(metadata['cuboid_quaternion']).as_matrix()
                        major_idx = 0 if dims[0] >= dims[1] else 1
                        minor_idx = 1 - major_idx
                        use_minor = bridge_node is not None and bridge_node.use_minor_axis
                        axis_dir = rot[:, minor_idx if use_minor else major_idx]
                        yaw = np.arctan2(axis_dir[1], axis_dir[0])
                        orientation_quat = R.from_euler('z', yaw).as_quat()
                    elif metadata['orientation_angle'] is not None:
                        orientation_quat = convert_2d_orientation_to_quaternion(
                            metadata['orientation_angle'], cam_quat, OPENCV_TO_CAMERA_QUAT
                        )
                    else:
                        orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
                    
                    # Store object name in metadata for label display
                    object_name = f"{color_name}_{object_index}"
                    metadata['object_name'] = object_name
                    
                    yolo_detected_objects.append({
                        "name": object_name,
                        "points": [point],
                        "position": point,
                        "quaternion": orientation_quat,
                        'inferred': False,
                    })
            
            # Update previous objects for next frame
            previous_yolo_objects = {}
            for obj in yolo_detected_objects:
                # Extract color_name and index from object name (e.g., "blue_object_0" -> "blue_object", 0)
                name_parts = obj["name"].rsplit("_", 1)
                if len(name_parts) == 2:
                    color_name = name_parts[0]
                    if color_name not in previous_yolo_objects:
                        previous_yolo_objects[color_name] = []
                    previous_yolo_objects[color_name].append({
                        "name": obj["name"],
                        "position": obj["position"]
                    })

            # Store cuboid params for temporal seeding in next frame
            _prev_cuboid_params.clear()
            for meta in detection_metadata:
                if 'cuboid_center' in meta:
                    cn = meta['color_name']
                    c = meta['cuboid_center']
                    q = meta['cuboid_quaternion']
                    d = meta['cuboid_dimensions']
                    yaw = R.from_quat(q).as_euler('ZYX')[0]
                    if cn not in _prev_cuboid_params:
                        _prev_cuboid_params[cn] = []
                    _prev_cuboid_params[cn].append((c[0], c[1], yaw, d[0], d[1]))
            
            # Camera pose is published by external package, we only subscribe to it
            # Publish all YOLO detections to objects_poses topic
            bridge_node.publish_object_poses(yolo_detected_objects)
            draw_text(frame, cam_pos, cam_quat, yolo_detected_objects, frame_idx, ee_pos, ee_quat)
            draw_object_lines(frame, CAMERA_MATRIX, cam_pos, cam_quat, yolo_detected_objects, [])

            # Draw YOLO detections with toggleable overlays (AFTER all other drawing)
            for detection in detection_metadata:
                box = detection['box']
                score = detection['score']
                class_name = detection['class_name']

                # Draw segmentation mask as semi-transparent overlay
                if bridge_node.show_seg_mask and 'mask_roi' in detection:
                    mroi = detection['mask_roi']
                    mbx1, mby1, mbx2, mby2 = detection['mask_bbox']
                    overlay = frame.copy()
                    mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                    mask_full[mby1:mby2, mbx1:mbx2] = (mroi > 0).astype(np.uint8) * 255
                    overlay[mask_full > 0] = (overlay[mask_full > 0] * 0.5 + np.array([128, 0, 255]) * 0.5).astype(np.uint8)
                    frame[:] = overlay

                # Draw bounding box + center dot + label
                if bridge_node.show_bbox:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    # Label with confidence and ID
                    if 'object_name' in detection:
                        label = f"{detection['object_name']}: {score:.2f}"
                    else:
                        class_name_display = class_name.replace(' ', '_')
                        label = f"{class_name_display}: {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 8),
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw 3D cuboid wireframe if available
                if 'cuboid_center' in detection:
                    if bridge_node.show_cuboid_wireframe:
                        draw_cuboid_wireframe(
                            frame, detection['cuboid_center'],
                            detection['cuboid_quaternion'],
                            detection['cuboid_dimensions'],
                            cam_pos, cam_quat, CAMERA_MATRIX,
                            OPENCV_TO_CAMERA_QUAT,
                            color=(0, 255, 255), thickness=2)
                    # Draw cuboid orientation axes based on toggle flags
                    if bridge_node.show_major_axis or bridge_node.show_minor_axis:
                        draw_cuboid_orientation_axes(
                            frame, detection['cuboid_center'],
                            detection['cuboid_quaternion'],
                            detection['cuboid_dimensions'],
                            cam_pos, cam_quat, CAMERA_MATRIX,
                            OPENCV_TO_CAMERA_QUAT,
                            show_major=bridge_node.show_major_axis,
                            show_minor=bridge_node.show_minor_axis)

            # Publish the annotated frame (same as what's displayed in OpenCV window)
            bridge_node.publish_annotated_image(frame)

            if not args.headless:
                cv2.imshow("YOLO-based Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
