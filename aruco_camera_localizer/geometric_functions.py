# geometric_function.py
import cv2
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
import numpy as np

def rvec_to_quat(rvec):
    """Convert OpenCV rotation vector to quaternion [x, y, z, w]"""
    rot, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot).as_quat()  # returns [x, y, z, w]

def quat_to_rvec(quat):
    """Convert quaternion [x, y, z, w] to OpenCV rotation vector"""
    rot = R.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rot)
    return rvec

# Per-object state for RPY smoothing (maintains continuity across frames)
# Using a dictionary to track multiple objects
_rpy_prev_dict = {}
# Per-object state for quaternion smoothing (reduces noise before RPY conversion)
_quat_smoothed_dict = {}

def quat_to_rpy_safe(quat, degrees=True, object_id=None, smoothing_alpha=0.75):
    """
    Convert quaternion [x, y, z, w] to roll, pitch, yaw (RPY) in a gimbal-lock-safe manner.
    
    This function handles gimbal lock by:
    1. Smoothing the input quaternion using SLERP to reduce noise (especially important near gimbal lock)
    2. Always using 'xyz' sequence for consistency (avoids discontinuities from sequence switching)
    3. Maintaining temporal continuity by unwrapping angles relative to previous frame
    4. Using per-object state tracking to avoid interference between different objects
    
    Note: Near gimbal lock (pitch ≈ ±90°), small quaternion noise causes large RPY variations.
    Quaternion smoothing reduces this noise before conversion, significantly improving stability.
    
    Args:
        quat: Quaternion [x, y, z, w]
        degrees: If True, return angles in degrees; if False, return in radians
        object_id: Optional identifier for per-object state tracking (e.g., object name)
        smoothing_alpha: Smoothing factor (0.0-1.0). Higher = more smoothing, less responsive.
                         Default 0.75 means 75% previous, 25% new quaternion.
        
    Returns:
        numpy array [roll, pitch, yaw] in the requested units
    """
    global _rpy_prev_dict, _quat_smoothed_dict
    
    # Normalize input quaternion
    quat = np.array(quat)
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-8:
        quat = quat / quat_norm
    else:
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion as fallback
    
    # Use object_id as key for per-object tracking
    key = object_id if object_id is not None else 'default'
    
    # Smooth quaternion using SLERP (spherical linear interpolation)
    if key in _quat_smoothed_dict:
        prev_quat = _quat_smoothed_dict[key]
        
        # Ensure quaternion continuity (q and -q represent same rotation)
        # Choose the quaternion closer to previous one
        dot1 = np.dot(prev_quat, quat)
        dot2 = np.dot(prev_quat, -quat)
        if abs(dot2) > abs(dot1):
            quat = -quat
        
        # Smooth using SLERP: blend between previous and current quaternion
        # smoothing_alpha = 0.75 means 75% previous, 25% new (more smoothing)
        # blend = 1 - smoothing_alpha means how much of the new quaternion to use
        blend = 1.0 - smoothing_alpha
        smoothed_quat = slerp_quat(prev_quat, quat, blend)
    else:
        # First frame: use quaternion as-is (no previous to smooth with)
        smoothed_quat = quat.copy()
    
    # Store smoothed quaternion for next frame
    _quat_smoothed_dict[key] = smoothed_quat.copy()
    
    # Convert smoothed quaternion to RPY
    r = R.from_quat(smoothed_quat)
    
    # Always use 'xyz' sequence for consistency
    # This avoids discontinuities from switching sequences
    rpy = r.as_euler('xyz', degrees=degrees)
    
    # Maintain temporal continuity by unwrapping angles relative to previous frame
    if key in _rpy_prev_dict:
        prev_rpy = _rpy_prev_dict[key]
        
        # Convert to radians for unwrapping
        if degrees:
            rpy_rad = np.deg2rad(rpy)
            prev_rad = np.deg2rad(prev_rpy)
        else:
            rpy_rad = rpy
            prev_rad = prev_rpy
        
        # Unwrap each angle relative to previous value
        # This prevents jumps from 180° to -180° or vice versa
        for i in range(3):
            diff = rpy_rad[i] - prev_rad[i]
            # If difference is > π, unwrap by adding/subtracting 2π
            if diff > np.pi:
                rpy_rad[i] -= 2 * np.pi
            elif diff < -np.pi:
                rpy_rad[i] += 2 * np.pi
        
        if degrees:
            rpy = np.rad2deg(rpy_rad)
        else:
            rpy = rpy_rad
    
    # Store for next frame
    _rpy_prev_dict[key] = rpy.copy()
    
    return rpy

def transform_points_world_to_img(points_world, cam_pos_world, cam_quat_world, camera_matrix):
    image_points = []
    for pt in points_world:
        cam_pt = transform_point_world_to_cam(pt, cam_pos_world, cam_quat_world)
        if cam_pt[2] <= 0.01:
            continue  # skip points behind the camera or too close
        u = int(camera_matrix[0, 0] * cam_pt[0] / cam_pt[2] + camera_matrix[0, 2])
        v = int(camera_matrix[1, 1] * cam_pt[1] / cam_pt[2] + camera_matrix[1, 2])
        image_points.append((u,v))
    return image_points

def transform_point_cam_to_world(point_cam, cam_pos_world, cam_quat_world):
    r_cam_world = R.from_quat(cam_quat_world)
    return cam_pos_world + r_cam_world.apply(point_cam)

def transform_point_world_to_cam(point_world, cam_pos_world, cam_quat_world):
    r_world_cam = R.from_quat(cam_quat_world).inv()
    return r_world_cam.apply(point_world - cam_pos_world)

def transform_orientation_cam_to_world(marker_quat_cam, cam_quat_world):
    r_marker_cam = R.from_quat(marker_quat_cam)
    r_cam_world = R.from_quat(cam_quat_world)
    r_marker_world = r_cam_world * r_marker_cam
    return r_marker_world.as_quat()

def transform_orientation_world_to_cam(marker_quat_world, cam_quat_world):
    """Transform orientation from world frame to camera frame"""
    r_marker_world = R.from_quat(marker_quat_world)
    r_cam_world = R.from_quat(cam_quat_world)
    r_marker_cam = r_cam_world.inv() * r_marker_world
    return r_marker_cam.as_quat()

def slerp_quat(q1, q2, blend=0.5):
    """Spherical linear interpolation between two quaternions"""
    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)
    rots = R.concatenate([rot1, rot2])
    slerp = scipy.spatial.transform.Slerp([0, 1], rots)
    return slerp(blend).as_quat()

def complete_triangle(p1, p2, side_lengths, tolerance=5.0):
    """
    Given two points p1 and p2, and triangle side lengths (in mm),
    return up to four 3D candidate positions for the third point p3
    that complete the triangle.

    Returns: list of np.array (3D points)
    """

    side_a, side_b, side_c = side_lengths
    d = np.linalg.norm(p1 - p2)
    sides = sorted([side_a, side_b, side_c])

    # Identify which side corresponds to p1-p2
    known_side = None
    for i, s in enumerate(sides):
        if abs(s - 1000 * d) < tolerance:
            known_side = i
            break
    if known_side is None:
        return None  # can't infer triangle

    # Other two sides
    idx = [0, 1, 2]
    idx.remove(known_side)
    s1, s2 = sides[idx[0]] / 1000, sides[idx[1]] / 1000  # convert to meters

    # Basis vectors
    e_x = (p2 - p1) / d

    # Orthogonal vector not aligned with e_x
    e_y = np.cross(e_x, np.array([0, 0, 1]))
    if np.linalg.norm(e_y) < 1e-6:
        e_y = np.cross(e_x, np.array([0, 1, 0]))
    e_y = e_y / np.linalg.norm(e_y)

    e_z = np.cross(e_x, e_y)  # Complete right-handed frame

    # Triangle geometry
    x = (s1**2 - s2**2 + d**2) / (2 * d)
    h_sq = s1**2 - x**2
    if h_sq < 0:
        return None  # no triangle possible
    h = np.sqrt(h_sq)

    # Four candidate points
    p3a = p1 + x * e_x + h * e_y
    p3b = p1 + x * e_x - h * e_y
    p3c = p2 - x * e_x + h * e_y
    p3d = p2 - x * e_x - h * e_y

    # Return unique ones only
    candidates = []
    for p in [p3a, p3b, p3c, p3d]:
        if not any(np.allclose(p, existing, atol=1e-6) for existing in candidates):
            candidates.append(p)

    return candidates

def pick_best_candidate(candidates, prev_position):
    """
    Given a list of candidate points and the previous position of the object,
    return the one closest to the previous position.
    """
    if prev_position is None or len(candidates) == 1:
        return candidates[0]

    distances = [np.linalg.norm(candidate - prev_position) for candidate in candidates]
    best_index = np.argmin(distances)
    return candidates[best_index]
