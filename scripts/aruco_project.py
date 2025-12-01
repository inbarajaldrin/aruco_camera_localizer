#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.config_loader import get_config
import argparse
import json

# Load robot configuration from YAML
robot_config = get_config()

# Note: We work directly in OpenCV frame for visualization
# No need to transform to camera frame - cv2.solvePnP and cv2.projectPoints use OpenCV frame

# Global configuration variables for box drawing
box_config = {
    # Using ArUco coordinate system: X=right, Y=down, Z=toward camera
    # All values are in METERS
    'top': {'X': 0.0, 'Y': -0.15, 'Z': 0.15},   # X=left/right (m), Y=up/down (m), Z=toward camera (m) for top row
    'mid': {'X': 0.0, 'Y': -0.095, 'Z': 0.1},   # X=left/right (m), Y=up/down (m), Z=toward camera (m) for middle row
    'bot': {'X': 0.0, 'Y': -0.025, 'Z': 0.05},   # X=left/right (m), Y=up/down (m), Z=toward camera (m) for bottom row
    'box': {'w': 0.03, 'h': 0.03, 'd': 0.03}  # Box dimensions in meters: width, height, depth (0.03m = 3cm)
}

# Marker visibility toggles (can be modified directly in code if needed)
visible_flags = {
    'top': True,
    'mid': True,
    'bot': True
}

# Camera intrinsics
# Wide-angle lens calibration parameters
# These are typical values for a 640x480 wide-angle camera
# Adjust these values based on your actual camera calibration

# Get camera dimensions from config
c_width = robot_config.get_camera_width()  # Typically 640 for robosort
c_height = robot_config.get_camera_height()  # Typically 480 for robosort

# Wide-angle cameras typically have focal lengths around 300-500 pixels
# Using moderate focal length: fx = fy = ~350 pixels
# Center point is typically at image center
fx = fy = 350.0  # Focal length (adjust if boxes appear too large/small)
cx = c_width / 2.0   # Principal point X (image center)
cy = c_height / 2.0  # Principal point Y (image center)

cameraMatrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)

# Standard pinhole distortion coefficients for wide-angle lens
# Format: [k1, k2, p1, p2, k3] - radial (k1, k2, k3) and tangential (p1, p2)
# Wide-angle lenses have moderate barrel distortion
distCoeffs = np.array([
    [-0.15],  # k1: Main radial distortion (barrel distortion - negative)
    [0.05],   # k2: Secondary radial distortion
    [0.0],    # p1: Tangential distortion (usually small)
    [0.0],    # p2: Tangential distortion (usually small)
    [-0.01]   # k3: Tertiary radial distortion
], dtype=np.float64)

marker_length = 0.032

# ArUco setup
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

def get_marker_properties(marker_id):
    """Returns (X_offset, Y_offset, Z_offset) in ArUco coordinate system: X=right, Y=down, Z=toward camera"""
    if marker_id in [1, 4, 7, 10, 13]:
        return box_config['top']['X'], box_config['top']['Y'], box_config['top']['Z']
    elif marker_id in [2, 5, 8, 11, 14]:
        return box_config['mid']['X'], box_config['mid']['Y'], box_config['mid']['Z']
    elif marker_id in [3, 6, 9, 12, 15]:
        return box_config['bot']['X'], box_config['bot']['Y'], box_config['bot']['Z']
    else:
        return 0.0, 0.0, 0.1

def compute_plane_normal_from_markers(rvecs_opencv, tvecs_opencv):
    """
    Compute plane normal from multiple markers assuming they're all in the same plane.
    Returns the normal vector in OpenCV frame.
    """
    if len(rvecs_opencv) < 3:
        return None
    
    # Get rotation matrices and positions
    points = []
    for rvec, tvec in zip(rvecs_opencv, tvecs_opencv):
        if rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            # Get marker center in camera frame
            tvec_flat = tvec.flatten()
            points.append(tvec_flat)
    
    if len(points) < 3:
        return None
    
    # Fit plane using SVD
    points = np.array(points)
    if len(points) >= 3:
        # Use SVD to fit plane
        # points is (N, 3), we want to find the plane normal
        centroid = np.mean(points, axis=0)
        centered = points - centroid  # (N, 3)
        # SVD of centered gives us the plane normal
        # The normal is the right singular vector corresponding to smallest singular value
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Vt is (3, 3), last row is the normal (corresponds to smallest singular value)
        normal = Vt[-1]  # Shape (3,)
        # Ensure normal points toward camera (positive Z in OpenCV frame)
        if normal[2] < 0:
            normal = -normal
        return normal  # Return as 1D array of shape (3,)
    return None

def draw_rectangular_box_on_marker(frame, rvec_opencv, tvec_opencv, marker_id, cam_matrix=None, dist_coeffs=None, reference_rvec=None, plane_normal=None, is_rectified=False):
    """
    Draw rectangular box on marker.
    rvec_opencv and tvec_opencv are in OpenCV frame (for visualization).
    If reference_rvec is provided, all boxes will use that orientation for alignment.
    If plane_normal is provided, boxes will be aligned perpendicular to the plane.
    """
    if marker_id in [1, 4, 7, 10, 13] and not visible_flags['top']:
        return
    elif marker_id in [2, 5, 8, 11, 14] and not visible_flags['mid']:
        return
    elif marker_id in [3, 6, 9, 12, 15] and not visible_flags['bot']:
        return

    # Use provided camera matrix or default to global one
    cam_mat = cam_matrix if cam_matrix is not None else cameraMatrix
    dist_coef = dist_coeffs if dist_coeffs is not None else distCoeffs

    box_w, box_h, box_d = box_config['box']['w'], box_config['box']['h'], box_config['box']['d']
    # Get offsets from config to position box relative to marker center
    # Offsets are in ArUco marker frame
    X_offset_aruco, Y_offset_aruco, Z_offset_aruco = get_marker_properties(marker_id)
    
    # Transform offsets from ArUco frame to camera frame
    # 180-degree rotation about X-axis: Y and Z are flipped
    R_aruco_to_cam = np.array([
        [1,  0,  0],  # X unchanged
        [0, -1,  0],  # Y flipped
        [0,  0, -1]   # Z flipped
    ], dtype=np.float64)
    offset_aruco = np.array([X_offset_aruco, Y_offset_aruco, Z_offset_aruco])
    offset_cam = R_aruco_to_cam @ offset_aruco
    X_offset, Y_offset, Z_offset = offset_cam[0], offset_cam[1], offset_cam[2]
    
    # Box corners: centered on marker, then offset by config values (now in camera frame)
    # Box extends from -box_d/2 to +box_d/2 in Z, then shifted by Z_offset
    corners_local = np.array([
        [-box_w/2 + X_offset, -box_h/2 + Y_offset, -box_d/2 + Z_offset], 
        [ box_w/2 + X_offset, -box_h/2 + Y_offset, -box_d/2 + Z_offset],
        [ box_w/2 + X_offset,  box_h/2 + Y_offset, -box_d/2 + Z_offset], 
        [-box_w/2 + X_offset,  box_h/2 + Y_offset, -box_d/2 + Z_offset],
        [-box_w/2 + X_offset, -box_h/2 + Y_offset,  box_d/2 + Z_offset], 
        [ box_w/2 + X_offset, -box_h/2 + Y_offset,  box_d/2 + Z_offset],
        [ box_w/2 + X_offset,  box_h/2 + Y_offset,  box_d/2 + Z_offset], 
        [-box_w/2 + X_offset,  box_h/2 + Y_offset,  box_d/2 + Z_offset]
    ], dtype=np.float32)

    # Determine box orientation
    # If plane_normal is provided, align all boxes perpendicular to the plane with consistent z-direction
    # Otherwise, boxes follow their marker's orientation
    if plane_normal is not None:
        # Align boxes perpendicular to the plane
        # Normal points toward camera, so boxes should extend along -normal (away from camera)
        # Ensure plane_normal is a 1D array of shape (3,)
        plane_normal_arr = np.array(plane_normal).flatten()
        if len(plane_normal_arr) == 3:
            # Box Z-axis extends perpendicular to plane (away from camera)
            z_axis = -plane_normal_arr / np.linalg.norm(plane_normal_arr)
            
            # Use fixed reference direction for x-axis (right in OpenCV frame: [1, 0, 0])
            # This ensures all boxes have consistent z-direction (right) when using plane normal
            fixed_x_ref = np.array([1.0, 0.0, 0.0])  # Right direction in OpenCV frame
            # Project fixed reference onto the plane (remove component along normal)
            x_axis = fixed_x_ref - np.dot(fixed_x_ref, z_axis) * z_axis
            if np.linalg.norm(x_axis) < 0.1:
                # If fixed reference is parallel to normal, use up direction [0, -1, 0]
                fixed_y_ref = np.array([0.0, -1.0, 0.0])  # Up direction in OpenCV frame
                x_axis = fixed_y_ref - np.dot(fixed_y_ref, z_axis) * z_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            R_opencv = np.column_stack([x_axis, y_axis, z_axis])
        else:
            # Invalid plane normal, fall back to marker's orientation
            R_opencv, _ = cv2.Rodrigues(rvec_opencv)
    elif reference_rvec is not None:
        # All boxes use the same orientation (aligned) from reference marker
        R_opencv, _ = cv2.Rodrigues(reference_rvec)
    else:
        # No plane normal and no reference: box follows marker's orientation
        # This is the natural behavior when all offsets are zero
        R_opencv, _ = cv2.Rodrigues(rvec_opencv)
    
    # Transform box corners from ArUco marker frame to camera frame
    # Box config offsets are in ArUco marker frame, so we need to:
    # 1. Rotate corners from ArUco frame using marker orientation
    # 2. Apply 180-degree rotation about X-axis to transform ArUco frame to camera frame
    R_aruco_to_cam = np.array([
        [1,  0,  0],  # X-axis unchanged
        [0, -1,  0],  # Y-axis flipped
        [0,  0, -1]   # Z-axis flipped
    ], dtype=np.float64)
    
    # Compose rotations: first rotate by marker orientation, then transform ArUco to camera frame
    R_opencv = R_opencv @ R_aruco_to_cam
    
    # Transform box corners from local frame (ArUco marker frame) to camera frame
    # Each box is positioned at its marker's location (tvec_opencv)
    tvec_opencv_flat = tvec_opencv.flatten()
    # Transform corners: rotate then translate
    # corners_local is (8, 3) in ArUco marker frame, we rotate to camera frame
    corners_opencv = (R_opencv @ corners_local.T).T + tvec_opencv_flat  # Broadcasting: (8, 3) + (3,) = (8, 3)
    
    # Project 3D points to image plane (using OpenCV frame coordinates)
    # For rectified images, use zero distortion; for unrectified, use actual distortion
    if is_rectified:
        # Rectified image - use standard projection with zero distortion
        img_pts, _ = cv2.projectPoints(corners_opencv, np.zeros((3,1)), np.zeros((3,1)), cam_mat, np.zeros((4,1)))
    else:
        # Unrectified image - use standard projection with distortion coefficients
        img_pts, _ = cv2.projectPoints(corners_opencv, np.zeros((3,1)), np.zeros((3,1)), cam_mat, dist_coef)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    
    # Get image dimensions for bounds checking
    h, w = frame.shape[:2]
    
    # Validate projected points: check if they're in front of camera and within image bounds
    valid_points = []
    for i, pt in enumerate(img_pts):
        # Check if point is behind camera (Z < 0 in camera frame)
        if corners_opencv[i, 2] < 0:
            valid_points.append(False)
            continue
        # Check if point is within image bounds (with some margin)
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            valid_points.append(True)
        else:
            valid_points.append(False)
    
    # Draw box edges only if both endpoints are valid
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for start, end in edges:
        if valid_points[start] and valid_points[end]:
            # Both points are valid, draw the edge
            cv2.line(frame, tuple(img_pts[start]), tuple(img_pts[end]), (0, 0, 255), 2)
        elif valid_points[start] or valid_points[end]:
            # One point is valid, clip to image bounds and draw
            pt1 = img_pts[start]
            pt2 = img_pts[end]
            # Clip points to image bounds
            pt1_clipped = (np.clip(pt1[0], 0, w-1), np.clip(pt1[1], 0, h-1))
            pt2_clipped = (np.clip(pt2[0], 0, w-1), np.clip(pt2[1], 0, h-1))
            cv2.line(frame, pt1_clipped, pt2_clipped, (0, 0, 255), 2)

def correct_z_axis_direction(rvec):
    R, _ = cv2.Rodrigues(rvec)
    z_axis = R[:, 2]
    if np.dot(z_axis, np.array([0, 0, 1])) < 0:
        R[:, 2] *= -1  # Flip z-axis if pointing in wrong direction
        R[:, 0] = np.cross(R[:, 1], R[:, 2])
        R[:, 1] = np.cross(R[:, 2], R[:, 0])
    corrected_rvec, _ = cv2.Rodrigues(R)
    return corrected_rvec

class ArUcoProjectNode(Node):
    def __init__(self):
        super().__init__('aruco_project_node')
        
        # Camera topic
        self.camera_topic = '/camera/image_rgb'
        
        # CvBridge for converting ROS Image messages to OpenCV format
        self.bridge = CvBridge()
        
        # Latest frame storage
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Rectification maps (computed on first frame)
        self.mapx = None
        self.mapy = None
        self.rectified_camera_matrix = None
        self.rectification_initialized = False
        self.rectification_successful = False
        
        # Subscribe to camera images
        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            10
        )
        self.get_logger().info(f"Subscribing to camera images on: {self.camera_topic}")
        
    def camera_callback(self, msg: Image):
        """Callback for receiving camera frames from ROS topic"""
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")
    
    def get_latest_frame(self):
        """Get the latest camera frame (thread-safe)"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def rectify_frame(self, frame):
        """Rectify the frame using camera matrix and distortion coefficients (pinhole)"""
        if frame is None:
            return None
        
        # Initialize rectification maps on first frame
        if not self.rectification_initialized:
            h, w = frame.shape[:2]
            
            # Standard pinhole camera undistortion
            # Get optimal new camera matrix
            self.rectified_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix, distCoeffs, (w, h), 1, (w, h)
            )
            # Compute rectification maps
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                cameraMatrix, distCoeffs, None, self.rectified_camera_matrix, (w, h), cv2.CV_32FC1
            )
            self.rectification_successful = True
            self.get_logger().info(f"Initialized rectification maps for image size: {w}x{h}")
            
            self.rectification_initialized = True
        
        # Apply rectification (only if maps are valid)
        if self.mapx is not None and self.mapy is not None:
            rectified_frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            return rectified_frame
        else:
            # Return original frame if rectification failed
            return frame
    
    def get_rectified_camera_matrix(self):
        """Get the rectified camera matrix (returns original if not initialized)"""
        return self.rectified_camera_matrix if self.rectified_camera_matrix is not None else cameraMatrix

def process_frame(frame, rectified_camera_matrix=None, is_rectified=False):
    """Process a single frame for ArUco marker detection and overlay"""
    # Use rectified camera matrix if provided, otherwise use original
    cam_matrix = rectified_camera_matrix if rectified_camera_matrix is not None else cameraMatrix
    
    # For rectified images, distortion coefficients are zero and we use standard solvePnP
    # For unrectified images, use standard solvePnP with distortion coefficients
    if is_rectified:
        # Rectified image - use standard solvePnP with zero distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    else:
        # Unrectified image - use standard solvePnP with distortion coefficients
        dist_coeffs = distCoeffs
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose for each marker using solvePnP
        # solvePnP returns pose in OpenCV frame
        rvecs_opencv = []
        tvecs_opencv = []
        half_size = marker_length / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        for corner in corners:
            image_points = corner[0].reshape(-1, 2)
            # Use standard solvePnP for pose estimation
            success, rvec_opencv, tvec_opencv = cv2.solvePnP(
                object_points, image_points, cam_matrix, dist_coeffs
            )
            if success:
                rvecs_opencv.append(rvec_opencv)
                tvecs_opencv.append(tvec_opencv)
            else:
                rvecs_opencv.append(None)
                tvecs_opencv.append(None)

        # Compute plane normal from all markers (since they're in the same plane)
        # Only use valid poses
        valid_rvecs = []
        valid_tvecs = []
        for r, t in zip(rvecs_opencv, tvecs_opencv):
            if r is not None and t is not None:
                valid_rvecs.append(r)
                valid_tvecs.append(t)
        
        plane_normal = compute_plane_normal_from_markers(valid_rvecs, valid_tvecs)
        
        # Verify plane normal was computed (for debugging)
        if plane_normal is None and len(valid_tvecs) >= 3:
            # Should have computed plane normal but didn't - might be an issue
            pass
        
        # Find reference marker (marker 7) for box alignment (fallback if plane normal fails)
        reference_rvec = None
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 7 and rvecs_opencv[i] is not None:
                reference_rvec = rvecs_opencv[i]
                break
        
        # If no reference marker found, use first marker's orientation
        if reference_rvec is None and len(rvecs_opencv) > 0 and rvecs_opencv[0] is not None:
            reference_rvec = rvecs_opencv[0]
        
        # Draw axes and boxes for each detected marker
        # Use OpenCV frame directly from solvePnP for visualization
        for i, marker_id in enumerate(ids.flatten()):
            if rvecs_opencv[i] is not None and tvecs_opencv[i] is not None:
                # Use OpenCV frame directly for visualization
                rvec = rvecs_opencv[i]
                tvec = tvecs_opencv[i]
                
                # Draw axes using marker's own orientation
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)
                # Draw box using plane normal for perfect alignment (or reference as fallback)
                # Each box is positioned at its marker's location (tvec)
                draw_rectangular_box_on_marker(frame, rvec, tvec, marker_id, cam_matrix, dist_coeffs, reference_rvec, plane_normal)
    
    return frame

def calibrate_camera_from_markers(node, num_frames=20):
    """
    Calibrate camera using ArUco markers.
    Collects corner points from multiple frames and computes camera matrix and distortion.
    """
    print("\n" + "="*60)
    print("CAMERA CALIBRATION MODE")
    print("="*60)
    print(f"Collecting data from {num_frames} frames...")
    print("Make sure all markers (1-15) are visible in the frame.")
    print("="*60 + "\n")
    
    # Storage for calibration data
    all_objpoints = []  # 3D points in marker coordinate system
    all_imgpoints = []  # 2D points in image plane
    
    frame_count = 0
    collected_frames = 0
    
    while collected_frames < num_frames:
        frame = node.get_latest_frame()
        if frame is None:
            import time
            time.sleep(0.01)
            continue
        
        # Don't rectify for calibration - use raw image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) >= 4:  # Need at least 4 markers
            # Prepare object points (marker corners in marker coordinate system)
            objpoints = []  # 3D points
            imgpoints = []  # 2D points
            
            half_size = marker_length / 2.0
            for corner, marker_id in zip(corners, ids.flatten()):
                # Object points for this marker (in marker's local frame)
                objp = np.array([
                    [-half_size,  half_size, 0],
                    [ half_size,  half_size, 0],
                    [ half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)
                
                # Image points (corner coordinates)
                imgp = corner[0].reshape(-1, 2).astype(np.float32)
                
                objpoints.append(objp)
                imgpoints.append(imgp)
            
            if len(objpoints) >= 4:
                all_objpoints.append(objpoints)
                all_imgpoints.append(imgpoints)
                collected_frames += 1
                print(f"Collected frame {collected_frames}/{num_frames} ({len(objpoints)} markers)")
        
        # Display progress
        cv2.putText(frame, f"Calibration: {collected_frames}/{num_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count > num_frames * 100:  # Timeout
            print("Timeout waiting for frames")
            break
    
    cv2.destroyAllWindows()
    
    if len(all_objpoints) < 5:
        print("ERROR: Not enough frames collected for calibration!")
        return None, None
    
    print("\nComputing calibration...")
    
    # Flatten the lists for calibration
    objpoints_flat = []
    imgpoints_flat = []
    for objp_list, imgp_list in zip(all_objpoints, all_imgpoints):
        for objp, imgp in zip(objp_list, imgp_list):
            objpoints_flat.append(objp)
            imgpoints_flat.append(imgp)
    
    # Get image size
    h, w = frame.shape[:2]
    
    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_flat, imgpoints_flat, (w, h), None, None
    )
    
    if ret:
        print("\n" + "="*60)
        print("CALIBRATION SUCCESSFUL!")
        print("="*60)
        print(f"Camera Matrix:\n{mtx}")
        print(f"\nDistortion Coefficients:\n{dist.flatten()}")
        print(f"\nReprojection Error: {ret:.4f} pixels")
        print("="*60 + "\n")
        
        # Save to file
        calibration_data = {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.flatten().tolist(),
            'image_size': [w, h],
            'reprojection_error': float(ret)
        }
        
        with open('camera_calibration.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print("Calibration saved to 'camera_calibration.json'")
        print("\nTo use these values, update the cameraMatrix and distCoeffs in the code.")
        
        return mtx, dist
    else:
        print("ERROR: Calibration failed!")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='ArUco Marker Detection with Box Overlay')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Run camera calibration using ArUco markers')
    parser.add_argument('--calibration-frames', type=int, default=20,
                       help='Number of frames to collect for calibration (default: 20)')
    args = parser.parse_args()
    
    rclpy.init()
    
    # Create the ROS2 node
    node = ArUcoProjectNode()
    
    # Spin ROS2 node in a separate thread
    def spin_node():
        rclpy.spin(node)
    
    spin_thread = threading.Thread(target=spin_node, daemon=True)
    spin_thread.start()
    
    # Run calibration if requested
    if args.calibrate:
        calibrate_camera_from_markers(node, args.calibration_frames)
        node.destroy_node()
        rclpy.shutdown()
        return
    
    print("="*60)
    print("ArUco Marker Detection")
    print("="*60)
    print(f"Waiting for camera frames on {node.camera_topic}...")
    print("Press 'q' in the OpenCV window to quit.")
    print("="*60 + "\n")
    
    try:
        while True:
            # Get the latest frame from the camera topic
            frame = node.get_latest_frame()
            
            # If no frame available yet, wait a bit and continue
            if frame is None:
                import time
                time.sleep(0.01)  # 10ms
                continue
            
            # Try to rectify the frame first
            rectified_frame = node.rectify_frame(frame)
            
            # Check if rectification actually happened
            is_rectified = node.rectification_successful and node.mapx is not None and node.mapy is not None
            
            # Get camera matrix for pose estimation
            if is_rectified:
                cam_matrix = node.get_rectified_camera_matrix()
            else:
                cam_matrix = cameraMatrix  # Use original if not rectified
            
            # Process frame and overlay markers
            # Pass is_rectified flag so we use correct solvePnP function
            processed_frame = process_frame(rectified_frame, cam_matrix, is_rectified)
            
            # Display the frame
            cv2.imshow("ArUco Marker Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
