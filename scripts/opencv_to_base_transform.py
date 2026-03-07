#!/usr/bin/env python3
"""
Transform Tool: OpenCV Frame to Base Frame
============================================
This script calculates the complete transformation from OpenCV pixel coordinates
to robot base frame coordinates, following the exact pipeline used in the codebase.

Usage:
    python3 opencv_to_base_transform.py --pixel-x 640 --pixel-y 360
    
Or for interactive mode:
    python3 opencv_to_base_transform.py --interactive
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse


# ============================================================================
# ROBOSORT CONFIGURATION (from robot_config.yaml)
# ============================================================================
ROBOT_CONFIG = {
    'active_robot': 'robosort',
    
    # Detection settings
    'detection': {
        'distance': 0.132,  # meters - distance from camera to place detected objects
    },
    
    # Camera parameters
    'camera': {
        'width': 1280,
        'height': 720,
        'horizontal_fov': 69.4,  # degrees
        'vertical_fov': 42.5,    # degrees
    },
    
    # Default/Home position of end-effector
    'ee_default': {
        'position': [-0.19275, 0.0, 0.170],  # meters [x, y, z]
        'quaternion': [0.0, 0.7071068, 0.0, 0.7071068],  # [x, y, z, w] - 90° Y rotation
    },
    
    # Default camera pose in base frame
    # You can change these values or provide them via command line
    'camera_default': {
        'position': [-0.058649, 0.000124, 0.042411],  # meters [x, y, z] in base frame
        'quaternion': [0.0, 0.0, 0.0, 1.0],  # [x, y, z, w] - identity (downward-facing) 
    },
    
    # OpenCV frame to Camera frame transformation
    # OpenCV convention: X=right, Y=down, Z=forward
    # Camera points along -X axis of base frame (forward-facing)
    # Transformation: Y-90° then Z-90° to align OpenCV Z-axis with camera optical axis
      'opencv_to_camera': {
          'quaternion': [-0.5, -0.5, 0.5, 0.5],  # [x, y, z, w] - Y-90° then Z-90°
      },
}


class OpenCVToBaseTransform:
    """Complete transformation pipeline from OpenCV pixels to base frame"""
    
    def __init__(self, cam_position=None, cam_quat=None, orientation_angle_2d=0.0):
        """
        Initialize the transformer with robot configuration
        
        Args:
            cam_position: Camera position [x,y,z] in base frame (None = use default)
            cam_quat: Camera quaternion [x,y,z,w] in base frame (None = use default)
            orientation_angle_2d: 2D orientation angle in radians (default=0.0)
        """
        # Camera parameters from config
        self.c_width = ROBOT_CONFIG['camera']['width']
        self.c_height = ROBOT_CONFIG['camera']['height']
        self.c_hfov = ROBOT_CONFIG['camera']['horizontal_fov']
        self.c_vfov = ROBOT_CONFIG['camera']['vertical_fov']
        
        # Calculate focal lengths
        self.fx = self.c_width / (2 * np.tan(np.deg2rad(self.c_hfov / 2)))
        self.fy = self.c_height / (2 * np.tan(np.deg2rad(self.c_vfov / 2)))
        
        # Camera matrix (intrinsics)
        self.camera_matrix = np.array([
            [self.fx,    0,      self.c_width/2],
            [0,       self.fy,   self.c_height/2],
            [0,          0,             1]
        ])
        
        # Camera pose in world frame (use provided or default)
        if cam_position is None:
            self.cam_pos_world = np.array(ROBOT_CONFIG['camera_default']['position'])
        else:
            self.cam_pos_world = np.array(cam_position)
            
        if cam_quat is None:
            self.cam_quat_world = np.array(ROBOT_CONFIG['camera_default']['quaternion'])
        else:
            self.cam_quat_world = np.array(cam_quat)
        
        # Calculate camera-to-EE offset from camera pose and EE pose
        self._calculate_camera_to_ee_offset()
        
        # 2D orientation angle for object
        self.orientation_angle_2d = orientation_angle_2d
        
        # Detection distance (default 0.1m, override with --distance CLI arg)
        self.detection_distance = ROBOT_CONFIG.get('detection', {}).get('distance', 0.1)
    
    def _calculate_camera_to_ee_offset(self):
        """
        Calculate camera-to-EE offset from camera pose and EE pose.
        
        Given:
            - Camera pose in base frame (cam_pos_world, cam_quat_world)
            - EE pose in base frame (ee_default)
        
        Calculate:
            - Camera position offset in EE frame
            - Camera orientation offset relative to EE
        """
        ee_pos = np.array(ROBOT_CONFIG['ee_default']['position'])
        ee_quat = np.array(ROBOT_CONFIG['ee_default']['quaternion'])
        
        # Rotation matrices
        R_ee = R.from_quat(ee_quat)  # EE orientation in base frame
        R_cam = R.from_quat(self.cam_quat_world)  # Camera orientation in base frame
        
        # Camera position offset in EE frame
        # cam_pos_world = ee_pos + R_ee * cam_offset_position
        # => cam_offset_position = R_ee^-1 * (cam_pos_world - ee_pos)
        self.cam_offset_position = R_ee.inv().apply(self.cam_pos_world - ee_pos)
        
        # Camera orientation offset relative to EE
        # R_cam = R_ee * R_cam_offset
        # => R_cam_offset = R_ee^-1 * R_cam
        R_cam_offset = R_ee.inv() * R_cam
        self.cam_offset_quat = R_cam_offset.as_quat()
        
        # Store EE pose for reference
        self.ee_position = ee_pos
        self.ee_quat = ee_quat
        
    def pixel_to_camera_ray(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to 3D ray in camera frame
        
        Args:
            pixel_x, pixel_y: Pixel coordinates in OpenCV frame
            
        Returns:
            ray_cam: Normalized direction vector in camera frame
        """
        # Step 1: Create homogeneous pixel coordinates
        pixel = np.array([pixel_x, pixel_y, 1.0])
        
        # Step 2: Apply inverse camera matrix to get ray direction in OpenCV frame
        ray_opencv = np.linalg.inv(self.camera_matrix) @ pixel
        
        # Step 3: Transform from OpenCV frame to camera frame
        opencv_to_cam = R.from_quat(ROBOT_CONFIG['opencv_to_camera']['quaternion'])
        ray_cam = opencv_to_cam.apply(ray_opencv)
        
        return ray_cam
    
    def camera_ray_to_world_point(self, ray_cam):
        """
        Convert camera frame ray to 3D world point at a fixed distance.
        
        Places the object at a distance of 'detection_distance' along the ray
        from the camera origin, regardless of camera orientation.
        
        Args:
            ray_cam: Ray direction in camera frame
            
        Returns:
            point_world: 3D point in base/world frame
        """
        # Transform ray to world frame
        R_wc = R.from_quat(self.cam_quat_world).as_matrix()
        ray_world = R_wc @ ray_cam
        cam_origin_world = np.array(self.cam_pos_world)
        
        # Normalize the ray and place object at detection_distance
        ray_normalized = ray_world / np.linalg.norm(ray_world)
        point_world = cam_origin_world + ray_normalized * self.detection_distance
        
        return point_world
    
    def orientation_2d_to_quaternion(self):
        """
        Convert 2D orientation angle to 3D quaternion in world frame
        
        Transformation chain:
        1. 2D angle → rotation in OpenCV frame (Z-axis rotation)
        2. OpenCV frame → Camera frame (opencv_to_camera)
        3. Camera frame → World/Base frame (camera quaternion)
        
        Returns:
            quat_world: Quaternion [x,y,z,w] in world frame
        """
        # Step 1: Create rotation around Z-axis in OpenCV frame
        z_rotation = R.from_euler('z', self.orientation_angle_2d)
        opencv_quat = z_rotation.as_quat()
        
        # Step 2: Transform from OpenCV frame to camera frame
        opencv_to_cam = R.from_quat(ROBOT_CONFIG['opencv_to_camera']['quaternion'])
        camera_orientation = opencv_to_cam * R.from_quat(opencv_quat)
        
        # Step 3: Transform from camera frame to world frame
        cam_rotation = R.from_quat(self.cam_quat_world)
        world_orientation = cam_rotation * camera_orientation
        
        return world_orientation.as_quat()
    
    def transform_pixel_to_pose(self, pixel_x, pixel_y, verbose=True):
        """
        Complete transformation pipeline from pixel to object pose
        
        Args:
            pixel_x, pixel_y: Pixel coordinates in OpenCV frame
            verbose: Print intermediate steps
            
        Returns:
            dict with position and quaternion in base frame
        """
        if verbose:
            print("\n" + "="*70)
            print("TRANSFORMATION PIPELINE: OpenCV Frame → Base Frame")
            print("="*70)
            
            print(f"\n{'CONFIGURATION':-^70}")
            print(f"Active Robot: {ROBOT_CONFIG['active_robot']}")
            print(f"\n📷 Camera Intrinsics:")
            print(f"  Resolution: {self.c_width} × {self.c_height} pixels")
            print(f"  FOV: {self.c_hfov:.1f}° (H) × {self.c_vfov:.1f}° (V)")
            print(f"  Focal Length: fx={self.fx:.2f}, fy={self.fy:.2f}")
            print(f"  Principal Point: cx={self.camera_matrix[0,2]:.1f}, cy={self.camera_matrix[1,2]:.1f}")
            print(f"\n📍 Detection Settings:")
            print(f"Detection Distance: {self.detection_distance} m")
            print(f"(Objects placed at fixed distance along camera's optical axis)")
            
            print(f"\n{'END-EFFECTOR POSE (DEFAULT)':-^70}")
            print(f"EE Position: [{self.ee_position[0]:.6f}, {self.ee_position[1]:.6f}, {self.ee_position[2]:.6f}] m")
            print(f"EE Quaternion: [{self.ee_quat[0]:.3f}, {self.ee_quat[1]:.3f}, {self.ee_quat[2]:.3f}, {self.ee_quat[3]:.3f}]")
            
            print(f"\n{'CAMERA-TO-EE OFFSET (CALCULATED)':-^70}")
            print(f"Position Offset (EE frame): [{self.cam_offset_position[0]:.6f}, {self.cam_offset_position[1]:.6f}, {self.cam_offset_position[2]:.6f}] m")
            print(f"Quaternion Offset: [{self.cam_offset_quat[0]:.3f}, {self.cam_offset_quat[1]:.3f}, {self.cam_offset_quat[2]:.3f}, {self.cam_offset_quat[3]:.3f}]")
            
            print(f"\n{'CAMERA POSE IN BASE FRAME':-^70}")
            print(f"Camera Position (world): [{self.cam_pos_world[0]:.6f}, {self.cam_pos_world[1]:.6f}, {self.cam_pos_world[2]:.6f}] m")
            print(f"                         [{self.cam_pos_world[0]*1000:.3f}, {self.cam_pos_world[1]*1000:.3f}, {self.cam_pos_world[2]*1000:.3f}] mm")
            print(f"Camera Quaternion (world): [{self.cam_quat_world[0]:.3f}, {self.cam_quat_world[1]:.3f}, {self.cam_quat_world[2]:.3f}, {self.cam_quat_world[3]:.3f}]")
            
            opencv_to_cam_quat = ROBOT_CONFIG['opencv_to_camera']['quaternion']
            print(f"\n{'OPENCV-TO-CAMERA TRANSFORM':-^70}")
            print(f"Quaternion: [{opencv_to_cam_quat[0]:.3f}, {opencv_to_cam_quat[1]:.3f}, {opencv_to_cam_quat[2]:.3f}, {opencv_to_cam_quat[3]:.3f}]")
            
            print(f"\n{'STEP 1: PIXEL COORDINATES':-^70}")
            print(f"Input Pixel: ({pixel_x:.1f}, {pixel_y:.1f})")
            print(f"Image Center: ({self.c_width/2:.1f}, {self.c_height/2:.1f})")
            print(f"Offset from center: ({pixel_x - self.c_width/2:.1f}, {pixel_y - self.c_height/2:.1f})")
        
        # For verbose output, show intermediate steps
        if verbose:
            pixel = np.array([pixel_x, pixel_y, 1.0])
            ray_opencv = np.linalg.inv(self.camera_matrix) @ pixel
            print(f"\n{'STEP 2A: RAY IN OPENCV FRAME':-^70}")
            print(f"Ray (OpenCV frame): [{ray_opencv[0]:.6f}, {ray_opencv[1]:.6f}, {ray_opencv[2]:.6f}]")
        
        # Step 1: Pixel to camera ray (includes OpenCV→Camera transformation)
        ray_cam = self.pixel_to_camera_ray(pixel_x, pixel_y)
        
        if verbose:
            print(f"\n{'STEP 2B: RAY IN CAMERA FRAME (AFTER OPENCV→CAM)':-^70}")
            print(f"Ray (camera frame): [{ray_cam[0]:.6f}, {ray_cam[1]:.6f}, {ray_cam[2]:.6f}]")
            print(f"Ray magnitude: {np.linalg.norm(ray_cam):.6f}")
            ray_normalized = ray_cam / np.linalg.norm(ray_cam)
            print(f"Ray (normalized): [{ray_normalized[0]:.6f}, {ray_normalized[1]:.6f}, {ray_normalized[2]:.6f}]")
        
        # Step 2C: Camera ray to world point
        position = self.camera_ray_to_world_point(ray_cam)
        
        if verbose:
            print(f"\n{'STEP 2C: 3D POSITION IN BASE FRAME (FIXED DISTANCE)':-^70}")
            print(f"Distance from camera: {self.detection_distance} m along ray direction")
            print(f"Position (meters): [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
            print(f"Position (mm):     [{position[0]*1000:.3f}, {position[1]*1000:.3f}, {position[2]*1000:.3f}]")
        
        # Step 3: 2D orientation to 3D quaternion
        quaternion = self.orientation_2d_to_quaternion()
        
        if verbose:
            print(f"\n{'STEP 3: 3D ORIENTATION IN BASE FRAME':-^70}")
            print(f"2D Orientation Angle: {self.orientation_angle_2d:.6f} rad ({np.rad2deg(self.orientation_angle_2d):.2f}°)")
            print(f"Quaternion (base frame): [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
            
            # Convert to Euler angles for easier interpretation
            rot = R.from_quat(quaternion)
            euler = rot.as_euler('xyz', degrees=True)
            print(f"Euler Angles (XYZ): Roll={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Yaw={euler[2]:.2f}°")
        
        if verbose:
            print(f"\n{'FINAL OBJECT POSE':-^70}")
            print(f"Object Name: object_0")
            print(f"Position: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}] m")
            print(f"Quaternion: [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
            print(f"\nThis is the pose that would be published to /objects_poses topic")
            print("="*70 + "\n")
        
        return {
            'position': position,
            'quaternion': quaternion,
            'euler_deg': R.from_quat(quaternion).as_euler('xyz', degrees=True)
        }
    
    def set_detection_distance(self, distance):
        """
        Change detection distance (distance from camera along optical axis)
        
        Args:
            distance: Distance in meters from camera
        """
        self.detection_distance = distance


def main():
    parser = argparse.ArgumentParser(
        description='Transform OpenCV pixel coordinates to robot base frame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Center of frame with default settings
  python3 opencv_to_base_transform.py --pixel-x 640 --pixel-y 360
  
  # Top-left corner
  python3 opencv_to_base_transform.py --pixel-x 0 --pixel-y 0
  
  # With custom camera pose and 45° object orientation
  python3 opencv_to_base_transform.py --pixel-x 640 --pixel-y 360 \\
      --cam-pos -0.058649 0.000124 0.042411 --cam-quat 0 0 0 1 --orientation 0.785
  
  # Custom detection distance (e.g., objects 50cm from camera)
  python3 opencv_to_base_transform.py --pixel-x 640 --pixel-y 360 --distance 0.5
  
  # Interactive mode
  python3 opencv_to_base_transform.py --interactive
        """)
    
    parser.add_argument('--pixel-x', type=float, help='X pixel coordinate')
    parser.add_argument('--pixel-y', type=float, help='Y pixel coordinate')
    parser.add_argument('--cam-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Camera position [x y z] in base frame (meters)')
    parser.add_argument('--cam-quat', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                        help='Camera quaternion [x y z w] in base frame')
    parser.add_argument('--orientation', type=float, default=0.0,
                        help='2D object orientation angle in radians (default: 0.0)')
    parser.add_argument('--distance', type=float, default=None,
                        help='Detection distance in meters from camera (default: from config)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Initialize transformer
    transformer = OpenCVToBaseTransform(
        cam_position=args.cam_pos,
        cam_quat=args.cam_quat,
        orientation_angle_2d=args.orientation
    )
    
    # Set detection distance (only override config if explicitly provided)
    if args.distance is not None:
        transformer.set_detection_distance(args.distance)
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE MODE: OpenCV to Base Frame Transformer")
        print("="*70)
        print("Enter pixel coordinates to see the transformation")
        print("Type 'quit' or 'exit' to quit")
        print("Type 'distance' to change detection distance")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("Enter pixel coordinates (x y) or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'distance':
                    distance_input = input(f"Detection distance (m) [{transformer.detection_distance}]: ").strip()
                    if distance_input:
                        distance = float(distance_input)
                        transformer.set_detection_distance(distance)
                        print(f"✓ Detection distance set to {distance}m\n")
                    continue
                
                if user_input.lower() == 'center':
                    pixel_x = transformer.c_width / 2
                    pixel_y = transformer.c_height / 2
                else:
                    coords = user_input.split()
                    if len(coords) != 2:
                        print("Error: Please enter exactly 2 numbers (x y)")
                        continue
                    pixel_x, pixel_y = float(coords[0]), float(coords[1])
                
                # Transform and display
                result = transformer.transform_pixel_to_pose(pixel_x, pixel_y, verbose=not args.quiet)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    else:
        # Single transformation mode
        if args.pixel_x is None or args.pixel_y is None:
            # Default to center if not specified
            pixel_x = transformer.c_width / 2
            pixel_y = transformer.c_height / 2
            print("No pixel coordinates specified, using image center")
        else:
            pixel_x = args.pixel_x
            pixel_y = args.pixel_y
        
        # Perform transformation
        result = transformer.transform_pixel_to_pose(pixel_x, pixel_y, verbose=not args.quiet)


if __name__ == '__main__':
    main()

