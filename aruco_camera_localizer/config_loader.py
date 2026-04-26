"""
Configuration Loader for ArUco Camera Localizer
Loads robot configuration from YAML file
"""

import yaml
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


class RobotConfig:
    """Loads and provides access to robot configuration from YAML file"""
    
    def __init__(self, config_file="robot_config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_file: Name of the config file (default: robot_config.yaml)
        """
        # Try to get package share directory, fall back to relative path
        try:
            pkg_dir = get_package_share_directory("aruco_camera_localizer")
            config_path = os.path.join(pkg_dir, "config", config_file)
        except:
            # Fallback for development/testing
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pkg_dir, "config", config_file)
        
        # Load YAML file
        with open(config_path, 'r') as f:
            self._full_config = yaml.safe_load(f)
        
        # Get active robot configuration
        self._active_robot = self._full_config.get('active_robot', 'ur5e')
        self._config = self._full_config['robots'][self._active_robot]
        
        print(f"Loaded configuration for robot: {self._active_robot}")
    
    def get_active_robot_name(self):
        """Get the name of the active robot"""
        return self._active_robot
    
    # ========== ROS Topics Configuration ==========
    
    def get_tcp_pose_topic(self):
        """Get the ROS topic name for TCP/end-effector pose"""
        return self._config['topics']['tcp_pose']
    
    # ========== Camera Configuration ==========
    
    def get_camera_width(self):
        """Get camera width in pixels"""
        return self._config['camera']['width']
    
    def get_camera_height(self):
        """Get camera height in pixels"""
        return self._config['camera']['height']
    
    def get_camera_hfov(self):
        """Get camera horizontal field of view in degrees"""
        return self._config['camera']['horizontal_fov']
    
    def get_camera_vfov(self):
        """Get camera vertical field of view in degrees"""
        return self._config['camera']['vertical_fov']
    
    def get_camera_matrix(self):
        """Calculate and return camera matrix from FOV and resolution"""
        width = self.get_camera_width()
        height = self.get_camera_height()
        hfov = self.get_camera_hfov()
        vfov = self.get_camera_vfov()
        
        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        
        return np.array([[fx, 0, width / 2],
                        [0, fy, height / 2],
                        [0, 0, 1]], dtype=np.float32)
    
    def get_opencv_to_camera_quaternion(self):
        """Get OpenCV frame to camera frame quaternion transformation"""
        return np.array(self._config['camera']['opencv_to_camera']['quaternion'])
    
    # ========== Detection Configuration ==========

    def get_ground_plane_z_offset(self):
        """Get ground plane Z offset in arm base frame (meters), or None if not configured"""
        return self._config['detection'].get('ground_plane_z_offset', None)

    def get_yolo_config(self):
        """Get YOLO detection config for the active robot, or empty dict if not configured.

        Returns a dict with optional keys:
          - prompts:     list[str]     text prompts fed to YOLOE (e.g. ['red object', ...])
          - prompt_map:  dict[str,str] class_name → color_name renames
                                       (e.g. {'red object': 'red'})
          - confidence:  float         YOLOE confidence threshold (overrides --yolo-conf default)

        Per-robot YOLO vocabulary lives next to active_robot so each
        physical setup can ship its own color set without CLI repetition.
        Mirrors how cup ArUco offsets live in aruco_config.json's per-robot
        block. CLI args (--yolo-prompts, --yolo-conf) still override this.
        """
        return dict(self._config.get('detection', {}).get('yolo', {}))
    
    # ========== Transform Configuration ==========
    
    def get_ee_default_position(self):
        """Get default end-effector position in base frame (meters)"""
        return np.array(self._config['transforms']['ee_default']['position'])
    
    def get_ee_default_quaternion(self):
        """Get default end-effector orientation in base frame (quaternion)"""
        return np.array(self._config['transforms']['ee_default']['quaternion'])
    
    def get_camera_default_position(self):
        """Get default camera position in base frame (meters)"""
        return np.array(self._config['transforms']['camera_default']['position'])
    
    def get_camera_default_quaternion(self):
        """Get default camera orientation in base frame (quaternion)"""
        return np.array(self._config['transforms']['camera_default']['quaternion'])
    
    def get_calibration_offset(self):
        """Get calibration offset for fine-tuning (meters)"""
        offset = self._config['transforms']['calibration_offset']
        return offset['x'], offset['y'], offset['z']

    # ========== Filter & Tracking Configuration ==========
    # These delegate to the standalone filter_config.yaml via get_filter_config().

    def get_z_range(self):
        fc = get_filter_config()
        return fc.get('z_range_min', 0.05), fc.get('z_range_max', 2.0)

    def get_stability_params(self):
        fc = get_filter_config()
        return (fc.get('max_movement', 0.05),
                fc.get('hold_required', 5),
                fc.get('ghost_timeout', 15))

    def get_kalman_process_noise(self):
        fc = get_filter_config()
        return {k: fc.get(k, d) for k, d in [
            ('q_pos_xy', 1e-8), ('q_pos_z', 1e-4), ('q_quat', 1e-2),
            ('q_velocity', 1e-2), ('q_acceleration', 1e-1),
            ('q_multiplier_moving', 10.0), ('q_multiplier_static', 0.1),
        ]}

    def get_kalman_measurement_noise(self):
        fc = get_filter_config()
        return {k: fc.get(k, d) for k, d in [
            ('r_pos_xy', 1e-4), ('r_pos_z', 5e-1), ('r_quat', 1e-2),
        ]}

    def get_blend_factor(self):
        return get_filter_config().get('blend_factor', 0.99)


# Singleton instance for easy access
_config_instance = None

def get_config():
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = RobotConfig()
    return _config_instance


# ============================================================================
# Filter config — loaded from config/filter_config.yaml (separate from robot)
# ============================================================================
_filter_config_instance = None

def get_filter_config():
    """Load filter_config.yaml once and return it as a flat dict."""
    global _filter_config_instance
    if _filter_config_instance is not None:
        return _filter_config_instance

    try:
        pkg_dir = get_package_share_directory("aruco_camera_localizer")
        path = os.path.join(pkg_dir, "config", "filter_config.yaml")
    except Exception:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(pkg_dir, "config", "filter_config.yaml")

    try:
        with open(path, 'r') as f:
            _filter_config_instance = yaml.safe_load(f) or {}
        print(f"Loaded filter config from {path}")
    except FileNotFoundError:
        print(f"filter_config.yaml not found at {path}, using defaults")
        _filter_config_instance = {}

    return _filter_config_instance

