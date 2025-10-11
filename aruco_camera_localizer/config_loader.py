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
    
    # ========== Transform Configuration ==========
    
    def get_camera_to_ee_position(self):
        """Get camera position offset from end-effector (meters)"""
        return np.array(self._config['transforms']['camera_to_ee']['position'])
    
    def get_camera_to_ee_quaternion(self):
        """Get camera orientation offset from end-effector (quaternion)"""
        return np.array(self._config['transforms']['camera_to_ee']['quaternion'])
    
    def get_calibration_offset(self):
        """Get calibration offset for fine-tuning (meters)"""
        offset = self._config['transforms']['calibration_offset']
        return offset['x'], offset['y'], offset['z']
    
    def get_ee_default_position(self):
        """Get default end-effector position in base frame (meters)"""
        return np.array(self._config['transforms']['ee_default']['position'])
    
    def get_ee_default_quaternion(self):
        """Get default end-effector orientation in base frame (quaternion)"""
        return np.array(self._config['transforms']['ee_default']['quaternion'])


# Singleton instance for easy access
_config_instance = None

def get_config():
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = RobotConfig()
    return _config_instance

