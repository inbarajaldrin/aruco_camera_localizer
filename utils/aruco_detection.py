#!/usr/bin/env python3
"""
Simple ArUco Localizer - Detects markers and displays their pose information

This is a simplified version that:
- Detects ArUco markers (4x4 and 5x5) in camera feed
- Shows marker ID and object name (from aruco-grasp-annotator data when available)
- Displays marker pose (position and orientation)
- Uses data_path_finder to load object/marker mapping from aruco-grasp-annotator data dir
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Optional
import argparse
import threading

# Allow importing aruco_camera_localizer when running from repo (e.g. python utils/aruco_detection.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from aruco_camera_localizer.data_path_finder import find_aruco_data_dir
    DATA_PATH_FINDER_AVAILABLE = True
except ImportError:
    DATA_PATH_FINDER_AVAILABLE = False
    find_aruco_data_dir = None

# ROS2 imports (optional)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


# ArUco dictionary name -> OpenCV dict constant (4x4 and 5x5)
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
}

DEFAULT_MARKER_SIZE_M = 0.021  # 21 mm default marker size in meters


def _load_marker_registry(data_dir: Path) -> dict:
    """
    Load (marker_id, dict_name) -> {object_name, marker_size} from data_dir/aruco/*_aruco.json.
    Returns a dict keyed by (marker_id, dict_name) with object_name and marker_size.
    """
    registry = {}
    aruco_dir = data_dir / "aruco"
    if not aruco_dir.exists():
        return registry
    for json_path in aruco_dir.glob("*_aruco.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        object_name = json_path.stem.replace("_aruco", "")
        dict_name = data.get("aruco_dictionary", "DICT_4X4_50")
        if dict_name not in ARUCO_DICTS:
            continue
        for m in data.get("markers", []):
            mid = m.get("aruco_id")
            if mid is None:
                continue
            registry[(int(mid), dict_name)] = {
                "object_name": object_name,
                "marker_size": DEFAULT_MARKER_SIZE_M,
            }
    return registry


class SimpleArUcoLocalizer:
    """Simple ArUco marker detector and pose reporter (4x4 and 5x5)."""
    
    def __init__(self, camera_id: int = 0, image_topic: Optional[str] = None,
                 dict_names: Optional[list] = None):
        self.camera_id = camera_id
        self.image_topic = image_topic
        self.use_ros2 = image_topic is not None
        
        # Marker registry from data path finder: (marker_id, dict_name) -> {object_name, marker_size}
        self.marker_registry = {}
        if DATA_PATH_FINDER_AVAILABLE and find_aruco_data_dir is not None:
            data_dir = find_aruco_data_dir()
            if data_dir is not None:
                self.marker_registry = _load_marker_registry(data_dir)
                print(f"✅ Loaded marker registry from {data_dir} ({len(self.marker_registry)} entries)")
            else:
                print("⚠️  aruco-grasp-annotator data dir not found; object names will be 'Unknown'")
        else:
            print("⚠️  data_path_finder not available; object names will be 'Unknown'")
        
        # Detectors: use dict_names or all 4x4 and 5x5
        self.aruco_params = cv2.aruco.DetectorParameters()
        names_to_use = dict_names if dict_names else list(ARUCO_DICTS)
        self.detectors = []
        for dict_name in names_to_use:
            if dict_name not in ARUCO_DICTS:
                continue
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
            detector = cv2.aruco.ArucoDetector(aruco_dict, self.aruco_params)
            self.detectors.append((dict_name, detector))
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ROS2 bridge (if using ROS2)
        if self.use_ros2:
            if not ROS2_AVAILABLE:
                raise RuntimeError("ROS2 not available. Install rclpy and cv_bridge to use --image-topic")
            self.bridge = CvBridge()
        
        dict_list = ", ".join(d[0] for d in self.detectors)
        print(f"✅ ArUco detector initialized ({dict_list})")
    
    def _get_object_and_size(self, marker_id: int, dict_name: str) -> tuple:
        """Get (object_name, marker_size) for (marker_id, dict_name)."""
        entry = self.marker_registry.get((marker_id, dict_name))
        if entry is None:
            return "Unknown", DEFAULT_MARKER_SIZE_M
        return entry["object_name"], entry["marker_size"]
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> tuple:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw) in radians."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    def _draw_axis(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, length: float = 0.03) -> np.ndarray:
        """Draw coordinate axes on the frame."""
        axis_points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float32)
        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        origin = tuple(image_points[0][0].astype(int))
        x_axis = tuple(image_points[1][0].astype(int))
        y_axis = tuple(image_points[2][0].astype(int))
        z_axis = tuple(image_points[3][0].astype(int))
        
        cv2.line(frame, origin, x_axis, (0, 0, 255), 3)  # X - Red
        cv2.line(frame, origin, y_axis, (0, 255, 0), 3)  # Y - Green
        cv2.line(frame, origin, z_axis, (255, 0, 0), 3)  # Z - Blue
        
        return frame
    
    def _initialize_camera_matrix(self, frame_width: int, frame_height: int):
        """Initialize camera matrix with default values if not calibrated."""
        if self.camera_matrix is None:
            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2)
            
            self.camera_matrix = np.array([[focal_length, 0, center[0]],
                                          [0, focal_length, center[1]],
                                          [0, 0, 1]], dtype=np.float32)
            
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            print(f"⚠️  Using default camera parameters")
    
    def load_camera_calibration(self, calibration_file: str):
        """Load camera calibration from file."""
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            self.camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
            self.dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float32)
            print(f"✅ Loaded camera calibration")
        except Exception as e:
            print(f"⚠️  Failed to load calibration: {e}")
    
    def process_frame(self, frame, show_axes=True, frame_count=0):
        """Process a single frame for marker detection (4x4 and 5x5)."""
        frame_height, frame_width = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect with all dictionaries and collect (corners, id, dict_name)
        all_corners, all_ids, all_dict_names = [], [], []
        for dict_name, detector in self.detectors:
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    all_corners.append(corners[i])
                    all_ids.append(int(marker_id))
                    all_dict_names.append(dict_name)
        
        detected_markers = len(all_ids)
        if all_corners:
            # Draw all detected markers (corners from all dicts)
            combined_corners = [c for c in all_corners]
            combined_ids = np.array(all_ids, dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(frame, combined_corners, combined_ids)
            
            for i, (marker_id, dict_name) in enumerate(zip(all_ids, all_dict_names)):
                object_name, marker_size = self._get_object_and_size(marker_id, dict_name)
                
                half_size = marker_size / 2
                obj_points = np.array([
                    [-half_size,  half_size, 0],
                    [ half_size,  half_size, 0],
                    [ half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)
                
                img_points = all_corners[i].reshape(4, 2)
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not success:
                    continue
                
                if show_axes:
                    self._draw_axis(frame, rvec, tvec, length=marker_size / 2)
                
                R_marker, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = self._rotation_matrix_to_euler_angles(R_marker)
                tvec_flat = tvec.flatten()
                roll_deg, pitch_deg, yaw_deg = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
                
                y_offset = 30 + (i * 180)
                cv2.putText(frame, f"Marker ID: {marker_id} ({dict_name})", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Object: {object_name}", (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Position (m): X:{tvec_flat[0]:.3f} Y:{tvec_flat[1]:.3f} Z:{tvec_flat[2]:.3f}",
                            (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Orientation (deg): R:{roll_deg:6.1f} P:{pitch_deg:6.1f} Y:{yaw_deg:6.1f}",
                            (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print(f"Marker ID: {marker_id:2d} ({dict_name:12s}) | Object: {object_name:20s} | "
                      f"Pos: [{tvec_flat[0]:6.3f}, {tvec_flat[1]:6.3f}, {tvec_flat[2]:6.3f}] | "
                      f"Rot: [R:{roll_deg:6.1f} P:{pitch_deg:6.1f} Y:{yaw_deg:6.1f}]")
        
        # Display info overlay
        cv2.putText(frame, f"FPS: {frame_count}", (10, frame_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Markers Detected: {detected_markers}", (10, frame_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Q: Quit | C: Toggle Axes", (10, frame_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run the simple marker detection and pose reporting."""
        if self.use_ros2:
            self._run_ros2()
        else:
            self._run_camera()
    
    def _run_camera(self):
        """Run with camera input."""
        print(f"\n🎥 Starting camera stream")
        print("📍 Detecting ArUco markers and reporting poses")
        print("Press 'q' to quit, 'c' to toggle coordinate axes")
        
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            print("   Possible reasons:")
            print("   - Camera is already in use by another application")
            print("   - Camera permissions not granted (check System Settings)")
            print("   - Invalid camera ID (try --camera 1, 2, etc.)")
            return
        
        # Test reading a frame
        ret, test_frame = cap.read()
        if not ret:
            print(f"⚠️  Camera {self.camera_id} opened but cannot read frames")
            print("   Camera may be in use or not responding")
            cap.release()
            return
        
        print(f"✅ Camera {self.camera_id} opened successfully")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self._initialize_camera_matrix(frame_width, frame_height)
        
        show_axes = True
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame from camera. Check if camera is in use or permissions are granted.")
                print("   Trying to continue...")
                # Wait a bit and try again instead of breaking immediately
                import time
                time.sleep(0.1)
                continue
            
            frame_count += 1
            frame = self.process_frame(frame, show_axes, frame_count)
            
            # Show frame
            cv2.imshow("Simple ArUco Localizer", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('c'):
                show_axes = not show_axes
                print(f"Coordinate axes: {'ON' if show_axes else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera stream stopped")
    
    def _run_ros2(self):
        """Run with ROS2 image topic input."""
        if not ROS2_AVAILABLE:
            print("❌ ROS2 not available. Install rclpy and cv_bridge to use --image-topic")
            return
        
        print(f"\n🎥 Starting ROS2 image subscriber")
        print(f"📍 Subscribing to topic: {self.image_topic}")
        print("📍 Detecting ArUco markers and reporting poses")
        print("Press 'q' to quit, 'c' to toggle coordinate axes")
        
        # Create ROS2 node
        rclpy.init()
        node = ROS2ArUcoNode(self, self.image_topic)
        
        try:
            # Wait for first frame to initialize camera matrix
            print("⏳ Waiting for first frame from ROS2 topic...")
            import time
            max_wait_time = 10.0  # Wait up to 10 seconds
            start_time = time.time()
            
            while node.last_frame is None and (time.time() - start_time) < max_wait_time:
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.last_frame is not None:
                    break
            
            if node.last_frame is None:
                print("❌ No frames received from ROS2 topic. Check if the topic is publishing.")
                return
            
            # Initialize camera matrix from first frame
            frame_height, frame_width = node.last_frame.shape[:2]
            self._initialize_camera_matrix(frame_width, frame_height)
            print(f"✅ Camera parameters initialized from ROS2 topic ({frame_width}x{frame_height})")
            
            show_axes = True
            frame_count = 0
            
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                
                if node.last_frame is not None:
                    frame_count += 1
                    frame = node.last_frame.copy()
                    frame = self.process_frame(frame, show_axes, frame_count)
                    
                    # Show frame
                    cv2.imshow("Simple ArUco Localizer (ROS2)", frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n👋 Quitting...")
                        break
                    elif key == ord('c'):
                        show_axes = not show_axes
                        print(f"Coordinate axes: {'ON' if show_axes else 'OFF'}")
        except KeyboardInterrupt:
            print("\n👋 Quitting...")
        finally:
            node.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()
            print("✅ ROS2 stream stopped")


# Define ROS2 node class only if ROS2 is available
if ROS2_AVAILABLE:
    class ROS2ArUcoNode(Node):
        """ROS2 node for subscribing to image topics."""
        
        def __init__(self, localizer: SimpleArUcoLocalizer, image_topic: str):
            super().__init__('simple_aruco_localizer_ros2')
            self.localizer = localizer
            self.last_frame = None
            self.frame_received = threading.Event()
            
            # Subscribe to image topic
            self.subscription = self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                10
            )
            self.get_logger().info(f'Subscribed to {image_topic}')
        
        def image_callback(self, msg):
            """Callback for processing ROS2 image messages."""
            try:
                # Convert ROS2 image to OpenCV
                frame = self.localizer.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.last_frame = frame
                self.frame_received.set()
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')


def main():
    """Main function to parse arguments and run the localizer."""
    parser = argparse.ArgumentParser(
        description="Simple ArUco Localizer - Detects markers and displays pose information"
    )
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--image-topic', type=str, help='ROS2 image topic (e.g., /intel_camera_rgb_sim/image_raw)')
    parser.add_argument('--calibration', type=str, help='Camera calibration JSON file (optional)')
    parser.add_argument('--dict', dest='aruco_dict', type=str, choices=list(ARUCO_DICTS),
                        help='Use only this ArUco dictionary (default: both 4x4 and 5x5)')
    args = parser.parse_args()
    
    try:
        print("🚀 Simple ArUco Localizer")
        print("=" * 60)
        
        # Validate arguments
        if args.image_topic and args.camera != 0:
            print("⚠️  Warning: Both --image-topic and --camera specified. Using --image-topic.")
        
        dict_names = [args.aruco_dict] if getattr(args, 'aruco_dict', None) else None
        localizer = SimpleArUcoLocalizer(
            camera_id=args.camera if not args.image_topic else 0,
            image_topic=args.image_topic,
            dict_names=dict_names
        )
        
        if args.calibration:
            localizer.load_camera_calibration(args.calibration)
        
        localizer.run()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())