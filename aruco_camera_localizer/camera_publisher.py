#!/usr/bin/env python3
"""
Camera Publisher Node

This node handles camera selection and publishes frames to ROS2 topics.
It acts as the camera driver layer, separating hardware access from processing.
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
from aruco_camera_localizer.camera_selection import detect_available_cameras, select_camera
from aruco_camera_localizer.config_loader import get_config


class CameraPublisher(Node):
    def __init__(self, camera_id, publish_topic='/camera/image_raw'):
        super().__init__('camera_publisher')
        
        # Load configuration
        self.config = get_config()
        self.c_width = self.config.get_camera_width()
        self.c_height = self.config.get_camera_height()
        
        # Create publishers
        self.raw_image_pub = self.create_publisher(Image, publish_topic, 10)
        self.bridge = CvBridge()
        self.publish_topic = publish_topic
        
        # Initialize camera
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera {self.camera_id}")
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.c_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.c_height)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
        
        self.get_logger().info(f"Camera {self.camera_id} opened successfully")
        self.get_logger().info(f"Resolution: {self.c_width}x{self.c_height}")
        self.get_logger().info(f"Publishing to: {self.publish_topic}")
        
        # Create timer for capturing and publishing frames
        # 30 FPS = 0.033 seconds
        self.timer = self.create_timer(0.033, self.capture_and_publish)
        
        self.frame_count = 0
        
    def capture_and_publish(self):
        """Capture a frame and publish it to ROS2 topic"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return
        
        # Convert OpenCV image to ROS2 Image message
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_optical_frame"
            
            # Publish the image
            self.raw_image_pub.publish(img_msg)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Published {self.frame_count} frames")
                
        except Exception as e:
            self.get_logger().error(f"Failed to publish frame: {e}")
    
    def destroy_node(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(description="Publish camera frames to ROS2 topics.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--publish-topic", type=str, default="/camera/image_raw",
                        help="ROS2 topic to publish camera images (default: /camera/image_raw)")
    return parser.parse_args()


def main(args=None):
    # Parse command line arguments (before rclpy.init)
    parsed_args = parse_args()
    
    # Determine which camera to use
    if parsed_args.camera_id is not None:
        cam_id = parsed_args.camera_id
        print(f"Using specified camera ID: {cam_id}")
    else:
        print("Detecting available cameras...")
        available = detect_available_cameras()
        if not available:
            print("No cameras detected!")
            return
        cam_id = select_camera(available)
        if cam_id is None:
            print("No camera selected!")
            return
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create and run the camera publisher node
        camera_publisher = CameraPublisher(cam_id, publish_topic=parsed_args.publish_topic)
        
        print(f"\n{'='*60}")
        print(f"Camera Publisher Node Started")
        print(f"{'='*60}")
        print(f"Camera ID: {cam_id}")
        print(f"Publishing to: {parsed_args.publish_topic}")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        rclpy.spin(camera_publisher)
        
    except KeyboardInterrupt:
        print("\nShutting down camera publisher...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'camera_publisher' in locals():
            camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

