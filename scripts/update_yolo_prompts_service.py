#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from max_camera_msgs.srv import UpdateYoloPrompts
import json
import sys

class YoloPromptUpdater(Node):
    def __init__(self):
        super().__init__('yolo_prompt_updater')
        self.client = self.create_client(UpdateYoloPrompts, '/update_yolo_prompts')
        
    def update_prompts(self, prompts, color_map=None):
        """Update YOLO prompts via service call"""
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service not available')
            return False
            
        request = UpdateYoloPrompts.Request()
        request.prompts_json = json.dumps(prompts)
        request.color_map_json = json.dumps(color_map) if color_map else "{}"
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Success: {response.message}")
                return True
            else:
                self.get_logger().error(f"Failed: {response.message}")
                return False
        else:
            self.get_logger().error("Service call failed")
            return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 update_yolo_prompts.py <prompt1> <prompt2> ... [--color-map prompt1:color1 prompt2:color2]")
        print("Example: python3 update_yolo_prompts.py 'blue object' 'red object' 'hand' --color-map 'blue object:blue' 'red object:red' 'hand:hand'")
        return
        
    rclpy.init()
    updater = YoloPromptUpdater()
    
    # Parse arguments
    prompts = []
    color_map = {}
    parsing_colors = False
    
    for arg in sys.argv[1:]:
        if arg == '--color-map':
            parsing_colors = True
            continue
        elif parsing_colors:
            if ':' in arg:
                prompt, color = arg.split(':', 1)
                color_map[prompt.strip()] = color.strip()
        else:
            prompts.append(arg)
    
    if not prompts:
        print("Error: No prompts provided")
        return
        
    print(f"Updating YOLO prompts to: {prompts}")
    if color_map:
        print(f"Color mapping: {color_map}")
    
    success = updater.update_prompts(prompts, color_map)
    
    if success:
        print("YOLO prompts updated successfully!")
    else:
        print("Failed to update YOLO prompts")
        sys.exit(1)
        
    updater.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
