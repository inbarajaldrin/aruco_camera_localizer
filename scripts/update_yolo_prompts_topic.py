#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys

class YoloPromptTopicUpdater(Node):
    def __init__(self):
        super().__init__('yolo_prompt_topic_updater')
        self.publisher = self.create_publisher(String, '/yolo_prompts_update', 10)
        
    def update_prompts(self, prompts, color_map=None):
        """Update YOLO prompts via topic"""
        data = {
            'prompts': prompts,
            'color_map': color_map if color_map else {}
        }
        
        msg = String()
        msg.data = json.dumps(data)
        
        self.publisher.publish(msg)
        self.get_logger().info(f"Published YOLO prompts update: {prompts}")
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 update_yolo_prompts_topic.py <prompt1> <prompt2> ... [--color-map prompt1:color1 prompt2:color2]")
        print("Example: python3 update_yolo_prompts_topic.py 'blue object' 'red object' 'hand' --color-map 'blue object:blue' 'red object:red' 'hand:hand'")
        return
        
    rclpy.init()
    updater = YoloPromptTopicUpdater()
    
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
        print("YOLO prompts update published!")
    else:
        print("Failed to publish YOLO prompts update")
        sys.exit(1)
        
    updater.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
