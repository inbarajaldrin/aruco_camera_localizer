#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys

class YoloPromptUpdater(Node):
    def __init__(self):
        super().__init__('yolo_prompt_updater')
        self.publisher = self.create_publisher(String, '/yolo_prompts_update', 10)

    def update_prompts(self, prompts, prompt_map=None):
        """Update YOLO prompts via topic publish"""
        msg = String()
        msg.data = json.dumps({
            'prompts': prompts,
            'prompt_map': prompt_map if prompt_map else {}
        })
        self.publisher.publish(msg)
        self.get_logger().info(f"Published YOLO prompt update: {prompts}")
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 update_yolo_prompts.py <prompt1> <prompt2> ... [--prompt-map prompt1:color1 prompt2:color2]")
        print("Example: python3 update_yolo_prompts.py 'blue object' 'red object' 'hand' --prompt-map 'blue object:blue' 'red object:red' 'hand:hand'")
        return

    rclpy.init()
    updater = YoloPromptUpdater()

    # Parse arguments
    prompts = []
    prompt_map = {}
    parsing_colors = False

    for arg in sys.argv[1:]:
        if arg == '--prompt-map':
            parsing_colors = True
            continue
        elif parsing_colors:
            if ':' in arg:
                prompt, color = arg.split(':', 1)
                prompt_map[prompt.strip()] = color.strip()
        else:
            prompts.append(arg)

    if not prompts:
        print("Error: No prompts provided")
        return

    print(f"Updating YOLO prompts to: {prompts}")
    if prompt_map:
        print(f"Prompt mapping: {prompt_map}")

    success = updater.update_prompts(prompts, prompt_map)

    if success:
        print("YOLO prompts update published!")
    else:
        print("Failed to publish YOLO prompts update")
        sys.exit(1)

    updater.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
