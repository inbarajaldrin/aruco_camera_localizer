# Dynamic YOLO Prompts - Usage Guide

This folder contains scripts for dynamically updating YOLO detection prompts while the camera localizer is running.

## Quick Start

### 1. Launch the Camera Localizer
```bash
ros2 run aruco_camera_localizer localize_yoloe
```

### 2. Add New Prompts Dynamically

#### Method A: Using Service Call (Recommended)
```bash
# Add "cork" prompt mapped to "jenga block"
ros2 service call /update_yolo_prompts max_camera_msgs/srv/UpdateYoloPrompts "{prompts_json: '[\"blue object\", \"red object\", \"green object\", \"yellow object\", \"hand\", \"ipad\", \"cork\"]', color_map_json: '{\"blue object\": \"blue\", \"red object\": \"red\", \"green object\": \"green\", \"yellow object\": \"yellow\", \"hand\": \"hand\", \"ipad\": \"box\", \"cork\": \"jenga block\"}'}"
```

#### Method B: Using Topic Publishing
```bash
# Add "cork" prompt mapped to "jenga block"
ros2 topic pub /yolo_prompts_update std_msgs/msg/String "data: '{\"prompts\": [\"blue object\", \"red object\", \"green object\", \"yellow object\", \"hand\", \"ipad\", \"cork\"], \"color_map\": {\"blue object\": \"blue\", \"red object\": \"red\", \"green object\": \"green\", \"yellow object\": \"yellow\", \"hand\": \"hand\", \"ipad\": \"box\", \"cork\": \"jenga block\"}}'"
```

## Available Scripts

### `update_yolo_prompts_service.py`
- **Purpose**: Service-based prompt updates with confirmation
- **Usage**: `python3 update_yolo_prompts_service.py "prompt1" "prompt2" --color-map "prompt1:color1"`
- **Pros**: Guaranteed delivery, error handling, confirmation

### `update_yolo_prompts_topic.py`
- **Purpose**: Topic-based prompt updates (fire-and-forget)
- **Usage**: `python3 update_yolo_prompts_topic.py "prompt1" "prompt2" --color-map "prompt1:color1"`
- **Pros**: Fast, real-time updates

## Examples

### Example 1: Add a new object type
```bash
# Add "bottle" detection
ros2 service call /update_yolo_prompts max_camera_msgs/srv/UpdateYoloPrompts "{prompts_json: '[\"blue object\", \"red object\", \"hand\", \"ipad\", \"bottle\"]', color_map_json: '{\"blue object\": \"blue\", \"red object\": \"red\", \"hand\": \"hand\", \"ipad\": \"box\", \"bottle\": \"green\"}'}"
```

### Example 2: Add multiple new objects
```bash
# Add "cup" and "plate" detections
ros2 topic pub /yolo_prompts_update std_msgs/msg/String "data: '{\"prompts\": [\"blue object\", \"red object\", \"hand\", \"ipad\", \"cup\", \"plate\"], \"color_map\": {\"blue object\": \"blue\", \"red object\": \"red\", \"hand\": \"hand\", \"ipad\": \"box\", \"cup\": \"yellow\", \"plate\": \"white\"}}'"
```

### Example 3: Using the Python scripts
```bash
# Service method
python3 update_yolo_prompts_service.py "blue object" "red object" "hand" "ipad" "cork" --color-map "blue object:blue" "red object:red" "hand:hand" "ipad:box" "cork:jenga block"

# Topic method
python3 update_yolo_prompts_topic.py "blue object" "red object" "hand" "ipad" "cork" --color-map "blue object:blue" "red object:red" "hand:hand" "ipad:box" "cork:jenga block"
```

## Monitoring Current Prompts

### Check current prompts via topic
```bash
ros2 topic echo /yolo_prompts
```

### Check service availability
```bash
ros2 service list | grep yolo
```

## Notes

- **No restart required**: Prompts are updated dynamically without stopping the localizer
- **Real-time updates**: Changes take effect immediately in the next detection cycle
- **Thread-safe**: Multiple updates can be sent safely
- **Backward compatible**: Original hardcoded prompts still work as defaults

## Troubleshooting

### Service not available
- Make sure the localizer is running
- Check if the service is listed: `ros2 service list | grep yolo`

### Prompts not updating
- Check the localizer console for error messages
- Verify JSON format in the service/topic calls
- Try restarting the localizer if needed

### Python script errors
- Ensure ROS2 environment is sourced: `source /opt/ros/humble/setup.bash`
- Check Python dependencies are installed
