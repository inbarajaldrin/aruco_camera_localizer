# Camera Publisher System

## Overview

The camera system uses a publisher-subscriber architecture that separates hardware camera access from processing logic. This allows for flexible camera management and enables multiple nodes to consume the same camera feed.

**Architecture:**
```
Hardware Camera → camera_publisher → ROS2 Topic → localize_yoloe → Detections
```

## Quick Start

### Basic Usage

**Terminal 1 - Start Camera Publisher:**
```bash
ros2 run aruco_camera_localizer camera_publisher --camera-id 8
```

**Terminal 2 - Start Localizer:**
```bash
ros2 run aruco_camera_localizer localize_yoloe
```

### Robosort with Custom Prompts (Most Common)

**Terminal 1 - Start Camera Publisher:**
```bash
ros2 run aruco_camera_localizer camera_publisher --camera-id 8 --publish-topic /rgb_aruco
```

**Terminal 2 - Start Localizer with Prompt Set Mode:**
```bash
ros2 run aruco_camera_localizer localize_yoloe \
    --camera-topic /rgb_aruco \
    --yolo-mode prompt-set \
    --yolo-prompts "red object"
```

### Interactive Camera Selection

If you don't know your camera ID, omit the `--camera-id` parameter:

```bash
ros2 run aruco_camera_localizer camera_publisher
```

The system will:
1. Scan for available cameras
2. Show a preview of each camera
3. Press **ESC** to select, or any other key to skip

## Examples

### Example 1: Default Setup
```bash
# Terminal 1
ros2 run aruco_camera_localizer camera_publisher --camera-id 8

# Terminal 2
ros2 run aruco_camera_localizer localize_yoloe
```

### Example 2: Custom Topic
```bash
# Terminal 1 - Publish to custom topic
ros2 run aruco_camera_localizer camera_publisher \
    --camera-id 8 \
    --publish-topic /my_robot/camera/raw

# Terminal 2 - Subscribe to custom topic
ros2 run aruco_camera_localizer localize_yoloe \
    --camera-topic /my_robot/camera/raw
```

### Example 3: Custom YOLO Settings with Prompt Set Mode
```bash
# Terminal 1
ros2 run aruco_camera_localizer camera_publisher --camera-id 8

# Terminal 2 - Custom YOLO detection with prompt set mode
ros2 run aruco_camera_localizer localize_yoloe \
    --camera-topic /rgb_aruco \
    --yolo-mode prompt-set \
    --yolo-prompts "red object" "blue object"
```

### Example 4: Multiple Cameras
```bash
# Terminal 1 - Camera 1
ros2 run aruco_camera_localizer camera_publisher \
    --camera-id 8 \
    --publish-topic /camera1/raw

# Terminal 2 - Camera 2
ros2 run aruco_camera_localizer camera_publisher \
    --camera-id 10 \
    --publish-topic /camera2/raw

# Terminal 3 - Process Camera 1
ros2 run aruco_camera_localizer localize_yoloe \
    --camera-topic /camera1/raw

# Terminal 4 - Process Camera 2 (if needed)
ros2 run aruco_camera_localizer localize_yoloe \
    --camera-topic /camera2/raw
```

### Example 5: No Pushers, Suppress Prints
```bash
# Terminal 1
ros2 run aruco_camera_localizer camera_publisher --camera-id 8

# Terminal 2
ros2 run aruco_camera_localizer localize_yoloe \
    --no-pushers \
    --suppress-prints
```

### Example 6: Push Recommendations
```bash
# Terminal 1
ros2 run aruco_camera_localizer camera_publisher --camera-id 8

# Terminal 2
ros2 run aruco_camera_localizer localize_yoloe \
    --recommend-push
```

## Command Reference

### camera_publisher

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--camera-id` | int | None | Camera device ID (e.g., 8) |
| `--publish-topic` | str | `/camera/image_raw` | Topic to publish images |

### localize_yoloe

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--camera-topic` | str | `/camera/image_raw` | Topic to subscribe for images |
| `--suppress-prints` | flag | False | Suppress console output |
| `--no-pushers` | flag | False | Disable pusher detection |
| `--recommend-push` | flag | False | Enable push recommendations |
| `--yolo-mode` | str | `prompt-set` | YOLO mode: 'prompt-set' for prompted detection |
| `--yolo-model` | str | `aruco_camera_localizer/yoloe-11s-seg.pt` | YOLO model path |
| `--yolo-conf` | float | 0.4 | YOLO confidence threshold |
| `--yolo-prompts` | list | `['hand']` | Detection prompts |
| `--yolo-color-map` | list | None | Custom color mappings |

## Published Topics

### By camera_publisher:
- `{publish-topic}` (default: `/camera/image_raw`) - Raw camera frames (sensor_msgs/Image)

### By localize_yoloe:
- `/camera_pose` - Camera pose in world frame
- `/objects_poses` - Detected object poses
- `/intel_camera_rgb_raw` - Raw camera image (republished)
- `/intel_camera_annotated` - Annotated image with detections
- `/yolo_prompts` - Current YOLO detection prompts
- `/pusher_data_{color}` - Pusher contact information
- `/recommended_pusher_{1,2}/position` - Recommended pusher positions
- `/recommended_pusher_{1,2}/normal` - Recommended pusher normals

## Monitoring

### Check if camera is publishing:
```bash
ros2 topic hz /camera/image_raw
```

### View camera feed:
```bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

### List all camera topics:
```bash
ros2 topic list | grep camera
```

### Echo camera info:
```bash
ros2 topic echo /camera/image_raw --once
```

## Troubleshooting

### "Waiting for camera frames..."
The localizer can't receive frames. Check:
```bash
# Is camera publisher running?
ros2 node list | grep camera_publisher

# Is the topic publishing?
ros2 topic hz /camera/image_raw

# Are topics matched?
ros2 topic info /camera/image_raw
```

### "Failed to open camera"
Camera is already in use or doesn't exist:
```bash
# List available cameras
v4l2-ctl --list-devices

# Check permissions
ls -l /dev/video*

# Kill other camera processes
pkill -f camera_publisher
```

### Camera shows but no detections
Check YOLO settings and camera pose:
```bash
# Monitor camera pose
ros2 topic echo /camera_pose

# Check detected objects
ros2 topic echo /objects_poses

# Adjust YOLO confidence
ros2 run aruco_camera_localizer localize_yoloe --yolo-conf 0.3
```

## Tips

1. **Camera ID**: Usually `/dev/video0`, `/dev/video2`, `/dev/video8`, etc.
2. **Frame Rate**: Camera publisher runs at ~30 FPS
3. **Latency**: Minimal - latest frame is always used
4. **Multiple Consumers**: Multiple nodes can subscribe to the same camera topic
5. **Testing**: Use `rqt_image_view` to verify camera feed before running localizer

## Integration with Launch Files

You can also integrate this into ROS2 launch files:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aruco_camera_localizer',
            executable='camera_publisher',
            name='camera_publisher',
            parameters=[
                {'camera_id': 8},
                {'publish_topic': '/camera/image_raw'}
            ]
        ),
        Node(
            package='aruco_camera_localizer',
            executable='localize_yoloe',
            name='localizer',
            parameters=[
                {'camera_topic': '/camera/image_raw'},
                {'yolo_conf': 0.4}
            ]
        )
    ])
```

## See Also

- [Update YOLO Prompts README](README.md) - How to dynamically update YOLO detection classes
- Main package README for overall system documentation

