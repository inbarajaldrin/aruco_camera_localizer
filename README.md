# Aruco-Based Localization for Camera mounted on UR End-Effector

This is a ROS 2 Humble Package that outputs aruco tag position and orientation *in the robot base frame* using camera inputs and ROS 2 communication with the UR driver. 

# Setup
Usage of this package will require the UR ROS 2 driver and cv2. 

To install, place this repository in the src directory of your workspace. Also install the repository `max_camera_msgs` in the same location. The repository is [here](https://github.com/MaxlGao/max_camera_msgs). Then, build your workspace. 

To run, first make sure a UR driver is running. Then, run `ros2 run aruco_camera_localizer localize` inside your workspace. 

A preview window will appear, picking one camera accessible to the system. Press ESC to select this camera, or press any other key to move onto the next camera. Once you select your camera, a more detailed window will appear marking detected objects and annotating details in the top left corner. If you know your camera ID and would like to skip this process, pass the additional arg `--camera-id [YOUR_ID]`. Be warned; USB cameras can change IDs between uses.  

While running, this script will publish estimated camera pose (in the UR frame) in `/camera_pose` as well as poses of any detected objects in `/object_poses/[object_name]`. Poses are given as `PoseStamped`.  

This script will doubly print its findings to console. If you wish, you may pass the argument `--suppress-prints` to not see this.

This script assumes a fixed translation from the end effector frame origin and the focal point of the camera. If the camera is ever repositioned, please measure or calibrate, and update the translation or rotation in `localization_bridge.py`

## Example Commands:
The following commands assume your camera is ID 8. Please remove this argument before first running the command. 

Run the following to display detected ArUco objects with wireframe visualization:

`ros2 run aruco_camera_localizer localize --camera-id 8 --suppress-prints`

## Workspace Structure:

```
YOUR_ROS2_WORKSPACE/
├── src/
│   └── aruco_camera_localizer/
│       ├── aruco_camera_localizer/
│       │   ├── trash/
│       │   │   └── [unused python files]
│       │   ├── camera_selection.py
│       │   ├── data_analyze.py
│       │   ├── data_predict.py
│       │   ├── detection_functions.py
│       │   ├── drawing_functions.py
│       │   ├── geometric_functions.py
│       │   ├── kalman_functions.py
│       │   ├── localizer_bridge.py
│       │   ├── merged_localization.py
│       │   ├── object_frame_definitions.py
│       │   └── process_stl.py
│       ├── STL/
│       │   ├── Allen Key.STL
│       │   └── Wrench.STL
│       ├── data/
│       │   ├── aruco/
│       │   ├── models/
│       │   └── wireframe/
│       ├── package.xml
│       ├── README.md
│       ├── setup.cfg
│       └── setup.py
```

## Important Parameters

Most of these scripts have parameters at the top of the file. Often, you will find references to `jenga`, `allen_key`, and `wrench` close to each other. These are the three objects of interest that we work with; you will need to add another reference to add another object.

### `data_analyze.py`

This script contains a variable `parent_folder` which references a rosbag directory and expects there to be a `*.db3` file and `metadata.yaml` inside a given folder. This directory *must* be changed before running this file on a new system. 

### `data_predict.py`

This script contains `file_folder` and `data_files`. As of now, `file_folder` *must* be changed before running *any* script needing pusher recommendations. (This will change in the future)

The script assumes the data files are structured as follows:

```
{file_folder}/
├── jenga/
│   ├── csv_output/
│   │   ├── _object_poses_jenga_[NUMBER].csv
│   │   ├── _pusher_data_green.csv
│   │   ├── _pusher_data_yellow.csv
│   │   ├── merged_pushers_with_pose.csv
│   ├── jenga_0.db3
│   └── metadata.yaml
├── allen_key/
│   ├── csv_output/
│   │   ├── _object_poses_allen_key.csv
│   │   ├── _pusher_data_green.csv
│   │   ├── _pusher_data_yellow.csv
│   │   ├── merged_pushers_with_pose.csv
│   ├── allen_key_0.db3
│   └── metadata.yaml
├── wrench/
│   ├── csv_output/
│   │   ├── _object_poses_wrench.csv
│   │   ├── _pusher_data_green.csv
│   │   ├── _pusher_data_yellow.csv
│   │   ├── merged_pushers_with_pose.csv
│   ├── wrench_0.db3
│   └── metadata.yaml
```

### `localizer_bridge.py`

Inside the `LocalizerBridge` class is a set of parameters inside `__init__()`. 

First are `self.cam_offset` for position and quaternion, describing the pose of the camera (focal point) relative to the end-effector (EE). 

Second is a set of initial values for EE pose, to be used if ROS2 fails to update the script with actual values. 

### `merged_localization.py`

`merged_localization.py` is a massive file with many parameters. Listing in current order of appearance:

1. Camera parameters: Image frame height and width, in pixels, plus angular FOV. Distortion coefficient is included. These are specific to the camera hardware used. 
2. `MARKER_SIZE` is the side length of the ArUco marker used for identifying Jenga blocks.
3. `BLOCK_*` are the side measures of the Jenga blocks. 
4. `TARGET_POSES` are the goal position and orientation for all objects.

### `object_frame_definitions.py`

The functions `define_body_frame_*` require careful changes for customization or extension to other objects. To summarize, each function does the following:

1. Extracts points `p1, p2, p3` and finds the distances between them. Sorts to find the short, medium, and long side. 
2. Assigns points `A, B, C` in relation to them being intersections of particular side lengths. For instance, `A` might be the point connecting the short and medium sides. 
3. Defines an origin, typically as one of the above points. 
4. Defines a coordinate frame, typically in relation to the above points. 
5. Extracts the object's pose.
6. [defunct, and to be removed later] defines manually-set contact points as points and normals. 
7. References a corresponding `CONTOUR_*` dictionary, transforms the contour's points and normals according to the object pose, and returns a current contour dictionary.

Other functions complete a subset of the above list. If adding objects, be sure to add it where "allen_key" and "wrench" are mentioned.

### `process_stl.py`

The first section of this script extracts trimeshes from given STL files (already installed locally in `./STL/`) and orients them according to the functions in `object_frame_definitions.py`. These are practically manual transformations.

Splines are made of 1000 points, as specified in `n_points`. Additional notable parameters are inside of `splprep()`, which uses an s-value of 1.

# Demonstration
The vision-based localizer, with a camera mounted on a robot's end-effector, can accurately yield the position and orientation of different objects using ArUco marker detection. The system uses CV2's ArUco functions to provide full 6D pose estimation in the robot base frame.

The system detects multiple ArUco markers per object and selects the best detection (closest to camera) for each object. Wireframe visualization is overlaid on the camera feed to show the detected object poses.

On the top of the following GIFs is a readout of robot end-effector pose, which is required to properly orient the camera in the base frame, which is in turn required to properly orient all objects that the camera sees. When objects come into view, a second readout appears, showing its estimated 6D pose.

Multiple objects may be detected at the same time. For best results, the working space should be kept as clean as possible.

![Objects Only](./media/Localizer%20Clip%201.gif)
