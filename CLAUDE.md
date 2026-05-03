# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package overview

ROS2 Humble `ament_python` package that estimates 6-DOF poses of objects (boards / blocks / pegs / sockets) from ArUco markers detected by a camera mounted on a UR robot's end-effector. Output is published in the **robot base frame**, requiring live EE pose from the UR driver.

Single console-script entry point: `localize` (defined in `setup.py` → `aruco_camera_localizer.merged_localization:main`).

## Build / run / test

This package lives inside a colcon workspace at `/home/aaugus11/Desktop/ros2_ws/`. **Always build with `--symlink-install`** — without it, edits to `setup.py` leave a stale `aruco_camera_localizer.egg-info/entry_points.txt` in `build/`, and `ros2 run` fails with `StopIteration` when loading the entry point.

```bash
# Build (run from workspace root, not the package)
cd /home/aaugus11/Desktop/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select aruco_camera_localizer --symlink-install

# Run
source install/setup.bash
ros2 run aruco_camera_localizer localize --suppress-prints

# Tests (default ament linting only — flake8, pep257, copyright)
colcon test --packages-select aruco_camera_localizer
colcon test-result --verbose
```

Useful CLI flags on `localize`:
- `--camera-id N` — bypass identity probe and use `/dev/videoN` directly
- `--image-topic /some/topic` — sim mode, consume `sensor_msgs/Image` instead of V4L2
- `--headless` — no OpenCV window (still publishes `/annotated_stream`)
- `--filter-tune` / `--robot-tune` — open tkinter trackbar panels with hot-reload + "Save as Default" that writes back to the YAML

## High-level architecture

The codebase splits into a **ROS I/O layer** and a **vision pipeline layer** that runs as a single-process main loop. There is no second node — `LocalizerBridge` is instantiated and spun on a background thread by the main loop.

### Two-config split (read at every `main()` invocation, no rebuild)

- `config/robot_config.yaml` → `RobotConfig` in `robot_config.py`. Camera intrinsics, EE→camera offset, exposure/WB modes, table Z, and the **camera-identity probe fields** (`camera_match_name`, `camera_match_vendor_id`, `camera_match_product_id`, `camera_prefer_format`, `camera_serial`).
- `config/filter_config.yaml` → `FilterConfig` in `filter_config.py`. Per-filter enable flags, EMA, motion pause, IPPE disambiguation, fold-symmetry snap, ArUco detector params, CLAHE.

Both configs use a flat key→attribute mapping. The classes have a `.save()` method that *preserves comments and ordering* by regex-rewriting the existing YAML lines — used by the tuning panels.

### Camera identity probe (don't bypass with hardcoded device IDs)

`camera_selection.probe_camera_by_identity()` walks `/sys/class/video4linux/` and matches by USB VID/PID/name from `RobotConfig`, then picks the sub-device whose V4L2 fourcc matches `camera_prefer_format`. This exists because Intel RealSense cameras expose **6 `/dev/videoN` sub-devices** (depth Z16, IR GREY, color YUYV, plus metadata), and `/dev/videoN` numbering is unstable across reboots. The default config matches Intel `8086:*` and prefers `YUYV` (color). On miss, it falls back to the legacy `select_camera()` interactive preview.

### Main loop pipeline (`merged_localization.py`)

1. Load both YAMLs → build camera intrinsics + ArUco detector params.
2. **Resolve the data directory** — `data_path_finder.find_aruco_data_dir()` searches `~/Documents/` recursively for an `aruco-grasp-annotator` repo's `data/` folder. **The package will exit early if this is missing.** Annotations (marker→object mappings, wireframes, grasp points, fold symmetries) live there, *not* in this repo.
3. For each model: load `<name>_aruco.json`, optional `<name>_wireframe.json`, optional `<name>_grasp_points_all_markers.json`, and (from `symmetry/`) `<name>_symmetry.json`.
4. Open the camera (probe-by-identity or `--camera-id` or ROS topic) and start `LocalizerBridge` rclpy node on a background thread.
5. Per-frame loop: capture → multi-dictionary ArUco detect → `estimate_poses` (per-marker `solvePnPGeneric` IPPE_SQUARE) → `estimate_board_pose_combined` (multi-marker board pose) → IPPE disambiguation (`pick_best_solution` keeps the IPPE branch closest to last accepted world quat or to a flat-on-table prior) → fold-symmetry orientation snap → optional EMA → publish + draw wireframe overlay.
6. Filters that may **suppress** publication: motion-pause (skip when EE speed > `motion_speed_threshold`), Z-range, reprojection-error, ghost timeout, active-marker tracking.

### LocalizerBridge ROS surface (`localizer_bridge.py`)

- Subscribes: `/tcp_pose_broadcaster/pose` (UR end-effector pose, configurable via `robot_config.ee_pose_topic`). Optionally subscribes to an image topic for sim mode.
- Publishes: `/camera_pose` (PoseStamped), `/objects_poses_real` (tf2_msgs/TFMessage), `/annotated_stream` (sensor_msgs/Image), `/intel_camera_rgb_raw` (raw Image), `/camera_frame_number` (Int32), and per-pusher `/pusher_info_<color>` topics dynamically created when `PusherInfo` messages arrive.
- The `from cv_bridge import CvBridge` is wrapped in `_import_cv_bridge_quietly()` to suppress the noisy "compiled using NumPy 1.x ... `_ARRAY_API not found`" diagnostic that appears when running NumPy 2.x against Humble's NumPy-1.x-built `cv_bridge` extension. The suppression uses `os.dup2(devnull, 2)` because the message is written *directly to fd 2* by the C extension, not through Python's warnings system. cv_bridge functions degrade to slower Python paths but remain usable — verified by `/annotated_stream` publishing at ~3.7 Hz.

### Geometry and frame conventions

`geometric_functions.py` is the single source of truth for transforms. Camera frame ↔ world frame transforms use a fixed cam-EE offset from `robot_config.cam_offset_position` / `cam_offset_quat`. Fold symmetry (`enable_fold_snap`) snaps an object's yaw to the nearest 360°/N orientation when it's resting flat on the table — only applied to subtypes listed in `fold_snap_subtypes` (default: `block`, `peg`).

The `euler_convention` filter config key (`intrinsic` or `extrinsic`) only affects the RPY display in the OpenCV overlay, not the underlying quaternion math.

## External dependencies (not pip-installable)

- **`max_camera_msgs`** (ROS2 package) — must be cloned into the same workspace `src/`. Provides `PusherInfo.msg`. Source: <https://github.com/MaxlGao/max_camera_msgs>.
- **`aruco-grasp-annotator/data/`** repository — must exist somewhere under `~/Documents/`. Contains the `aruco/`, `wireframe/`, `grasp/`, and `symmetry/` JSON files that define every detectable object. The package is non-functional without it.
- **UR ROS2 driver** publishing to `/tcp_pose_broadcaster/pose` — without this, the `LocalizerBridge` will subscribe but never publish (no EE pose → no world-frame transform).

## Gotchas worth knowing before editing

- **README.md is partially stale.** It references files that no longer exist in `aruco_camera_localizer/`: `data_analyze.py`, `data_predict.py`, `object_frame_definitions.py`, `process_stl.py`, and a `trash/` subdir. The current pipeline is consolidated in `merged_localization.py` (see commit `e6af5f0` "simplify pipeline, add combined multi-marker solvePnP for boards, externalize data"). Treat the README's "Important Parameters" section as historical.
- The `install/aruco_camera_localizer/lib/aruco_camera_localizer/` may still contain orphan launcher scripts (`localize_aruco`, `localize_yoloe`, `camera_publisher`) from older `setup.py` versions. They're harmless but will fail if invoked. Only `localize` is current.
- `tuning_panel.py` opens **tkinter** windows alongside the OpenCV window. Per `~/.claude/CLAUDE.md` X11 safety rule: stop the node with SIGTERM (e.g. `pkill -SIGTERM -f "ros2 run aruco_camera_localizer"`), never SIGKILL — both windows need graceful teardown.
- The `merged_localization.py` `main()` is a long monolithic loop (~600 lines after argparse). When extending the pipeline, add new steps inline rather than refactoring — the order between IPPE disambiguation, EMA, fold-snap, motion-pause, and wireframe rendering is load-bearing and undocumented except by reading the code.
- Both `RobotConfig` and `FilterConfig` print `Warning: unknown key 'X'` for keys in YAML that aren't in `_DEFAULTS`. When adding a new tunable, you must add it to both the `_DEFAULTS` dict in the Python class **and** the YAML file.
