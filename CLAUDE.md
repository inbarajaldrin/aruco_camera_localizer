## Project

ArUco marker + YOLOE detection pipeline. Two ROS2 nodes (`localize_aruco`,
`localize_yoloe`) consume `/wrist_camera_rgb_sim` (or any camera image
topic), run their detectors, and publish 6D object poses in the robot's
base frame. The pipeline is robot-agnostic — per-robot config
(`config/aruco_config.json`, `config/robot_config.yaml`) declares which
markers map to which objects, and which camera intrinsics + frame
conventions apply.

Currently used by SO-ARM101 (active), JETANK (legacy), UR5e + RoboSort (other configs).

## Cross-repo

This package is consumed by:
- `~/Documents/isaac-sim-mcp` (so-arm101 branch) — Isaac Sim digital twin
  publishes the camera image and `/drop_poses` ground truth.
- `~/Projects/Exploring-VLAs/vla_SO-ARM101` — control GUI subscribes to
  `/objects_poses_real`, `/drop_poses_real` for real-mode pick-place.

**Two clones of this repo exist on this machine** — `~/Desktop/ros2_ws/src/aruco_camera_localizer/` (LIVE, runtime imports from this clone via colcon symlink-install) and `~/Projects/RoboSort/aruco_camera_localizer/` (stale, NOT used at runtime). Always edit Desktop. Verify which clone is on the Python path with `python3 -c "import aruco_camera_localizer; print(aruco_camera_localizer.__file__)"` before assuming edits will take effect.

## Build & launch

```bash
cd ~/Desktop/ros2_ws && colcon build --packages-select aruco_camera_localizer --symlink-install
```

`--symlink-install` is essential — it symlinks `build/.../aruco_camera_localizer` to `src/.../aruco_camera_localizer`, so source edits become live without rebuild. Confirm with `ls -la build/aruco_camera_localizer/aruco_camera_localizer` (should show `->` to src).

Launchers (run from your own terminal so cv2.imshow has D-bus / XDG_SESSION context):
- ArUco: `bash ~/Documents/isaac-sim-mcp/scripts/restart_aruco_localizer.sh` (kills stale + launches with sim wrist cam + so_arm101 robot config + drop mode)
- YOLOE: `bash ~/Documents/isaac-sim-mcp/scripts/restart_yoloe.sh` (kills stale + launches with config-driven prompts from `robot_config.yaml`)

Both scripts have `--headless`, `--bg`, `--camera`, `--prompts` (yoloe), `--conf` (yoloe) overrides.

## Architecture — two-process detection model

```
camera image (/wrist_camera_rgb_sim)
       |
       +─→ localize_aruco  (PnP on ArUco markers)
       |       ├─ camera_pose subscriber (/camera_pose)
       |       ├─ ee_pose subscriber (/ee_pose)
       |       ├─ marker_geometry.py (computes marker→object transform)
       |       └─ publishes: /aruco_poses_real, /drop_poses_real
       |
       +─→ localize_yoloe  (CLIP-prompted detection)
               ├─ same /camera_pose, /ee_pose subscribers
               ├─ text prompts from config (or CLI override)
               └─ publishes: /objects_poses_real, /objects_bbox_real
```

Each process subscribes to the camera independently — there's no shared image-decode pass. Same node name `/localizer_bridge` registers from both processes (causes a "duplicate name" warning from `ros2 node info`; safe to ignore).

There is **NO merged YOLOE+ArUco mode** despite the `merged_localization_*.py` filenames — those are individual entrypoints, the "merged" suffix refers to merging PnP + segmentation + cuboid fitting *within* one detector.

## Configuration files

| File | Purpose | Schema notes |
|------|---------|--------------|
| `config/robot_config.yaml` | Per-robot camera intrinsics, TF frame conventions, default poses, YOLOE detection block | `active_robot` selects which sub-block is used. Each robot has `camera`, `transforms`, `detection`. `detection.yolo` (optional) provides prompt set for `localize_yoloe`. |
| `config/aruco_config.json` | Per-robot ArUco marker → object mapping with offsets/orientation | Each `marker_rows.<row>` declares either `marker_to_object: {method, params}` (preferred) or legacy `position_offset: {X, Y, Z}`. |
| `config/bbox_catalog.json` | Static object dimensions republished on `/objects_bbox_real` at 1Hz | Keys are object names (sized: `red_lego_2x4`; or color-only fallback: `red`). Consumers pick the right entry via name or color-prefix lookup. |
| `config/filter_config.yaml` | Kalman + stability params (shared across robots) | Tune carefully; see `kalman_functions.py`. |

## marker_to_object schema (preferred for new markers)

`aruco_config.json` rows declare a marker→object_body transform via a registered geometry method:

```json
"marker_rows": {
  "cups": {
    "marker_ids": [0, 1, 2],
    "marker_to_object": {
      "method": "cylinder_side_marker",
      "params": {
        "object_radius_m": 0.039,
        "object_height_m": 0.0965,
        "marker_height_pct": 0.45,
        "marker_inset_m": 0.006,
        "marker_y_axis": "up"
      }
    }
  }
}
```

`aruco_camera_localizer/marker_geometry.py` exports:
- `cylinder_side_marker(R, H, h_pct, inset, marker_y_axis)` — returns BOTH `offset` (X/Y/Z in marker frame) AND `orientation_quat_marker_to_object` (xyzw). The offset positions the cup body center; the orientation rotation aligns the cup mesh's local axes with world (so the mesh stands upright in MoveIt instead of being drawn in marker frame).
- `MARKER_GEOMETRY_METHODS` — registry. Add new methods (`box_top_marker`, `sphere_pole_marker`, etc.) here and they're immediately available via the schema.

`localize_aruco` dispatches: when `marker_to_object` is present, applies BOTH offset (position) and orientation rotation (`drop_quat = R_marker_world * R_marker_to_object`). Legacy `position_offset` rows continue to work and skip the orientation correction.

## Adding a new marker / object — use scripts/derive_marker_config.py

Don't hand-derive offsets. The script empirically derives the config by querying USD ground truth + capturing live PnP detection:

```bash
python3 scripts/derive_marker_config.py \
  --marker-id 0 \
  --cup-prim /World/Containers/cup_red \
  --marker-prim /World/Containers/cup_red/aruco_000/aruco_marker_mesh \
  --mode cylinder    # or 'explicit'
```

Cylinder mode reverse-engineers `cylinder_side_marker` params from USD bbox queries; explicit mode emits raw offset + quaternion. Either way: paste the output JSON snippet into the appropriate `marker_rows` entry, restart `localize_aruco`. The script validates round-trip math (residuals should be ~0) and warns if cylinder fit diverges by >2mm from the empirical truth (signals PnP/camera bias or non-standard marker mounting).

Sharp edge: when the marker is **out of FOV**, the Kalman filter extrapolates wildly (positions in the kilometer range). Make sure the marker is in camera FOV during the script run.

## YOLOE prompts via robot_config.yaml

Per-robot YOLOE vocabulary lives in `robot_config.yaml`'s `detection.yolo` block (added Apr 2026):

```yaml
so_arm101:
  detection:
    yolo:
      prompts: ["red object", "blue object", "green object"]
      prompt_map: {red object: red, blue object: blue, green object: green}
      confidence: 0.25
```

`localize_yoloe.main()` precedence: CLI args (`--yolo-prompts`, `--yolo-conf`) > config > argparse default. So `restart_yoloe.sh` doesn't pass prompts (config drives them); explicit override is `--prompts "..."`.

Runtime prompt updates without restart — publish to `/yolo_prompts_update` (std_msgs/String, JSON payload):
```bash
ros2 topic pub --once /yolo_prompts_update std_msgs/msg/String \
  'data: "{\"prompts\": [\"X\"], \"prompt_map\": {\"X\": \"x_alias\"}}"'
```

## Topic names — sim vs real

Topic names are configured via `--ros-args -p X_topic:=Y` parameters (not topic remapping). Defaults:
- `/aruco_poses` (raw markers), `/drop_poses` (cup body-centers), `/objects_poses` (lego bodies), `/objects_bbox` (catalog)
- `/aruco_poses_real`, `/drop_poses_real`, `/objects_poses_real`, `/objects_bbox_real` (when launched with `_real` suffix params, e.g. via `restart_*.sh`)

Frame: all transforms publish in `frame_id='base'` (so_arm101) or whatever `tcp_pose` topic the `robot_config.yaml` block declares.

## Gotchas

- **`--symlink-install` is required for hot-reload-style dev**. `ros2 run` imports the installed package; if build/ has copies (not symlinks) of source, edits are invisible at runtime.
- **`bbox_catalog.json` is loaded at process start, not via subscription** — edit + restart `localize_yoloe` for changes to appear on `/objects_bbox_real`. Verify with `ros2 topic echo /objects_bbox_real --once`.
- **Static text-embedding cache** in `aruco_camera_localizer/.text_embeddings_*.pt` files (gitignore candidates). Generated from MD5 of sorted prompts; on first run with new prompts CLIP recomputes (~90s); subsequent runs load in ~5-10s.
- **PnP's marker pose ambiguity** — `solvePnP` returns two solutions for ArUco; the temporal-consistency seed in `detection_functions.py` picks one. If markers occasionally "flip" (orientation jumps 180°), the seed got reset.
- **Kalman extrapolation** — when a marker leaves FOV, the filter's `predict()` continues for several frames using the last velocity. If `derive_marker_config.py` or any downstream consumer sees nonsense positions (>5m magnitude), the filter has drifted; bring the marker back into FOV and reseed.
- **`--suppress-prints` doesn't silence ROS2 INFO logs** — only the cv2 `print()` debug spam. Use `RCUTILS_LOGGING_MIN_SEVERITY=WARN` env to fully quiet.
- **Two clones of this repo on this workstation** (`Desktop/ros2_ws/src/...` is live, `Projects/RoboSort/...` is stale and divergent). All edits land on Desktop. Don't propagate to RoboSort unless explicitly resolving the divergence.

## Required reading for this codebase

1. `aruco_camera_localizer/marker_geometry.py` — the function module + method registry. Read before adding new objects.
2. `config/aruco_config.json` — current robot/marker mappings. The schema is documented inline via `_comment` fields.
3. `aruco_camera_localizer/merged_localization_aruco.py:main()` — the main loop. Where PnP, camera→world transform, and `marker_to_object` dispatch happen.
4. `scripts/README.md` — script index (or just `ls scripts/` for a quick survey).
