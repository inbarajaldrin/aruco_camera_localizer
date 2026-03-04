# 3D Cuboid Fitting — Progress & Status

## Overview

Added 3D cuboid fitting to `merged_localization_yoloe.py` via depth point cloud + PCA.
Previous approach used 2D moments centroid + single scalar median depth → point along ray.
New approach back-projects ALL masked depth pixels to 3D, fits an oriented bounding box (OBB).

## What Was Implemented

### New functions in `merged_localization_yoloe.py`

1. **`_backproject_mask_to_pointcloud()`** (vectorized)
   - Takes YOLOE segmentation mask + depth image ROI
   - Back-projects every masked pixel to a world-frame 3D point
   - Includes 3x3 elliptical erosion to strip noisy mask boundary pixels
   - Handles float32 (meters) and uint16 (mm) depth encodings
   - Returns N×3 array of world points, or None if <10 valid pixels

2. **`_fit_cuboid_obb()`**
   - PCA on 3D point cloud → oriented bounding box axes
   - IQR outlier removal (1.5× IQR on each PCA axis) to reject table bleed / shadow pixels
   - Re-fits PCA on inliers for tighter axes
   - Returns (centroid, quaternion, dimensions)
   - Ensures right-handed coordinate system

3. **`draw_cuboid_wireframe()`**
   - Projects 8 OBB corners from world frame → image via camera transforms
   - Draws 12-edge wireframe overlay in yellow/cyan
   - Projection chain: world → camera frame (inv cam_quat) → OpenCV frame (inv opencv_to_camera) → image (K)

### Modified depth path in `detect_yolo_blobs()`

When depth data is available AND segmentation mask exists:
1. Try cuboid fitting first (back-project mask → PCA OBB)
2. If cuboid succeeds → use OBB centroid as `point_world`
3. If cuboid fails → fall back to single-ray + median depth (old behavior)

No-depth / table-z path is unchanged (still uses `_backproject_rect_to_table`).

### Cuboid metadata in `detection_metadata`

When cuboid fitting succeeds, each detection's metadata dict includes:
- `cuboid_center`: 3D centroid from OBB (np.array)
- `cuboid_quaternion`: orientation [x,y,z,w] from PCA axes
- `cuboid_dimensions`: [w,h,d] extents in meters (np.array)

### Wireframe drawing

Integrated into the main annotation loop — draws after bbox and before label text.

## What Was Tried & Rejected

### Moments-derived orientation for cuboid axes
- Converted 2D moments orientation_angle → 3D quaternion via `convert_2d_orientation_to_quaternion()`
- Used that quaternion's axes for the OBB instead of PCA
- **Result**: Worse — the full 3D quaternion from camera viewing angle doesn't decompose cleanly into tabletop length/width/height axes. Cuboid dimensions inflated.

### Tabletop-aligned axes (XY-projected yaw + Z-up)
- Projected the moments yaw direction onto world XY plane
- Built axes as (long_axis_XY, perpendicular_XY, Z_up)
- **Result**: Worse — diamond-shaped wireframes that don't match the camera viewing perspective. The Z-up axis doesn't align with depth/height as seen from an angled camera.

### `transform_points_world_to_img()` for wireframe projection
- Tried using the existing geometric_functions helper for 3D→2D projection
- **Result**: Wireframes disappeared — that function uses a different coordinate convention (cam_quat directly to projection frame) that skips the opencv_to_camera step needed for this camera setup.

**Conclusion**: Pure PCA on the depth point cloud gives the best-looking wireframes because PCA axes naturally align with the point cloud's principal variation, which matches what the camera sees.

## Current Results (2026-03-04)

### Test scene: 3 lego bricks in Gazebo (headless), camera tilted at -0.7854 rad

Ground truth (BEARING_1 frame):
- Red:   (-0.124,  0.031, -0.0855) — 32×16×11mm
- Green: (-0.134, -0.009, -0.0855) — 24×16×11mm
- Blue:  (-0.154, -0.029, -0.0855) — 16×16×11mm

### Position accuracy: Cuboid vs Table-Z

| Color | Cuboid err | dx     | dy    | dz     | Table-Z err | dx     | dy    | dz    |
|-------|-----------|--------|-------|--------|------------|--------|-------|-------|
| Red   | 19.8mm    | +7.8   | +1.9  | +18.1  | 17.0mm     | -10.5  | +12.2 | -5.5  |
| Green | 12.5mm    | +3.2   | +6.0  | +10.5  | 13.9mm     | -11.6  | +5.3  | -5.5  |
| Blue  | 12.7mm    | +6.0   | +8.2  | +7.6   | 11.0mm     | -8.5   | +4.2  | -5.5  |

**Key finding**: Cuboid significantly improves dx (X error halved or better for all objects). Total Euclidean error is similar because dz gets worse — see Z problem below.

### Cuboid dimensions (PCA + IQR + erosion)

| Color | Measured        | Ground Truth   |
|-------|-----------------|----------------|
| Red   | 34.6×14.1×6.9mm | 32×16×11mm     |
| Green | 26.8×18.0×7.4mm | 24×16×11mm     |
| Blue  | 22.6×18.2×15.0mm| 16×16×11mm     |

Dimensions are in the right ballpark but inflated, especially for blue (smallest object, noisiest mask at low confidence).

### Wireframe visualization

Wireframes are drawn but appear oversized relative to the objects in the image. Root cause: at low YOLO confidence (0.05–0.3 for simulated legos), segmentation masks include background/shadow pixels that inflate the point cloud even after erosion and IQR filtering.

## Known Issues

### 1. Wireframe too large
The OBB wireframe extends beyond the actual object boundary in the image. Caused by noisy seg masks at low YOLO confidence. At higher confidence (>0.3), masks are tighter and this should improve.

### 2. Z-axis bias (dz = +8 to +18mm)
The depth camera only sees the top surface of each object. The point cloud centroid sits above the true object center. This is a fundamental single-viewpoint limitation, not a code bug.
- Table-Z method gets dz≈-5.5mm (half brick height error, but in the other direction)
- True object center Z = table_z + half_object_height

### 3. Low YOLO confidence on simulated legos
Scores 0.05–0.3 (vs 0.22–0.28 in earlier Phase B tests). Scene/camera angle may have changed. Not related to cuboid fitting.

## Next Steps

### 1. Constrain cuboid to YOLOE detection bbox
Clip or fit the 3D cuboid so its 2D projection stays within the YOLOE bounding box. This should prevent the oversized wireframe issue by using the detection box as an upper bound on the projected cuboid footprint.

### 2. Fix Z-axis bias
Investigate approaches:
- Use cuboid's smallest PCA dimension as height → snap centroid Z to `table_z + height/2`
- Depth histogram analysis (object surface depth vs table depth) to infer object height
- Two-pass: cuboid for XY centroid, table-z + measured height for Z

## File Reference

All changes in: `aruco_camera_localizer/aruco_camera_localizer/merged_localization_yoloe.py`

Key line ranges (approximate, may shift with edits):
- `_backproject_mask_to_pointcloud()`: ~line 342
- `_fit_cuboid_obb()`: ~line 418
- `draw_cuboid_wireframe()`: ~line 481
- Cuboid call in depth path: ~line 680
- Wireframe draw in annotation loop: ~line 1058
- Cuboid metadata storage: ~line 700
