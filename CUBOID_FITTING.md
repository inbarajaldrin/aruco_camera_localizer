# 3D Cuboid Fitting — Progress & Status

## Overview

The cuboid fitting system estimates 3D oriented bounding boxes for detected objects and projects them as wireframe overlays on the camera image.

**Current approach**: 2D silhouette-based fitting — optimizes a 3D cuboid's pose so its projected 2D convex hull maximizes IoU with the YOLOE segmentation mask. No depth data required; uses known table plane height.

**Previous approach** (kept as fallback): Depth point cloud + PCA for oriented bounding box fitting.

## Current Implementation (2026-03-04)

### Silhouette-Based Cuboid Fitting (`_fit_cuboid_from_silhouette`)

**Algorithm**: Analysis-by-synthesis — render candidate cuboids, compare against observation.

1. **Initialize** from `_backproject_rect_to_table()`: back-project minAreaRect corners onto table plane for initial centroid (x, y), yaw, and edge lengths (w, l).
2. **Build reference mask** in a padded ROI for efficient IoU computation.
3. **Optimize** with `scipy.optimize.minimize(Nelder-Mead, maxiter=200)`:
   - Parameters: `[x, y, yaw, w, l]` (h fixed at known_height, default 11mm for lego bricks)
   - Z derived from table plane: `z = table_z + h/2`
   - Objective: negative IoU between projected cuboid convex hull and segmentation mask
   - Extra yaw seeds (yaw+π/4, yaw+π/2) tried if IoU < 0.3
4. **Return** `(center, quaternion, dimensions)` — same format as `_fit_cuboid_obb()`.

### Supporting Functions

- **`_project_cuboid_corners()`**: Extracted projection chain (world → camera → OpenCV → image). Returns (8, 2) float64 array. Used by both the optimizer and `draw_cuboid_wireframe()`.
- **`_fit_cuboid_obb()`**: PCA-based OBB fitting (depth path fallback). Unchanged.
- **`_backproject_rect_to_table()`**: Ray-table intersection for minAreaRect corners. Used for initialization.

### Integration in `detect_yolo_blobs()`

- **Depth path** (actual_distance != distance or table_z is None):
  - First tries PCA cuboid via depth point cloud (existing behavior)
  - If table_z is available and mask exists, also tries silhouette fitting — prefers it over PCA if it succeeds
  - Falls back to single ray + median depth

- **No-depth path** (table_z provided, depth returns -inf):
  - Primary: silhouette-based cuboid fitting
  - Fallback: back-projected rect centroid
  - Last resort: ray-table intersection from moments centroid

### Results (2026-03-04)

Test scene: 3 lego bricks in Gazebo (headless), camera tilted at -0.7854 rad, `--table-z -0.091`.

**Z-axis**: Exact at -85.5mm (= table_z + h/2 = -91.0 + 5.5), matching ground truth perfectly.

**XY-axis**: Consistent with known camera calibration bias (~60-70mm Y-axis systematic offset documented as a separate issue). X error ~15mm.

**Wireframes**: Rendering correctly as 3D cuboid projections. Somewhat oversized compared to actual block outlines — the Nelder-Mead optimizer may converge to a local minimum with inflated dimensions.

## Historical Approaches (tried and rejected)

### Depth PCA — Original approach
- Back-project masked depth pixels → 3D point cloud → PCA for OBB
- **Problem**: PCA orientation doesn't align with object edges → wireframes overshoot
- **Status**: Kept as fallback in depth path, but silhouette fitting preferred when table_z available

### Z-bias fix + Bbox constraint — REVERTED
- Shifted centroid by Z-offset, constrained wireframe to fit inside bbox
- **Failed**: Cascading interaction between fixes. Centroid shift moved projection off-center in bbox, causing bbox constraint to over-shrink dimensions.
- **Lesson**: Implement ONE change at a time. Verify before combining.

### Mask-guided rotation — REVERTED
- Used 2D segmentation mask to guide PCA orientation
- **Failed**: Interacted poorly with depth-derived dimensions.

### Tabletop-aligned axes — REJECTED
- Forced Z-up axis + XY-projected yaw
- **Failed**: Diamond-shaped wireframes from camera perspective mismatch.

## Known Issues

### 1. Wireframe slightly oversized
The Nelder-Mead optimizer may settle at a local minimum with inflated dimensions. Possible improvements:
- Add dimension regularization to the objective (penalty for large w*l)
- Use mask area as a constraint
- Reduce maxiter and use tighter initial bounds

### 2. Y-axis systematic bias (~60-70mm)
Known camera calibration issue, not related to cuboid fitting. The Gazebo camera config has vfov=42.5° (compensating for a deeper ray direction error). See MEMORY.md for details.

### 3. Red block sometimes missed
YOLOE confidence varies frame-to-frame (0.1–0.6 for simulated legos). Red block at image edge may drop below threshold.

## File Reference

All in: `aruco_camera_localizer/aruco_camera_localizer/merged_localization_yoloe.py`

Key functions:
- `_project_cuboid_corners()`: ~line 482
- `_fit_cuboid_from_silhouette()`: ~line 538
- `_fit_cuboid_obb()`: ~line 419 (PCA fallback)
- `_backproject_rect_to_table()`: ~line 275
- `draw_cuboid_wireframe()`: ~line 516
- Integration in `detect_yolo_blobs()`: ~line 863
- Wireframe draw in annotation loop: ~line 1265
