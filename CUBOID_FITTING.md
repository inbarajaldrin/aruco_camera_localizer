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

## Failed Attempt: Z-bias Fix + Bbox Constraint (2026-03-04) — REVERTED

### What was tried
1. **Z-bias fix**: `centroid_z = inlier_min_z - min(dimensions)/2` — shift centroid downward by half the smallest PCA dimension, using the minimum Z of inlier points as the "top surface" estimate.
2. **Bbox constraint**: Project all 8 OBB corners to 2D, compute per-corner scale factor relative to the projected centroid, uniformly shrink dimensions so all corners fit inside the YOLOE detection bbox.

### Process mistake — both fixes were implemented together without verifying either one
The next steps listed bbox constraint and Z-fix as **separate items**. Instead of implementing #1, verifying it visually, then implementing #2, both were coded in a single pass. This made it impossible to tell which fix caused the regression, and they interacted badly.

**RULE: Implement ONE change at a time. Verify visually (capture annotated image) and check position accuracy BEFORE starting the next change. Do NOT commit without user approval.**

### Why it failed — cascading interaction between the two fixes
1. **Z-fix shifts 3D centroid downward** → its 2D projection moves toward the bottom of the image.
2. **Projected centroid ends up off-center in the bbox** — for objects near image edges (e.g., red block at bottom-right), the projected center can land near a bbox boundary.
3. **Bbox constraint then over-shrinks**: it scales corner offsets relative to the projected centroid. When the centroid is near a bbox edge, corners on the opposite side have large offsets → the scale factor becomes tiny (e.g., 0.1) → dimensions collapse to near-zero → wireframes become tiny displaced shapes that don't wrap the object at all.

### Visual result
- Before: wireframes properly wrap objects (oversized but correct shape/orientation)
- After: wireframes collapsed into tiny diamonds/squares displaced from the objects. Red block had cyan lines shooting off-screen.

### Key lessons for future attempts
1. **Don't chain fixes that interact with each other's assumptions.** The bbox constraint assumes the projected centroid is roughly centered in the bbox. The Z-fix violates that assumption by moving the centroid.
2. **Z-fix via `min(point_cloud Z)` is unreliable.** At 45° camera angle, the point cloud is an oblique slab. `min Z` depends on mask boundary extent, not object geometry. It's NOT the top surface.
3. **Bbox constraint by uniform scaling from projected centroid cannot work when the centroid projection is off-center.** Some sides get over-shrunk while others may still stick out. The fundamental problem: scaling dimensions doesn't reposition the cuboid.
4. **PCA centroid position quality is dominated by mask quality**, not geometric corrections. YOLOE mask noise (±5mm XY per frame) is the primary error source, not Z-bias.
5. **Do NOT modify the centroid position and then try to constrain the wireframe in the same pass.** These must be independent, or the constraint must account for the centroid shift.

### If retrying these fixes
- Apply bbox constraint and Z-fix **independently**, not together
- For bbox constraint: instead of scaling dimensions from projected centroid, consider clipping the wireframe drawing to the bbox (purely visual) or constraining the 3D point cloud before PCA fitting
- For Z-fix: consider using known object height from a catalog instead of PCA smallest dim. Or use a two-pass approach: cuboid for XY only, separate depth method for Z
- Test each fix in isolation and verify wireframes visually before combining

## Next Steps

### 1. Fix cuboid orientation (the actual problem)
**The seg mask is accurate** — it fits tightly on the lego bodies. The mask is NOT the problem. The problem is **PCA orientation doesn't align with the object edges**. PCA finds axes of maximum variance in the 3D point cloud, which can be rotated relative to the actual object geometry. This causes wireframe corners to extend beyond the object/bbox even though the underlying point cloud is tight.

**Key insight**: if even 2 edges of the cuboid aligned with the segmentation mask edges, the whole cuboid would fit inside the YOLOE bbox. The fix is in the **fitting orientation**, not point cloud cleanup.

**Do NOT scale dimensions from projected centroid** (see failed attempt above). **Do NOT visually clip the wireframe** — that hides wrong values behind a correct-looking overlay.

Approaches to fix orientation:
- Use `cv2.minAreaRect()` on the 2D seg mask to get the object's 2D orientation, then use that to constrain the cuboid's yaw/roll axes in 3D
- Fit 2D OBB from mask edges first, back-project the 2D OBB edges to 3D, use those as the cuboid's principal axes instead of PCA
- Hybrid: use 2D mask orientation for the two in-plane axes, use PCA only for the depth axis

### 2. Fix Z-axis bias
**Do NOT use `min(point_cloud Z)` as top surface** (unreliable at oblique viewing angles). Better approaches:
- Two-pass: use cuboid centroid for XY, use table-z + known object height for Z
- If object height is unknown: use the depth range of the point cloud along the camera's viewing direction (not world Z) as the height estimate
- Accept the Z-bias as inherent to single-viewpoint depth and document the expected error

## File Reference

All changes in: `aruco_camera_localizer/aruco_camera_localizer/merged_localization_yoloe.py`

Key line ranges (approximate, may shift with edits):
- `_backproject_mask_to_pointcloud()`: ~line 342
- `_fit_cuboid_obb()`: ~line 418
- `draw_cuboid_wireframe()`: ~line 481
- Cuboid call in depth path: ~line 680
- Wireframe draw in annotation loop: ~line 1058
- Cuboid metadata storage: ~line 700
