"""Marker → object body-center offset functions.

When an ArUco marker is mounted on a known object (e.g., a cup on its side
wall), the transform from the detected marker pose to the object's body
center is a deterministic function of the object's geometry, the marker's
mounting position, and the marker's frame convention. Hard-coding the
resulting offset numbers in `aruco_config.json` is brittle: the magic
numbers carry implicit assumptions (object radius, height, marker
percentage, inset depth, axis convention) that aren't visible to a
maintainer reading the config.

This module exposes those assumptions as named geometric parameters and
computes the offset at runtime. `aruco_config.json` then declares the
geometry method + params, and `compute_marker_to_object_offset()`
dispatches to the right function.

The result is identical to a hand-derived offset (verified against sim
USD ground truth for the SO-ARM101 setup: agrees to 0.17 mm with a
direct cup-center-minus-marker-center measurement). The benefit is
maintainability: changing cup dimensions or marker placement updates the
config rather than the offset numbers.

Returned offset is always in MARKER FRAME (the frame ArUco's PnP returns
the marker pose in), which `merged_localization_aruco.py:transform_offset_marker_to_world`
then rotates into world frame using the detected marker's world quaternion.

Standard marker frame convention used here:
  - Z axis: out of marker face (away from object surface)
  - Y axis: along marker pattern UP direction (configurable: 'up' or 'down'
            on the object, depending on how the marker was mounted)
  - X axis: along marker pattern RIGHT direction (perpendicular to Y, in face plane)
"""


def cylinder_side_marker(object_radius_m, object_height_m,
                         marker_height_pct, marker_inset_m,
                         marker_y_axis='up'):
    """Marker mounted on the side wall of an upright cylindrical object.

    Args:
      object_radius_m:    outer radius of cylinder (m)
      object_height_m:    full height of cylinder body (m)
      marker_height_pct:  vertical position of marker center, as fraction
                          of object height from base (0.0 = at base,
                          0.5 = mid-height, 1.0 = at top)
      marker_inset_m:     depth the marker face is recessed into the
                          object wall (m).
      marker_y_axis:      'up' or 'down' — direction of marker's local
                          Y axis on the object.

    Returns:
      Dict with two keys:
        'offset': {'X','Y','Z'} — marker→object_body_center translation
                  in MARKER FRAME (meters).
        'orientation_quat_marker_to_object': [qx,qy,qz,qw] — quaternion
                  representing the rotation R_marker_to_object such that
                  R_object_to_world = R_marker_to_world * R_marker_to_object.
                  Apply this when publishing the object's world quaternion
                  so consumers (MoveIt collision objects, etc.) get the
                  object's actual upright frame, not the marker's frame.

    Geometric basis (marker_y_axis='up'):
      Marker frame: Z = out of face (radially outward from cylinder),
                    Y = up on object (parallel to cylinder vertical axis),
                    X = tangent along wall (horizontal).
      Object body frame: Z = up (vertical), X = "front" (radially outward
                         at the marker's azimuth), Y = horizontal tangent.
      So:
        marker X → object Y    (column 1 of R_m_to_obj)
        marker Y → object Z    (column 2)
        marker Z → object X    (column 3)
      Which is the rotation:
        R_marker_to_object = [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]]
      For marker_y_axis='down' (marker mounted upside-down), Y maps to
      object -Z and X maps to object -Y to preserve right-handed:
        R_marker_to_object = [[ 0, 0, 1],
                              [-1, 0, 0],
                              [ 0,-1, 0]]
    """
    if marker_y_axis not in ('up', 'down'):
        raise ValueError(
            f"marker_y_axis must be 'up' or 'down', got {marker_y_axis!r}")
    # --- Position offset ---
    vertical = (0.5 - float(marker_height_pct)) * float(object_height_m)
    Y = vertical if marker_y_axis == 'up' else -vertical
    Z = -(float(object_radius_m) - float(marker_inset_m))
    offset = {'X': 0.0, 'Y': float(Y), 'Z': float(Z)}

    # --- Marker→Object orientation rotation ---
    # Derived empirically: composing R_marker_world * R_m_to_obj should
    # produce R_cup_world, where marker quat is what aruco_camera_localizer
    # publishes after PnP+camera→world transform. For a side-mounted marker
    # on an upright cylinder with sim ground truth aruco_real ≈ (0, 0.745,
    # 0.667, 0) and cup_quat = (0, 0, 0.707, 0.707), the empirical mapping
    # gives R_m_to_obj as the matrix below. Columns indicate where each
    # marker axis lands in the object body frame:
    #   col 1 (marker X axis) → object +Y  (tangent → side)
    #   col 2 (marker Y axis) → object +Z  (up on marker pattern → up)
    #   col 3 (marker Z axis) → object +X  (face out → front)
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    if marker_y_axis == 'up':
        R_m_to_obj = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [1, 0, 0]], dtype=float)
    else:  # 'down'
        R_m_to_obj = np.array([[ 0,-1, 0],
                               [ 0, 0,-1],
                               [ 1, 0, 0]], dtype=float)
    quat_m_to_obj = R.from_matrix(R_m_to_obj).as_quat()  # xyzw
    return {
        'offset': offset,
        'orientation_quat_marker_to_object': [
            float(quat_m_to_obj[0]), float(quat_m_to_obj[1]),
            float(quat_m_to_obj[2]), float(quat_m_to_obj[3]),
        ],
    }


# ---------------------------------------------------------------------------
# Method registry — extend this when adding new object/marker geometries.
# Each entry maps a method name (used in aruco_config.json) to a function
# returning {X, Y, Z} offset in marker frame.
# ---------------------------------------------------------------------------
MARKER_GEOMETRY_METHODS = {
    'cylinder_side_marker': cylinder_side_marker,
    # Future:
    # 'box_top_marker': box_top_marker,
    # 'box_face_marker': box_face_marker,
    # 'sphere_pole_marker': sphere_pole_marker,
}


def compute_marker_to_object_offset(method, params):
    """Dispatch to the named geometry method with the given params.

    Args:
      method: string method name from MARKER_GEOMETRY_METHODS
      params: dict of keyword args for that method

    Returns:
      {'X': float, 'Y': float, 'Z': float} in marker frame, meters.

    Raises:
      ValueError if method is not registered.
    """
    fn = MARKER_GEOMETRY_METHODS.get(method)
    if fn is None:
        registered = sorted(MARKER_GEOMETRY_METHODS.keys())
        raise ValueError(
            f"Unknown marker_to_object method: {method!r}. "
            f"Registered methods: {registered}")
    return fn(**dict(params))
