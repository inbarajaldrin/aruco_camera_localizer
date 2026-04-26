#!/usr/bin/env python3
"""Derive a `marker_to_object` config block for aruco_config.json from a
running Isaac Sim scene + a live aruco_camera_localizer detection.

Use case: you've attached a new ArUco marker to an object in Isaac Sim
(or in real life) and need to compute the marker→object transform for
aruco_config.json. Manually deriving the geometric params is error-prone
(easy to flip sign on Y axis, easy to confuse marker frame vs object
body frame). This script does it empirically: query USD ground truth +
live PnP output, compute the relations, emit a ready-to-paste config
block.

Workflow:
  1. Place the new marker + object in Isaac Sim. Make sure cups USD
     and the marker mesh have proper transforms (parent-child, Xform).
  2. Run quick_start so /drop_poses publishes the object's pose AND
     localize_aruco is detecting the marker → /aruco_poses_real.
  3. Move the wrist camera to a pose where the new marker is in FOV
     (or use a stationary camera).
  4. Run this script — it captures both topics + queries USD, computes
     the empirical R_marker_to_object and offset, emits a JSON block.
  5. Paste the block into aruco_config.json's robot[robot].marker_rows
     entry, restart localize_aruco.

The script supports two output modes:

  - cylinder_side_marker: when the object is an upright cylinder with
    a marker on its side wall, emit geometry params (radius, height,
    marker_height_pct, marker_inset, marker_y_axis). Easier to maintain
    when geometry changes (edit dimensions, not raw offset numbers).

  - explicit: emit raw offset (X,Y,Z in marker frame) and orientation
    quat (R_marker_to_object as xyzw). Use when the object isn't a
    standard shape or when the cylinder fit is off.

Reference: ~/Desktop/ros2_ws/src/aruco_camera_localizer/aruco_camera_localizer/marker_geometry.py
"""

import argparse
import json
import math
import socket
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_msgs.msg import TFMessage

MCP_HOST = 'localhost'
MCP_PORT = 8767
MCP_TIMEOUT = 8.0


# ---------------------------------------------------------------------------
# USD prim queries via Isaac Sim MCP
# ---------------------------------------------------------------------------

def query_isaac_sim(code: str) -> dict:
    """Run Python in Isaac Sim's Kit context. Set `result = ...` to return."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(MCP_TIMEOUT)
    s.connect((MCP_HOST, MCP_PORT))
    req = json.dumps({"type": "execute_python_code",
                      "params": {"code": code}}) + "\n"
    s.sendall(req.encode())
    data = b''
    while True:
        chunk = s.recv(8192)
        if not chunk:
            break
        data += chunk
        try:
            return json.loads(data.decode().strip())
        except json.JSONDecodeError:
            continue


def get_prim_world_bbox(prim_path: str) -> dict:
    """Returns world-axis-aligned bbox center, size, and L2W rotation quat
    for a USD prim. Raises RuntimeError on missing prim."""
    code = f"""
from pxr import UsdGeom, Usd
import omni.usd
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath({prim_path!r})
if not prim or not prim.IsValid():
    result = {{'error': 'prim not found or invalid'}}
else:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    L2W = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    pos = L2W.ExtractTranslation()
    q = L2W.ExtractRotationQuat()
    result = {{
        'bbox_min': [mn[0], mn[1], mn[2]],
        'bbox_max': [mx[0], mx[1], mx[2]],
        'bbox_center': [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2],
        'bbox_size': [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]],
        'prim_pos_world': [pos[0], pos[1], pos[2]],
        'prim_quat_xyzw': [q.GetImaginary()[0], q.GetImaginary()[1],
                           q.GetImaginary()[2], q.GetReal()],
    }}
"""
    resp = query_isaac_sim(code)
    inner = resp.get('result', {}).get('result', {})
    if 'error' in inner:
        raise RuntimeError(f"USD query failed for {prim_path}: {inner['error']}")
    return inner


# ---------------------------------------------------------------------------
# ROS2 capture — marker pose from /aruco_poses_real, cup pose from /drop_poses
# ---------------------------------------------------------------------------

class TFCapture(Node):
    """Capture latest TFMessage transforms by child_frame_id from a topic."""

    def __init__(self, name, topic):
        super().__init__(name)
        self.captured = {}
        self.create_subscription(TFMessage, topic, self._cb, 10)

    def _cb(self, msg):
        for tf in msg.transforms:
            self.captured[tf.child_frame_id] = (
                np.array([tf.transform.translation.x,
                          tf.transform.translation.y,
                          tf.transform.translation.z]),
                np.array([tf.transform.rotation.x,
                          tf.transform.rotation.y,
                          tf.transform.rotation.z,
                          tf.transform.rotation.w]),
            )


def capture_topic(topic: str, expected_keys: set, timeout: float = 5.0):
    """Subscribe to a TFMessage topic, wait until we've seen ALL
    expected_keys. Returns dict {child_frame_id: (pos, quat)}."""
    rclpy.init()
    n = TFCapture(f'derive_capture_{topic.replace("/", "_")}', topic)
    t0 = time.time()
    while time.time() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
        if expected_keys.issubset(n.captured.keys()):
            break
    captured = dict(n.captured)
    n.destroy_node()
    rclpy.shutdown()
    return captured


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def compute_offset_in_marker_frame(cup_pos_world, marker_pos_world,
                                    marker_quat_world):
    """Δ_marker = R_marker_world.inv() * (cup_world - marker_world)"""
    delta_world = np.array(cup_pos_world) - np.array(marker_pos_world)
    R_mw = R.from_quat(marker_quat_world)
    return R_mw.inv().apply(delta_world)


def compute_R_marker_to_object(marker_quat_world, cup_quat_world):
    """R_obj_world = R_marker_world * R_marker_to_object → solve for R_m_to_obj"""
    R_mw = R.from_quat(marker_quat_world)
    R_cw = R.from_quat(cup_quat_world)
    return (R_mw.inv() * R_cw).as_quat()


def fit_cylinder_side_marker_params(cup_bbox, marker_bbox):
    """Reverse-engineer cylinder_side_marker params from the bboxes.
    Best-effort: assumes cup is upright (Z = vertical), marker on side wall.
    Caller should sanity-check the output and override params if needed."""
    cup_h = cup_bbox['bbox_size'][2]
    cup_r = max(cup_bbox['bbox_size'][0], cup_bbox['bbox_size'][1]) / 2.0
    cup_base_z = cup_bbox['bbox_min'][2]
    marker_z = marker_bbox['bbox_center'][2]
    marker_height_pct = (marker_z - cup_base_z) / cup_h
    cup_center_xy = np.array(cup_bbox['bbox_center'][:2])
    marker_center_xy = np.array(marker_bbox['bbox_center'][:2])
    marker_radial_dist = np.linalg.norm(marker_center_xy - cup_center_xy)
    marker_inset = max(0.0, cup_r - marker_radial_dist)
    return {
        'object_radius_m': round(cup_r, 4),
        'object_height_m': round(cup_h, 4),
        'marker_height_pct': round(marker_height_pct, 3),
        'marker_inset_m': round(marker_inset, 4),
        'marker_y_axis': 'up',  # convention; flip to 'down' if cup ends up inverted
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--marker-id', type=int, required=True,
                   help='ArUco marker id (also matches /drop_poses child_frame_id drop_<id>)')
    p.add_argument('--cup-prim', required=True,
                   help='USD prim path of the object (e.g. /World/Containers/cup_red)')
    p.add_argument('--marker-prim', required=True,
                   help='USD prim path of the marker mesh (e.g. /World/Containers/cup_red/aruco_000/aruco_marker_mesh)')
    p.add_argument('--mode', choices=['cylinder', 'explicit'], default='cylinder',
                   help='Output mode: cylinder_side_marker geometry params (default) or raw explicit offset+quat')
    p.add_argument('--timeout', type=float, default=8.0,
                   help='Seconds to wait for /aruco_poses_real to publish the marker (default 8s)')
    args = p.parse_args()

    drop_id = f'drop_{args.marker_id}'
    aruco_id = f'aruco_{args.marker_id}'

    print(f"=== Step 1: USD ground truth ===")
    cup_bbox = get_prim_world_bbox(args.cup_prim)
    marker_bbox = get_prim_world_bbox(args.marker_prim)
    print(f"  Cup '{args.cup_prim}':")
    print(f"    bbox_center: {cup_bbox['bbox_center']}")
    print(f"    bbox_size:   {cup_bbox['bbox_size']}")
    print(f"    prim_quat:   {cup_bbox['prim_quat_xyzw']}")
    print(f"  Marker '{args.marker_prim}':")
    print(f"    bbox_center: {marker_bbox['bbox_center']}")
    print(f"    bbox_size:   {marker_bbox['bbox_size']}")

    print(f"\n=== Step 2: Capture /drop_poses (sim cup body-center pose) ===")
    sim_caps = capture_topic('/drop_poses', {drop_id}, timeout=args.timeout)
    if drop_id not in sim_caps:
        sys.exit(f"ERROR: /drop_poses didn't publish {drop_id} within {args.timeout}s. "
                 f"Run quick_start to spawn drop_poses publisher.")
    cup_pos_sim, cup_quat_sim = sim_caps[drop_id]
    print(f"  {drop_id}: pos={cup_pos_sim.round(4).tolist()}  quat={cup_quat_sim.round(4).tolist()}")

    print(f"\n=== Step 3: Capture /aruco_poses_real (live PnP marker pose) ===")
    real_caps = capture_topic('/aruco_poses_real', {aruco_id}, timeout=args.timeout)
    if aruco_id not in real_caps:
        sys.exit(f"ERROR: /aruco_poses_real didn't publish {aruco_id} within {args.timeout}s. "
                 f"Make sure localize_aruco is running and the marker is in camera FOV.")
    marker_pos_real, marker_quat_real = real_caps[aruco_id]
    print(f"  {aruco_id}: pos={marker_pos_real.round(4).tolist()}  quat={marker_quat_real.round(4).tolist()}")

    print(f"\n=== Step 4: Empirical derivation ===")
    # Position offset: cup body center − marker world position, expressed in marker frame
    offset = compute_offset_in_marker_frame(
        cup_pos_world=cup_pos_sim,
        marker_pos_world=marker_pos_real,
        marker_quat_world=marker_quat_real,
    )
    print(f"  Offset (marker frame, mm): X={offset[0]*1000:+.2f}, Y={offset[1]*1000:+.2f}, Z={offset[2]*1000:+.2f}")

    # Orientation correction: R_marker_to_object such that
    # R_marker_world * R_marker_to_object = R_cup_world
    R_m_to_obj_quat = compute_R_marker_to_object(marker_quat_real, cup_quat_sim)
    R_m_to_obj = R.from_quat(R_m_to_obj_quat)
    print(f"  R_marker_to_object quat (xyzw): {R_m_to_obj_quat.round(4).tolist()}")
    print(f"  R_marker_to_object matrix:")
    for row in R_m_to_obj.as_matrix(): print(f"    {row.round(3).tolist()}")

    # Sanity: compute residuals
    print(f"\n=== Step 5: Round-trip verification ===")
    R_mw = R.from_quat(marker_quat_real)
    cup_pos_back = marker_pos_real + R_mw.apply(offset)
    cup_pos_residual_mm = (cup_pos_back - cup_pos_sim) * 1000
    print(f"  Position round-trip residual: {cup_pos_residual_mm.round(3).tolist()} mm (should be ~0)")
    cup_quat_back = (R_mw * R_m_to_obj).as_quat()
    quat_diff_deg = (R.from_quat(cup_quat_back).inv() * R.from_quat(cup_quat_sim)).magnitude() * 180 / np.pi
    print(f"  Orientation round-trip residual: {quat_diff_deg:.4f}° (should be ~0)")

    print(f"\n=== Step 6: aruco_config.json snippet ===")
    if args.mode == 'cylinder':
        params = fit_cylinder_side_marker_params(cup_bbox, marker_bbox)
        print(f"  Cylinder fit (best-effort; check that cup is upright + marker on side wall):")
        snippet = {
            "marker_ids": [args.marker_id],
            "marker_to_object": {
                "method": "cylinder_side_marker",
                "params": params,
            },
        }
        # Compare cylinder fit's output vs empirical
        try:
            sys.path.insert(0, '/home/aaugus11/Desktop/ros2_ws/src/aruco_camera_localizer')
            from aruco_camera_localizer.marker_geometry import cylinder_side_marker
            fit_result = cylinder_side_marker(**params)
            fit_offset = fit_result['offset']
            fit_offset_mm = [fit_result['offset']['X']*1000,
                             fit_result['offset']['Y']*1000,
                             fit_result['offset']['Z']*1000]
            empirical_mm = [offset[0]*1000, offset[1]*1000, offset[2]*1000]
            diff_mm = [round(f - e, 2) for f, e in zip(fit_offset_mm, empirical_mm)]
            print(f"  Cylinder fit offset (mm):    {[round(v,2) for v in fit_offset_mm]}")
            print(f"  Empirical offset (mm):       {[round(v,2) for v in empirical_mm]}")
            print(f"  Diff (cylinder − empirical): {diff_mm} mm")
            if max(abs(d) for d in diff_mm) > 2:
                print(f"  ⚠ WARN: cylinder fit diverges by >2mm — consider --mode explicit")
        except Exception as e:
            print(f"  (couldn't validate cylinder fit: {e})")
    else:
        snippet = {
            "marker_ids": [args.marker_id],
            "marker_to_object": {
                "method": "explicit_offset_and_orientation",
                "params": {
                    "offset": {
                        "X": round(float(offset[0]), 5),
                        "Y": round(float(offset[1]), 5),
                        "Z": round(float(offset[2]), 5),
                    },
                    "orientation_quat_marker_to_object": [
                        round(float(R_m_to_obj_quat[0]), 5),
                        round(float(R_m_to_obj_quat[1]), 5),
                        round(float(R_m_to_obj_quat[2]), 5),
                        round(float(R_m_to_obj_quat[3]), 5),
                    ],
                },
            },
        }
        print(f"  ⚠ NOTE: 'explicit_offset_and_orientation' method needs to be added to "
              f"MARKER_GEOMETRY_METHODS in marker_geometry.py if not already present.")

    print()
    print(json.dumps(snippet, indent=2))
    print(f"\n  Paste the above into the appropriate row of aruco_config.json")
    print(f"  (under robots.<robot_name>.marker_rows.<row_name>) then restart localize_aruco.")


if __name__ == '__main__':
    main()
