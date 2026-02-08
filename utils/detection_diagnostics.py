#!/usr/bin/env python3
"""
Detection Diagnostics Logger

Runs the ArUco detection pipeline and logs per-frame data to a CSV + summary
so you can diagnose jitter, pose flipping, close-range dropout, and Z vs height.

Usage:
    python3 utils/detection_diagnostics.py --camera-id 8
    python3 utils/detection_diagnostics.py --camera-id 8 --duration 30
    python3 utils/detection_diagnostics.py --image-topic /camera/image_raw

Output:
    logs/detection_diag_<timestamp>.csv   - per-frame data
    logs/detection_diag_<timestamp>.txt   - summary report
"""

import sys
import os
import csv
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime

import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

# Add parent to path so we can import the localizer modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aruco_camera_localizer.detection_functions import detect_markers, estimate_poses
from aruco_camera_localizer.geometric_functions import (
    rvec_to_quat, transform_point_cam_to_world, transform_orientation_cam_to_world
)
from aruco_camera_localizer.data_path_finder import find_aruco_data_dir

# UR5e DH parameters (from ik_solver.py)
DH_PARAMS = [
    (0, 0.1625, 0, np.pi / 2),
    (0, 0, -0.425, 0),
    (0, 0, -0.3922, 0),
    (0, 0.1333, 0, np.pi / 2),
    (0, 0.0997, 0, -np.pi / 2),
    (0, 0.0996, 0, 0),
]


def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])


def forward_kinematics(joint_angles):
    T = np.eye(4)
    for i, (theta, d, a, alpha) in enumerate(DH_PARAMS):
        T = T @ dh_transform(joint_angles[i] + theta, d, a, alpha)
    return T


# ---------------------------------------------------------------------------
# Camera / marker constants (same as merged_localization.py)
# ---------------------------------------------------------------------------
C_WIDTH, C_HEIGHT = 1280, 720
C_HFOV, C_VFOV = 69.4, 42.5
FX = C_WIDTH / (2 * np.tan(np.deg2rad(C_HFOV / 2)))
FY = C_HEIGHT / (2 * np.tan(np.deg2rad(C_VFOV / 2)))
FX_SIM, FY_SIM = 731.78, 731.78

MARKER_SIZE_MM = 21
BORDER_PCT = 5
TOTAL_MARKER_SIZE = (MARKER_SIZE_MM - 2 * MARKER_SIZE_MM * BORDER_PCT / 100) / 1000.0

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_50": aruco.DICT_5X5_50,
}

CAM_OFFSET_POS = np.array([0.0, 0.0, 0.08])
CAM_OFFSET_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# ROS2 node for joint states + EE pose
# ---------------------------------------------------------------------------
class DiagNode(Node):
    def __init__(self, image_topic=None):
        super().__init__("detection_diagnostics")
        self.lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.bridge = CvBridge()

        # EE pose
        self.ee_pos = np.array([0.064125, -0.384832, 0.480763])
        self.ee_quat = np.array([0.0, -1.0, 0.0, 0.0])
        self.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose",
                                 self._ee_cb, 10)

        # Joint states
        self.joint_positions = np.zeros(6)
        self.joint_names = []
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # Optional image topic
        self.latest_frame = None
        self.frame_available = False
        if image_topic:
            self.create_subscription(Image, image_topic, self._img_cb, 10)

    def _ee_cb(self, msg):
        with self.lock:
            self.ee_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def _joint_cb(self, msg):
        with self.lock:
            self.joint_names = list(msg.name)
            self.joint_positions = np.array(msg.position[:6])

    def _img_cb(self, msg):
        with self.image_lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_available = True

    def get_ee_pose(self):
        with self.lock:
            return self.ee_pos.copy(), self.ee_quat.copy()

    def get_joint_state(self):
        with self.lock:
            return self.joint_names[:], self.joint_positions.copy()

    def get_camera_pose(self):
        with self.lock:
            r_ee = R.from_quat(self.ee_quat)
            r_off = R.from_quat(CAM_OFFSET_QUAT)
            pos = self.ee_pos + r_ee.apply(CAM_OFFSET_POS)
            quat = (r_ee * r_off).as_quat()
        return pos, quat

    def get_frame(self):
        with self.image_lock:
            if self.frame_available:
                return self.latest_frame.copy(), True
            return None, False


# ---------------------------------------------------------------------------
# Detector parameters (same as merged_localization.py)
# ---------------------------------------------------------------------------
def create_detector_parameters():
    p = aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 43
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.008
    p.maxMarkerPerimeterRate = 0.4
    p.polygonalApproxAccuracyRate = 0.03
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.1
    return p


def marker_pixel_size(corner):
    """Average side length in pixels for a detected marker corner."""
    pts = corner[0]
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return np.mean(sides)


def rvec_z_component(rvec):
    """Z-axis direction of marker in camera frame (for flip detection)."""
    R_mat, _ = cv2.Rodrigues(rvec)
    return R_mat[:, 2]  # third column = marker Z-axis in camera coords


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ArUco detection diagnostics logger.")
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--image-topic", type=str, default=None)
    parser.add_argument("--duration", type=int, default=60, help="Seconds to log (default 60)")
    parser.add_argument("--sim", action="store_true", help="Use sim camera intrinsics")
    args = parser.parse_args()

    use_sim = args.sim or args.image_topic is not None

    if use_sim:
        cam_matrix = np.array([[FX_SIM, 0, C_WIDTH / 2],
                               [0, FY_SIM, C_HEIGHT / 2],
                               [0, 0, 1]], dtype=np.float32)
    else:
        cam_matrix = np.array([[FX, 0, C_WIDTH / 2],
                               [0, FY, C_HEIGHT / 2],
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # ROS2 init
    rclpy.init()
    node = DiagNode(args.image_topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Camera
    cap = None
    use_ros_topic = args.image_topic is not None
    if not use_ros_topic:
        cam_id = args.camera_id if args.camera_id is not None else 0
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Cannot open camera {cam_id}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C_HEIGHT)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    else:
        print("Waiting for first frame from ROS topic...")
        while True:
            f, ok = node.get_frame()
            if ok:
                break
            time.sleep(0.1)
        print("Got frame.")

    params = create_detector_parameters()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Output files
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"detection_diag_{ts}.csv"
    txt_path = log_dir / f"detection_diag_{ts}.txt"

    csv_fields = [
        "frame", "time_s",
        # Robot
        "ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw",
        "cam_x", "cam_y", "cam_z",
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
        "fk_ee_z",  # FK-computed EE Z for independent verification
        # Detection summary
        "n_markers_detected", "n_markers_4x4", "n_markers_5x5",
        # Per-marker (up to 8 markers logged inline; most frames have fewer)
        # Each marker gets: id, dict, pixel_size, tvec_x/y/z, rvec_z_x/y/z, reproj_err
        *[f"m{i}_{f}" for i in range(8)
          for f in ("id", "dict", "px_size", "tx", "ty", "tz",
                    "rz_x", "rz_y", "rz_z", "reproj")],
        # Per-object (up to 4 objects)
        *[f"obj{i}_{f}" for i in range(4)
          for f in ("name", "wx", "wy", "wz", "qx", "qy", "qz", "qw",
                    "marker_id", "marker_dict")],
        # Jitter tracking (frame-to-frame delta for first object)
        "obj0_delta_pos_mm", "obj0_delta_rot_deg",
        # Flip detection: did marker rvec Z flip sign from last frame?
        "any_flip_detected",
    ]

    csvfile = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()

    # Load marker annotations
    data_dir = find_aruco_data_dir()
    if data_dir is None:
        print("Could not find aruco-grasp-annotator data directory in Documents folder")
        return

    marker_annotations = {}
    aruco_dir = data_dir / "aruco"
    if aruco_dir.exists():
        for jf in aruco_dir.glob("*_aruco.json"):
            import json
            with open(jf) as f:
                d = json.load(f)
            model_name = jf.stem.replace("_aruco", "")
            dict_name = d.get("aruco_dictionary", "DICT_4X4_50")
            for ann in d["markers"]:
                marker_annotations[(ann["aruco_id"], dict_name)] = {
                    "annotation": ann, "model_name": model_name
                }

    # State for tracking
    prev_rvec_z = {}       # (marker_id, dict) -> last rvec Z-axis
    prev_obj_world = {}    # model_name -> (pos, quat) for EMA smoothing
    prev_obj_world_raw = {}  # model_name -> (pos, quat) for jitter tracking
    stats = {
        "total_frames": 0,
        "frames_with_markers": 0,
        "flip_count": 0,
        "per_marker_detections": {},   # (id, dict) -> count
        "obj_z_values": {},            # model_name -> list of z
        "cam_z_values": [],            # camera Z per frame
        "jitter_pos": [],              # mm per frame
        "jitter_rot": [],              # deg per frame
    }

    print(f"Logging to {csv_path}")
    print(f"Recording for {args.duration}s ... press 'q' in window to stop early.")

    start_time = time.time()
    frame_idx = 0

    while time.time() - start_time < args.duration:
        # Grab frame
        if use_ros_topic:
            frame, ok = node.get_frame()
            if not ok:
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1
        t = time.time() - start_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        ee_pos, ee_quat = node.get_ee_pose()
        cam_pos, cam_quat = node.get_camera_pose()
        joint_names, joint_pos = node.get_joint_state()

        # FK verification
        fk_ee_z = np.nan
        if len(joint_pos) >= 6 and np.any(joint_pos != 0):
            T_fk = forward_kinematics(joint_pos)
            fk_ee_z = T_fk[2, 3]

        # Detect
        corners, ids, dict_names = detect_markers(frame, gray, ARUCO_DICTS, params)
        marker_poses = estimate_poses(
            corners, ids, dict_names, cam_matrix, dist_coeffs, TOTAL_MARKER_SIZE,
            talk=False
        )

        n_4x4 = sum(1 for _, dn in marker_poses if dn == "DICT_4X4_50")
        n_5x5 = sum(1 for _, dn in marker_poses if dn == "DICT_5X5_50")

        stats["total_frames"] += 1
        if len(marker_poses) > 0:
            stats["frames_with_markers"] += 1
        stats["cam_z_values"].append(cam_pos[2] if cam_pos is not None else np.nan)

        row = {
            "frame": frame_idx, "time_s": f"{t:.3f}",
            "ee_x": f"{ee_pos[0]:.6f}", "ee_y": f"{ee_pos[1]:.6f}", "ee_z": f"{ee_pos[2]:.6f}",
            "ee_qx": f"{ee_quat[0]:.6f}", "ee_qy": f"{ee_quat[1]:.6f}",
            "ee_qz": f"{ee_quat[2]:.6f}", "ee_qw": f"{ee_quat[3]:.6f}",
            "cam_x": f"{cam_pos[0]:.6f}", "cam_y": f"{cam_pos[1]:.6f}", "cam_z": f"{cam_pos[2]:.6f}",
            "fk_ee_z": f"{fk_ee_z:.6f}",
            "n_markers_detected": len(marker_poses),
            "n_markers_4x4": n_4x4, "n_markers_5x5": n_5x5,
            "any_flip_detected": 0,
        }

        # Joint angles
        for ji in range(min(6, len(joint_pos))):
            row[f"joint_{ji+1}"] = f"{np.degrees(joint_pos[ji]):.2f}"

        # Per-marker data
        any_flip = False
        corner_lookup = {}
        for ci, (mid, dn) in enumerate(zip(ids, dict_names)):
            corner_lookup[(mid, dn)] = corners[ci]

        for mi, ((mid, dn), pd) in enumerate(marker_poses.items()):
            if mi >= 8:
                break
            key = (mid, dn)

            stats["per_marker_detections"][key] = stats["per_marker_detections"].get(key, 0) + 1

            px_sz = marker_pixel_size(corner_lookup.get(key, np.zeros((1, 4, 2)))) if key in corner_lookup else 0
            rz = rvec_z_component(pd["rvec"])

            # Flip detection: check if marker Z-axis flipped from last frame
            if key in prev_rvec_z:
                dot = np.dot(rz, prev_rvec_z[key])
                if dot < 0:  # Z-axis reversed -> pose ambiguity flip
                    any_flip = True
                    stats["flip_count"] += 1
            prev_rvec_z[key] = rz.copy()

            prefix = f"m{mi}_"
            row[prefix + "id"] = mid
            row[prefix + "dict"] = dn
            row[prefix + "px_size"] = f"{px_sz:.1f}"
            row[prefix + "tx"] = f"{pd['tvec'][0]:.5f}"
            row[prefix + "ty"] = f"{pd['tvec'][1]:.5f}"
            row[prefix + "tz"] = f"{pd['tvec'][2]:.5f}"
            row[prefix + "rz_x"] = f"{rz[0]:.4f}"
            row[prefix + "rz_y"] = f"{rz[1]:.4f}"
            row[prefix + "rz_z"] = f"{rz[2]:.4f}"
            row[prefix + "reproj"] = f"{pd['reproj_error']:.3f}"

        row["any_flip_detected"] = 1 if any_flip else 0

        # Map to objects: collect all marker candidates per model
        from aruco_camera_localizer.merged_localization import estimate_object_pose_from_marker
        from aruco_camera_localizer.geometric_functions import slerp_quat
        candidates_per_model = {}  # model_name -> [(mkey, tvec, rvec, reproj)]
        for mkey, pd in marker_poses.items():
            if mkey not in marker_annotations:
                continue
            ann_data = marker_annotations[mkey]
            try:
                ot, orv = estimate_object_pose_from_marker(
                    (pd["tvec"], pd["rvec"]), ann_data["annotation"]
                )
            except ValueError:
                continue
            mn = ann_data["model_name"]
            if mn not in candidates_per_model:
                candidates_per_model[mn] = []
            candidates_per_model[mn].append((mkey, ot, orv, pd["reproj_error"]))

        # Weighted average of all visible markers per object + EMA smoothing
        EMA_ALPHA = 0.5
        obj_idx = 0
        for mn, marker_list in candidates_per_model.items():
            if obj_idx >= 4:
                break

            if len(marker_list) == 1:
                mkey, ot, orv, _ = marker_list[0]
            else:
                weights = np.array([1.0 / max(e[3], 1e-6) for e in marker_list])
                weights /= weights.sum()
                tvecs = np.array([e[1] for e in marker_list])
                ot = np.average(tvecs, axis=0, weights=weights)
                rotations = R.from_rotvec([e[2].flatten() for e in marker_list])
                orv = rotations.mean(weights=weights).as_rotvec().reshape(3, 1)
                mkey = marker_list[0][0]  # report first marker key

            oq = rvec_to_quat(orv)
            if cam_pos is not None and cam_quat is not None:
                pw = transform_point_cam_to_world(ot, cam_pos, cam_quat)
                qw = transform_orientation_cam_to_world(oq, cam_quat)
            else:
                pw, qw = ot, oq

            # EMA smoothing
            if mn in prev_obj_world:
                pp, pq = prev_obj_world[mn]
                pw = (1.0 - EMA_ALPHA) * pp + EMA_ALPHA * pw
                qw = slerp_quat(pq, qw, blend=EMA_ALPHA)
            prev_obj_world[mn] = (pw.copy(), qw.copy())

            prefix = f"obj{obj_idx}_"
            row[prefix + "name"] = mn
            row[prefix + "wx"] = f"{pw[0]:.5f}"
            row[prefix + "wy"] = f"{pw[1]:.5f}"
            row[prefix + "wz"] = f"{pw[2]:.5f}"
            row[prefix + "qx"] = f"{qw[0]:.5f}"
            row[prefix + "qy"] = f"{qw[1]:.5f}"
            row[prefix + "qz"] = f"{qw[2]:.5f}"
            row[prefix + "qw"] = f"{qw[3]:.5f}"
            row[prefix + "marker_id"] = mkey[0]
            row[prefix + "marker_dict"] = mkey[1]

            # Track Z values per object
            if mn not in stats["obj_z_values"]:
                stats["obj_z_values"][mn] = []
            stats["obj_z_values"][mn].append(pw[2])

            # Jitter: frame-to-frame delta for first object
            if obj_idx == 0:
                if mn in prev_obj_world_raw:
                    pp, pq = prev_obj_world_raw[mn]
                    d_pos = np.linalg.norm(pw - pp) * 1000  # mm
                    qd = np.abs(np.dot(qw, pq))
                    qd = np.clip(qd, -1.0, 1.0)
                    d_rot = np.degrees(2 * np.arccos(qd))
                    row["obj0_delta_pos_mm"] = f"{d_pos:.2f}"
                    row["obj0_delta_rot_deg"] = f"{d_rot:.2f}"
                    stats["jitter_pos"].append(d_pos)
                    stats["jitter_rot"].append(d_rot)
                prev_obj_world_raw[mn] = (pw.copy(), qw.copy())

            obj_idx += 1

        writer.writerow(row)

        # Live display
        info = f"F{frame_idx} | {len(marker_poses)} markers"
        if any_flip:
            info += " | FLIP!"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Diagnostics", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    csvfile.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # Write summary report
    # -----------------------------------------------------------------------
    lines = []
    lines.append("=" * 60)
    lines.append("DETECTION DIAGNOSTICS SUMMARY")
    lines.append(f"Recorded: {frame_idx} frames over {time.time()-start_time:.1f}s")
    lines.append(f"CSV log:  {csv_path}")
    lines.append("=" * 60)

    det_rate = stats["frames_with_markers"] / max(1, stats["total_frames"]) * 100
    lines.append(f"\nDetection rate: {det_rate:.1f}% ({stats['frames_with_markers']}/{stats['total_frames']} frames)")

    lines.append(f"\nPose flips detected: {stats['flip_count']} total")
    if stats["flip_count"] > 0:
        lines.append("  -> This indicates solvePnP pose ambiguity (marker Z-axis reversal).")
        lines.append("  -> Common when viewing markers close to head-on.")
        lines.append("  -> Fix: use solvePnPGeneric to get both solutions and pick")
        lines.append("     the one consistent with the previous frame.")

    lines.append("\nPer-marker detection counts:")
    for key, cnt in sorted(stats["per_marker_detections"].items(), key=lambda x: -x[1]):
        mid, dn = key
        lines.append(f"  ID {mid:3d} ({dn}): {cnt} frames")

    if stats["jitter_pos"]:
        jp = np.array(stats["jitter_pos"])
        jr = np.array(stats["jitter_rot"])
        lines.append(f"\nPosition jitter (frame-to-frame, first object):")
        lines.append(f"  Mean:   {np.mean(jp):.2f} mm")
        lines.append(f"  Median: {np.median(jp):.2f} mm")
        lines.append(f"  Std:    {np.std(jp):.2f} mm")
        lines.append(f"  Max:    {np.max(jp):.2f} mm")
        lines.append(f"  P95:    {np.percentile(jp, 95):.2f} mm")
        lines.append(f"\nRotation jitter (frame-to-frame, first object):")
        lines.append(f"  Mean:   {np.mean(jr):.2f} deg")
        lines.append(f"  Median: {np.median(jr):.2f} deg")
        lines.append(f"  Max:    {np.max(jr):.2f} deg")
        lines.append(f"  P95:    {np.percentile(jr, 95):.2f} deg")

    for mn, zvals in stats["obj_z_values"].items():
        z = np.array(zvals)
        lines.append(f"\nObject '{mn}' world-Z statistics (should be stable if object is stationary):")
        lines.append(f"  Mean: {np.mean(z)*1000:.1f} mm")
        lines.append(f"  Std:  {np.std(z)*1000:.2f} mm")
        lines.append(f"  Range: [{np.min(z)*1000:.1f}, {np.max(z)*1000:.1f}] mm")

    cam_z = np.array(stats["cam_z_values"])
    cam_z_valid = cam_z[~np.isnan(cam_z)]
    if len(cam_z_valid) > 0:
        lines.append(f"\nCamera Z (height) during recording:")
        lines.append(f"  Mean: {np.mean(cam_z_valid)*1000:.1f} mm")
        lines.append(f"  Range: [{np.min(cam_z_valid)*1000:.1f}, {np.max(cam_z_valid)*1000:.1f}] mm")

        # Z correlation: does object Z track camera Z?
        for mn, zvals in stats["obj_z_values"].items():
            if len(zvals) > 10:
                # Align lengths (object may not be detected every frame)
                n = min(len(zvals), len(cam_z_valid))
                if n > 10:
                    cc = np.corrcoef(cam_z_valid[:n], np.array(zvals[:n]))[0, 1]
                    lines.append(f"\n  Correlation(cam_Z, {mn}_Z) = {cc:.3f}")
                    if abs(cc) > 0.7:
                        lines.append(f"    -> Strong correlation: object Z drifts with camera height.")
                        lines.append(f"    -> This is expected for monocular depth (Z = f*size/px_size).")

    lines.append("\n" + "=" * 60)
    lines.append("CLOSE-RANGE ANALYSIS")
    lines.append("=" * 60)
    lines.append("Check the CSV column 'm0_px_size' (marker pixel size).")
    lines.append("If markers are NOT detected when close, likely causes:")
    lines.append("  1. Marker fills >25% of image perimeter (maxMarkerPerimeterRate too low)")
    lines.append("  2. Marker partially out of frame")
    lines.append("  3. Adaptive threshold window too small for large marker")
    lines.append(f"\nCurrent maxMarkerPerimeterRate = 0.25 (max marker side ~{0.25*2*(C_WIDTH+C_HEIGHT)/4:.0f}px)")
    lines.append(f"If your marker exceeds this at close range, increase it.")

    report = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nFiles written:")
    print(f"  CSV: {csv_path}")
    print(f"  TXT: {txt_path}")

    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
