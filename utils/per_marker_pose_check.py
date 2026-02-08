#!/usr/bin/env python3
"""Per-marker object pose diagnostic — guided one-at-a-time evaluation.
Steps through each marker. Shows both IPPE solutions (green [A] / red [B]).
Y = one of them is correct (quaternion OK), N = neither is correct (quaternion BAD).
Automatically advances to the next marker after rating.

Usage:
    python3 utils/per_marker_pose_check.py --camera-id 12 --object u_orange
"""

import sys
import time
import argparse
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aruco_camera_localizer.merged_localization import (
    estimate_object_pose_from_marker, create_detector_parameters
)
from aruco_camera_localizer.detection_functions import detect_markers
from aruco_camera_localizer.data_path_finder import find_aruco_data_dir

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_50": aruco.DICT_5X5_50,
}

C_WIDTH, C_HEIGHT = 1280, 720
C_HFOV, C_VFOV = 69.4, 42.5
FX = C_WIDTH / (2 * np.tan(np.deg2rad(C_HFOV / 2)))
FY = C_HEIGHT / (2 * np.tan(np.deg2rad(C_VFOV / 2)))
MARKER_SIZE_MM = 21
BORDER_PCT = 5
TOTAL_MARKER_SIZE = (MARKER_SIZE_MM - 2 * MARKER_SIZE_MM * BORDER_PCT / 100) / 1000.0


def draw_wireframe(frame, obj_tvec, obj_rvec, vertices, edges,
                   cam_matrix, dist_coeffs, color, thickness=2):
    """Draw wireframe mesh projected onto the frame."""
    pts_3d = np.array(vertices, dtype=np.float32)
    projected, _ = cv2.projectPoints(pts_3d, obj_rvec, obj_tvec, cam_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)

    for edge in edges:
        if len(edge) >= 2:
            si, ei = edge[0], edge[1]
            if si < len(projected) and ei < len(projected):
                p1 = tuple(projected[si])
                p2 = tuple(projected[ei])
                if all(-2000 < c < 4000 for c in p1 + p2):
                    cv2.line(frame, p1, p2, color, thickness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", type=int, default=12)
    parser.add_argument("--object", type=str, default="u_orange")
    args = parser.parse_args()

    cam_matrix = np.array([[FX, 0, C_WIDTH / 2],
                           [0, FY, C_HEIGHT / 2],
                           [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera_id}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)

    print("Warming up camera (2s)...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        cap.read()

    params = create_detector_parameters()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Load data
    data_dir = find_aruco_data_dir()
    if data_dir is None:
        print("No data dir found")
        return
    aruco_file = data_dir / "aruco" / f"{args.object}_aruco.json"
    wireframe_file = data_dir / "wireframe" / f"{args.object}_wireframe.json"
    if not aruco_file.exists():
        print(f"No annotation file: {aruco_file}")
        return
    if not wireframe_file.exists():
        print(f"No wireframe file: {wireframe_file}")
        return

    with open(wireframe_file) as f:
        wf = json.load(f)
    vertices, edges = wf["vertices"], wf["edges"]

    print(f"Loading: {aruco_file}")
    with open(aruco_file) as f:
        d = json.load(f)
    dict_name = d.get("aruco_dictionary", "DICT_4X4_50")

    # Build ordered marker list
    marker_list = []
    for ann in d["markers"]:
        mkey = (ann["aruco_id"], dict_name)
        q = ann["T_object_to_marker"]["rotation"]["quaternion"]
        marker_list.append({"mkey": mkey, "ann": ann})
        print(f"  ID{ann['aruco_id']:3d} ({ann['face_type']:6s})  "
              f"quat=({q['x']:+.4f}, {q['y']:+.4f}, {q['z']:+.4f}, {q['w']:+.4f})")

    # solvePnP object points
    half = TOTAL_MARKER_SIZE / 2.0
    marker_obj_pts = np.array([
        [-half, half, 0], [half, half, 0],
        [half, -half, 0], [-half, -half, 0]
    ], dtype=np.float32)

    current_idx = 0
    results = {}

    print(f"\nWill test {len(marker_list)} markers one at a time.")
    print(f"  Green = solution [A]   Red = solution [B]")
    print(f"  Y = one wireframe aligns (quaternion OK)")
    print(f"  N = neither aligns (quaternion BAD)")
    print(f"  S = skip    Q = quit\n")

    while current_idx < len(marker_list):
        current = marker_list[current_idx]
        target_mkey = current["mkey"]
        target_ann = current["ann"]
        face = target_ann["face_type"]
        aruco_id = target_ann["aruco_id"]

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        corners, ids, dict_names_det = detect_markers(frame, gray, ARUCO_DICTS, params)

        # Only process the target marker
        detected = False
        for corner, mid, dname in zip(corners, ids, dict_names_det):
            if (mid, dname) != target_mkey:
                continue
            detected = True
            img_pts = corner[0].reshape(-1, 2)

            # Draw the detected marker outline
            cv2.aruco.drawDetectedMarkers(frame, [corner])

            try:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    marker_obj_pts, img_pts, cam_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
            except Exception:
                break

            for sol_idx in range(min(n_sol, 2)):
                m_tvec = tvecs[sol_idx].flatten()
                if m_tvec[2] < 0.05:
                    continue
                try:
                    obj_tvec, obj_rvec = estimate_object_pose_from_marker(
                        (m_tvec, rvecs[sol_idx]), target_ann
                    )
                except ValueError:
                    continue

                color = (0, 255, 0) if sol_idx == 0 else (0, 0, 255)
                thickness = 2 if sol_idx == 0 else 1
                draw_wireframe(frame, obj_tvec, obj_rvec, vertices, edges,
                               cam_matrix, dist_coeffs, color, thickness)

        # HUD - instructions
        status = "DETECTED" if detected else "waiting..."
        cv2.putText(frame, f"[{current_idx+1}/{len(marker_list)}] Show ID{aruco_id} ({face}) to camera   {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Green=[A]  Red=[B]  |  Y=OK  N=BAD  S=skip  Q=quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show previous results
        y_off = 85
        for face_type, verdict in results.items():
            v_color = (0, 255, 0) if verdict == "OK" else (0, 0, 255)
            cv2.putText(frame, f"  {face_type}: {verdict}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, v_color, 2)
            y_off += 22

        cv2.imshow("Marker Pose Check", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('y'):
            results[face] = "OK"
            print(f"  {face:8s} (ID{aruco_id}) -> OK")
            current_idx += 1
        elif key == ord('n'):
            results[face] = "BAD"
            print(f"  {face:8s} (ID{aruco_id}) -> BAD")
            current_idx += 1
        elif key == ord('s'):
            results[face] = "SKIP"
            print(f"  {face:8s} (ID{aruco_id}) -> SKIP")
            current_idx += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"  RESULTS — {args.object}")
    print(f"{'='*50}")
    for face_type, verdict in results.items():
        print(f"  {face_type:8s} -> {verdict}")
    print(f"{'='*50}")

    # --- PASS 2: Fix BAD markers using a good reference marker ---
    bad_markers = [m for m in marker_list if results.get(m["ann"]["face_type"]) == "BAD"]
    ok_markers = [m for m in marker_list if results.get(m["ann"]["face_type"]) == "OK"]

    if not bad_markers:
        print("\nNo BAD markers to fix.")
        cap.release()
        cv2.destroyAllWindows()
        return
    if not ok_markers:
        print("\nNo OK markers to use as reference!")
        cap.release()
        cv2.destroyAllWindows()
        return

    ok_mkeys = {m["mkey"] for m in ok_markers}
    ok_ann_by_mkey = {m["mkey"]: m["ann"] for m in ok_markers}

    print(f"\n--- PASS 2: Fixing {len(bad_markers)} BAD markers ---")
    print(f"  Show a BAD marker + any OK marker simultaneously.")
    print(f"  C = capture & compute corrected quaternion")
    print(f"  S = skip    Q = quit\n")

    fix_idx = 0
    corrections = {}
    samples = {}  # face -> list of sampled T_object_to_marker matrices

    while fix_idx < len(bad_markers):
        bad = bad_markers[fix_idx]
        bad_mkey = bad["mkey"]
        bad_ann = bad["ann"]
        bad_face = bad_ann["face_type"]
        bad_id = bad_ann["aruco_id"]

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        corners, ids_det, dict_names_det = detect_markers(frame, gray, ARUCO_DICTS, params)

        # Find raw solvePnP for all detected markers
        marker_raw = {}  # mkey -> (rvec, tvec) best solution
        for corner, mid, dname in zip(corners, ids_det, dict_names_det):
            mkey = (mid, dname)
            img_pts = corner[0].reshape(-1, 2)
            cv2.aruco.drawDetectedMarkers(frame, [corner])
            try:
                n_sol, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                    marker_obj_pts, img_pts, cam_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
            except Exception:
                continue
            if n_sol > 0 and tvecs[0].flatten()[2] > 0.05:
                marker_raw[mkey] = (rvecs[0], tvecs[0])

        # Check what we see
        bad_detected = bad_mkey in marker_raw
        ref_detected = [mk for mk in marker_raw if mk in ok_mkeys]

        status_parts = []
        if bad_detected:
            status_parts.append(f"BAD ID{bad_id} detected")
        if ref_detected:
            ref_ids = [str(mk[0]) for mk in ref_detected]
            status_parts.append(f"REF ID{','.join(ref_ids)} detected")
        status = " | ".join(status_parts) if status_parts else "waiting..."

        # If both visible, show the corrected wireframe in yellow preview
        if bad_detected and ref_detected:
            # Use first available reference marker
            ref_mkey = ref_detected[0]
            ref_ann = ok_ann_by_mkey[ref_mkey]
            ref_rvec, ref_tvec = marker_raw[ref_mkey]
            try:
                obj_tvec, obj_rvec = estimate_object_pose_from_marker(
                    (ref_tvec.flatten(), ref_rvec), ref_ann
                )
                # Draw wireframe from reference (green = ground truth)
                draw_wireframe(frame, obj_tvec, obj_rvec, vertices, edges,
                               cam_matrix, dist_coeffs, (0, 255, 0), 2)
            except ValueError:
                pass

        # HUD
        cv2.putText(frame, f"FIX [{fix_idx+1}/{len(bad_markers)}] ID{bad_id} ({bad_face})   {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        n_samp = len(samples.get(bad_face, []))
        cv2.putText(frame, f"C=capture  S=skip  Q=quit  D=done ({n_samp} samples)",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Marker Pose Check", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and bad_detected and ref_detected:
            # Capture: compute corrected T_object_to_marker for the bad marker
            ref_mkey = ref_detected[0]
            ref_ann = ok_ann_by_mkey[ref_mkey]
            ref_rvec, ref_tvec = marker_raw[ref_mkey]
            bad_rvec, bad_tvec = marker_raw[bad_mkey]

            try:
                # True object pose from reference marker
                obj_tvec, obj_rvec = estimate_object_pose_from_marker(
                    (ref_tvec.flatten(), ref_rvec), ref_ann
                )
                # Build T_cam_to_object
                R_cam_obj, _ = cv2.Rodrigues(obj_rvec)
                T_cam_obj = np.eye(4)
                T_cam_obj[:3, :3] = R_cam_obj
                T_cam_obj[:3, 3] = obj_tvec.flatten()

                # Build T_cam_to_marker (bad marker raw)
                R_cam_bad, _ = cv2.Rodrigues(bad_rvec)
                T_cam_bad = np.eye(4)
                T_cam_bad[:3, :3] = R_cam_bad
                T_cam_bad[:3, 3] = bad_tvec.flatten()

                # T_object_to_marker = inv(T_cam_to_object) @ T_cam_to_marker
                T_obj_marker = np.linalg.inv(T_cam_obj) @ T_cam_bad

                samples.setdefault(bad_face, []).append(T_obj_marker)
                n = len(samples[bad_face])

                q_new = R.from_matrix(T_obj_marker[:3, :3]).as_quat()  # [x,y,z,w]
                pos = T_obj_marker[:3, 3]
                print(f"  Sample {n} for {bad_face} (ref=ID{ref_mkey[0]}): "
                      f"quat=({q_new[0]:+.4f}, {q_new[1]:+.4f}, {q_new[2]:+.4f}, {q_new[3]:+.4f})  "
                      f"pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"  Error: {e}")

        elif key == ord('d'):
            # Done with this marker, compute average
            if bad_face in samples and samples[bad_face]:
                quats = [R.from_matrix(T[:3, :3]).as_quat() for T in samples[bad_face]]
                positions = [T[:3, 3] for T in samples[bad_face]]
                avg_quat = R.from_quat(quats).mean().as_quat()
                avg_pos = np.mean(positions, axis=0)
                corrections[bad_face] = {"quat": avg_quat, "pos": avg_pos, "n": len(quats)}
                print(f"\n  >>> {bad_face} CORRECTED ({len(quats)} samples averaged):")
                print(f"      quat=({avg_quat[0]:+.4f}, {avg_quat[1]:+.4f}, {avg_quat[2]:+.4f}, {avg_quat[3]:+.4f})")
                print(f"      pos =({avg_pos[0]:+.4f}, {avg_pos[1]:+.4f}, {avg_pos[2]:+.4f})")
                q_old = bad_ann["T_object_to_marker"]["rotation"]["quaternion"]
                print(f"      was =({q_old['x']:+.4f}, {q_old['y']:+.4f}, {q_old['z']:+.4f}, {q_old['w']:+.4f})")
            fix_idx += 1

        elif key == ord('s'):
            print(f"  {bad_face} (ID{bad_id}) -> SKIPPED")
            fix_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    if corrections:
        print(f"\n{'='*50}")
        print(f"  CORRECTED QUATERNIONS — {args.object}")
        print(f"{'='*50}")
        for face, data in corrections.items():
            q = data["quat"]
            p = data["pos"]
            print(f"  {face:8s} ({data['n']} samples):")
            print(f"    quat: x={q[0]:+.6f}, y={q[1]:+.6f}, z={q[2]:+.6f}, w={q[3]:+.6f}")
            print(f"    pos : x={p[0]:+.6f}, y={p[1]:+.6f}, z={p[2]:+.6f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
