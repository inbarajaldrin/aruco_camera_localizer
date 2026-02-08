import cv2
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import (
    transform_points_world_to_img, quat_to_rpy
)
import numpy as np


def draw_text(frame, cam_pos, cam_quat, object_data, frame_idx, ee_pos, ee_quat, euler_convention='intrinsic'):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    x0 = 10
    y = 30

    def put_line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), font, 0.6, color, 2)
        y += line_height

    put_line(f"Frame: {frame_idx}", (200, 200, 200))

    ee_euler = quat_to_rpy(ee_quat, degrees=True, euler_convention=euler_convention)
    put_line(f"EE xyz: ({1000*ee_pos[0]:.1f}, {1000*ee_pos[1]:.1f}, {1000*ee_pos[2]:.1f}) mm")
    put_line(f"EE rpy: ({ee_euler[0]: 5.1f}, {ee_euler[1]: 5.1f}, {ee_euler[2]: 5.1f}) deg")

    y += 10

    for obj in object_data:
        name = obj["name"]
        pos = obj["position"]
        quat = obj["quaternion"]

        euler = quat_to_rpy(quat, degrees=True, euler_convention=euler_convention)
        put_line(f"{name} xyz: ({1000*pos[0]:.1f}, {1000*pos[1]:.1f}, {1000*pos[2]:.1f}) mm", (0, 255, 0))
        put_line(f"{name} rpy: ({euler[0]: 5.1f}, {euler[1]: 5.1f}, {euler[2]: 5.1f}) deg", (0, 255, 0))
        y += 5


def draw_object_lines(frame, camera_matrix, cam_pos, cam_quat, identified_objects, nearest_pushers):
    for obj in identified_objects:
        if obj.get('no_display', False) or obj.get('ghost_tracked', False):
            continue

        # Draw axes using the object pose
        origin = obj["position"]
        rot = R.from_quat(obj["quaternion"])
        axes_world = [
            origin,
            origin + rot.apply([0.01, 0, 0]),
            origin + rot.apply([0, 0.01, 0]),
            origin + rot.apply([0, 0, 0.01])
        ]

        axes_image = transform_points_world_to_img(axes_world, cam_pos, cam_quat, camera_matrix)

        if len(axes_image) == 4:
            o, x, y, z = axes_image
            if o and x:
                cv2.arrowedLine(frame, o, x, (0, 0, 255), 2, tipLength=0.3)
            if o and y:
                cv2.arrowedLine(frame, o, y, (0, 255, 0), 2, tipLength=0.3)
            if o and z:
                cv2.arrowedLine(frame, o, z, (255, 0, 0), 2, tipLength=0.3)

    return frame


def draw_grasp_points(frame, camera_matrix, cam_pos, cam_quat, identified_objects, model_data):
    """Draw grasp points for identified objects."""
    for obj in identified_objects:
        model_name = obj["name"]

        if obj.get('no_display', False) or obj.get('ghost_tracked', False):
            continue

        if model_name not in model_data or model_data[model_name]['grasp_points'] is None:
            continue

        grasp_points = model_data[model_name]['grasp_points']
        object_pos = obj["position"]
        object_quat = obj["quaternion"]

        rot_matrix = R.from_quat(object_quat).as_matrix()

        world_grasp_points = []
        for grasp_point in grasp_points:
            grasp_pos_local = np.array([
                grasp_point['position']['x'],
                grasp_point['position']['y'],
                grasp_point['position']['z']
            ])
            grasp_pos_world = object_pos + rot_matrix @ grasp_pos_local
            world_grasp_points.append(grasp_pos_world)

        grasp_points_img = transform_points_world_to_img(world_grasp_points, cam_pos, cam_quat, camera_matrix)

        for i, grasp_point_img in enumerate(grasp_points_img):
            if grasp_point_img is not None:
                cv2.circle(frame, grasp_point_img, 8, (255, 0, 0), -1)
                cv2.putText(frame, f"G{i+1}",
                           (grasp_point_img[0] + 10, grasp_point_img[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
