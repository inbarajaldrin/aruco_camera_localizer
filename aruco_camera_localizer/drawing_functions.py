import cv2
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
    transform_points_world_to_img, transform_point_world_to_cam, quat_to_rpy_safe
import numpy as np

def canonicalize_euler(orientation):
    """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
    roll, pitch, yaw = orientation
    if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
        return (0.0, 180.0, (yaw % 360)-180)
    else:
        return orientation

def draw_text(frame, cam_pos, cam_quat, object_data, frame_idx, ee_pos, ee_quat):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    x0 = 10
    y = 30

    def put_line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), font, 0.6, color, 2)
        y += line_height

    # Frame Index
    put_line(f"Frame: {frame_idx}", (200, 200, 200))

    # End Effector
    ee_euler = quat_to_rpy_safe(ee_quat, degrees=True, object_id='end_effector')
    put_line(f"EE xyz: ({1000*ee_pos[0]:.1f}, {1000*ee_pos[1]:.1f}, {1000*ee_pos[2]:.1f}) mm")
    put_line(f"EE rpy: ({ee_euler[0]: 5.1f}, {ee_euler[1]: 5.1f}, {ee_euler[2]: 5.1f}) deg")

    y += 10  # small gap

    # # Camera Info
    # cam_euler = R.from_quat(cam_quat).as_euler('xyz', degrees=True)
    # put_line(f"Camera Pos: x={1000*cam_pos[0]:.1f} mm, y={1000*cam_pos[1]:.1f} mm, z={1000*cam_pos[2]:.1f} mm", (255, 255, 0))
    # put_line(f"Camera Euler: r={cam_euler[0]: 5.1f} deg, p={cam_euler[1]: 5.1f} deg, y={cam_euler[2]: 5.1f} deg", (255, 255, 0))

    # y += 10  # small gap

    # Generic Object Info
    for obj in object_data:
        name = obj["name"]
        pos = obj["position"]
        quat = obj["quaternion"]

        # Use object name as ID for per-object RPY state tracking
        euler = quat_to_rpy_safe(quat, degrees=True, object_id=name)
        put_line(f"{name} xyz: ({1000*pos[0]:.1f}, {1000*pos[1]:.1f}, {1000*pos[2]:.1f}) mm", (0, 255, 0))
        put_line(f"{name} rpy: ({euler[0]: 5.1f}, {euler[1]: 5.1f}, {euler[2]: 5.1f}) deg", (0, 255, 0))
        y += 5

def draw_object_lines(frame, camera_matrix, cam_pos, cam_quat, identified_objects, nearest_pushers):
    color_map = {
        "allen_key": (0, 255, 0),   # Green
        "wrench": (0, 0, 255),      # Red
    }

    for obj in identified_objects:
        # Skip all visualization if no_display flag is set (timeout exceeded)
        if obj.get('no_display', False):
            continue  # Don't draw anything for objects that exceeded timeout
        
        name = obj["name"]
        world_pts = obj["points"]
        color = color_map.get(name, (255, 255, 255))  # default to white
        if obj.get("inferred"):
            color = (0, 255, 255)  # Yellow for inferred

        image_pts = transform_points_world_to_img(world_pts, cam_pos, cam_quat, camera_matrix)

        if len(image_pts) == 3:
            # Draw triangle edges
            for i in range(3):
                pt1 = image_pts[i]
                pt2 = image_pts[(i + 1) % 3]
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw axes using the object pose
        origin = obj["position"]
        rot = R.from_quat(obj["quaternion"])
        axes_world = [
            origin,                          # origin
            origin + rot.apply([0.01, 0, 0]),  # X axis
            origin + rot.apply([0, 0.01, 0]),  # Y axis
            origin + rot.apply([0, 0, 0.01])   # Z axis
        ]

        axes_image = transform_points_world_to_img(axes_world, cam_pos, cam_quat, camera_matrix)

        o, x, y, z = axes_image
        if o and x:
            cv2.arrowedLine(frame, o, x, (0, 0, 255), 2, tipLength=0.3)  # X: Red
        if o and y:
            cv2.arrowedLine(frame, o, y, (0, 255, 0), 2, tipLength=0.3)  # Y: Green
        if o and z:
            cv2.arrowedLine(frame, o, z, (255, 0, 0), 2, tipLength=0.3)  # Z: Blue

        # Draw contact points
        # contact_points = obj["contacts"]
        # contact_poses = [contact[1] for contact in contact_points]
        # contact_norms = [contact[2] for contact in contact_points]
        # contact_axes_start = [contact_pos - 0.02*contact_norm for (contact_pos, contact_norm) in zip(contact_poses, contact_norms)]
        # contact_poses_img = transform_points_world_to_img(contact_poses, cam_pos, cam_quat, camera_matrix)
        # contact_axes_img = transform_points_world_to_img(contact_axes_start, cam_pos, cam_quat, camera_matrix)
        # for pos, ax in zip(contact_poses_img, contact_axes_img):
        #     cv2.arrowedLine(frame, ax, pos, (255, 255, 255), 2, tipLength=0.3)

        # Draw low-res Contour (only if contour data exists)
        if 'contour' in obj and obj['contour'] is not None:
            contour = obj["contour"]
            contour_xyz = contour["xyz"]
            contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, camera_matrix)
            contour_img = np.array(contour_img)
            contour_img.reshape((-1, 1, 2))
            contour_img = contour_img[::20]
            cv2.polylines(frame,[contour_img],False,color)

        # Draw label with background
        # label = f"{name}"
        # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # offset = [-40, 40]
        # cv2.rectangle(frame, (o[0] + offset[0], o[1] - h + offset[1] - 5), 
        #                      (o[0] + w + offset[0], o[1] + offset[1] + 5), (0, 0, 0), -1)
        # cv2.putText(frame, label, (o[0] + offset[0], o[1] + offset[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    for nearest_pusher in nearest_pushers:
        label = f"pusher_{nearest_pusher['pusher_name']} @ contour {nearest_pusher['local_contour_index']}"
        pusher_point_world = nearest_pusher['pusher_location']
        pusher_point_img = transform_points_world_to_img([pusher_point_world], cam_pos, cam_quat, camera_matrix)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - h - 20 - 5), (pusher_point_img[0][0] + w - 20, pusher_point_img[0][1] - 20 + 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (pusher_point_img[0][0] - 20, pusher_point_img[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, nearest_pusher["color"], 2)

        contour_point_world = nearest_pusher['nearest_point']
        contour_point_img = transform_points_world_to_img([contour_point_world], cam_pos, cam_quat, camera_matrix)
        cv2.arrowedLine(frame, pusher_point_img[0], contour_point_img[0], nearest_pusher["color"], 2, tipLength=0.3)

    return frame

def draw_grasp_points(frame, camera_matrix, cam_pos, cam_quat, identified_objects, model_data):
    """Draw grasp points for identified objects (without approach vectors)"""
    for obj in identified_objects:
        model_name = obj["name"]
        
        # Skip grasp points if no_display flag is set (timeout exceeded)
        if obj.get('no_display', False):
            continue  # Don't draw grasp points for objects that exceeded timeout
        
        # Check if this model has grasp points data
        if model_name not in model_data or model_data[model_name]['grasp_points'] is None:
            continue
            
        grasp_points = model_data[model_name]['grasp_points']
        object_pos = obj["position"]
        object_quat = obj["quaternion"]
        
        # Transform object rotation to rotation matrix
        rot_matrix = R.from_quat(object_quat).as_matrix()
        
        # Coordinate system transformation matrix (same as wireframe)
        coord_transform = np.array([
            [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
            [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
            [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
        ])
        
        # Transform each grasp point from object center frame to world frame
        world_grasp_points = []
        for grasp_point in grasp_points:
            # Get grasp point position relative to object center
            grasp_pos_local = np.array([
                grasp_point['position']['x'],
                grasp_point['position']['y'], 
                grasp_point['position']['z']
            ])
            
            # Apply coordinate system transformation (same as wireframe)
            grasp_pos_transformed = coord_transform @ grasp_pos_local
            
            # Transform to world frame
            grasp_pos_world = object_pos + rot_matrix @ grasp_pos_transformed
            world_grasp_points.append(grasp_pos_world)
        
        # Transform grasp points to image coordinates
        grasp_points_img = transform_points_world_to_img(world_grasp_points, cam_pos, cam_quat, camera_matrix)
        
        # Draw grasp points (without approach vectors)
        for i, grasp_point_img in enumerate(grasp_points_img):
            if grasp_point_img is not None:
                # Draw grasp point as a circle (Blue color)
                cv2.circle(frame, grasp_point_img, 8, (255, 0, 0), -1)  # Blue circle, larger size
                
                # Draw grasp point ID
                cv2.putText(frame, f"G{i+1}", 
                           (grasp_point_img[0] + 10, grasp_point_img[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text with thicker stroke