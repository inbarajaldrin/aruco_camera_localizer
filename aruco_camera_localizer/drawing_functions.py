import cv2
from scipy.spatial.transform import Rotation as R
from aruco_camera_localizer.geometric_functions import rvec_to_quat, transform_orientation_cam_to_world, transform_point_cam_to_world, \
transform_points_world_to_img, transform_point_world_to_cam
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
    line_height = 18
    x0 = 10
    y = 30

    def put_line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), font, 0.55, color, 2)
        y += line_height

    # Frame Index
    put_line(f"Frame: {frame_idx}", (200, 200, 200))

    # End Effector
    ee_euler = R.from_quat(ee_quat).as_euler('xyz', degrees=True)
    ee_euler = canonicalize_euler(ee_euler)
    put_line(f"EE xyz: ({1000*ee_pos[0]:.1f}, {1000*ee_pos[1]:.1f}, {1000*ee_pos[2]:.1f}) mm")
    put_line(f"EE rpy: ({ee_euler[0]: 5.1f}, {ee_euler[1]: 5.1f}, {ee_euler[2]: 5.1f}) deg")

    y += 10  # small gap

    # Generic Object Info
    for obj in object_data:
        name = obj["name"]
        pos = obj["position"]
        quat = obj["quaternion"]

        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        put_line(f"{name} xyz: ({1000*pos[0]:.1f}, {1000*pos[1]:.1f}, {1000*pos[2]:.1f}) mm", (0, 255, 0))
        put_line(f"{name} rpy: ({euler[0]: 5.1f}, {euler[1]: 5.1f}, {euler[2]: 5.1f}) deg", (0, 255, 0))
        y += 5

def draw_object_lines(frame, camera_matrix, cam_pos, cam_quat, identified_objects, nearest_pushers):
    color_map = {
        # Add object-specific colors here if needed
    }

    for obj in identified_objects:
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

        # Handle cases where some axes points might be behind the camera
        if len(axes_image) >= 1:
            o = axes_image[0]  # origin
            x = axes_image[1] if len(axes_image) > 1 else None  # X axis
            y = axes_image[2] if len(axes_image) > 2 else None  # Y axis  
            z = axes_image[3] if len(axes_image) > 3 else None  # Z axis
            
            if o and x:
                cv2.arrowedLine(frame, o, x, (0, 0, 255), 2, tipLength=0.3)  # X: Red
            if o and y:
                cv2.arrowedLine(frame, o, y, (0, 255, 0), 2, tipLength=0.3)  # Y: Green
            if o and z:
                cv2.arrowedLine(frame, o, z, (255, 0, 0), 2, tipLength=0.3)  # Z: Blue

        # Draw low-res Contour (only if contour data exists)
        if 'contour' in obj and obj['contour'] is not None:
            contour = obj["contour"]
            contour_xyz = contour["xyz"]
            contour_img = transform_points_world_to_img(contour_xyz, cam_pos, cam_quat, camera_matrix)
            contour_img = np.array(contour_img)
            contour_img.reshape((-1, 1, 2))
            contour_img = contour_img[::20]
            cv2.polylines(frame,[contour_img],False,color)

    return frame

def draw_color_dot_poses(frame, camera_matrix, cam_pos, cam_quat, detected_color_points, color_visualization):
    """Draw all detected color dot poses on the OpenCV frame"""
    if not detected_color_points:
        return frame
    
    # Draw dots for each detected color
    for color_name, world_points in detected_color_points.items():
        if not world_points:
            continue
            
        # Get visualization color for this color
        color_bgr = color_visualization.get(color_name, (255, 255, 255))
        
        # Transform world points to image coordinates
        image_points = transform_points_world_to_img(world_points, cam_pos, cam_quat, camera_matrix)
        
        for i, (world_pt, img_pt) in enumerate(zip(world_points, image_points)):
            if img_pt is None:
                continue
                
            # Draw colored dot as a filled circle
            cv2.circle(frame, img_pt, 8, color_bgr, -1)  # Colored filled circle
            cv2.circle(frame, img_pt, 10, (255, 255, 255), 2)  # White outline
            
            # Draw label with background
            label = f"{color_name.capitalize()}_{i}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background rectangle for text
            cv2.rectangle(frame, 
                         (img_pt[0] - w//2 - 5, img_pt[1] - h - 20 - 5), 
                         (img_pt[0] + w//2 + 5, img_pt[1] - 20 + 5), 
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, 
                       (img_pt[0] - w//2, img_pt[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw coordinates text
            coord_text = f"({world_pt[0]*1000:.1f}, {world_pt[1]*1000:.1f})"
            (cw, ch), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # Background for coordinates
            cv2.rectangle(frame, 
                         (img_pt[0] - cw//2 - 3, img_pt[1] + 5), 
                         (img_pt[0] + cw//2 + 3, img_pt[1] + ch + 10), 
                         (0, 0, 0), -1)
            
            # Draw coordinates
            cv2.putText(frame, coord_text, 
                       (img_pt[0] - cw//2, img_pt[1] + ch + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # Yellow text
    
    return frame