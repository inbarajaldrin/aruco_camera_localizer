#!/usr/bin/env python3
"""
Read the pose of a named object from /objects_poses_real and /objects_poses_sim.

Both topics use tf2_msgs/TFMessage; each transform's child_frame_id is the object name.
Pose = translation (x,y,z) + rotation quaternion (x,y,z,w).

Usage:
    python3 read_object_pose.py [--object NAME] [--once]
    python3 read_object_pose.py --object fork_orange
    python3 read_object_pose.py --object fork_orange --once
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from tf2_msgs.msg import TFMessage
import argparse


def pose_from_transform(t):
    """Extract (x,y,z), (qx,qy,qz,qw) from a TransformStamped."""
    trans = t.transform.translation
    rot = t.transform.rotation
    return (
        (trans.x, trans.y, trans.z),
        (rot.x, rot.y, rot.z, rot.w),
    )


class PoseReader(Node):
    def __init__(self, object_name: str, once: bool):
        super().__init__("read_object_pose")
        self.object_name = object_name
        self.once = once
        self.real_pose = None  # (position, quat) or None
        self.sim_pose = None
        self.real_received = False
        self.sim_received = False

        self.sim_sub = self.create_subscription(
            TFMessage, "/objects_poses_sim", self.sim_callback, 10
        )
        self.real_sub = self.create_subscription(
            TFMessage, "/objects_poses_real", self.real_callback, 10
        )
        self.get_logger().info(
            f"Subscribed to /objects_poses_sim and /objects_poses_real for object '{object_name}'"
        )

    def sim_callback(self, msg: TFMessage):
        for t in msg.transforms:
            if t.child_frame_id == self.object_name:
                self.sim_pose = pose_from_transform(t)
                self.sim_received = True
                self.print_if_ready()
                return

    def real_callback(self, msg: TFMessage):
        for t in msg.transforms:
            if t.child_frame_id == self.object_name:
                self.real_pose = pose_from_transform(t)
                self.real_received = True
                self.print_if_ready()
                return

    def print_if_ready(self):
        if not (self.real_received and self.sim_received):
            return
        self.print_poses()
        if self.once:
            raise KeyboardInterrupt

    def print_poses(self):
        print("\n" + "=" * 60)
        print(f"Object: {self.object_name}")
        print("=" * 60)
        if self.real_pose is not None:
            pos, quat = self.real_pose
            print("/objects_poses_real:")
            print(f"  position:  x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
            print(f"  quaternion: x={quat[0]:.4f}, y={quat[1]:.4f}, z={quat[2]:.4f}, w={quat[3]:.4f}")
        else:
            print("/objects_poses_real: (no data yet)")
        if self.sim_pose is not None:
            pos, quat = self.sim_pose
            print("/objects_poses_sim:")
            print(f"  position:  x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
            print(f"  quaternion: x={quat[0]:.4f}, y={quat[1]:.4f}, z={quat[2]:.4f}, w={quat[3]:.4f}")
        else:
            print("/objects_poses_sim: (no data yet)")
        print("=" * 60 + "\n")


def main(args=None):
    parser = argparse.ArgumentParser(description="Read object pose from objects_poses_real and objects_poses_sim")
    parser.add_argument("--object", "-o", type=str, default="fork_orange", help="Object name (default: fork_orange)")
    parser.add_argument("--once", action="store_true", help="Print once when both topics have data, then exit")
    parsed = parser.parse_args()

    rclpy.init(args=args)
    node = PoseReader(object_name=parsed.object, once=parsed.once)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
