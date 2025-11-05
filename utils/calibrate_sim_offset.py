#!/usr/bin/env python3
"""
Utility to calibrate the sim_offset by comparing /objects_poses_sim and /objects_poses_real.

This script subscribes to both topics, compares the same objects, and calculates
the position offset needed to align simulation with real world.

Usage:
    python3 calibrate_sim_offset.py [--object-name OBJECT_NAME ...]
    
Examples:
    python3 calibrate_sim_offset.py                                    # Compare all common objects
    python3 calibrate_sim_offset.py --object-name line_brown           # Only compare line_brown
    python3 calibrate_sim_offset.py --object-name line_brown --object-name fork_orange  # Compare specific objects
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
import numpy as np
from collections import defaultdict
import sys
import argparse
import re
import os
from pathlib import Path

class SimOffsetCalibrator(Node):
    def __init__(self, filter_objects=None, current_offset=None):
        super().__init__('sim_offset_calibrator')
        
        # Storage for poses
        self.sim_poses = {}  # {object_name: [x, y, z]}
        self.real_poses = {}  # {object_name: [x, y, z]}
        self.has_computed = False
        self.filter_objects = set(filter_objects) if filter_objects else None
        self.current_offset = current_offset
        
        # Subscribers
        self.sim_sub = self.create_subscription(
            TFMessage,
            '/objects_poses_sim',
            self.sim_callback,
            10
        )
        self.real_sub = self.create_subscription(
            TFMessage,
            '/objects_poses_real',
            self.real_callback,
            10
        )
        
        if self.filter_objects:
            print(f"\nWaiting for objects: {', '.join(sorted(self.filter_objects))}...")
        else:
            print("\nWaiting for data from /objects_poses_sim and /objects_poses_real...")
    
    def sim_callback(self, msg: TFMessage):
        """Callback for simulation poses"""
        for transform in msg.transforms:
            name = transform.child_frame_id
            # Only store if no filter or name is in filter
            if self.filter_objects is None or name in self.filter_objects:
                self.sim_poses[name] = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
        self.check_and_compute()
    
    def real_callback(self, msg: TFMessage):
        """Callback for real world poses"""
        for transform in msg.transforms:
            name = transform.child_frame_id
            # Only store if no filter or name is in filter
            if self.filter_objects is None or name in self.filter_objects:
                self.real_poses[name] = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
        self.check_and_compute()
    
    def check_and_compute(self):
        """Check if we have data from both topics and compute once"""
        if self.has_computed:
            return
        
        if self.sim_poses and self.real_poses:
            common_objects = set(self.sim_poses.keys()) & set(self.real_poses.keys())
            
            # Apply filter if specified
            if self.filter_objects:
                common_objects = common_objects & self.filter_objects
            
            if common_objects:
                self.compute_offset(common_objects)
                self.has_computed = True
                raise KeyboardInterrupt  # Exit the spin loop
    
    def compute_offset(self, common_objects):
        """Compute and display the sim offset"""
        if not common_objects:
            return
        
        # Compute offsets for each common object (only the filtered ones)
        offsets = []
        print("\n" + "=" * 60)
        for obj_name in sorted(common_objects):
            if obj_name not in self.sim_poses or obj_name not in self.real_poses:
                continue
            sim_pos = self.sim_poses[obj_name]
            real_pos = self.real_poses[obj_name]
            offset = real_pos - sim_pos
            offsets.append(offset)
            
            print(f"{obj_name:15s} | Sim: [{sim_pos[0]:7.4f}, {sim_pos[1]:7.4f}, {sim_pos[2]:7.4f}]")
            print(f"{'':15s} | Real:[{real_pos[0]:7.4f}, {real_pos[1]:7.4f}, {real_pos[2]:7.4f}]")
            print(f"{'':15s} | Diff:[{offset[0]:7.4f}, {offset[1]:7.4f}, {offset[2]:7.4f}] m")
        
        # Compute average offset
        # offset = real - sim_published (difference from published sim to real)
        avg_offset = np.mean(offsets, axis=0)  # This is: real - sim_published
        
        # /objects_poses_sim = ground truth from Isaac Sim (no offset)
        # /objects_poses_real = detected_position + current_sim_offset (offset already applied)
        # 
        # We want: detected_position + new_offset = ground_truth (sim)
        # We know: real_published = detected_position + current_offset
        # So: detected_position = real_published - current_offset
        # Therefore: new_offset = sim_published - detected_position
        #                       = sim_published - (real_published - current_offset)
        #                       = sim_published - real_published + current_offset
        #                       = -avg_offset + current_offset
        
        if self.current_offset is not None:
            total_offset = self.current_offset - avg_offset
        else:
            # If no current offset, assume real = detected (no offset applied)
            total_offset = -avg_offset
        
        print("=" * 60)
        print(f"Difference (real - sim):        [{avg_offset[0]:.6f}, {avg_offset[1]:.6f}, {avg_offset[2]:.6f}]")
        if self.current_offset is not None:
            print(f"Current offset in code:         [{self.current_offset[0]:.6f}, {self.current_offset[1]:.6f}, {self.current_offset[2]:.6f}]")
            print(f"New offset (current - diff):    [{total_offset[0]:.6f}, {total_offset[1]:.6f}, {total_offset[2]:.6f}]")
        print(f"\n" + "=" * 60)
        print("REPLACE IN localizer_bridge.py:")
        print("=" * 60)
        print(f"self.sim_offset = np.array([{total_offset[0]:.6f}, {total_offset[1]:.6f}, {total_offset[2]:.6f}])")
        print("=" * 60 + "\n")


def read_current_offset():
    """Read current sim_offset from localizer_bridge.py"""
    # Find the localizer_bridge.py file
    script_dir = Path(__file__).parent
    bridge_file = script_dir.parent / "aruco_camera_localizer" / "localizer_bridge.py"
    
    if not bridge_file.exists():
        return None
    
    try:
        with open(bridge_file, 'r') as f:
            content = f.read()
        
        # Find sim_offset line
        pattern = r'self\.sim_offset\s*=\s*np\.array\(\[([^\]]+)\]\)'
        match = re.search(pattern, content)
        
        if match:
            # Parse the values
            values_str = match.group(1)
            values = [float(v.strip()) for v in values_str.split(',')]
            return np.array(values)
    except Exception as e:
        pass
    
    return None


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calibrate sim_offset by comparing sim and real poses')
    parser.add_argument('--object-name', action='append', dest='objects', help='Object names to compare (e.g., --object-name line_brown). Can be specified multiple times. If not specified, compares all common objects.')
    parsed_args = parser.parse_args()
    
    # Read current offset
    current_offset = read_current_offset()
    if current_offset is not None:
        print(f"Current sim_offset: [{current_offset[0]:.6f}, {current_offset[1]:.6f}, {current_offset[2]:.6f}]")
    
    rclpy.init(args=args)
    node = SimOffsetCalibrator(filter_objects=parsed_args.objects if parsed_args.objects else None, 
                               current_offset=current_offset)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

