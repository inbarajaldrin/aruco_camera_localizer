class FilterConfig:
    """Minimal configuration for the ArUco localizer pipeline."""

    def __init__(self):
        # Z-range validation (rejects markers behind camera or too far)
        self.z_range_min = 0.05   # meters
        self.z_range_max = 2.0    # meters

        # Optional EMA smoothing on output world-frame poses
        self.enable_ema_smoothing = True
        self.ema_alpha = 0.5  # 0.0 = full smoothing, 1.0 = no smoothing

        # Multi-marker selection for objects:
        #   'single'   — pick one marker per object (closest_z)
        #   'combined'  — combine all visible markers into one solvePnP (like boards)
        self.object_pose_mode = 'combined'

        # Board: constrain to yaw-only (flat on table, no roll/pitch)
        self.board_yaw_only = True

        # Board: snap Z to TABLE_Z + object_height/2
        self.board_snap_z = True

        # Board sticky marker: stay on one marker unless reproj error exceeds threshold
        self.board_reproj_threshold = 1.0  # pixels — switch marker if error exceeds this

        # Active marker timeout (clear marker if object not seen for N frames)
        self.active_marker_timeout = 60  # frames

        # Fold symmetry snapping for objects on table
        #   Blocks: snap constrained axes to block_snap_angle (90° for rectangular faces)
        #   Pegs: snap constrained axes to 360/fold per axis using symmetry data
        self.enable_fold_snap = True
        self.fold_snap_subtypes = ['block', 'peg']  # which subtypes get snapping
        self.block_snap_angle = 90.0  # degrees — blocks always 90° (rectangular faces)

        # Euler angle convention: 'intrinsic' (body-frame XYZ)
        #                         'extrinsic' (fixed-frame XYZ)
        self.euler_convention = 'intrinsic'
