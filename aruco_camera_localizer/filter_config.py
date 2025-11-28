"""
Filter configuration module for ArUco camera localizer.

This module provides a centralized configuration system for all filtering mechanisms
used in the localization pipeline. Each filter can be individually enabled or disabled,
and all threshold/parameter values are configurable.
"""


class FilterConfig:
    """
    Configuration class for all filtering mechanisms in the ArUco localizer.
    
    All filters are enabled by default to maintain current behavior.
    Individual filters can be disabled by setting their enable flag to False.
    """
    
    def __init__(self):
        # =====================================================================
        # FILTER ENABLE/DISABLE FLAGS
        # =====================================================================
        
        # 1. Kalman Filtering - Temporal smoothing of position and orientation
        self.enable_kalman_filter = True
        
        # 2. Z-Range Validation - Rejects markers outside valid depth range
        self.enable_z_range_validation = True
        
        # 3. Mahalanobis Distance Outlier Rejection - Statistical outlier detection
        self.enable_mahalanobis_outlier_rejection = True
        
        # 4. Simple Distance/Rotation Outlier Rejection - Fallback outlier detection
        self.enable_simple_outlier_rejection = True
        
        # 5. Quality Threshold Filtering - Rejects low-quality detections
        self.enable_quality_threshold = True
        
        # 6. Movement Validation - Validates object movement between frames
        self.enable_movement_validation = True
        
        # 7. SLERP Smoothing - Quaternion smoothing for orientation
        self.enable_slerp_smoothing = True
        
        # 8. Marker Stability Confirmation Smoothing - Temporal smoothing of confirmed pose
        self.enable_marker_stability_smoothing = True
        
        # 9. Ghost Tracking - Pose holding when markers not detected
        self.enable_ghost_tracking = True
        
        # 10. Marker Confirmation Reset - Resets confirmation after missed frames
        self.enable_marker_confirmation_reset = True
        
        # =====================================================================
        # FILTER PARAMETERS
        # =====================================================================
        
        # Z-Range Validation Parameters
        self.z_range_min = 0.05  # meters - minimum depth
        self.z_range_max = 2.0   # meters - maximum depth
        
        # Quality Threshold Parameters
        self.min_quality_threshold = 0.3  # Reject if quality < 0.3 (~3.5 pixel RMS error)
        self.max_acceptable_error = 5.0   # pixels - for quality calculation
        self.quality_history_size = 20     # Number of quality values to track
        
        # Mahalanobis Distance Outlier Rejection Parameters
        self.mahalanobis_base_threshold_moving = 4.5      # sigma for moving robot
        self.mahalanobis_base_threshold_stationary = 5.0  # sigma for stationary robot
        self.mahalanobis_rot_threshold_moving = 0.50       # radians for moving
        self.mahalanobis_rot_threshold_stationary = 0.40   # radians for stationary
        self.mahalanobis_variance_factor_cap = 3.0        # Cap variance factor
        self.mahalanobis_rot_factor_cap = 0.8             # Cap rotation factor
        self.mahalanobis_measurement_history_size = 20    # Number of measurements to track
        self.mahalanobis_rejection_count_threshold = 20    # Rejections before clearing history
        
        # Simple Outlier Rejection Parameters
        self.simple_outlier_movement_threshold_moving = 0.150    # meters - moving
        self.simple_outlier_movement_threshold_stationary = 0.100  # meters - stationary
        self.simple_outlier_rotation_threshold_moving = 0.50    # radians - moving
        self.simple_outlier_rotation_threshold_stationary = 0.40  # radians - stationary
        self.simple_outlier_rejection_count_threshold = 20     # Rejections before clearing
        
        # Movement Validation Parameters
        self.movement_max_velocity_moving = 2.0           # m/s - maximum linear velocity when moving
        self.movement_max_angular_velocity_moving = 5.0   # rad/s - maximum angular velocity when moving
        self.movement_max_velocity_stationary = 0.5       # m/s - maximum linear velocity when stationary
        self.movement_max_angular_velocity_stationary = 2.0  # rad/s - maximum angular velocity when stationary
        self.movement_max_absolute_movement = 0.3         # meters - reject if moved more than this
        self.movement_max_absolute_rotation = 1.0         # radians (~57 degrees)
        self.movement_fps = 30.0                         # Frame rate for velocity calculations
        
        # SLERP Smoothing Parameters
        self.slerp_blend_factor = 0.8  # 0.0 = use previous, 1.0 = use current (0.8 = 80% current, 20% previous)
        
        # Marker Stability Confirmation Smoothing Parameters
        self.stability_smoothing_alpha_moving = 0.3     # 30% new, 70% old when moving
        self.stability_smoothing_alpha_stationary = 0.15  # 15% new, 85% old when stationary
        
        # Ghost Tracking Parameters
        self.ghost_tracking_timeout = 90  # frames - ~3 seconds at 30fps
        
        # Marker Confirmation Reset Parameters
        self.marker_confirmation_missed_frames_threshold = 3  # Reset after N consecutive missed frames
        
        # Pose Consistency Check Parameters (for marker switching)
        self.pose_consistency_max_position_diff_moving = 0.15    # meters - moving
        self.pose_consistency_max_position_diff_stationary = 0.08  # meters - stationary
        self.pose_consistency_max_rotation_diff_moving = 0.5     # radians - moving
        self.pose_consistency_max_rotation_diff_stationary = 0.3  # radians - stationary
        self.pose_consistency_clear_timeout = 60  # frames - clear active marker after N frames
        
        # Robot Movement Detection Parameters
        self.robot_slow_movement_threshold = 0.01  # meters - 10mm threshold for slow movement
        self.stationary_settle_time = 30          # frames - time to wait after movement stops
    
    def update_from_args(self, args):
        """
        Update filter configuration from command-line arguments.
        
        Args:
            args: argparse.Namespace with filter flags (e.g., filter_kalman, filter_z_range, etc.)
        """
        if hasattr(args, 'filter_kalman'):
            self.enable_kalman_filter = args.filter_kalman
        if hasattr(args, 'filter_z_range'):
            self.enable_z_range_validation = args.filter_z_range
        if hasattr(args, 'filter_mahalanobis'):
            self.enable_mahalanobis_outlier_rejection = args.filter_mahalanobis
        if hasattr(args, 'filter_simple_outlier'):
            self.enable_simple_outlier_rejection = args.filter_simple_outlier
        if hasattr(args, 'filter_quality'):
            self.enable_quality_threshold = args.filter_quality
        if hasattr(args, 'filter_movement'):
            self.enable_movement_validation = args.filter_movement
        if hasattr(args, 'filter_slerp'):
            self.enable_slerp_smoothing = args.filter_slerp
        if hasattr(args, 'filter_stability_smoothing'):
            self.enable_marker_stability_smoothing = args.filter_stability_smoothing
        if hasattr(args, 'filter_ghost_tracking'):
            self.enable_ghost_tracking = args.filter_ghost_tracking
        if hasattr(args, 'filter_confirmation_reset'):
            self.enable_marker_confirmation_reset = args.filter_confirmation_reset

