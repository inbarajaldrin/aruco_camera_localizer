import cv2
import numpy as np
import time
from aruco_camera_localizer.geometric_functions import rvec_to_quat, quat_to_rvec

class QuaternionKalman:
    def __init__(self):
        # 13 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz, ax, ay, az]
        self.kf = cv2.KalmanFilter(13, 7)

        # Track time for dynamic dt calculation
        self.last_update_time = None
        self.default_dt = 1.0 / 30.0  # Default 30 fps
        
        # Initialize transition matrix with default dt
        self._update_transition_matrix(self.default_dt)

        # H: Measurement matrix (7x13) - only measures position and orientation, not velocity/acceleration
        self.kf.measurementMatrix = np.zeros((7, 13), dtype=np.float32)
        self.kf.measurementMatrix[0:7, 0:7] = np.eye(7)

        # Process noise covariance (Q). Lower Values = More Inertia
        # Will be adjusted adaptively based on robot_moving state
        self.base_process_noise = np.eye(13, dtype=np.float32) * 1e-5
        # X and Y are stable
        self.base_process_noise[0, 0] = 1e-8  # X position
        self.base_process_noise[1, 1] = 1e-8  # Y position
        # Z is much more uncertain in process model
        self.base_process_noise[2, 2] = 1e-4  # Z position - much higher uncertainty
        for i in range(3, 7):  # quaternion x, y, z, w - orientation uncertainty
            self.base_process_noise[i, i] = 1e-2
        for i in range(7, 10):  # vx, vy, vz - velocity uncertainty
            self.base_process_noise[i, i] = 1e-2
        for i in range(10, 13):  # ax, ay, az - acceleration uncertainty
            self.base_process_noise[i, i] = 1e-1  # Acceleration is more uncertain
        
        # Initialize with base process noise
        self.kf.processNoiseCov = self.base_process_noise.copy()
        self.robot_moving = True  # Track robot state for adaptive noise

        # Measurement noise covariance (R). Lower Values = More Trust = You have good cameras
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        # Anisotropic noise: X,Y are stable but Z (depth) is very noisy in monocular vision
        self.kf.measurementNoiseCov[0, 0] = 1e-4  # X position - stable
        self.kf.measurementNoiseCov[1, 1] = 1e-4  # Y position - stable  
        self.kf.measurementNoiseCov[2, 2] = 5e-1  # Z position - 5000x noisier (EXTREMELY aggressive)
        for i in range(3, 7):  # quaternion - orientation measurements are less reliable
            self.kf.measurementNoiseCov[i, i] = 1e-2
        
        self.kf.errorCovPost = np.eye(13, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((13, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion
        
        self.needs_initialization = True  # Flag to track if filter needs initialization after reset
    
    def _update_transition_matrix(self, dt):
        """Update transition matrix with actual dt value"""
        # A: Transition matrix (13x13)
        # State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, ax, ay, az]
        self.kf.transitionMatrix = np.eye(13, dtype=np.float32)
        
        # Position updates: x += vx*dt + 0.5*ax*dt^2
        for i in range(3):
            self.kf.transitionMatrix[i, i+7] = dt  # velocity contribution
            self.kf.transitionMatrix[i, i+10] = 0.5 * dt * dt  # acceleration contribution
        
        # Velocity updates: vx += ax*dt
        for i in range(3):
            self.kf.transitionMatrix[i+7, i+10] = dt  # acceleration contribution to velocity
    
    def update_dt(self, dt):
        """Update transition matrix with new dt value"""
        self._update_transition_matrix(dt)

    def correct(self, tvec, rvec, robot_moving=True):
        # Calculate actual dt from timestamps
        current_time = time.time()
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
            # Clamp dt to reasonable range (avoid huge jumps)
            dt = max(0.001, min(dt, 0.1))  # Between 1ms and 100ms
            self.update_dt(dt)
        else:
            dt = self.default_dt
        self.last_update_time = current_time
        
        # Update adaptive process noise based on robot_moving state
        if robot_moving != self.robot_moving:
            self.robot_moving = robot_moving
            if robot_moving:
                # Higher process noise when moving (more uncertainty)
                self.kf.processNoiseCov = self.base_process_noise * 10.0
            else:
                # Lower process noise when stationary (objects don't move)
                self.kf.processNoiseCov = self.base_process_noise * 0.1
        
        # Store raw measurements for get_raw_measurement()
        self.last_measurement_tvec = tvec.copy()
        self.last_measurement_rvec = rvec.copy()
        
        # If filter needs initialization after reset, initialize state directly with first measurement
        if self.needs_initialization:
            quat = rvec_to_quat(rvec)
            # Initialize state directly with the measurement (no smoothing for first measurement)
            # This ensures accurate pose immediately after reset
            self.kf.statePost[0:3] = tvec.reshape(3, 1).astype(np.float32)
            self.kf.statePost[3:7] = np.array(quat).reshape(4, 1).astype(np.float32)
            self.kf.statePost[7:10] = np.zeros((3, 1), dtype=np.float32)  # Initialize velocities to zero
            self.kf.statePost[10:13] = np.zeros((3, 1), dtype=np.float32)  # Initialize accelerations to zero
            # Normalize quaternion
            quat_norm = np.linalg.norm(self.kf.statePost[3:7])
            if quat_norm > 1e-8:
                self.kf.statePost[3:7] /= quat_norm
            # Update error covariance to reflect that we have a measurement
            # Use measurement directly, so set covariance to measurement noise
            # errorCovPost is 13x13, measurementNoiseCov is 7x7
            # Set the measured states (0:7) to measurement noise, velocities/accelerations (7:13) to default
            self.kf.errorCovPost[0:7, 0:7] = self.kf.measurementNoiseCov
            self.kf.errorCovPost[7:10, 7:10] = np.eye(3, dtype=np.float32) * 1e-2  # Velocity uncertainty
            self.kf.errorCovPost[10:13, 10:13] = np.eye(3, dtype=np.float32) * 1e-1  # Acceleration uncertainty
            self.needs_initialization = False
            # Still call kf.correct() to properly update the filter
            measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
            self.kf.correct(measurement)
            return  # Skip additional processing for first measurement after reset
        
        quat = rvec_to_quat(rvec)
        measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
        self.kf.correct(measurement)

    def predict(self, dt=None):
        """Predict next state. If dt is provided, use it; otherwise use last calculated dt"""
        if dt is not None:
            self.update_dt(dt)
        pred = self.kf.predict()
        pred_tvec = pred[0:3].flatten()
        pred_quat = pred[3:7].flatten()
        # Normalize quaternion to prevent drift
        pred_quat /= np.linalg.norm(pred_quat)
        pred_rvec = quat_to_rvec(pred_quat).flatten()
        return pred_tvec, pred_rvec
    
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.kf.statePost[7:10].flatten()
    
    def get_acceleration(self):
        """Get current acceleration estimate"""
        return self.kf.statePost[10:13].flatten()
    
    def get_raw_measurement(self):
        """Get the last raw measurement without Kalman prediction"""
        # Return the last measurement that was used for correction
        if hasattr(self, 'last_measurement_tvec') and hasattr(self, 'last_measurement_rvec'):
            return self.last_measurement_tvec, self.last_measurement_rvec
        else:
            # Fallback to prediction if no raw measurement available
            return self.predict()
    
    def reset(self):
        """Reset the Kalman filter to initial state"""
        self.kf.statePost = np.zeros((13, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion
        self.kf.errorCovPost = np.eye(13, dtype=np.float32)
        # Reset time tracking
        self.last_update_time = None
        # Mark that filter needs initialization with first measurement
        self.needs_initialization = True