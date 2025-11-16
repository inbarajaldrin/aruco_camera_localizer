import cv2
import numpy as np
import time
from aruco_camera_localizer.geometric_functions import rvec_to_quat, quat_to_rvec
from scipy.spatial.distance import cdist

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
        
        # Z value tracking for minimum z selection when robot is stationary
        self.z_measurements = []  # Store recent z measurements
        self.max_z_history = 10  # Maximum number of z values to track
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
            return  # Skip additional smoothing for first measurement after reset
        
        # Get current Z state without advancing the filter
        current_z = self.kf.statePost[2, 0]
        
        # When robot is not moving or moving slowly, track z measurements and use minimum
        if not robot_moving:
            # Add current z measurement to history
            self.z_measurements.append(tvec[2])
            # Keep only recent measurements
            if len(self.z_measurements) > self.max_z_history:
                self.z_measurements.pop(0)
            
            # Use minimum z from recent measurements
            if len(self.z_measurements) > 0:
                min_z = min(self.z_measurements)
                smoothed_tvec = tvec.copy()
                smoothed_tvec[2] = min_z
            else:
                smoothed_tvec = tvec.copy()
        else:
            # Robot is moving - clear z history and use normal smoothing
            self.z_measurements = []
            
            # Outlier rejection for Z: reject measurements that are too far from current state
            z_error = abs(tvec[2] - current_z) if current_z != 0 else 0
            outlier_threshold = 0.05  # Reject Z measurements more than 25mm from current state
            
            if z_error > outlier_threshold and current_z != 0:
                # Outlier detected - use current state instead of measurement
                smoothed_tvec = tvec.copy()
                smoothed_tvec[2] = current_z
            else:
                # Apply exponential smoothing to Z only
                # Very aggressive exponential smoothing - only trust 2% of new measurement
                alpha = 0.02  # Only trust 2% of new Z measurement (98% from current state)
                if current_z == 0:
                    # First measurement - use it directly
                    smoothed_z = tvec[2]
                else:
                    smoothed_z = alpha * tvec[2] + (1 - alpha) * current_z
                
                # Create smoothed tvec with exponential smoothing on Z
                smoothed_tvec = tvec.copy()
                smoothed_tvec[2] = smoothed_z
        
        quat = rvec_to_quat(rvec)
        measurement = np.vstack((smoothed_tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
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
        # Clear z measurements history
        self.z_measurements = []
        # Reset time tracking
        self.last_update_time = None
        # Mark that filter needs initialization with first measurement
        self.needs_initialization = True
    

class BlobKalman:
    def __init__(self, dt=1.0):
        # State: [x, y, z, vx, vy, vz]
        self.kf = cv2.KalmanFilter(6, 3)
        
        # Transition matrix (A)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = dt  # x += vx*dt, etc.

        # Measurement matrix (H)
        self.kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1

        # Process noise (Q): controls filter smoothness
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
        self.kf.processNoiseCov[3:, 3:] *= 10  # more uncertainty in velocity

        # Measurement noise (R): trust in measurement
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-3

        # Initial error covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Start at zero
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

        self.age = 0          # number of frames since creation
        self.time_since_update = 0

    def predict(self):
        prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return prediction[:3].flatten()

    def correct(self, pos):
        """Input is 3D position in world coordinates"""
        measurement = np.array(pos, dtype=np.float32).reshape(3, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0

    def get_position(self):
        return self.kf.statePost[:3].flatten()


class BlobTrackerManager:
    def __init__(self, dist_thresh=0.03, max_missed=5):
        self.trackers = []
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def update(self, detections):
        updated_trackers = []

        if len(self.trackers) == 0:
            # Initialize new trackers
            for det in detections:
                tracker = BlobKalman()
                tracker.correct(det)
                updated_trackers.append(tracker)
        else:
            predicted = np.array([trk.predict() for trk in self.trackers])
            dists = cdist(predicted, detections)

            assigned_dets = set()
            for i, trk in enumerate(self.trackers):
                min_j = np.argmin(dists[i])
                if dists[i, min_j] < self.dist_thresh and min_j not in assigned_dets:
                    trk.correct(detections[min_j])
                    assigned_dets.add(min_j)
                    updated_trackers.append(trk)
                else:
                    updated_trackers.append(trk)

            # Add unassigned detections as new trackers
            for j, det in enumerate(detections):
                if j not in assigned_dets:
                    trk = BlobKalman()
                    trk.correct(det)
                    updated_trackers.append(trk)

        # Remove stale trackers
        self.trackers = [trk for trk in updated_trackers if trk.time_since_update < self.max_missed]
        return self.trackers