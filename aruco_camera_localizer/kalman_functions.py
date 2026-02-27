import cv2
import numpy as np
import time
from aruco_camera_localizer.geometric_functions import rvec_to_quat, quat_to_rvec
from scipy.spatial.distance import cdist


class QuaternionKalman:
    """13-state Kalman filter with acceleration, dynamic dt, anisotropic noise.

    State vector: [x, y, z, qx, qy, qz, qw, vx, vy, vz, ax, ay, az]
    Measurement:  [x, y, z, qx, qy, qz, qw]

    Noise parameters are loaded from robot_config.yaml via config_loader.
    Pass q_params/r_params dicts to override, or omit to use defaults.
    """

    def __init__(self, q_params=None, r_params=None):
        # 13 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz, ax, ay, az]
        self.kf = cv2.KalmanFilter(13, 7)

        # Track time for dynamic dt calculation
        self.last_update_time = None
        self.default_dt = 1.0 / 30.0  # Default 30 fps

        # Initialize transition matrix with default dt
        self._update_transition_matrix(self.default_dt)

        # H: Measurement matrix (7x13)
        self.kf.measurementMatrix = np.zeros((7, 13), dtype=np.float32)
        self.kf.measurementMatrix[0:7, 0:7] = np.eye(7)

        # --- Process noise (Q) from config ---
        q = q_params or {}
        q_pos_xy = q.get('q_pos_xy', 1e-8)
        q_pos_z = q.get('q_pos_z', 1e-4)
        q_quat = q.get('q_quat', 1e-2)
        q_vel = q.get('q_velocity', 1e-2)
        q_acc = q.get('q_acceleration', 1e-1)
        self.q_multiplier_moving = q.get('q_multiplier_moving', 10.0)
        self.q_multiplier_static = q.get('q_multiplier_static', 0.1)

        self.base_process_noise = np.eye(13, dtype=np.float32) * 1e-5
        self.base_process_noise[0, 0] = q_pos_xy
        self.base_process_noise[1, 1] = q_pos_xy
        self.base_process_noise[2, 2] = q_pos_z
        for i in range(3, 7):
            self.base_process_noise[i, i] = q_quat
        for i in range(7, 10):
            self.base_process_noise[i, i] = q_vel
        for i in range(10, 13):
            self.base_process_noise[i, i] = q_acc

        self.kf.processNoiseCov = self.base_process_noise.copy()
        self.robot_moving = True

        # --- Measurement noise (R) from config ---
        r = r_params or {}
        r_pos_xy = r.get('r_pos_xy', 1e-4)
        r_pos_z = r.get('r_pos_z', 5e-1)
        r_quat = r.get('r_quat', 1e-2)

        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.measurementNoiseCov[0, 0] = r_pos_xy
        self.kf.measurementNoiseCov[1, 1] = r_pos_xy
        self.kf.measurementNoiseCov[2, 2] = r_pos_z
        for i in range(3, 7):
            self.kf.measurementNoiseCov[i, i] = r_quat

        self.kf.errorCovPost = np.eye(13, dtype=np.float32)

        # Initial state — identity quaternion, everything else zero
        self.kf.statePost = np.zeros((13, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)

        self.needs_initialization = True

    def _update_transition_matrix(self, dt):
        """Update transition matrix for constant-acceleration model."""
        # State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, ax, ay, az]
        self.kf.transitionMatrix = np.eye(13, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 7] = dt               # x += vx*dt
            self.kf.transitionMatrix[i, i + 10] = 0.5 * dt * dt   # x += 0.5*ax*dt^2
        for i in range(3):
            self.kf.transitionMatrix[i + 7, i + 10] = dt          # vx += ax*dt

    def update_dt(self, dt):
        """Public dt setter for transition matrix."""
        self._update_transition_matrix(dt)

    def correct(self, tvec, rvec, robot_moving=True):
        # Dynamic dt from wall-clock timestamps, clamped to [1ms, 100ms]
        current_time = time.time()
        if self.last_update_time is not None:
            dt = max(0.001, min(current_time - self.last_update_time, 0.1))
            self.update_dt(dt)
        self.last_update_time = current_time

        # Adaptive process noise — respond to robot motion state
        if robot_moving != self.robot_moving:
            self.robot_moving = robot_moving
            if robot_moving:
                self.kf.processNoiseCov = self.base_process_noise * self.q_multiplier_moving
            else:
                self.kf.processNoiseCov = self.base_process_noise * self.q_multiplier_static

        # Store raw measurements
        self.last_measurement_tvec = tvec.copy()
        self.last_measurement_rvec = rvec.copy()

        quat = rvec_to_quat(rvec)

        # First measurement after creation/reset — initialize state directly
        if self.needs_initialization:
            self.kf.statePost[0:3] = tvec.reshape(3, 1).astype(np.float32)
            self.kf.statePost[3:7] = np.array(quat).reshape(4, 1).astype(np.float32)
            self.kf.statePost[7:10] = np.zeros((3, 1), dtype=np.float32)
            self.kf.statePost[10:13] = np.zeros((3, 1), dtype=np.float32)
            quat_norm = np.linalg.norm(self.kf.statePost[3:7])
            if quat_norm > 1e-8:
                self.kf.statePost[3:7] /= quat_norm
            self.kf.errorCovPost[0:7, 0:7] = self.kf.measurementNoiseCov
            self.kf.errorCovPost[7:10, 7:10] = np.eye(3, dtype=np.float32) * 1e-2
            self.kf.errorCovPost[10:13, 10:13] = np.eye(3, dtype=np.float32) * 1e-1
            self.needs_initialization = False
            measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
            self.kf.correct(measurement)
            return

        measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
        self.kf.correct(measurement)

    def predict(self, dt=None):
        if dt is not None:
            self.update_dt(dt)
        pred = self.kf.predict()
        pred_tvec = pred[0:3].flatten()
        pred_quat = pred[3:7].flatten()
        pred_quat /= np.linalg.norm(pred_quat)
        pred_rvec = quat_to_rvec(pred_quat).flatten()
        return pred_tvec, pred_rvec

    def get_velocity(self):
        return self.kf.statePost[7:10].flatten()

    def get_acceleration(self):
        return self.kf.statePost[10:13].flatten()

    def get_state(self):
        """Current smoothed state (after correction, no prediction step)."""
        state_tvec = self.kf.statePost[0:3].flatten()
        state_quat = self.kf.statePost[3:7].flatten()
        state_quat /= np.linalg.norm(state_quat)
        state_rvec = quat_to_rvec(state_quat).flatten()
        return state_tvec, state_rvec

    def get_raw_measurement(self):
        """Last raw measurement before Kalman smoothing."""
        if hasattr(self, 'last_measurement_tvec') and hasattr(self, 'last_measurement_rvec'):
            return self.last_measurement_tvec, self.last_measurement_rvec
        return self.predict()

    def reset(self):
        """Reset filter to initial state."""
        self.kf.statePost = np.zeros((13, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(13, dtype=np.float32)
        self.last_update_time = None
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