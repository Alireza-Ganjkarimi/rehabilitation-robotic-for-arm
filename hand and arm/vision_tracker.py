import cv2
import mediapipe as mp
import numpy as np
import math
import time

class KalmanKinematicFilter:
    def __init__(self, num_vars=7, process_noise=1e-3, measurement_noise=1e-1):
        """
        Initializes the Kalman Filter for tracking position and estimating velocity.
        num_vars: Number of kinematic variables to track (e.g., 7 for this use case).
        process_noise: Trust in the mathematical model (lower = smoother, but more lag).
        measurement_noise: Trust in the sensor measurement (higher = smoother, ignores jitter).
        """
        self.num_vars = num_vars
        self.dt = 0
        self.t_prev = None

        # State vector: [x1, x2, ..., x7, v1, v2, ..., v7].T
        # Top half represents positions/angles, bottom half represents velocities
        self.state = np.zeros((2 * num_vars, 1), dtype=np.float32)

        # State covariance matrix (P)
        self.P = np.eye(2 * num_vars, dtype=np.float32)

        # Process noise covariance (Q)
        self.Q = np.eye(2 * num_vars, dtype=np.float32) * process_noise

        # Measurement noise covariance (R)
        self.R = np.eye(num_vars, dtype=np.float32) * measurement_noise

        # Measurement matrix (H) - we only observe positions, not velocities
        self.H = np.zeros((num_vars, 2 * num_vars), dtype=np.float32)
        for i in range(num_vars):
            self.H[i, i] = 1.0

    def __call__(self, measurement, t=None):
        """
        Applies the Kalman Filter to the current measurement.
        Returns the smoothed positions and the estimated velocities.
        """
        if t is None:
            t = time.time()
            
        if self.t_prev is None:
            self.t_prev = t
            # Initialize state with the first measurement
            self.state[:self.num_vars, 0] = measurement
            return measurement, np.zeros(self.num_vars)
            
        self.dt = t - self.t_prev
        if self.dt <= 0:
            self.dt = 1e-5  # Prevent zero time step division/errors
            
        # Update State Transition Matrix (A) with current dynamic time step (dt)
        A = np.eye(2 * self.num_vars, dtype=np.float32)
        for i in range(self.num_vars):
            A[i, i + self.num_vars] = self.dt

        # --- 1. PREDICT STEP ---
        # Predict next state: X = A * X
        self.state = np.dot(A, self.state)
        # Predict state covariance: P = A * P * A^T + Q
        self.P = np.dot(np.dot(A, self.P), A.T) + self.Q

        # --- 2. UPDATE STEP ---
        Z = np.array(measurement, dtype=np.float32).reshape(-1, 1)

        # Innovation (Residual): Y = Z - H * X
        Y = Z - np.dot(self.H, self.state)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman Gain: K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state with measurement: X = X + K * Y
        self.state = self.state + np.dot(K, Y)

        # Update state covariance: P = (I - K * H) * P
        I = np.eye(2 * self.num_vars, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        self.t_prev = t

        # Extract smoothed positions and estimated velocities
        smoothed_positions = self.state[:self.num_vars, 0]
        estimated_velocities = self.state[self.num_vars:, 0]

        return smoothed_positions, estimated_velocities


class VisionTracker:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Tracking 7 kinematic variables: 5 fingers, 1 elbow, 1 wrist
        self.kalman_filter = KalmanKinematicFilter(
            num_vars=7, 
            process_noise=process_noise, 
            measurement_noise=measurement_noise
        )
        self.filtered_kinematics = np.zeros(7)
        self.estimated_velocities = np.zeros(7)

    def calculate_3d_angle(self, a, b, c):
        """Calculates the 3D angle formed by three spatial landmarks."""
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 180.0
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        current_kinematics = None
        
        if hands_results.multi_hand_landmarks and pose_results.pose_landmarks:
            # Isolate target hand; accounts for horizontal mirroring ('Right' label maps to physical left hand)
            target_hand_landmark = None
            if hands_results.multi_handedness:
                for idx, handedness in enumerate(hands_results.multi_handedness):
                    if handedness.classification[0].label == 'Right':
                        target_hand_landmark = hands_results.multi_hand_landmarks[idx]
                        break
            
            if target_hand_landmark:
                self.mp_drawing.draw_landmarks(frame, target_hand_landmark, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract 3D pose coordinates for the left arm assembly
                pose_lms = pose_results.pose_landmarks.landmark
                shoulder = pose_lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                elbow = pose_lms[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                wrist = pose_lms[self.mp_pose.PoseLandmark.LEFT_WRIST]
                index = pose_lms[self.mp_pose.PoseLandmark.LEFT_INDEX]
                
                # Map arm landmarks to 2D image plane for visualization
                h, w, _ = frame.shape
                arm_pts = [shoulder, elbow, wrist, index]
                pixel_pts = []
                for pt in arm_pts:
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    pixel_pts.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                
                # Render skeletal connections
                cv2.line(frame, pixel_pts[0], pixel_pts[1], (255, 0, 0), 2)
                cv2.line(frame, pixel_pts[1], pixel_pts[2], (255, 0, 0), 2)
                cv2.line(frame, pixel_pts[2], pixel_pts[3], (255, 0, 0), 2)

                current_kinematics = np.zeros(7)
                
                # --- 1. Finger Kinematics ---
                landmarks = target_hand_landmark.landmark
                dx = landmarks[9].x - landmarks[0].x
                dy = landmarks[9].y - landmarks[0].y
                dz = landmarks[9].z - landmarks[0].z
                palm_size = max(math.sqrt(dx**2 + dy**2 + dz**2), 1e-5)
                
                tips = [4, 8, 12, 16, 20]  
                bases = [1, 5, 9, 13, 17]  
                
                for i in range(5):
                    tx, ty, tz = landmarks[tips[i]].x, landmarks[tips[i]].y, landmarks[tips[i]].z
                    bx, by, bz = landmarks[bases[i]].x, landmarks[bases[i]].y, landmarks[bases[i]].z
                    dist = math.sqrt((tx-bx)**2 + (ty-by)**2 + (tz-bz)**2)
                    ratio = dist / palm_size
                    max_r, min_r = 1.3, 0.4
                    flex = 1.0 - ((ratio - min_r) / (max_r - min_r))
                    current_kinematics[i] = np.clip(flex, 0.0, 1.0)
                
                # --- 2. Elbow and Wrist Kinematics ---
                # Map elbow flexion: 180 degrees (fully extended) to 30 degrees (fully flexed)
                elbow_angle = self.calculate_3d_angle(shoulder, elbow, wrist)
                elbow_flex = 1.0 - np.clip((elbow_angle - 30) / (180 - 30), 0.0, 1.0)
                current_kinematics[5] = elbow_flex
                
                # Calculate and normalize wrist flexion
                wrist_angle = self.calculate_3d_angle(elbow, wrist, index)
                wrist_flex_corrected = np.clip((170.0 - wrist_angle) / (170.0 - 110.0), 0.0, 1.0)
                current_kinematics[6] = wrist_flex_corrected
                
                # Apply Kalman filter to smooth kinematics and estimate velocity
                filtered_positions, estimated_velocities = self.kalman_filter(current_kinematics)
                self.filtered_kinematics = filtered_positions
                self.estimated_velocities = estimated_velocities

        # Returning frame, filtered positions, and velocities
        if current_kinematics is not None:
            return frame, self.filtered_kinematics.tolist(), self.estimated_velocities.tolist()
        else:
            return frame, None, None