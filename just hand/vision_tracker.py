import cv2
import mediapipe as mp
import numpy as np
import math
import time

class KalmanFilter:
    def __init__(self, num_vars=5, process_noise=1e-2, measurement_noise=1e-1):
        """
        Initializes the Kalman Filter for tracking multiple variables and their velocities.
        num_vars: Number of variables to track (5 for fingers).
        process_noise: Q matrix multiplier (lower = smoother, but lags).
        measurement_noise: R matrix multiplier (higher = trust model more than measurements).
        """
        self.num_vars = num_vars
        self.is_initialized = False
        self.t_prev = None
        
        # State vector [positions (flexions), velocities]
        self.x = np.zeros(num_vars * 2)
        
        # Error Covariance Matrix
        self.P = np.eye(num_vars * 2)
        
        # Measurement Matrix (We only measure positions/flexions, not velocities directly)
        self.H = np.zeros((num_vars, num_vars * 2))
        for i in range(num_vars):
            self.H[i, i] = 1.0
            
        # Process Noise Covariance Matrix
        self.Q = np.eye(num_vars * 2) * process_noise
        
        # Measurement Noise Covariance Matrix
        self.R = np.eye(num_vars) * measurement_noise

    def __call__(self, measurement, t=None):
        if t is None:
            t = time.time()
            
        if not self.is_initialized:
            # Initialize positions with the first measurement, velocities stay 0
            self.x[:self.num_vars] = measurement
            self.t_prev = t
            self.is_initialized = True
            return self.x[:self.num_vars], self.x[self.num_vars:]
            
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-5  # Prevent division/multiplication by zero issues
        self.t_prev = t

        # 1. PREDICT STEP
        # State Transition Matrix (A)
        # Position = Position + Velocity * dt
        # Velocity = Velocity (Constant velocity model)
        A = np.eye(self.num_vars * 2)
        for i in range(self.num_vars):
            A[i, i + self.num_vars] = dt
            
        # Predict the next state
        self.x = np.dot(A, self.x)
        # Predict the next error covariance
        self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q

        # 2. UPDATE STEP
        # Calculate Kalman Gain (K)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        
        # Calculate measurement residual (y)
        y = measurement - np.dot(self.H, self.x)
        
        # Update the state estimate
        self.x = self.x + np.dot(K, y)
        
        # Update the error covariance
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

        # Return smoothed flexions and estimated velocities
        return self.x[:self.num_vars], self.x[self.num_vars:]


class VisionTracker:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize the Kalman filter
        self.kalman_filter = KalmanFilter(
            num_vars=5, 
            process_noise=process_noise, 
            measurement_noise=measurement_noise
        )
        
        self.filtered_flexion = np.zeros(5)
        self.estimated_velocity = np.zeros(5)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = self.hands.process(rgb_frame)
        
        current_flexion = None
        
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                
                # Calculate reference palm size
                dx = landmarks[9].x - landmarks[0].x
                dy = landmarks[9].y - landmarks[0].y
                dz = landmarks[9].z - landmarks[0].z
                palm_size = math.sqrt(dx**2 + dy**2 + dz**2)
                palm_size = max(palm_size, 1e-5)
                
                tips = [4, 8, 12, 16, 20]  
                bases = [1, 5, 9, 13, 17]  
                
                current_flexion = np.zeros(5)
                
                for i in range(5):
                    # Calculate distance from fingertip to its base
                    tx, ty, tz = landmarks[tips[i]].x, landmarks[tips[i]].y, landmarks[tips[i]].z
                    bx, by, bz = landmarks[bases[i]].x, landmarks[bases[i]].y, landmarks[bases[i]].z
                    dist = math.sqrt((tx-bx)**2 + (ty-by)**2 + (tz-bz)**2)
                    
                    ratio = dist / palm_size
                    
                    # Normalize flexion based on empirical limits
                    max_r, min_r = 1.3, 0.4
                    flex = 1.0 - ((ratio - min_r) / (max_r - min_r))
                    current_flexion[i] = np.clip(flex, 0.0, 1.0)
                
                # Apply Kalman filter to smooth data AND estimate motion velocity
                self.filtered_flexion, self.estimated_velocity = self.kalman_filter(current_flexion)

        # Return frame, positions, and velocities
        if current_flexion is not None:
            return frame, self.filtered_flexion.tolist(), self.estimated_velocity.tolist()
        else:
            return frame, None, None