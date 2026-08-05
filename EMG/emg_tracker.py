import numpy as np
import scipy.io as sio
import joblib
import pywt
import cv2

class EMGTracker:
    def __init__(self, model_path='emg_lgbm_model_combined.pkl', scaler_path='glove_scaler_combined.pkl', window_size=400):
        # Load pre-trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load and merge EMG datasets 
        data_A = sio.loadmat('S1_E1_A1.mat')
        data_C = sio.loadmat('S1_E2_A1.mat')
        
        restim_A = data_A['restimulus'].flatten()
        restim_C = data_C['restimulus'].flatten()
        
        self.emg_full = np.vstack((data_A['emg'], data_C['emg']))
        self.restimulus = np.concatenate((restim_A, restim_C))
        self.repetition = np.concatenate((data_A['repetition'].flatten(), data_C['repetition'].flatten()))
        
        self.restimulus = np.nan_to_num(self.restimulus, nan=0).astype(int)
        
        self.smoothed_flexion = np.zeros(5)
        self.smoothing_alpha = 0.15 
        self.window_size = window_size
        
        # Map target classes to specific grasp types
        self.target_classes = {
            0: 0,   # Rest
            1: 11,  # Thumb Flexion
            2: 1,   # Index Flexion
            3: 3,   # Middle Flexion
            4: 5,   # Ring Flexion
            5: 7,   # Pinky Flexion
            6: 19,  # Power Grip 
            7: 27,  # Power Sphere 
            8: 32   # Tip Pinch 
        }
        
        # Pre-compute valid window indices for each target movement
        self.movement_indices = {k: [] for k in self.target_classes.values()}
        test_repetitions = [5] 
        
        for i in range(0, len(self.emg_full) - self.window_size, 50):
            window_labels = self.restimulus[i : i + self.window_size]
            window_reps = self.repetition[i : i + self.window_size]
            
            dominant_rep = np.argmax(np.bincount(np.clip(window_reps, 0, None)))
            dominant_movement = np.argmax(np.bincount(np.clip(window_labels, 0, None)))
            
            # Filter out non-test repetitions and rest states
            if dominant_rep not in test_repetitions and dominant_movement != 0:
                continue 
                
            if dominant_movement in self.movement_indices:
                self.movement_indices[dominant_movement].append(i)
                
        # Handle cases where a movement might have no valid indices
        for k in self.movement_indices:
            if not self.movement_indices[k]: 
                self.movement_indices[k] = [0]
                
        self.index_pointer = 0
        self.last_movement = -1
        print("EMG Tracker Initialized with Exact Grasp Mapping and Dynamic Synergy!")

    def extract_features(self, window, threshold=1e-4, wavelet='db4', level=3):
        # Time-domain features
        mav = np.mean(np.abs(window), axis=0)
        rms = np.sqrt(np.mean(window**2, axis=0))
        var = np.var(window, axis=0)
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
        
        # Zero crossings and slope sign changes
        sign_change_zc = (window[:-1] * window[1:]) < 0
        amplitude_diff_zc = np.abs(window[:-1] - window[1:]) >= threshold
        zc = np.sum(sign_change_zc & amplitude_diff_zc, axis=0)
        
        diff = np.diff(window, axis=0)
        sign_change_ssc = (diff[:-1] * diff[1:]) < 0
        amplitude_diff_ssc = (np.abs(diff[:-1]) >= threshold) | (np.abs(diff[1:]) >= threshold)
        ssc = np.sum(sign_change_ssc & amplitude_diff_ssc, axis=0)
        
        # Frequency-domain features via wavelet transform
        coeffs = pywt.wavedec(window, wavelet, level=level, axis=0)
        
        energies = []
        for c in coeffs:
            energy_level = np.sum(c**2, axis=0)
            energies.append(energy_level)
            
        total_energy = np.sum(energies, axis=0)
        total_energy = np.where(total_energy == 0, 1e-12, total_energy)
        
        entropy = np.zeros(window.shape[1])
        for e in energies:
            p = e / total_energy
            p = np.where(p == 0, 1e-12, p)
            entropy -= p * np.log2(p)
            
        return np.concatenate([mav, rms, var, wl, zc, ssc] + energies + [entropy])

    def generate_plot(self, window_x, plot_w=400, plot_h=300):
        # Render a multi-channel line plot for the current EMG window
        img = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
        colors = [(0,255,0), (0,255,255), (255,0,0), (255,255,0), (0,165,255), (255,0,255)] * 2
        step_y = plot_h // 12
        x_coords = np.linspace(0, plot_w-1, self.window_size, dtype=int)
        
        for ch in range(12):
            y_offset = int((ch + 0.5) * step_y)
            ch_data = window_x[:, ch]
            norm_data = (ch_data / (np.max(np.abs(ch_data)) + 1e-8)) * (step_y * 0.45)
            y_coords = np.clip(y_offset - norm_data, 0, plot_h - 1).astype(int)
            pts = np.column_stack((x_coords, y_coords)).astype(np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=colors[ch], thickness=1)
            
        return img

    def apply_dynamic_synergy(self, flexions, activation_thresh=0.15, pinch_rest_thresh=0.15):
        # Enforce functional grasp synergies to stabilize raw predictions
        processed = flexions.copy()
        fingers_mean = np.mean(processed[1:5])
        
        # 1. Power Grasp (Grip/Sphere): Thumb is active and general finger flexion is high
        if processed[0] > 0.15 and fingers_mean > activation_thresh:
            mean_grasp_flexion = np.mean(processed)
            processed[:] = mean_grasp_flexion
            return processed
            
        # 2. Tip Pinch: Thumb and Index active, while other fingers remain mostly resting
        is_others_resting = np.all(processed[2:] < pinch_rest_thresh)
        if processed[0] > 0.2 and processed[1] > 0.2 and is_others_resting:
            mean_pinch_flexion = (processed[0] + processed[1]) / 2.0
            processed[0] = mean_pinch_flexion
            processed[1] = mean_pinch_flexion
            processed[2:] = 0.0
            return processed
            
        # 3. Isolated finger movements (no override needed)
        return processed

    def process_stream(self, requested_movement):
        # Reset stream tracking if the movement target changes
        if requested_movement != self.last_movement:
            self.index_pointer = 0
            self.last_movement = requested_movement
            
        ninapro_class = self.target_classes.get(requested_movement, 0)
        valid_starts = self.movement_indices[ninapro_class]
        
        # Loop back to the start if we run out of valid windows
        if self.index_pointer >= len(valid_starts):
            self.index_pointer = 0
            
        start_idx = valid_starts[self.index_pointer]
        window_x = self.emg_full[start_idx : start_idx + self.window_size, :]
        self.index_pointer += 1
        
        # Extract features and predict
        features = self.extract_features(window_x).reshape(1, -1)
        predicted_flexion = self.model.predict(features)[0]
        
        # Amplify ring finger signal if it's the dominant movement but weak overall
        if predicted_flexion[3] > np.max(np.delete(predicted_flexion, 3)) and np.mean(predicted_flexion) < 0.25:
            predicted_flexion[3] *= 6
        predicted_flexion = np.clip(predicted_flexion, 0.0, 1.0)
                
        # Apply finger-specific deadzones to filter cross-talk and baseline noise
        deadzone_threshold = np.array([0.25, 0.3, 0.20, 0.4, 0.20])
        predicted_flexion[predicted_flexion < deadzone_threshold] = 0.0
        
        # Normalize flexions that pass the deadzone check
        mask = predicted_flexion >= deadzone_threshold
        predicted_flexion[mask] = (predicted_flexion[mask] - deadzone_threshold[mask]) / (1.0 - deadzone_threshold[mask])
        
        # Apply synergy corrections and temporal smoothing
        predicted_flexion = self.apply_dynamic_synergy(predicted_flexion)
        self.smoothed_flexion = (self.smoothing_alpha * predicted_flexion) + ((1 - self.smoothing_alpha) * self.smoothed_flexion)
        
        plot_img = self.generate_plot(window_x)
        
        return plot_img, self.smoothed_flexion.tolist()