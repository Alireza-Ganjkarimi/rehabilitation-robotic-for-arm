import scipy.io as sio
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib 
import time

def load_ninapro_file(file_path):
    data = sio.loadmat(file_path)
    emg = data['emg']                  
    glove = data['glove']              
    repetition = data['repetition'].flatten()    
    restimulus = data['restimulus'].flatten()
    
    # Extract MCP joint sensors for the 5 fingers
    target_fingers_raw = glove[:, [1, 4, 8, 12, 16]]
    
    return emg, target_fingers_raw, repetition, restimulus

def prepare_combined_data():
    # Load and merge datasets
    emg_A, glove_A, rep_A, restim_A = load_ninapro_file("S1_E1_A1.mat")
    emg_C, glove_C, rep_C, restim_C = load_ninapro_file("S1_E2_A1.mat")    
    
    emg_combined = np.vstack((emg_A, emg_C))
    glove_combined = np.vstack((glove_A, glove_C))
    rep_combined = np.concatenate((rep_A, rep_C))
    restim_combined = np.concatenate((restim_A, restim_C))
    
    # Normalize glove data
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    glove_normalized = scaler.fit_transform(glove_combined)
    
    # Initialize target array for synergistic mapping
    target_fingers_synergized = np.zeros_like(glove_normalized)
    
    # Define movement masks based on restimulus classes
    is_thumb = (restim_combined == 11) | (restim_combined == 12)
    is_index = (restim_combined == 1)  | (restim_combined == 2)
    is_middle = (restim_combined == 3) | (restim_combined == 4)
    is_ring = (restim_combined == 5)   | (restim_combined == 6)
    is_pinky = (restim_combined == 7)  | (restim_combined == 8)
    
    is_power_grip_sphere = (restim_combined == 19) | (restim_combined == 27)
    is_tip_pinch = (restim_combined == 32)
    
    # Map isolated finger movements
    target_fingers_synergized[is_thumb, 0] = glove_normalized[is_thumb, 0]
    target_fingers_synergized[is_index, 1] = glove_normalized[is_index, 1]
    target_fingers_synergized[is_middle, 2] = glove_normalized[is_middle, 2]
    target_fingers_synergized[is_ring, 3] = glove_normalized[is_ring, 3]
    target_fingers_synergized[is_pinky, 4] = glove_normalized[is_pinky, 4]
    
    # Map synergistic/functional movements
    target_fingers_synergized[is_power_grip_sphere, :] = glove_normalized[is_power_grip_sphere, :]
    
    # Use actual independent values for thumb and index during tip pinch
    target_fingers_synergized[is_tip_pinch, 0] = glove_normalized[is_tip_pinch, 0]
    target_fingers_synergized[is_tip_pinch, 1] = glove_normalized[is_tip_pinch, 1]

    # Average thumb and index flexion for tip pinch
    mean_pinch_flexion = np.mean(glove_normalized[is_tip_pinch][:, [0, 1]], axis=1)
    target_fingers_synergized[is_tip_pinch, 0] = mean_pinch_flexion
    target_fingers_synergized[is_tip_pinch, 1] = mean_pinch_flexion
    
    # Split data by repetition (train: 0,1,3,4,6 | test: 2,5)
    train_mask = np.isin(rep_combined, [1, 3, 4, 6]) | (rep_combined == 0)
    test_mask = np.isin(rep_combined, [2, 5])
    
    X_train = emg_combined[train_mask]
    y_train = target_fingers_synergized[train_mask]
    X_test = emg_combined[test_mask]
    y_test = target_fingers_synergized[test_mask]
    
    return X_train, y_train, X_test, y_test, scaler

def extract_features(window, threshold=1e-4, wavelet='db4', level=3):
    # Time-domain features
    mav = np.mean(np.abs(window), axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0))
    var = np.var(window, axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    
    # Zero crossings
    sign_change_zc = (window[:-1] * window[1:]) < 0
    amplitude_diff_zc = np.abs(window[:-1] - window[1:]) >= threshold
    zc = np.sum(sign_change_zc & amplitude_diff_zc, axis=0)
    
    # Slope sign changes
    diff = np.diff(window, axis=0)
    sign_change_ssc = (diff[:-1] * diff[1:]) < 0
    amplitude_diff_ssc = (np.abs(diff[:-1]) >= threshold) | (np.abs(diff[1:]) >= threshold)
    ssc = np.sum(sign_change_ssc & amplitude_diff_ssc, axis=0)
    
    # Frequency-domain (Wavelet) features
    coeffs = pywt.wavedec(window, wavelet, level=level, axis=0)
    
    energies = []
    for c in coeffs:
        energy_level = np.sum(c**2, axis=0)
        energies.append(energy_level)
        
    total_energy = np.sum(energies, axis=0)
    total_energy = np.where(total_energy == 0, 1e-12, total_energy)
    
    # Wavelet entropy
    entropy = np.zeros(window.shape[1])
    for e in energies:
        p = e / total_energy
        p = np.where(p == 0, 1e-12, p)
        entropy -= p * np.log2(p)
        
    return np.concatenate([mav, rms, var, wl, zc, ssc] + energies + [entropy])

def create_feature_dataset(X_raw, y_raw, window_size, step_size):
    # Extract features using a sliding window approach
    n_samples = X_raw.shape[0]
    X_features, y_windows = [], []
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        X_features.append(extract_features(X_raw[start_idx:end_idx, :]))
        y_windows.append(np.mean(y_raw[start_idx:end_idx, :], axis=0))
    return np.array(X_features), np.array(y_windows)

if __name__ == "__main__":
    WINDOW_SIZE = 400
    STEP_SIZE = 100
    
    print("1. Loading Combined Data and Applying Label Synergy...")
    X_train, y_train, X_test, y_test, scaler = prepare_combined_data()
    
    print("2. Extracting Features...")
    X_train_feat, y_train_feat = create_feature_dataset(X_train, y_train, WINDOW_SIZE, STEP_SIZE)
    X_test_feat, y_test_feat = create_feature_dataset(X_test, y_test, WINDOW_SIZE, STEP_SIZE)
    
    print("\n3. Training LightGBM Model...")
    lgbm_base = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, num_leaves=31, n_jobs=-1, random_state=42)
    lgbm_model = MultiOutputRegressor(lgbm_base)
    lgbm_model.fit(X_train_feat, y_train_feat)
    
    print("\n4. Evaluating the Model...")
    y_train_pred = lgbm_model.predict(X_train_feat)
    y_test_pred = lgbm_model.predict(X_test_feat)

    # Evaluation on train
    train_r2 = r2_score(y_train_feat, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_feat, y_train_pred))
    train_mae = mean_absolute_error(y_train_feat, y_train_pred)
    
    # Evaluation on test
    test_r2 = r2_score(y_test_feat, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_feat, y_test_pred))
    test_mae = mean_absolute_error(y_test_feat, y_test_pred)
    
    print("-" * 40)
    print("        REGRESSION METRICS REPORT       ")
    print("-" * 40)
    print(" TRAINING SET:")
    print(f"  * R-squared (R2): {train_r2:.4f}")
    print(f"  * RMSE:           {train_rmse:.4f}")
    print(f"  * MAE:            {train_mae:.4f}")
    print("\n TESTING SET:")
    print(f"  * R-squared (R2): {test_r2:.4f}")
    print(f"  * RMSE:           {test_rmse:.4f}")
    print(f"  * MAE:            {test_mae:.4f}")
    print("-" * 40)
    
    print("\n4. Saving Model and Scaler...")
    joblib.dump(lgbm_model, 'emg_lgbm_model_combined.pkl')
    joblib.dump(scaler, 'glove_scaler_combined.pkl')
    print("Done!")