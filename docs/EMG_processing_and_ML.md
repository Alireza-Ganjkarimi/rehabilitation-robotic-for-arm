# Muscle Signal Processing (sEMG) and Machine Learning
This module is responsible for receiving raw surface electromyography (sEMG) signals, extracting meaningful features from them, and predicting the degree of flexion for the five fingers of the hand. Given the noisy and complex nature of biological signals, this system utilizes a combination of a powerful LightGBM machine learning model and post-processing algorithms to stabilize the robot's motion.

# 1. Dataset Architecture
The machine learning model in this project was trained on data from subject 1 within the standard NinaPro Dataset (DB2 version). During the data recording process, subjects were asked to repeat movements displayed on a screen while their sEMG signals and hand joint angles were recorded simultaneously. The structure of the dataset used is as follows:

•	**Muscle Signals (sEMG):** Muscle activity was recorded using 12 surface electrode leads at a sampling rate of 2 kHz. 8 electrodes were placed with equal spacing around the forearm (near the radiohumeral joint), 2 electrodes were placed on the primary muscle activity sites for finger flexors and extensors (Flexor/Extensor Digitorum), and 2 additional electrodes were placed on the biceps and triceps muscles.

•	**Joint Kinematics:** Finger motion angles were synchronously recorded using a smart glove (CyberGlove II) to serve as the target variable for model training. In this project, the metacarpophalangeal (MCP) joint angles (the base joints connecting the fingers to the palm) were extracted for all 5 fingers and defined as the target variables. To align and facilitate the learning process, the recorded joint angle data were scaled individually to a range of 0 to 1 based on each joint's minimum and maximum angles. 

•	**Motion Classes:** Out of the 49 movements available in the database, exactly 9 practical motion classes were extracted for controlling the simulator: Rest, 5 isolated movements for individual fingers, and 3 functional movements including Power Grip, Power Sphere, and Tip Pinch.

# 2. Target Mapping & Motor Synergy
A key challenge in controlling rehabilitation robots via sEMG is the coupling effect—meaning that when an individual intends to flex only a single finger, other fingers involuntarily move slightly as well. If the model is trained on raw glove data, it will learn these unwanted noises. To resolve this issue and train the model accurately, motion targets were engineered through two approaches rather than using raw labels directly:

•	**Isolated Movements:** For movements specific to a single finger (e.g., flexing only the index finger), only the normalized angle of that specific finger from the glove sensor is recorded in the target data array. The angles for the remaining 4 fingers are forcibly set to zero. This forces the machine learning model to ignore movement noise in the other fingers and learn muscle patterns specific strictly to that single finger.

•	**Functional Grasps & Synergy:** Motor synergy refers to the coordinated activation patterns of joints and muscles to execute a unified task (such as grasping a glass). In this project, synergy logic is implemented for functional movements as follows:

o	Power Grip and Power Sphere: Because the entire hand is engaged during these movements, angles for all 5 fingers are read directly from glove data and fed synchronously to the model as targets (None of them is forcibly set to zero).

<img width="1355" height="366" alt="image" src="https://github.com/user-attachments/assets/2f6fa0fd-84d2-4672-951e-0839354bd0ea" />

o	Tip Pinch: Only the thumb and index finger are engaged in this movement. To ensure a stable and precise pinch in the simulator, the system calculates the average flexion of the thumb and index finger, assigning this uniform average as the target for both digits. To prevent interference, target values for the remaining 3 fingers (middle, ring, and pinky) are set to zero during this movement.

<img width="1497" height="345" alt="image" src="https://github.com/user-attachments/assets/53fda8a2-4d5b-49b1-aafb-519b2edaf6b9" />


# 3. Data Splitting for Training
To ensure model generalization and prevent overfitting, the dataset was split based on the repetition index of each movement. Repetitions 0, 1, 3, 4, and 6 were designated as training data (Train), while repetitions 2 and 5 were reserved as testing data (Test) to evaluate model performance.

# 4. Signal Processing and Feature Extraction
To extract features, initially, the continuous sEMG signal is segmented using a sliding window technique (windows of 400 samples with a step size of 100), and a rich set of time-domain and time-frequency domain features is extracted from each window. In this phase, for each 400-sample window, the mean value of the target joint angles is defined as the ground-truth label for that window.

# 4.1. Time-Domain Features
Due to their low computational complexity, time-domain features play a key role in real-time processing of biological signals. Assuming $x_i$ is the signal amplitude at the $i$-th sample and $N$ is the length of the time window, the following features are extracted:

•	Mean Absolute Value (MAV): A linear measure of signal amplitude and muscle contraction effort.

$$MAV = \frac{1}{N} \sum_{i=1}^{N} \vert{}x_i\vert{}$$

•	Root Mean Square (RMS): One of the most standard metrics for evaluating signal power, which places greater weight on signal peaks than MAV due to its quadratic structure.

$$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$

•	Variance (VAR): Represents the dispersion of signal values around the mean ($\mu$) and indicates the power of the AC muscle signal.

$$VAR = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

•	Waveform Length (WL): A measure of signal complexity, fluctuation, and amplitude variation across a time window.

$$WL = \sum_{i=1}^{N-1} \vert{}x_{i+1} - x_i\vert{}$$

•	Zero Crossings (ZC): The number of times the signal changes sign and crosses the zero axis. To filter out background noise and prevent false counts, a threshold of $10^{-4}$ is applied for this feature.

•	Slope Sign Changes (SSC): The number of times the sign of the signal slope changes direction (local peaks and valleys). Similar to ZC, this feature is calculated with an applied threshold of $10^{-4}$.

# 4.2. Time-Frequency Features
Given the non-stationary nature of sEMG signals, using purely frequency-based transforms (such as the Fourier Transform) leads to the loss of temporal information. To overcome this limitation, this project utilizes the Discrete Wavelet Transform (DWT). By providing multi-resolution analysis, this transform achieves an optimal balance between time resolution and frequency resolution.
In this architecture, the Daubechies 4 (db4) mother wavelet is used, and the signal is decomposed down to Level 3.

# Decomposition Mechanism (Filter Banks):
Based on Mallat's Algorithm, the input signal passes sequentially through a digital filter bank. At each level of decomposition, the signal passes through a low-pass filter and a high-pass filter, followed by downsampling by a factor of 2. By performing this operation up to Level 3, the initial signal is decomposed into a set of sub-band signals consisting of one set of low-frequency approximation coefficients ($A_3$) and three sets of higher-frequency detail coefficients ($D_1, D_2, D_3$).

To convert these coefficients into interpretable features for the machine learning model, two critical metrics are extracted:

•	**Wavelet Energy:**

The energy of the signal across each obtained frequency band is calculated. If $C_j$ represents the vector of coefficients in a specific sub-band and $M$ is the length of that vector, the energy of that band ($E_j$) is equal to the sum of the squared coefficients:

$$E_j = \sum_{k=1}^{M} \vert{}C_j[k]\vert{}^2$$

The total energy ($E_{total}$) is then computed as the sum of energies across all bands.

•	**Wavelet Entropy:**

This metric evaluates the degree of disorder and energy distribution across the frequency spectrum of the signal. First, the relative energy contribution of each frequency band ($p_j$) is calculated:

$$p_j = \frac{E_j}{E_{total}}$$

Then, using Shannon's Entropy formula, the frequency complexity of the signal is extracted:

$$Entropy = - \sum_{j=1}^{4} p_j \log_2(p_j)$$

Lower entropy values indicate energy concentration within a specific frequency band (rhythmic, coordinated contraction), while higher values reflect an even distribution of energy across all bands (such as rest state). This distinction provides a set of effective features for motion pattern classification by the LightGBM model.

# 5. Machine Learning Model and Kinematics Prediction 
To map the features extracted from muscle signals to the continuous angles of hand joints, this project addresses a multi-target regression problem. To solve this problem, the LightGBM (Light Gradient Boosting Machine) algorithm was utilized due to its strength in modeling complex relationships, high speed, and computational efficiency.

Since standard regression algorithms are typically capable of predicting only a single output value, the base LightGBM model is wrapped inside the `MultiOutputRegressor` class. This class manages the architecture by creating 5 completely independent LightGBM models under the hood instead of a single monolithic model. In other words, the system feeds the exact same input feature array to all models, but each of the 5 models is specialized, trained, and optimized in parallel to exclusively predict the angle of one specific finger.
### Evaluation Results
The evaluation results regarding the performance of the machine learning model in estimating joint angles for the training samples (corresponding to repetitions 0, 1, 3, 4, and 6) and test samples (corresponding to repetitions 2 and 5 of the subject) are presented in the table below.
<p align="center">
  <b>Table 1: Evaluation Results</b><br>
  <img src="https://github.com/user-attachments/assets/96279aef-a602-4363-886e-81f07f72a1e1" width="100%">
</p>


# 6. Real-Time Data Streaming
In this simulator, when a user selects a motion type (e.g., Power Sphere or Index Finger Movement) via the dashboard, the robot is not directly controlled by this selection. Instead, this trigger simply instructs the `EMGTracker` module to query the database and locate the raw sEMG signals corresponding to the test samples (held-out repetition 5) for that specific movement. The system then uses an index pointer to extract these signals as sliding time windows, extracts their features, and feeds them into the machine learning model. In fact, the machine learning model predicts joint flexion completely blindly, solely based on the incoming muscle signal stream.

# 7. Post-Processing Pipeline and Kinematic Stabilization
The raw outputs predicted by the machine learning model (i.e., estimated flexion values for the five fingers) may contain jitter or noise. To convert these model predictions into smooth, joint-safe movement angles for the robot, the estimates pass through a multi-stage processing pipeline:

•	**Deadzone:** To eliminate model noise during rest-state predictions and prevent signal crosstalk, a dedicated deadzone threshold is defined for each finger's predicted flexion. If the predicted flexion for a finger falls below this threshold, it is treated as noise and forced directly to zero.

•	**Renormalization:** To prevent sudden jumps in finger position once predicted flexion exceeds the deadzone, active values are renormalized to transition smoothly starting from zero up to one:

$$Flex_{norm} = \frac{Flex_{pred} - Threshold}{1.0 - Threshold}$$

•	**Dynamic Synergy Application:** At this stage, the system evaluates the pattern of values predicted by the machine learning model to detect the functional movement type, applying synergy logic (similar to training time) for stabilization:

o	Power Grasp: If the algorithm detects that the predicted thumb flexion exceeds a specific threshold (active) while the average model prediction for the other fingers is also high, it interprets the user's intent as a full hand closure. In this case, all finger angles are locked to their collective average, closing the robotic hand uniformly.

o	Tip Pinch: If predicted values for the thumb and index finger are high (active) while predicted outputs for the other three fingers remain at rest (below threshold), the system identifies a pinch pattern. Consequently, thumb and index angles are set to their joint average, and the outputs for the remaining three fingers are forced to zero to prevent interference.

•	Temporal Smoothing: In the final step, to remove high-frequency fluctuations and ensure fluid robot movement, the corrected model predictions pass through an Exponential Moving Average (EMA) low-pass filter. This filter blends the new predicted flexion with the previous joint state:

$$Flex_t = \alpha \cdot Flex_{new\_pred} + (1 - \alpha) \cdot Flex_{t-1}$$

where $\alpha$ represents the system's smoothing factor.

# 8. Signal Visualization
To provide visual feedback to the user on the interactive dashboard, this module generates a multi-channel graphical plot of the raw sEMG data corresponding to the user-selected movement within each time window. This chart plots the values of all 12 channels using distinct colors as line graphs and updates continuously, allowing the real-time muscle contraction status to be observed live by the user or therapist.


