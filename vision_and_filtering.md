# Vision and Kinematic Filtering
This module is responsible for processing video frames, extracting the 3D position of hand and arm joints, and finally filtering this data to eliminate noise and estimate velocity.

**Note for Developers:** The architecture described in this document pertains to the comprehensive and hybrid mode—namely, the simultaneous control of fingers, elbow, and wrist (totaling 7 degrees of freedom). If you intend to use this project solely for a robotic glove (only 5 fingers), you may ignore the calculations related to MediaPipe Pose and the elbow/wrist joints, and use only the equations in the first section.

# 1. Finger Kinematics Extraction
To detect finger movement, the MediaPipe Hands model is used, which outputs 21 keypoints (landmarks) of the hand in 3D space (x, y, z). The main challenge in image processing is that the hand size changes depending on the user's distance from the camera. To solve this problem and extract scale-invariant data, the following calculations are used:

# A) Calculation of the Scaling Factor (Palm Size)
First, a fixed baseline of the patient's hand called "Palm Size" is calculated. This value is equal to the Euclidean distance between the wrist joint (point 0) and the base of the middle finger (point 9):

$$PalmSize = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2 + (z_9 - z_0)^2}$$ 

# B) Calculation of Each Finger's Flexion Ratio
For each of the 5 fingers, the 3D distance between the fingertip (Tip) and the base of that same finger (Base) is measured and divided by the palm size to obtain a dimensionless ratio (Ratio):

$$D_{finger} = \sqrt{(x_{tip} - x_{base})^2 + (y_{tip} - y_{base})^2 + (z_{tip} - z_{base})^2}$$
$$Ratio = \frac{D_{finger}}{PalmSize}$$

# C) Normalization
This ratio is mapped to a standard range $[0, 1]$ using two empirical thresholds: maximum extension ($R_{max} = 1.3$) and maximum flexion ($R_{min} = 0.4$), where 0.0 represents a fully open hand and 1.0 represents a fully closed finger:

$$Flexion = 1.0 - \left( \frac{Ratio - R_{min}}{R_{max} - R_{min}} \right)$$

# 2. Arm & Wrist Kinematics Extraction
The MediaPipe Pose model is used for the arm. Unlike the fingers, which are calculated based on the tip-to-base distance, the elbow and wrist movements are calculated based on the spatial angle between three joints.

# A) Calculation of the 3D Spatial Angle
Suppose we have three points $A$ (shoulder), $B$ (elbow), and $C$ (wrist). To find the angle of the middle joint ($B$), we first construct the two vectors $\vec{BA}$ and $\vec{BC}$. The angle between these two vectors is obtained using the dot product rule:

$$\theta = \arccos\left( \frac{\vec{BA} \cdot \vec{BC}}{\vert{}\vec{BA}\vert{} \vert{}\vec{BC}\vert{}} \right)$$

This angle is converted from radians to degrees.

# B) Mapping the Angle to Control Space

•	Elbow: The physiological movement of the elbow typically ranges between 180 degrees (fully extended) and 30 degrees (fully flexed). This range is converted to a normalized value $[0, 1]$ using the following equation:

$$Flex_{elbow} = 1.0 - \left( \frac{\theta_{elbow} - 30}{180 - 30} \right)$$

•	Wrist: The wrist angle is calculated by examining the alignment of the forearm (the line from the elbow to the wrist) and the alignment of the palm (the line from the wrist to the index finger). In this calibration, an angle of 170 degrees represents the wrist in a neutral or fully extended state (Flexion = 0.0), and an angle of 110 degrees represents the wrist fully flexed inward (Flexion = 1.0).

# 3. Kalman Kinematic Filter
To eliminate the intrinsic noise (jitter) of machine vision data, the system’s approach was upgraded from a One-Euro low-pass filter to a Kalman filter. As an optimal state estimator, this filter does not merely smooth the signal; rather, by leveraging position and velocity dynamics, it predicts the upcoming joint movement. 
# A) State-Space Model
The state vector $X$ in our system includes position ($x$) and velocity ($v$) for all 7 degrees of freedom (a $14 \times 1$ column matrix):

$$X = \begin{bmatrix} x_1, x_2, \dots, x_7, v_1, v_2, \dots, v_7 \end{bmatrix}^T$$

The state transition matrix ($A$) is constructed based on physical equations of motion ($x_{new} = x_{old} + v \cdot dt$). In this block matrix, the relationship between position and velocity is established via the time variable ($dt$).

# B) Prediction Step
Based on our mathematical model (linear kinematics), the system predicts the state and uncertainty (covariance $P$) at the new time step:

$$X_{pred} = A \cdot X_{prev}$$
$$P_{pred} = A \cdot P_{prev} \cdot A^T + Q$$

Here, $Q$ is the process noise covariance matrix, representing our confidence in the mathematical model.


