# rehabilitation-robotic-for-arm
This project is a simulator for an upper-limb rehabilitation robot covering the arm, forearm, hand, and fingers. The system evaluates the user's motion intent through two completely distinct approaches (hand video analysis or sEMG muscle signal processing). If the force applied by the user is insufficient to complete the movement, the robot increases its assistive force to complete the intended motion. Therefore, for real-world applications, the primary goal of this robot is to maximize the user's active muscular participation and prevent muscle slacking.

# Assist-As-Needed Rehabilitation Exoskeleton Simulator
Assist-As-Needed Rehabilitation Exoskeleton Simulator
The goal of this project is to develop an intelligent simulator for an arm and hand rehabilitation exoskeleton. Since physical hardware was unavailable during this phase, the system was designed to detect the user's motion intent through one of the following two perception modules:

**1-Computer Vision:** Tracking hand joints via camera frames.

**2-Surface Electromyography (sEMG) Signals:** Interpreting sEMG data using machine learning models.
# Project Structure
To avoid complexity, the project is divided into three separate folders with distinct capabilities:

`just_hand/`: Controls robot finger movements exclusively through image processing (camera).

`hand_and_arm/`: Simultaneous control of robot fingers and elbow via image processing.

`EMG/`: Controls robot finger movements through muscle signal processing (sEMG).

# Assist-As-Needed (AAN) Mechanism
This project is built upon the AAN (Assist-As-Needed) control strategy. The core logic of this rehabilitation robot is as follows:

•	Independent Movement: If the system detects that the individual possesses sufficient capability during a movement process and performs the movement well enough on their own, the robot does not intervene.

•	Applying Assistive Force: However, if the control algorithms detect that the individual experiences weakness at certain moments or faces difficulty completing specific angles, the robotic system steps in. In this state, the robot applies a calculated assistive force to aid the patient's hand in completing the movement.
# Key Features
•	Dual Isolated Input Support: Kinematic extraction of hand and arm angles using a real-time camera, or grasp pattern prediction using sEMG signals.

•	Motion Intent Detection (ML): Utilization of the LightGBM machine learning model for processing temporal and frequency (wavelet) features of muscle signals.

•	Intelligent AAN Controller: Continuously evaluates movement speed and distance to the target to automatically adjust the degree of robotic assistance (Alpha parameter).

•	Physical Simulation: Implements the URDF model of the hand and arm in the PyBullet simulation environment for 3D visualization of the robot's performance.

•	Multiprocessing Architecture: Complete isolation of compute-heavy perception tasks from the robot physics engine to prevent frame drops and ensure smooth UI performance.
# Prerequisites and Installation
To run this simulator, you will need Python and the following libraries:

```bash
pip install numpy opencv-python mediapipe pybullet pillow scikit-learn lightgbm PyWavelets scipy
```
Note: The Graphical User Interface is built using Tkinter, which is typically installed by default alongside Python.
# How to Run
First, navigate to your desired directory (one of the three folders mentioned above), then simply execute the main file:

```Bash
python main.py
```
Upon execution, the dashboard window will open, comprising two main sections:
1.	Sensor/Camera View: Displays live data streams (camera frames or sEMG signal plots).
2.	Simulator View: Displays the real-time status of the exoskeleton inside the virtual environment and visualizes the robot's assistive force level.

# Results and Demonstrations
The performance of the AAN rehabilitation robot simulator has been validated using both Computer Vision and EMG signal processing. Below are visual demonstrations of the system in action.

## 1. EMG-Based Control
In this mode, utilizing test dataset, the LightGBM model predicts the user's intent purely from 12-channel surface EMG signals. The dynamic synergy filter ensures stable functional grasps. 
Power Grip (Dynamic Synergy) | Tip Pinch (Dynamic Synergy) |
| :---: | :---: |
| <img src="docs/assets/emg_power_grip.jpg" width="400"> | <img src="docs/assets/emg_tip_pinch.jpg" width="400"> |
# Technical Documentation
For an in-depth study of the project's technical concepts, please refer to the docs/ folder:

•	architecture.md: Overview of software structure and process synchronization.

•	control_algorithm.md: Mathematics behind the AAN controller and how Alpha is calculated.

•	vision_kinematics.md: Angle calculation methodology and the operation of the Kalman filter.

•	EMG_processing_and_ML.md: Details regarding feature extraction from EMG signals and machine learning model training.

•	simulation.md: Details of the URDF file and position control in PyBullet.
# Credits & Attribution
The 3D simulation of the robot's hand region used in this project incorporates parts of the [Realhand_description](https://github.com/RealHand-Robotics/Realhand_description) by RealHand-Robotics, used under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

Additionally, the EMG dataset utilized in this work is sourced from the [Ninapro dataset](https://ninapro.hevs.ch/instructions/DB2.html) (DB2 Data version) [1].

[[1] Electromyography data for non-invasive naturally-controlled robotic hand prostheses](https://www.nature.com/articles/sdata201453)
