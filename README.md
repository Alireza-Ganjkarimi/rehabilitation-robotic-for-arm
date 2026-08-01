# rehabilitation-robotic-for-arm
This project is a simulator for a upper-limb rehabilitation robot covering the arm, forearm, hand, and fingers. By analyzing video of the user’s hand, it estimates the intended movement. If the user's applied force is insufficient, the robot increases its force to assist in completing the desired motion.
# Hand Rehabilitation Exoskeleton Simulator (Assist-As-Needed)
The objective of this project is to develop an intelligent simulator for a hand and arm rehabilitation exoskeleton. Since physical hardware equipment was not available during this phase of the project, the system has been designed so that the user's Motion Intent is detected directly through video camera frames and computer vision algorithms.
# Assist-As-Needed (AAN) Mechanism
This project is built upon the AAN (Assist-As-Needed) control strategy. The core logic of this rehabilitation robot is as follows:

•	Independent Movement: If the system detects that the individual possesses sufficient capability during a movement process and performs the movement well enough on their own, the robot does not intervene.

•	Applying Assistive Force: However, if the control algorithms detect that the individual experiences weakness at certain moments or faces difficulty completing specific angles, the robotic system steps in. In this state, the robot applies a calculated assistive force to aid the patient's hand in completing the movement.
# Key Features
•	Real-Time Motion Tracking: Uses a camera to extract the kinematic angles of the 5 fingers, wrist, and elbow in real time.

•	Intelligent AAN Controller: Continuously evaluates movement speed and distance to the target to automatically adjust the degree of robotic assistance (Alpha parameter).

•	Physical Simulation: Implements the URDF model of the hand and arm in the PyBullet simulation environment for 3D visualization of the robot's performance.

•	Multiprocessing Architecture: Separates computer vision processing from the robot physics engine to prevent frame drops and provide a seamless, smooth dashboard experience.
# Prerequisites and Installation
To run this simulator, you will need Python and the following libraries:

```bash
pip install numpy opencv-python mediapipe pybullet pillow
```
(Note: The Graphical User Interface is built using Tkinter, which is typically installed by default alongside Python.)
# How to Run
To run the rehabilitation dashboard, simply execute the main file:

```Bash
python main.py
```
Upon execution, the dashboard window will open, comprising two main sections:
1.	Camera View: Tracks your hand and extracts joint locations.
2.	Simulator View: Displays the exoskeleton's status in a virtual environment and assists you in flexing your fingers or arm when needed.
# Technical Documentation
For an in-depth study of the project's technical concepts, please refer to the docs/ folder:
•	architecture.md: Overview of software structure and process synchronization.

•	control_algorithm.md: Mathematics behind the AAN controller and how Alpha is calculated.

•	vision_and_filtering.md: Angle calculation methodology and the operation of the One-Euro filter for noise reduction.

•	simulation.md: Details of the URDF file and position control in PyBullet.
# Credits & Attribution
This project incorporates components from [Realhand_description](https://github.com/RealHand-Robotics/Realhand_description) 
by RealHand-Robotics, used under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
