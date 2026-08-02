# Simulation Documentation, URDF Architecture, and Physics Engine
This section forms the core of physical validation and visualization for the rehabilitation robot. In this module, using the PyBullet physics engine, the mathematical outputs of the AAN controller are converted into physical joint torques, and mechanical constraints, inertia, and gravity are realistically simulated.

Note for Developers: This simulation environment is designed as a comprehensive (hybrid) system and includes physical definitions for both the hand (fingers) and the arm (elbow and wrist). If your project focuses solely on a rehabilitation glove (only fingers), you may ignore the sections related to arm dynamics and use only the equations and constraints from the finger section.

# 1. URDF Architecture and Custom Modifications
The physical structure of the robot is defined using the Unified Robot Description Format (URDF):

•	Hand and Fingers Assembly: The 3D mesh files, inertia matrices, and joint limits for the 5 fingers and the palm are based on the open-source Realhand_description project.

•	Arm Assembly (Custom Extension): To build a comprehensive rehabilitation simulator for the upper limb, new links and revolute joints—including the shoulder, upper arm, elbow, and forearm—were mathematically modeled, assigned mass and inertial properties, and appended to the base URDF file. 

# 2. Initial Physics Environment Configuration
To maintain real-time performance and synchronization with image processing, the PyBullet physics engine runs in DIRECT mode (without the default graphical GUI). Key variables of the physics environment are configured as follows:

•	Simulation Time Step ($dt$): Set to $\frac{1}{60}$ seconds to match the operating frequency of standard real-time controllers.

•	Gravity Vector: Applied with an acceleration of $-9.81 \text{ m/s}^2$ along the Z-axis to accurately simulate the weight and natural sagging of the user's hand in virtual space.

