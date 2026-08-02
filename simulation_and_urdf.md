# URDF Architecture and Custom Modifications
The physical structure of the robot is defined using the Unified Robot Description Format (URDF):

•	Hand and Fingers Assembly: The 3D mesh files, inertia matrices, and joint limits for the 5 fingers and the palm are based on the open-source Realhand_description project.

•	Arm Assembly (Custom Extension): To build a comprehensive rehabilitation simulator for the upper limb, new links and revolute joints—including the shoulder, upper arm, elbow, and forearm—were mathematically modeled, assigned mass and inertial properties, and appended to the base URDF file. 


