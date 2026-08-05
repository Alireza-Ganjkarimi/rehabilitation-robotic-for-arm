# Simulation Documentation, URDF Architecture, and Physics Engine
This section forms the core of physical validation and visualization for the rehabilitation robot. In this module, using the PyBullet physics engine, the mathematical outputs of the AAN controller are converted into physical joint torques, and mechanical constraints, inertia, and gravity are realistically simulated.

Note for Developers: This simulation environment is designed as a comprehensive (hybrid) system and includes physical definitions for both the hand (fingers) and the arm (elbow and wrist). If your project focuses solely on a rehabilitation glove (only fingers), you may ignore the sections related to arm dynamics and use only the equations and constraints from the finger section.

# 1. URDF Architecture and Custom Modifications
The physical structure of the robot is defined using the Unified Robot Description Format (URDF):

•	Hand and Fingers Assembly: The 3D mesh files, inertia matrices, and joint limits for the 5 fingers and the palm are based on the open-source Realhand_description project.

•	Arm Assembly (Custom Extension): To build a comprehensive rehabilitation simulator for the upper limb, new links and revolute joints—including the shoulder, upper arm, elbow, and forearm—were mathematically modeled, assigned mass and inertial properties, and appended to the base URDF file. 

# 2. Initial Physics Environment Configuration
To maintain real-time performance and synchronization with vision tracker module, the PyBullet physics engine runs in DIRECT mode (without the default graphical GUI). Key variables of the physics environment are configured as follows:

•	Simulation Time Step ($dt$): Set to $\frac{1}{60}$ seconds to match the operating frequency of standard real-time controllers.

•	Gravity Vector: Applied with an acceleration of $-9.81 \text{ m/s}^2$ along the Z-axis to accurately simulate the weight and natural sagging of the user's hand in virtual space.

# Kinematic Mapping (Normalized Space to Radians)
The output of the AAN controller is a dimensionless signal in the range $[0, 1]$, where $0.0$ represents full extension and $1.0$ represents full flexion. The simulator maps these values to the physical joint ranges in radians using linear interpolation.

Given the normalized control signal $u \in [0, 1]$, the target joint angle ($\theta_{target}$) is calculated as follows:

# A) Finger Joints (Index, Middle, Ring, Pinky)
Each finger features two primary actuated joints, the base joint (MCP) and the middle joint (PIP).

•	Base Joint (MCP): $\theta_{mcp} = u_{finger} \times 1.4$

•	Middle Joint (PIP): $\theta_{pip} = u_{finger} \times 1.57$

Note: The distal finger joints (DIP) are mechanically coupled to the PIP joints via the <mimic> tag in the URDF file and do not require an independent controller.

# B) Thumb Joints
the thumb features two joints, thumb base joint and 	thumb rotational joint.

•	Thumb Base Joint: $\theta_{thumb\_{mcp}} = u_{thumb} \times 1.05$

•	Thumb Rotational Joint: $\theta_{thumb\_{cmc}} = u_{thumb} \times 0.79$

# C) Elbow and Wrist Joints (Arm Section)

The elbow features a wider range of motion. The input signal maps directly to a maximum elbow flexion of $2.5$ radians (approximately 143 degrees):

•	Elbow Joint: $\theta_{elbow} = u_{elbow} \times 2.5$

Note: In the current configuration, the wrist joint is locked in a neutral position at $\theta_{wrist} = 0$, but its control mechanism is provisioned for future expansion up to a range of $1.57$ radians.

# 4. Actuator Dynamics: PD Controller Mechanism
The simulator does not instantaneously "teleport" the joints to the target angle. Instead, it uses PyBullet's POSITION_CONTROL mode, which functions mathematically as a Proportional-Derivative (PD) controller.

To reach the position $\theta_{target}$, the virtual motors apply a torque ($\tau$) based on the error between the target position and the current joint position ($\theta_{current}$), as well as the current joint velocity ($\dot{\theta}_{current}$):

$$\tau = K_p (\theta_{target} - \theta_{current}) - K_d (\dot{\theta}_{current})$$

Where $K_p$ is the stiffness coefficient (proportional gain) and $K_d$ is the damping coefficient (derivative gain).

# Clipping Limits (Power and Velocity Constraints)
To simulate the physical limitations of real actuators (exoskeleton servo motors), the output of the PD controller is strictly constrained by two parameters: 

maximum velocity ($v_{max}$) and maximum applied torque ($\tau_{max}$). The physics engine clamps these equations accordingly:

$$\vert \tau \vert \le \tau_{max}$$
$$\vert \dot{\theta}_{current} \vert \le v_{max}$$

Given the mechanical and inertial differences between the heavy arm segments and the lightweight fingers, these constraints are applied with varying thresholds for each joint group:

<p align="center">
  <b>Table 1: Maximum Allowed Torque and Velocity for Each Joint</b><br>
  <img src="https://github.com/user-attachments/assets/afc74869-d6e9-4c17-bf3f-11e6f4c52044" width="100%">
</p>
By capping the torque to 20 N.m for the elbow and 0.5 N.m for the fingers, the simulator accurately reflects a real-world scenario where the robot cannot drive the joint with infinite torque if severe physical resistance (e.g., patient muscle spasticity) is encountered. This guarantees Human-Robot Interaction safety within the simulation.

# 5. Synthetic Rendering and Real-Time Synchronization
To display the robot's status on the user dashboard, a virtual camera is mathematically positioned in 3D space using a View Matrix.

To prevent excessive CPU overhead, the rendering pipeline offloads computation to GPU hardware accelerators using the ER_BULLET_HARDWARE_OPENGL flag. This configuration enables the simulator to extract RGB matrices efficiently and stream them to the user interface at a high, stable frame rate. 

Despite hardware acceleration, a common issue in multi-process physics simulations is time dilation, where GUI and rendering overhead can cause the physics engine to lag behind real-world time. To ensure the AAN controller operates accurately based on strict real-time derivatives, a dynamic catch-up mechanism is implemented alongside the rendering pipeline. By calculating the elapsed real time (current_time - last_sim_time) and dividing it by the fixed time step ($dt$), the simulator executes multiple localized internal steps (p.stepSimulation()) within a single cycle. This guarantees that the physics engine perfectly syncs with the real-world clock, maintaining a highly stable 60Hz loop for precision.

