# Assist-As-Needed (AAN) Control Algorithm
The control algorithm in this project is designed based on the Assist-As-Needed (AAN) strategy. The primary goal of this controller is to encourage the user to utilize their maximum muscular capability and to prevent muscle slacking. The system applies force to joint movements only when the user is unable to execute the movement or when their velocity falls below a desired threshold.

**Note for Developers:** This controller is extended to simultaneously manage 7 degrees of freedom (DOF), including 5 fingers, 1 elbow joint, and 1 wrist joint. If you intend to use this system solely for a rehabilitation glove (only 5 fingers), you can limit the calculations to the first 5 elements. However, the formulas and mathematical logic remain identical for all joints, with the only difference being the values of the motion thresholds. 

# 1. Velocity Estimation and Initial Filtering
The AAN controller requires real-time flexion velocity estimation for each DOF to detect patient intent.

The raw velocity of each DOF is calculated using time-based differentiation (the difference between the current and previous positions relative to elapsed time):

$$v_{raw} = \frac{x_t - x_{t-1}}{\Delta t}$$

Note if you want to use vision-based tracking you can also use the velocity that the Kalman filter estimates.

To prevent high-frequency noise from affecting controller decision-making, this raw velocity is passed through an exponential smoothing low-pass filter with a coefficient of $\beta$:

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot v_{raw}$$

This filtering step ensures that momentary velocity spikes are ignored, yielding more stable system behavior.

# 2. Phase 1: Intent Detection
The system must identify whether the patient intends to open their hand (Extension), close it (Flexion), or remain stationary (Idle). This is accomplished via a finite state machine (FSM) with three states: 1, -1, and 0.

State transitions are governed by comparing the filtered velocity ($v_t$) against an intent threshold ($\tau_{intent}$):

•	Movement Initiation: If $\vert{}v_t\vert{} > \tau_{intent}$, the system logs motion intent toward the target.

•	Movement Cancellation: If the user suddenly exerts an opposing force exceeding $\tau_{cancel}$ during motion, the system detects the stop or change of intent and resets the state to 0.

**Hand vs. Arm Joint Differences:** Due to their significantly higher mass and inertia compared to the fingers, the elbow and wrist joints require greater effort to overcome static friction. Consequently, in this algorithm, the intent detection threshold for the elbow is set $0.05$ higher, and for the wrist $0.03$ higher than the fingers to prevent unwanted robot activation caused by natural hand tremulousness or involuntary motion.

# 3. Phase 2: Dynamic Trajectory Generation
Unlike classical controllers that drive the arm directly to a pre-defined fixed target, this system utilizes the "Virtual Carrot" concept (a dynamic target).
Once the user's motion intent is confirmed (e.g., flexion movement), the system generates a virtual target that moves slightly ahead of the user's current position at a fixed offset (Max Lead).
The target step increment per time cycle is calculated as follows:

$$Step_{target} = 0.2 \times \Delta t$$

Then, the dynamic target position ($Target_{dynamic}$) is updated considering the user's current position ($x_t$) and the maximum allowable offset ($Lead_{max} = 0.4$):

$$Target_{t} = \min(1.0, \min(x_t + Step_{target}, x_t + Lead_{max}))$$

This strategy allows the robot to pull the user forward without propelling them unsafely to the end of their range of motion. If the user stops, the virtual target also halts and smoothly recedes back to the user's position.

# 4. Phase 3: Effort Evaluation and Alpha Parameter Tuning
This module constitutes the core logic of the AAN algorithm. The variable alpha ($\alpha$) determines the degree of user independence:

•	$\alpha = 1.0$: The user is fully independent, and the robot applies no assistance force.

•	$\alpha = 0.0$: The user is completely unable to move, and the robot performs 100% of the movement.

The system continuously monitors whether the user is tracking toward the virtual target ($Dist = Target - x_t$). If the user stays on trajectory, their effective velocity is evaluated against the effort threshold:

•	Struggling User: If the user's effective velocity drops below the effort threshold, muscle weakness at that specific angle is indicated. In this case, alpha decays at a set rate ($Decay$) so the robot can intervene:

$$\alpha_{t} = \alpha_{t-1} - (\lambda_{decay} \times \Delta t)$$

•	Succeeding User: If the user executes the movement at an adequate speed or is in an idle state, alpha increases at a recovery rate ($Recovery$), prompting the robot to disengage quickly and yield control back to the user:

$$\alpha_{t} = \alpha_{t-1} + (\lambda_{recovery} \times \Delta t)$$

The final value of $\alpha$ is continuously clamped between 0.0 and 1.0.


# 5. Kinematic Blending
In the final step, the reference target angle sent to the exoskeleton actuators in the simulation ($\theta_{assist}$) is calculated as a linear combination of the user's actual hand position ($x_t$) and the dynamic target position ($Target_t$), weighted by the $\alpha$ factor:

$$\theta_{assist} = \alpha \cdot x_t + (1 - \alpha) \cdot Target_t$$

Controller Summary:

•	If the patient performs the movement effectively ($\alpha \approx 1$), the control angle output to the robot matches the patient's actual hand angle, resulting in zero resistive or assistive force felt by the user.

•	If the patient exhibits weakness ($\alpha \to 0$), the control angle output shifts toward the virtual target, causing the robot to apply joint torque and guide the patient's hand toward the destination.


