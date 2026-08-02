# Assist-As-Needed (AAN) Control Algorithm
The control algorithm in this project is designed based on the Assist-As-Needed (AAN) strategy. The primary goal of this controller is to encourage the user to utilize their maximum muscular capability and to prevent muscle slacking. The system applies force to joint movements only when the user is unable to execute the movement or when their velocity falls below a desired threshold.

**Note for Developers:** This controller is extended to simultaneously manage 7 degrees of freedom (DOF), including 5 fingers, 1 elbow joint, and 1 wrist joint. If you intend to use this system solely for a rehabilitation glove (only 5 fingers), you can limit the calculations to the first 5 elements. However, the formulas and mathematical logic remain identical for all joints, with the only difference being the values of the motion thresholds. 

# 1. Velocity Estimation and Initial Filtering
The AAN controller requires real-time flexion velocity estimation for each DOF to detect patient intent.

The raw velocity of each DOF is calculated using time-based differentiation (the difference between the current and previous positions relative to elapsed time):

$$v_{raw} = \frac{x_t - x_{t-1}}{\Delta t}$$

Note you can also use the velocity that the Kalman filter estimates.

To prevent high-frequency noise from affecting controller decision-making, this raw velocity is passed through an exponential smoothing low-pass filter with a coefficient of $\beta = 0.4$:

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot v_{raw}$$

This filtering step ensures that momentary velocity spikes are ignored, yielding more stable system behavior.

# 2. Phase 1: Intent Detection
The system must identify whether the patient intends to open their hand (Extension), close it (Flexion), or remain stationary (Idle). This is accomplished via a finite state machine (FSM) with three states: 1, -1, and 0.

State transitions are governed by comparing the filtered velocity ($v_t$) against an intent threshold ($\tau_{intent}$):

•	Movement Initiation: If $\vert{}v_t\vert{} > \tau_{intent}$, the system logs motion intent toward the target.

•	Movement Cancellation: If the user suddenly exerts an opposing force exceeding $\tau_{cancel}$ during motion, the system detects the stop or change of intent and resets the state to 0.
