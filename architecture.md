# Software Architecture and Process Synchronization
The `main.py` file in this project is not merely a graphical user interface (GUI), but serves as the central orchestrator of the entire rehabilitation system. The primary role of this module is to establish real-time communication among the perception layer (Vision), the control layer (AAN), and the physics layer (PyBullet), ensuring the GUI updates seamlessly without frame drops or latency.
# 1. Multiprocessing Approach
The single biggest challenge in developing machine vision-based robotic systems in Python is the Global Interpreter Lock (GIL), which prevents true parallel execution of compute-heavy code across multiple CPU cores when using multithreading.
To overcome this limitation and achieve a high operating frequency, this module employs a Multiprocessing Architecture. Under this architecture, the system is decomposed into three completely independent processes, each operating with its own isolated memory space:

**Process 1: Vision Layer** (`vision_worker_process`)

This process is exclusively dedicated to communicating with the webcam and executing the hand-tracking model (`VisionTracker`).

•	Frame Management: Once joint angles (flexion) are extracted and the output image is prepared, the data is pushed to communication queues.

•	Latency Prevention: To ensure the system always processes the latest frame, non-blocking `try/except` blocks are used. If a queue is full, the process immediately discards stale data (`get_nowait`) and replaces it with the newly arrived data.

**Process 2: Robot Physics & Controller** (`robot_worker_process`)

This process forms the computational and physical engine of the system. Operating in parallel with the camera, it incorporates several critical features:

•	**Queue Draining:** This process drains all pending data from the vision queue until it reaches the most recent kinematic sample. This prevents data accumulation and eliminates "time stretch" effects in the controller.

•	**Velocity Spike Prevention:** Velocity calculations within the AAN controller are triggered only when new sensor data has actually been received (`new_data_received`).

•	**Isolated 60 Hz Loop:** By tracking the elapsed loop execution time (`elapsed_loop_time`) and applying dynamic sleep intervals (`time.sleep(sleep_time)`), the physics engine clock is tightly locked at 60 Hz.

•	**Controlled Rendering:** To prevent performance degradation of the physics engine, image extraction from the simulator is capped at ~30 FPS (every 0.033 seconds).
