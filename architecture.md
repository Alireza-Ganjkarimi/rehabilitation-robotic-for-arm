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

# Process 3: Main Thread & Rehabilitation Dashboard UI
The application's main thread handles the Tkinter user interface.

•	**Non-Blocking Loop:** The `update_ui_loop` method continuously polls the queues using the `.after(16)` callback (equivalent to ~60 FPS). Because `get_nowait` is utilized, the UI never freezes while awaiting data from the physics engine or camera.

•	**High-Performance Plotting:** Rather than using heavy plotting packages such as Matplotlib—which introduce UI frame drops—the robot assistance metric ($\alpha$) is rendered directly on a raw canvas using native primitives (lines and geometric shapes), drastically reducing computational overhead.

# 2. Inter-Process Communication (IPC)

Safe data exchange among these three isolated processes is handled via the following mechanisms:

•	**Vision Queue** (`vision_queue`): Transfers processed camera frames from the vision processor to the dashboard.

•	**Kinematics Queue** (`flexion_queue`): Transfers the joint angle array from the vision processor to the robot processor.

•	**Robot Queue** (`robot_queue`): Transfers rendered simulator frames and the output alpha metric ($\alpha$, indicating the degree of robot intervention) from the robot processor to the dashboard.

**Bounded Queues Strategy**
All communication queues are defined with a strict maximum size (`maxsize=2`). This small capacity represents a deliberate decision to prevent data stagnation within the pipes. In real-time control systems, dropping an outdated frame (frame dropping) is significantly safer than processing stale frames with accumulated lag.

# 3. Graceful System Shutdown
When the user closes the dashboard (invoking the `on_close` handler), the system avoids abrupt termination by signaling through a shared `mp.Event` named `running_event`.

Clearing this event (`clear`) stops the infinite processing loops inside both worker processes (vision and robot). The main system then grants them a 1-second grace period (`join`) to properly release hardware resources (webcam and physics engine context). If a process fails to respond within this window, the system forcefully terminates it (`terminate`) to prevent orphan (zombie) processes from persisting in the operating system background.

Note: The `mp.freeze_support()` invocation is embedded in the main execution entry point to ensure this multiprocessing architecture compiles and executes correctly under Windows environments without spawning infinite recursive loops.

