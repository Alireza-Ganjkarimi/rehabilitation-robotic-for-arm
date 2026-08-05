import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from collections import deque
import multiprocessing as mp
import time
import queue

# Local project modules
from emg_tracker import EMGTracker
from aan_controller import FingerAANController
from simulator import RobotSimulator

# ---------------------------------------------------------
# EMG Worker Process
# ---------------------------------------------------------
def emg_worker_process(sensor_queue, flexion_queue, running_event, active_movement):
    # Initialize the EMG tracking model
    try:
        tracker = EMGTracker()
    except Exception as e:
        print(f"Failed to load EMG Tracker: {e}")
        return

    # Continuous data acquisition loop
    while running_event.is_set():
        current_movement = active_movement.value
        img, predicted_flexion = tracker.process_stream(current_movement)
        
        # Flush stale data if the queue is full to prioritize real-time updates
        if sensor_queue.full():
            try: sensor_queue.get_nowait()
            except queue.Empty: pass
        try: sensor_queue.put_nowait((img, predicted_flexion))
        except queue.Full: pass
        
        if flexion_queue.full():
            try: flexion_queue.get_nowait()
            except queue.Empty: pass
        try: flexion_queue.put_nowait(predicted_flexion)
        except queue.Full: pass
        
        time.sleep(0.05) 

# ---------------------------------------------------------
# Robot Control Worker Process
# ---------------------------------------------------------
def robot_worker_process(flexion_queue, robot_queue, running_event):
    controller = FingerAANController()
    simulator = RobotSimulator(render=False)
    
    last_render_time = 0
    current_flexion = None
    target_loop_time = 1.0 / 60.0 # Target 60Hz physics loop
    
    while running_event.is_set():
        loop_start_time = time.time()
        
        # Drain queue to ensure we process the most recent intention signal
        while not flexion_queue.empty():
            try:
                current_flexion = flexion_queue.get_nowait()
            except queue.Empty:
                break
                
        if current_flexion is not None:
            # Continuously tick the AAN controller to maintain stable time dynamics (dt)
            assisted_flexion, avg_alpha = controller.update(current_flexion)
            
            simulator.set_finger_angles(assisted_flexion)
            current_time = time.time()
            
            # Throttle camera rendering to ~30 FPS to reduce IPC overhead
            if current_time - last_render_time >= 0.033:
                sim_img_array = simulator.get_camera_image(width=400, height=300)
                if robot_queue.full():
                    try: robot_queue.get_nowait() 
                    except queue.Empty: pass
                try: robot_queue.put_nowait((sim_img_array, avg_alpha)) 
                except queue.Full: pass
                
                last_render_time = current_time
                
        # Sleep to maintain the target physics loop frequency
        sleep_time = target_loop_time - (time.time() - loop_start_time)
        if sleep_time > 0: time.sleep(sleep_time)
            
    simulator.disconnect()

# ---------------------------------------------------------
# Main UI Dashboard
# ---------------------------------------------------------
class RehabDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Hand Rehabilitation AAN (EMG Controlled)")
        self.root.geometry("1400x750") 
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Main layout structure
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel: Sensor Stream ---
        left_frame = ttk.LabelFrame(main_frame, text="12-Channel EMG Signal Streaming", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.sensor_label = tk.Label(left_frame, bg="black")
        self.sensor_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        control_frame = ttk.LabelFrame(left_frame, text="Simulate User Intention", padding=5)
        control_frame.pack(fill=tk.X)
        
        # Shared memory for UI-to-process intention overrides
        self.active_movement = mp.Value('i', 0)
        
        buttons = [
            ("Rest", 0), ("Thumb", 1), ("Index", 2), 
            ("Middle", 3), ("Ring", 4), ("Pinky", 5),
            ("Power Grip", 6), ("Power Sphere", 7), ("Tip Pinch", 8)
        ]
        
        # Split buttons into two rows to prevent UI clutter
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, pady=2)
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, pady=2)
        
        for i, (txt, val) in enumerate(buttons):
            target_frame = row1_frame if i < 5 else row2_frame
            btn = ttk.Button(target_frame, text=txt, command=lambda v=val: self.set_movement(v))
            btn.pack(side=tk.LEFT, padx=3, pady=2, expand=True, fill=tk.X)

        # --- Center Panel: Simulation ---
        center_frame = ttk.LabelFrame(main_frame, text="Robot Exoskeleton", padding=5)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sim_label = tk.Label(center_frame, bg="gray")
        self.sim_label.pack(fill=tk.BOTH, expand=True)

        # --- Right Panel: Dashboard Stats ---
        right_frame = ttk.LabelFrame(main_frame, text="Rehab Dashboard", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(right_frame, text="Status: Ready", font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        self.info_label = ttk.Label(right_frame, text="Intent Detection: LightGBM Model\nMultiprocessing Engine Active", foreground="gray")
        self.info_label.pack(pady=5)
        
        self.force_label = ttk.Label(right_frame, text="Robot Assist Force: 0.0%", font=('Arial', 12, 'bold'), foreground="#0055ff")
        self.force_label.pack(pady=(15, 5))
        
        self.force_bar = ttk.Progressbar(right_frame, orient='horizontal', length=300, mode='determinate')
        self.force_bar.pack(pady=5)
        self.force_bar['maximum'] = 100

        # Assist level (Alpha) tracking chart
        self.canvas_width = 320
        self.canvas_height = 220
        self.alpha_canvas = tk.Canvas(right_frame, width=self.canvas_width, height=self.canvas_height, bg="#f0f0f0", highlightthickness=1, highlightbackground="#cccccc")
        self.alpha_canvas.pack(pady=(10, 0), fill=tk.X)
        self.alpha_history = deque([1.0]*100, maxlen=100)

        # Process management and IPC queues
        self.running_event = mp.Event()
        self.running_event.set()
        
        self.sensor_queue = mp.Queue(maxsize=2)
        self.flexion_queue = mp.Queue(maxsize=2)
        self.robot_queue = mp.Queue(maxsize=2)

        self.latest_alpha = 1.0
        self.latest_flexion = None

        # Spawn background workers
        self.emg_process = mp.Process(target=emg_worker_process, args=(self.sensor_queue, self.flexion_queue, self.running_event, self.active_movement))
        self.robot_process = mp.Process(target=robot_worker_process, args=(self.flexion_queue, self.robot_queue, self.running_event))
        
        self.emg_process.start()
        self.robot_process.start()

        # Kick off the main GUI refresh loop
        self.update_ui_loop()

    def set_movement(self, val):
        self.active_movement.value = val

    def draw_canvas_chart(self):
        # Render a lightweight real-time chart for the alpha (assist) value
        self.alpha_canvas.delete("all")
        margin_top, margin_bottom, margin_x = 25, 15, 30
        plot_height = self.canvas_height - margin_top - margin_bottom
        
        # Draw background grid/labels
        for v in [0.0, 0.5, 1.0]:
            y_pos = self.canvas_height - margin_bottom - (v * plot_height)
            self.alpha_canvas.create_text(5, y_pos, text=f"{v}", anchor="w", fill="#888888", font=('Arial', 8))
            self.alpha_canvas.create_line(margin_x, y_pos, self.canvas_width, y_pos, fill="#e0e0e0", dash=(4, 4))
        
        self.alpha_canvas.create_text(margin_x, 5, text="Alpha (1=Patient, 0=Robot Assist)", anchor="nw", fill="#666666", font=('Arial', 9))
        
        # Project historical data points onto the canvas
        dx = (self.canvas_width - margin_x) / (len(self.alpha_history) - 1)
        points = []
        for i, val in enumerate(self.alpha_history):
            x = margin_x + (i * dx)
            y = self.canvas_height - margin_bottom - (val * plot_height)
            points.extend([x, y])
            
        if len(points) >= 4:
            self.alpha_canvas.create_line(points, fill="#0055ff", width=2, smooth=False)

    def update_ui_loop(self):
        # Non-blocking state fetch from worker processes
        try:
            emg_img, flexion = self.sensor_queue.get_nowait()
            self.latest_flexion = flexion
            pil_img = Image.fromarray(emg_img)
            self.tk_emg_img = ImageTk.PhotoImage(image=pil_img)
            self.sensor_label.configure(image=self.tk_emg_img)
        except queue.Empty:
            pass

        try:
            sim_frame, alpha = self.robot_queue.get_nowait()
            self.latest_alpha = alpha
            pil_sim = Image.fromarray(sim_frame)
            self.tk_sim_img = ImageTk.PhotoImage(image=pil_sim)
            self.sim_label.configure(image=self.tk_sim_img)
        except queue.Empty:
            pass

        # Update indicators based on the current robot assist level
        if self.latest_flexion is not None:
            force_percentage = (1.0 - self.latest_alpha) * 100.0
            self.force_label.config(text=f"Robot Assist Force: {force_percentage:.1f}%")
            self.force_bar['value'] = force_percentage
            
            if self.latest_alpha < 0.3:
                self.status_label.config(text="Status: High Robot Assistance!", foreground="red")
                self.force_label.config(foreground="red")
            elif self.latest_alpha < 0.8:
                self.status_label.config(text="Status: Partial Assist...", foreground="orange")
                self.force_label.config(foreground="orange")
            else:
                self.status_label.config(text="Status: Independent Movement", foreground="green")
                self.force_label.config(foreground="green")

        self.alpha_history.append(self.latest_alpha)
        self.draw_canvas_chart()
        
        # Re-schedule UI update (~60 FPS target)
        self.root.after(16, self.update_ui_loop)

    def on_close(self):
        # Graceful shutdown of background workers
        self.running_event.clear()
        self.emg_process.join(timeout=1.0)
        self.robot_process.join(timeout=1.0)
        
        # Force terminate if they hang
        if self.emg_process.is_alive(): self.emg_process.terminate()
        if self.robot_process.is_alive(): self.robot_process.terminate()
        
        self.root.destroy()

if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app = RehabDashboard(root)
    root.mainloop()