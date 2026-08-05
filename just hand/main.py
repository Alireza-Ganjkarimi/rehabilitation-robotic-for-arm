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
from vision_tracker import VisionTracker
from aan_controller import FingerAANController
from simulator import RobotSimulator


def vision_worker_process(vision_queue, flexion_queue, running_event):
    """Worker process handling camera I/O and vision tracking."""
    cap = cv2.VideoCapture(0)
    vision = VisionTracker()
    
    while running_event.is_set():
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            processed_frame, fingers_flexion, _ = vision.process_frame(frame)
            
            rgb_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            fast_resized_cam = cv2.resize(rgb_img, (400, 300), interpolation=cv2.INTER_LINEAR)
            
            # Push processed frame to the UI queue (Drop old if full)
            if vision_queue.full():
                try: vision_queue.get_nowait() 
                except queue.Empty: pass
            try: vision_queue.put_nowait((fast_resized_cam, fingers_flexion))
            except queue.Full: pass
            
            # Forward kinematic data to the robot control process
            # ALWAYS drop the oldest data and keep the absolute newest frame
            if fingers_flexion is not None:
                if flexion_queue.full():
                    try: flexion_queue.get_nowait()
                    except queue.Empty: pass
                try: flexion_queue.put_nowait(fingers_flexion)
                except queue.Full: pass
                        
        time.sleep(0.01)
        
    cap.release()


def robot_worker_process(flexion_queue, robot_queue, running_event):
    """Worker process running the PyBullet physics engine and control algorithm."""
    controller = FingerAANController(effort_threshold=0.15, intent_threshold=0.15)
    simulator = RobotSimulator(render=False)
    
    last_render_time = 0
    current_flexion = None
    assisted_flexion = None # Keep track of the last known control output
    target_loop_time = 1.0 / 60.0 
    
    while running_event.is_set():
        loop_start_time = time.time()
        
        # Detect if we actually received a NEW frame from the camera ---
        new_data_received = False
        while not flexion_queue.empty():
            try:
                current_flexion = flexion_queue.get_nowait()
                new_data_received = True
            except queue.Empty:
                break
                
        if current_flexion is not None:
            
            # ONLY calculate velocity and intent if the physics loop received fresh data.
            # This prevents artificial velocity spikes caused by dt becoming 16ms for 0 distance.
            if new_data_received or assisted_flexion is None:
                assisted_flexion, avg_alpha = controller.update(current_flexion)
            
            # Physics runs at 60Hz safely, using the last calculated assisted_flexion
            simulator.set_finger_angles(assisted_flexion)
            
            current_time = time.time()
            
            if current_time - last_render_time >= 0.033:
                sim_img_array = simulator.get_camera_image(width=400, height=300)
                if robot_queue.full():
                    try: robot_queue.get_nowait() 
                    except queue.Empty: pass
                try: robot_queue.put_nowait((sim_img_array, avg_alpha)) 
                except queue.Full: pass
                
                last_render_time = current_time
                
        elapsed_loop_time = time.time() - loop_start_time
        sleep_time = target_loop_time - elapsed_loop_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    simulator.disconnect()


class RehabDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Hand Rehabilitation AAN (High-Performance Multiprocessing)")
        self.root.geometry("1400x750") 
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="Patient Hand Tracking", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        center_frame = ttk.LabelFrame(main_frame, text="Robot Exoskeleton", padding=5)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sim_label = tk.Label(center_frame)
        self.sim_label.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.LabelFrame(main_frame, text="Rehab Dashboard", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(right_frame, text="Status: Ready", font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        self.info_label = ttk.Label(right_frame, text="Intent Detection: Automatic\nMultiprocessing Engine Active", foreground="gray")
        self.info_label.pack(pady=5)
        
        self.force_label = ttk.Label(right_frame, text="Robot Assist Force: 0.0%", font=('Arial', 12, 'bold'), foreground="#0055ff")
        self.force_label.pack(pady=(15, 5))
        
        self.force_bar = ttk.Progressbar(right_frame, orient='horizontal', length=300, mode='determinate')
        self.force_bar.pack(pady=5)
        self.force_bar['maximum'] = 100

        self.canvas_width = 320
        self.canvas_height = 220
        self.alpha_canvas = tk.Canvas(right_frame, width=self.canvas_width, height=self.canvas_height, bg="#f0f0f0", highlightthickness=1, highlightbackground="#cccccc")
        self.alpha_canvas.pack(pady=(10, 0), fill=tk.X)
        self.alpha_history = deque([1.0]*100, maxlen=100)

        self.running_event = mp.Event()
        self.running_event.set()
        
        self.vision_queue = mp.Queue(maxsize=2)
        self.flexion_queue = mp.Queue(maxsize=2)
        self.robot_queue = mp.Queue(maxsize=2)

        self.latest_alpha = 1.0
        self.latest_flexion = None

        self.vision_process = mp.Process(target=vision_worker_process, args=(self.vision_queue, self.flexion_queue, self.running_event))
        self.robot_process = mp.Process(target=robot_worker_process, args=(self.flexion_queue, self.robot_queue, self.running_event))
        
        self.vision_process.start()
        self.robot_process.start()

        self.update_ui_loop()

    def draw_canvas_chart(self):
        self.alpha_canvas.delete("all")
        
        margin_top = 25
        margin_bottom = 15
        margin_x = 30
        plot_height = self.canvas_height - margin_top - margin_bottom
        
        for v in [0.0, 0.5, 1.0]:
            y_pos = self.canvas_height - margin_bottom - (v * plot_height)
            self.alpha_canvas.create_text(5, y_pos, text=f"{v}", anchor="w", fill="#888888", font=('Arial', 8))
            self.alpha_canvas.create_line(margin_x, y_pos, self.canvas_width, y_pos, fill="#e0e0e0", dash=(4, 4))
        
        self.alpha_canvas.create_text(margin_x, 5, text="Alpha (1=Patient, 0=Robot Assist)", anchor="nw", fill="#666666", font=('Arial', 9))
        
        dx = (self.canvas_width - margin_x) / (len(self.alpha_history) - 1)
        points = []
        
        for i, val in enumerate(self.alpha_history):
            x = margin_x + (i * dx)
            y = self.canvas_height - margin_bottom - (val * plot_height)
            points.extend([x, y])
            
        if len(points) >= 4:
            self.alpha_canvas.create_line(points, fill="#0055ff", width=2, smooth=False)

    def update_ui_loop(self):
        try:
            cam_frame, flexion = self.vision_queue.get_nowait()
            self.latest_flexion = flexion
            pil_img = Image.fromarray(cam_frame)
            self.tk_cam_img = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=self.tk_cam_img)
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
        else:
            self.status_label.config(text="Status: Hand not detected!", foreground="black")
            self.force_label.config(text="Robot Assist Force: 0.0%", foreground="black")
            self.force_bar['value'] = 0

        self.alpha_history.append(self.latest_alpha)
        self.draw_canvas_chart()

        self.root.after(16, self.update_ui_loop)

    def on_close(self):
        self.running_event.clear()
        
        self.vision_process.join(timeout=1.0)
        self.robot_process.join(timeout=1.0)
        
        if self.vision_process.is_alive():
            self.vision_process.terminate()
        if self.robot_process.is_alive():
            self.robot_process.terminate()
            
        self.root.destroy()

if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app = RehabDashboard(root)
    root.mainloop()