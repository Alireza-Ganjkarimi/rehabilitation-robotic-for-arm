import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class RobotSimulator:
    def __init__(self, render=False, dt=1.0/60.0):
        self.dt = dt
        self.render = render
        
        # Init PyBullet (DIRECT mode)
        self.physicsClient = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Set sim time step
        p.setTimeStep(self.dt)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "realhand_l20_right.urdf")
        
        # Load URDF using absolute path
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        
        # Map joint names to indices
        self.joint_map = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            self.joint_map[joint_name] = i
            
        # Init real-time tracker for catch-up logic
        self.last_sim_time = time.time()
            
    def set_finger_angles(self, finger_flexion):
        thumb, index, middle, ring, pinky = finger_flexion
        
        target_angles = {
            'thumb_mcp': thumb * 1.05, 
            'thumb_cmc_pitch': thumb * 0.79,
            'index_mcp_pitch': index * 1.4,
            'index_pip': index * 1.57,
            'middle_mcp_pitch': middle * 1.4,
            'middle_pip': middle * 1.57,
            'ring_mcp_pitch': ring * 1.4,
            'ring_pip': ring * 1.57,
            'pinky_mcp_pitch': pinky * 1.4,
            'pinky_pip': pinky * 1.57
        }
        
        # Apply position control
        for joint_name, angle in target_angles.items():
            if joint_name in self.joint_map:
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id, 
                    jointIndex=self.joint_map[joint_name], 
                    controlMode=p.POSITION_CONTROL, 
                    targetPosition=angle,
                    force=1,      # Max torque (N.m)
                    maxVelocity=3.0 # Max velocity (rad/s)
                )
                
        # --- Catch-up logic ---
        current_time = time.time()
        elapsed_real_time = current_time - self.last_sim_time
        self.last_sim_time = current_time
        
        # Calc required sim steps
        num_steps = int(elapsed_real_time / self.dt)
        
        # Cap steps to prevent spiral of death
        num_steps = min(num_steps, 10) 
        
        # Sync physics with real-time
        if num_steps == 0:
            # Force at least one step
            p.stepSimulation()
        else:
            for _ in range(num_steps):
                p.stepSimulation()

    def disconnect(self):
        p.disconnect(self.physicsClient)
        
    def get_camera_image(self, width=400, height=300):
        # Configure camera view
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.1],  
            distance=0.4,                      
            yaw=60,                            
            pitch=-20, 
            roll=0, 
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width)/height, nearVal=0.01, farVal=10.0)
        
        # Render via OpenGL
        (_, _, px, _, _) = p.getCameraImage(
            width, height, 
            viewMatrix=view_matrix, 
            projectionMatrix=proj_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract RGB array
        rgb_array = np.reshape(np.array(px, dtype=np.uint8), (height, width, 4))[:, :, :3]
        return rgb_array