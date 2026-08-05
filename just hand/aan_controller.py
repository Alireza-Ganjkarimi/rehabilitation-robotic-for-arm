import numpy as np
import time

class FingerAANController:
    def __init__(self, effort_threshold=0.05, intent_threshold=0.10, cancel_threshold=0.1, alpha_decay=0.4, alpha_recovery=0.8):
        self.effort_threshold = effort_threshold 
        self.intent_threshold = intent_threshold 
        
        # Velocity threshold in the opposite direction indicating intent to brake/stop
        self.cancel_threshold = cancel_threshold 
        # Maximum distance the virtual target can lead the user's hand (virtual rubber band effect)
        self.max_lead = 0.4
        
        self.alpha_decay = alpha_decay
        self.alpha_recovery = alpha_recovery
        
        self.alphas = np.ones(5, dtype=float)
        self.dynamic_targets = np.zeros(5, dtype=float)
        self.intent_state = np.zeros(5, dtype=int) 
        
        self.prev_user_flex = None
        self.prev_time = None
        self.filtered_velocities = np.zeros(5, dtype=float)
        self.velocity_filter_beta = 0.4

    def update(self, current_user_flex):
        current_user_flex = np.array(current_user_flex, dtype=float)
        current_time = time.time()

        if self.prev_user_flex is None:
            self.prev_user_flex = current_user_flex
            self.dynamic_targets = current_user_flex.copy()
            self.prev_time = current_time
            return current_user_flex, np.mean(self.alphas) 

        dt = current_time - self.prev_time
        if dt <= 0: dt = 1e-4  

        raw_velocities = (current_user_flex - self.prev_user_flex) / dt
        self.filtered_velocities = (self.velocity_filter_beta * self.filtered_velocities) + \
                                   ((1 - self.velocity_filter_beta) * raw_velocities)

        assisted_flexion = np.zeros(5)

        for i in range(5):
            v = self.filtered_velocities[i]
            c_flex = current_user_flex[i]

            # --- 1. Intent Detection and Active Braking ---
            if v > self.intent_threshold:
                self.intent_state[i] = 1
            elif v < -self.intent_threshold:
                self.intent_state[i] = -1
                
            # Cancel intent if the user applies force in the opposite direction (braking)
            if self.intent_state[i] == 1 and v < -self.cancel_threshold:
                self.intent_state[i] = 0
            elif self.intent_state[i] == -1 and v > self.cancel_threshold:
                self.intent_state[i] = 0

            # --- 2. Trajectory Generation (Virtual Rubber Band) ---
            target_step = 0.15 * dt 
            
            if self.intent_state[i] == 1:
                # Step 1: Ensure the virtual target does not fall behind the user's hand
                desired_target = max(c_flex, self.dynamic_targets[i] + target_step)
                
                # Step 2: Apply the virtual rubber band constraint and physical upper limit
                self.dynamic_targets[i] = min(1.0, min(desired_target, c_flex + self.max_lead))
                
                if c_flex >= 0.95: 
                    self.intent_state[i] = 0 
                    
            elif self.intent_state[i] == -1:
                # Step 1: Ensure the virtual target does not lag behind during extension
                desired_target = min(c_flex, self.dynamic_targets[i] - target_step)
                
                # Step 2: Apply the virtual rubber band constraint and physical lower limit
                self.dynamic_targets[i] = max(0.0, max(desired_target, c_flex - self.max_lead))
                
                if c_flex <= 0.4: 
                    self.intent_state[i] = 0
                    
            else:
                # Soft stop mechanism
                self.dynamic_targets[i] += (c_flex - self.dynamic_targets[i]) * 4.0 * dt

            # --- 3. User Effort Evaluation ---
            dist_to_target = self.dynamic_targets[i] - c_flex
            
            if abs(dist_to_target) > 0.05 and self.intent_state[i] != 0:  
                is_moving_towards_target = (dist_to_target > 0 and v > 0) or (dist_to_target < 0 and v < 0)
                effective_velocity = abs(v) if is_moving_towards_target else 0.0

                if effective_velocity < self.effort_threshold:
                    # User is struggling; increase assistance
                    self.alphas[i] -= self.alpha_decay * dt
                else:
                    self.alphas[i] += self.alpha_recovery * dt
            else:
                self.alphas[i] += self.alpha_recovery * dt
                
            self.alphas[i] = np.clip(self.alphas[i], 0.0, 1.0)

            # --- 4. Final Blending ---
            assisted_flexion[i] = (self.alphas[i] * c_flex) + ((1.0 - self.alphas[i]) * self.dynamic_targets[i])

        self.prev_user_flex = current_user_flex
        self.prev_time = current_time

        return assisted_flexion, np.mean(self.alphas)

    def reset(self):
        self.alphas = np.ones(5, dtype=float)
        self.intent_state = np.zeros(5, dtype=int)
        self.prev_user_flex = None
        self.prev_time = None