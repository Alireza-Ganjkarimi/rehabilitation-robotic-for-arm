import numpy as np
import time

class FingerAANController:
    # Kept as FingerAANController for backward compatibility with main.py
    def __init__(self,  effort_threshold=0.1, intent_threshold=0.10, cancel_threshold=0.1, alpha_decay=0.2, alpha_recovery=0.8):
        
        # Increased DOFs to 7
        self.dof = 7
        
        # Array mapping: 0-4 (Fingers, default base), 5 (Elbow, increased threshold), 6 (Wrist, tuned threshold)
        self.effort_thresholds = np.array([effort_threshold] * 5 + [effort_threshold +0.2, effort_threshold], dtype=float)
        self.intent_thresholds = np.array([intent_threshold] * 5 + [intent_threshold + 0.05, intent_threshold + 0.03], dtype=float)
        self.cancel_thresholds = np.array([cancel_threshold] * 5 + [cancel_threshold +0.1, cancel_threshold + 0.01], dtype=float)
        
        self.max_lead = 0.4
        self.alpha_decay = alpha_decay
        self.alpha_recovery = alpha_recovery
        
        self.alphas = np.ones(self.dof, dtype=float)
        self.dynamic_targets = np.zeros(self.dof, dtype=float)
        self.intent_state = np.zeros(self.dof, dtype=int) 
        
        self.prev_user_flex = None
        self.prev_time = None
        self.filtered_velocities = np.zeros(self.dof, dtype=float)
        self.velocity_filter_beta = 0.5

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

        assisted_flexion = np.zeros(self.dof)

        for i in range(self.dof): 
            v = self.filtered_velocities[i]
            c_flex = current_user_flex[i]

            # Phase 1: Intent Detection
            if v > self.intent_thresholds[i]:
                self.intent_state[i] = 1
            elif v < -0.5*self.intent_thresholds[i]:
                self.intent_state[i] = -1
                
            if self.intent_state[i] == 1 and v < -self.cancel_thresholds[i]:
                self.intent_state[i] = 0
            elif self.intent_state[i] == -1 and v > self.cancel_thresholds[i]:
                self.intent_state[i] = 0

            # Phase 2: Dynamic Trajectory Generation
            target_step = 0.2 * dt 
            
            if self.intent_state[i] == 1:
                desired_target = max(c_flex, self.dynamic_targets[i] + target_step)
                self.dynamic_targets[i] = min(1.0, min(desired_target, c_flex + self.max_lead))
                if c_flex >= 0.95: 
                    self.intent_state[i] = 0 
            elif self.intent_state[i] == -1:
                desired_target = min(c_flex, self.dynamic_targets[i] - target_step)
                self.dynamic_targets[i] = max(0.0, max(desired_target, c_flex - self.max_lead))
                if c_flex <= 0.15: 
                    self.intent_state[i] = 0
            else:
                self.dynamic_targets[i] += (c_flex - self.dynamic_targets[i]) * 4.0 * dt

            # Phase 3: Effort Evaluation
            dist_to_target = self.dynamic_targets[i] - c_flex
            
            if abs(dist_to_target) > 0.05 and self.intent_state[i] != 0:  
                is_moving_towards_target = (dist_to_target > 0 and v > 0) or (dist_to_target < 0 and v < 0)
                effective_velocity = abs(v) if is_moving_towards_target else 0.0

                if effective_velocity < self.effort_thresholds[i]:
                    self.alphas[i] -= self.alpha_decay * dt
                else:
                    self.alphas[i] += self.alpha_recovery * dt
            else:
                self.alphas[i] += self.alpha_recovery * dt
                
            self.alphas[i] = np.clip(self.alphas[i], 0.0, 1.0)
            assisted_flexion[i] = (self.alphas[i] * c_flex) + ((1.0 - self.alphas[i]) * self.dynamic_targets[i])

        self.prev_user_flex = current_user_flex
        self.prev_time = current_time

        return assisted_flexion, np.mean(self.alphas[5])

    def reset(self):
        self.alphas = np.ones(self.dof, dtype=float)
        self.intent_state = np.zeros(self.dof, dtype=int)
        self.prev_user_flex = None
        self.prev_time = None