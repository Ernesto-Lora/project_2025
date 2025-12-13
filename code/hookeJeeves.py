import numpy as np
import pandas as pd

# ==========================================
# 2. The Hooke-Jeeves Pattern Search Optimizer
# ==========================================
class HookeJeevesOptimizer:
    def __init__(self, func, dim, max_iter=20, step_size=0.05, step_decay=0.5, min_step=1e-6):
        self.func = func       # The objective function (to MINIMIZE)
        self.dim = dim
        self.max_iter = max_iter
        
        # Hooke-Jeeves specific parameters
        self.step_size = step_size  # Initial step length
        self.step_decay = step_decay # Factor to reduce step size (alpha in some texts)
        self.min_step = min_step     # Convergence criteria for step size

    def _exploratory_move(self, base_point, current_step_size, current_score):
        """
        Explores the neighborhood of base_point along each axis.
        Returns: (new_point, new_score, success_flag)
        """
        point = np.copy(base_point)
        best_score = current_score
        improved = False

        # Iterate through every dimension (parameter)
        for i in range(self.dim):
            # 1. Try moving forward (+ step)
            old_val = point[i]
            point[i] = old_val + current_step_size
            score = self.func(point)

            if score < best_score:
                best_score = score
                improved = True
            else:
                # 2. If forward didn't work, try moving backward (- step)
                point[i] = old_val - current_step_size
                score = self.func(point)
                
                if score < best_score:
                    best_score = score
                    improved = True
                else:
                    # 3. If neither worked, reset this dimension
                    point[i] = old_val
        
        return point, best_score, improved

    def optimize(self, start_point):
        print(f"--- Starting Hooke-Jeeves Optimization (Dim: {self.dim}) ---")
        
        # Initialize points
        base_point = np.copy(start_point)
        best_score = self.func(base_point)
        
        # Previous base point (used for Pattern Move calculation)
        prev_base_point = np.copy(base_point)
        
        current_step = self.step_size

        print(f"Initial Score: {-best_score:.4f} (CLL)")
        print("-" * 30)

        for it in range(self.max_iter):
            # Check for convergence based on step size
            if current_step < self.min_step:
                print("Step size too small. Converged.")
                break

            print(f"Iter {it+1}: Best: {-best_score:.4f} | Step: {current_step:.5f}", end=" | ")

            # --- 1. Exploratory Move from Base Point ---
            new_point, new_score, improved = self._exploratory_move(base_point, current_step, best_score)

            if improved:
                # --- 2. Pattern Move (Acceleration) ---
                # We improved, so let's try to jump further in this direction
                # Formula: P = New + (New - Old)
                pattern_point = new_point + (new_point - prev_base_point)
                
                # We must evaluate the pattern point itself or do an exploration around it.
                # Standard HJ does an exploration *around* the pattern point.
                pat_expl_point, pat_expl_score, pat_success = self._exploratory_move(
                    pattern_point, current_step, self.func(pattern_point) # We treat pattern point as temporary base
                )

                if pat_expl_score < new_score:
                    # Pattern Move Succeeded significantly!
                    print(f"Action: PATTERN MOVE (Accepted). New Score: {-pat_expl_score:.4f}")
                    prev_base_point = np.copy(base_point) # Update history
                    base_point = pat_expl_point           # Update current best
                    best_score = pat_expl_score
                else:
                    # Pattern move failed, but the initial exploration was good.
                    # Just take the result of the first exploration.
                    print(f"Action: EXPLORATION (Accepted). New Score: {-new_score:.4f}")
                    prev_base_point = np.copy(base_point)
                    base_point = new_point
                    best_score = new_score

            else:
                # --- 3. Step Reduction ---
                # Exploration failed to find a better point around current base.
                # We are likely in a local valley, so we refine the search grid.
                current_step *= self.step_decay
                print(f"Action: STEP REDUCTION. New Step: {current_step:.5f}")
                
                # Note: In standard HJ, we do NOT update the base_point here. 
                # We stay put and try again with smaller steps.

        print("-" * 30)
        print(f"Optimization Finished. Best Score found: {-best_score:.4f}")
        return base_point