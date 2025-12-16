import numpy as np
import pandas as pd


import numpy as np

class HookeJeevesOptimizer:
    def __init__(self, func, dim, max_iter=20, step_size=0.05, step_decay=0.5, min_step=1e-6, ftol=1e-1):
        self.func = func       # The objective function (to MINIMIZE)
        self.dim = dim
        self.max_iter = max_iter
        
        # Hooke-Jeeves specific parameters
        self.step_size = step_size  # Initial step length
        self.step_decay = step_decay # Factor to reduce step size
        self.min_step = min_step     # Tolerance for step size (Grid precision)
        self.ftol = ftol             # Tolerance for function value improvement

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
            # Track score at the start of the iteration to check improvement later
            score_start_of_iter = best_score
            
            # --- CONVERGENCE CONDITION 1: Step Size ---
            # If the grid is too fine, we assume we are at the minimum.
            if current_step < self.min_step:
                print(f"CONVERGENCE ACHIEVED: Step size ({current_step:.7f}) < Min Step ({self.min_step})")
                break


            # --- 1. Exploratory Move from Base Point ---
            new_point, new_score, improved = self._exploratory_move(base_point, current_step, best_score)

            if improved:
                # --- 2. Pattern Move (Acceleration) ---
                # Formula: P = New + (New - Old)
                pattern_point = new_point + (new_point - prev_base_point)
                
                pat_expl_point, pat_expl_score, pat_success = self._exploratory_move(
                    pattern_point, current_step, self.func(pattern_point)
                )

                if pat_expl_score < new_score:
                    prev_base_point = np.copy(base_point) 
                    base_point = pat_expl_point           
                    best_score = pat_expl_score
                else:
                    prev_base_point = np.copy(base_point)
                    base_point = new_point
                    best_score = new_score

                # --- CONVERGENCE CONDITION 2: Function Value Tolerance ---
                # If we improved, but the improvement is tiny, we might be done.
                score_improvement = abs(score_start_of_iter - best_score)
                if score_improvement < self.ftol:
                    print(f"\nCONVERGENCE ACHIEVED: Improvement ({score_improvement:.6f}) < ftol ({self.ftol})")
                    break

            else:
                # --- 3. Step Reduction ---
                current_step *= self.step_decay
                # We do NOT check ftol here because score didn't change (diff is 0),
                # but we are not converged yet; we just need a finer grid.

        print("-" * 30)
        print(f"Optimization Finished. Best Score found: {-best_score:.4f}")
        return base_point