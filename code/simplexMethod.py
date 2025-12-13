import numpy as np
import pandas as pd

# ==========================================
# 2. The Simplex (Nelder-Mead) Optimizer 
# ==========================================
class NelderMeadOptimizer:
    def __init__(self, func, dim, max_iter=5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.func = func # The objective function (to MINIMIZE)
        self.dim = dim
        self.max_iter = max_iter
        # Standard coefficients
        self.alpha = alpha # Reflection
        self.gamma = gamma # Expansion
        self.rho = rho     # Contraction
        self.sigma = sigma # Shrink

    def optimize(self, start_point):
        print(f"--- Starting Simplex Optimization (Dim: {self.dim}) ---")
        
        # 1. Initialize Simplex: N+1 points
        simplex = [start_point]
        for i in range(self.dim):
            point = np.copy(start_point)
            # Perturb one dimension significantly enough to create a volume
            point[i] = point[i] + 0.05 
            simplex.append(point)
        
        # Evaluate all points initially
        scores = [(self.func(p), p) for p in simplex]
        scores.sort(key=lambda x: x[0])
        
        print(f"Initial Best Score: {-scores[0][0]:.4f} (CLL)")
        print(f"Initial Worst Score: {-scores[-1][0]:.4f} (CLL)")
        print("-" * 30)

        for it in range(self.max_iter):
            # Sort: Best (lowest score) to Worst
            scores.sort(key=lambda x: x[0])
            
            best_score, best_point = scores[0]
            worst_score, worst_point = scores[-1]
            second_worst_score = scores[-2][0]
            
            # Print status at start of iteration
            print(f"Iter {it+1}: Best: {-best_score:.4f} | Worst: {-worst_score:.4f}", end=" | ")

            # Calculate Centroid of all except worst
            points_matrix = np.array([x[1] for x in scores[:-1]])
            centroid = np.mean(points_matrix, axis=0)
            
            # --- Attempt Reflection ---
            xr = centroid + self.alpha * (centroid - worst_point)
            r_score = self.func(xr)
            
            if best_score <= r_score < second_worst_score:
                scores[-1] = (r_score, xr)
                print(f"Action: REFLECTION (Accepted). New Score: {-r_score:.4f}")
                continue

            # --- Attempt Expansion ---
            if r_score < best_score:
                xe = centroid + self.gamma * (xr - centroid)
                e_score = self.func(xe)
                
                if e_score < r_score:
                    scores[-1] = (e_score, xe)
                    print(f"Action: EXPANSION (Accepted). New Score: {-e_score:.4f}")
                else:
                    scores[-1] = (r_score, xr)
                    print(f"Action: EXPANSION (Reverted to Reflection). New Score: {-r_score:.4f}")
                continue
                
            # --- Attempt Contraction ---
            # If reflection was worse than the second worst point, try contracting
            xc = centroid + self.rho * (worst_point - centroid)
            c_score = self.func(xc)
            
            if c_score < worst_score:
                scores[-1] = (c_score, xc)
                print(f"Action: CONTRACTION (Accepted). New Score: {-c_score:.4f}")
                continue
            
            # --- Shrink ---
            # If all else fails, shrink the whole simplex towards the best point
            print(f"Action: SHRINK (Simplex Reduction)")
            new_scores = [(scores[0][0], scores[0][1])]
            for i in range(1, len(scores)):
                p = scores[0][1] + self.sigma * (scores[i][1] - scores[0][1])
                new_scores.append((self.func(p), p))
            scores = new_scores
            
        print("-" * 30)
        print(f"Optimization Finished. Best Score found: {-scores[0][0]:.4f}")
        return scores[0][1] # Return best point