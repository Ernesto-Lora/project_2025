import numpy as np
from scipy.optimize import minimize
import pandas as pd

class LBFGSOptimizerWrapper:
    """
    Wraps the highly efficient L-BFGS-B algorithm from SciPy.
    Requires both the objective function (CLL) and its analytical gradient.
    """
    def __init__(self, func, gradient_func, dim, max_iter=1000, bounds=None, ftol=0.01, gtol=0.01):
        # The objective function (to MINIMIZE, typically -CLL)
        self.func = func
        # The function that computes the analytical gradient (vector of partial derivatives)
        self.gradient_func = gradient_func 
        self.dim = dim
        self.max_iter = max_iter
        self.bounds = bounds 
        
        # --- TOLERANCE PARAMETERS ---
        # ftol: Stops if (f^k - f^{k+1}) / max{|f^k|,|f^{k+1}|,1} <= ftol
        self.ftol = ftol  
        # gtol: Stops if the projected gradient norm < gtol (Slope is effectively zero)
        self.gtol = gtol

    def optimize(self, start_point):
        print(f"--- Starting L-BFGS Optimization (Dim: {self.dim}) ---")
        
        # 1. Initial Evaluation
        initial_score = self.func(start_point)
        print(f"Initial Best Score: {-initial_score:.4f} (CLL)")
        print("-" * 30)

        # 2. Perform Optimization using SciPy's minimize
        result = minimize(
            fun=self.func,              # The objective function
            x0=start_point,             # The starting parameter vector
            method='L-BFGS-B',          
            jac=self.gradient_func,     # Analytical gradient
            bounds=self.bounds,         
            options={
                'maxiter': self.max_iter, 
                'disp': True,           # Print convergence messages from Fortran backend
                'ftol': self.ftol,      # <--- Explicit Function Tolerance
                'gtol': self.gtol       # <--- Explicit Gradient Tolerance
            } 
        )
        
        # 3. Output Results
        print("-" * 30)
        
        if result.success:
            print("Optimization Success!")
        else:
            print(f"Optimization Finished. Reason: {result.message}")

        final_score = result.fun
        print(f"Total Iterations: {result.nit}")
        print(f"Final Best Score: {-final_score:.4f} (CLL)")
        
        # Return the optimized parameter vector
        return result.x