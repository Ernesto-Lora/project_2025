# Settings
import numpy as np
import pandas as pd
from pgmpy.readwrite import XMLBIFReader
import random
import copy
from discriminativeBN import DiscriminativeBN
from simplexMethod import NelderMeadOptimizer
from hookeJeeves import HookeJeevesOptimizer
import pickle

def get_params(xml_file, csv_file, target_var, Optimizer_fun,
                max_iter,
               params_file_name, betas_file_name ):
    ## xml_file = "models/chessTAN.xml" 
    ## csv_file = "datasets/chess_data_3196.csv" 
    ## target_var = "Class"        

    # 1. Load BN Wrapper
    # Note: Ensure your XML and CSV have matching column names/outcomes
    bn = DiscriminativeBN(xml_file, target_var)

    # 2. Load Data
    df = pd.read_csv(csv_file)

    # 3. Define Objective Function
    # We want to Maximize CLL, but Optimizer Minimizes. So minimize Negative CLL.
    def objective(betas):
        cll = bn.calculate_cll(betas, df)
        return -cll # Negative because we minimize

    # 4. Initialize Parameters (Betas)
    # Start with random small values close to 0 (implies roughly uniform probs)
    np.random.seed(42)
    start_betas = np.random.normal(0, 0.1, bn.total_params)

    print(f"Starting Optimization with {bn.total_params} parameters...")

    # 5. Run optimizer
    optimizer = Optimizer_fun(objective, dim=bn.total_params, max_iter=max_iter) # Low iters for demo
    best_betas = optimizer.optimize(start_betas)

    # 6. Result
    print("\nOptimization Complete.")
    final_cll = bn.calculate_cll(best_betas, df)
    print(f"Final Conditional Log Likelihood: {final_cll:.4f}")

    # Show one resulting table
    final_tables = bn.betas_to_probabilities(best_betas)
    print(f"\nLearned Table for {target_var}:")
    print(final_tables[target_var])

    ## params_file_name = "final_tablesChessTAN_hooke.pkl"
    ## betas_file_name = "final_betasChessTAN_hooke.pkl"
    with open(params_file_name, "wb") as f:
        pickle.dump(final_tables, f)

    with open(betas_file_name, "wb") as f:
        pickle.dump(best_betas, f)