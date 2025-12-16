import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time

# --- Your Custom Imports ---
from discriminativeBN import DiscriminativeBN
# Import your L-BFGS optimizer wrapper here
# from lbfgsMethod import LbfgsOptimizer (Example)

# ==========================================
# 1. Prediction & Evaluation Helper Functions
# ==========================================

def predict_row(bn, tables, row):
    """
    Predicts the Class for a single row of data.
    """
    target = bn.target_col
    target_outcomes = bn.var_info[target]['outcomes'] 
    
    best_class = None
    best_log_prob = -float('inf')
    
    for t_val in target_outcomes:
        temp_row = row.copy()
        temp_row[target] = t_val
        
        current_log_prob = 0
        
        for node in bn.nodes:
            p_idx = bn._get_parent_config_index(temp_row, node)
            try:
                val_str = str(temp_row[node]) 
                val_idx = bn.var_info[node]['outcomes'].index(val_str)
            except ValueError:
                val_idx = 0 
            
            prob = tables[node][p_idx, val_idx]
            current_log_prob += np.log(prob + 1e-10)
            
        if current_log_prob > best_log_prob:
            best_log_prob = current_log_prob
            best_class = t_val
            
    return best_class

def calculate_accuracy(bn, tables, df):
    y_true = df[bn.target_col].astype(str).tolist()
    y_pred = []
    
    records = df.to_dict('records')
    for row in records:
        prediction = predict_row(bn, tables, row)
        y_pred.append(prediction)
        
    acc = accuracy_score(y_true, y_pred)
    return acc

# ==========================================
# 2. K-Fold Validation Function (Gradient Version)
# ==========================================

def k_fold_validation_gradient(xml_file, csv_file, target_var, Optimizer_fun, max_iter, results_file="validation_log_lbfgs.txt"):
    
    # 1. Load Data and BN Structure
    # Note: Ensure XML and CSV match
    df = pd.read_csv(csv_file)
    bn = DiscriminativeBN(xml_file, target_var)
    
    print(f"Starting 10-Fold Validation (Gradient-Based) on {csv_file}...")
    print(f"Optimizer: {Optimizer_fun.__name__} | Max Iter: {max_iter}")
    
    # 2. Prepare K-Fold
    kf = KFold(n_splits=10, shuffle=True, random_state=42) 
    accuracies = []

    # Initialize Log File
    with open(results_file, "w") as f:
        f.write(f"Gradient-Based Results for {csv_file} using {Optimizer_fun.__name__}\n")
        f.write("------------------------------------------------\n")

    # 3. Iterate Folds
    fold_count = 1
    for train_index, test_index in kf.split(df):
        # Split Data
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        
        # --- Optimization Phase ---
        
        # Define Objective: Minimize Negative CLL
        def objective(betas):
            cll = bn.calculate_cll(betas, df_train)
            return -cll 

        # Define Gradient Function
        # Note: Depending on your Optimizer implementation, you might need 
        # to return the negative gradient here if it expects the gradient of the objective.
        # Based on your snippet, we pass it directly:
        gradient_func = lambda betas: bn.calculate_gradient(betas, df_train)

        # Initialize Parameters
        np.random.seed(42 + fold_count)
        start_betas = np.random.normal(0, 0.1, bn.total_params)
        
        # Run Optimizer (passing gradient_func)
        optimizer = Optimizer_fun(objective, gradient_func, dim=bn.total_params, max_iter=max_iter)
        best_betas = optimizer.optimize(start_betas)
        
        # Convert betas to Probability Tables
        final_tables = bn.betas_to_probabilities(best_betas)
        
        # --- Evaluation Phase ---
        acc = calculate_accuracy(bn, final_tables, df_test)
        accuracies.append(acc)
        
        # Log Result
        result_str = f"{fold_count} fold {acc*100:.2f}%"
        print(result_str)
        
        with open(results_file, "a") as f:
            f.write(result_str + "\n")
            
        fold_count += 1

    # 4. Final Average
    avg_acc = np.mean(accuracies)
    final_str = f"average {avg_acc*100:.2f}%"
    
    print("\n" + final_str)
    with open(results_file, "a") as f:
        f.write("------------------------------------------------\n")
        f.write(final_str + "\n")

# ==========================================
# 3. Usage Call
# ==========================================

if __name__ == "__main__":
    start_time = time.time()

    # Make sure to import your LBFGS optimizer class and pass it here
    # from lbfgsMethod import LbfgsOptimizer
    
    # k_fold_validation_gradient(
    #    xml_file="models/australianTAN.xml",
    #    csv_file="datasets/AustralianDisc.csv",
    #    target_var="A15",
    #    Optimizer_fun=LbfgsOptimizer,  # <--- Your LBFGS Class here
    #    max_iter=100,
    #    results_file="results_AustralianTAN_LBFGS.txt"
    # )

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nElapsed time: {elapsed_minutes:.2f} minutes")