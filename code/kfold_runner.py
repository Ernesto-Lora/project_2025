import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time

# --- Your Custom Imports ---
from discriminativeBN import DiscriminativeBN
# Ensure your optimizer classes are importable here
from simplexMethod import NelderMeadOptimizer
# from hookeJeeves import HookeJeevesOptimizer 

# ==========================================
# 1. Prediction & Evaluation Helper Functions
# ==========================================

def predict_row(bn, tables, row):
    """
    Predicts the Class for a single row of data.
    """
    target = bn.target_col
    # Get outcomes, e.g., ['0', '1'] or ['nowin', 'won']
    target_outcomes = bn.var_info[target]['outcomes'] 
    
    best_class = None
    best_log_prob = -float('inf')
    
    # Try every possible class value
    for t_val in target_outcomes:
        # Create a temporary row with this class assumption
        temp_row = row.copy()
        temp_row[target] = t_val
        
        # Calculate Log Joint Probability for this assumption
        # Log P(Class | Features) proportional to Sum( Log P(Node | Parents) )
        current_log_prob = 0
        
        for node in bn.nodes:
            # 1. Identify parent config index
            p_idx = bn._get_parent_config_index(temp_row, node)
            
            # 2. Identify outcome index
            try:
                # value in the current row for this node
                val_str = str(temp_row[node]) 
                val_idx = bn.var_info[node]['outcomes'].index(val_str)
            except ValueError:
                # Fallback for unseen values
                val_idx = 0 
            
            # 3. Get probability
            prob = tables[node][p_idx, val_idx]
            current_log_prob += np.log(prob + 1e-10) # Avoid log(0)
            
        # Keep track of the best class
        if current_log_prob > best_log_prob:
            best_log_prob = current_log_prob
            best_class = t_val
            
    return best_class

def calculate_accuracy(bn, tables, df):
    """
    Iterates over the DataFrame, predicts, and returns accuracy.
    """
    y_true = df[bn.target_col].astype(str).tolist()
    y_pred = []
    
    # Convert to dict for faster iteration than iterrows
    records = df.to_dict('records')
    
    for row in records:
        prediction = predict_row(bn, tables, row)
        y_pred.append(prediction)
        
    acc = accuracy_score(y_true, y_pred)
    return acc

# ==========================================
# 2. K-Fold Validation Function
# ==========================================

def k_fold_validation(xml_file, csv_file, target_var, Optimizer_fun, max_iter, results_file="validation_log.txt"):
    
    # 1. Load Data and BN Structure
    df = pd.read_csv(csv_file)
    bn = DiscriminativeBN(xml_file, target_var)
    
    print(f"Starting 10-Fold Validation on {csv_file}...")
    print(f"Optimizer: {Optimizer_fun.__name__} | Max Iter: {max_iter}")
    
    # 2. Prepare K-Fold
    # shuffle=True is important to ensure random distribution
    kf = KFold(n_splits=10, shuffle=True, random_state=42) 
    accuracies = []

    # Clear/Create the results file
    with open(results_file, "w") as f:
        f.write(f"Results for {csv_file} using {Optimizer_fun.__name__}\n")
        f.write("------------------------------------------------\n")

    # 3. Iterate Folds
    fold_count = 1
    for train_index, test_index in kf.split(df):
        # Split Data
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        
        # --- Optimization Phase ---
        
        # Define Objective (Closure captures current df_train)
        def objective(betas):
            cll = bn.calculate_cll(betas, df_train)
            return -cll # Minimize negative CLL

        # Initialize Betas (Random restart per fold is usually good practice)
        np.random.seed(42 + fold_count) 
        start_betas = np.random.normal(0, 0.1, bn.total_params)
        
        # Run Optimizer
        optimizer = Optimizer_fun(objective, dim=bn.total_params, max_iter=max_iter)
        best_betas = optimizer.optimize(start_betas)
        
        # Convert betas to Probability Tables
        final_tables = bn.betas_to_probabilities(best_betas)
        
        # --- Evaluation Phase ---
        acc = calculate_accuracy(bn, final_tables, df_test)
        accuracies.append(acc)
        
        # Log Result
        result_str = f"{fold_count} fold {acc*100:.2f}%"
        print(result_str) # Print to console
        
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

