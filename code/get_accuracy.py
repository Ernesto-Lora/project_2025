import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def predict_row(bn, tables, row):
    """
    Predicts the Class for a single row of data.
    bn: The DiscriminativeBN object (needed for structure/parents)
    tables: The dictionary of learned probabilities (final_tables)
    row: A dictionary or pandas Series containing the features (x1, x2, ...)
    """
    target = bn.target_col
    target_outcomes = bn.var_info[target]['outcomes'] # e.g., ['f', 't'] or ['nowin', 'won']
    
    best_class = None
    best_log_prob = -float('inf')
    
    # Try every possible class value
    for t_val in target_outcomes:
        # Create a temporary row with this class assumption
        temp_row = row.copy()
        temp_row[target] = t_val
        
        # Calculate Log Joint Probability for this assumption
        # Log P(All Variables) = Sum( Log P(Node | Parents) )
        current_log_prob = 0
        
        for node in bn.nodes:
            # 1. Identify which row in the probability table to look at (based on parents)
            p_idx = bn._get_parent_config_index(temp_row, node)
            
            # 2. Identify which column (outcome) to look at
            # We must handle cases where test data has values not seen in training
            try:
                val_idx = bn.var_info[node]['outcomes'].index(str(temp_row[node]))
            except ValueError:
                # Fallback for unseen values (rare, but good for safety)
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
    Iterates over the entire DataFrame, predicts, and returns accuracy.
    """
    y_true = df[bn.target_col].astype(str).tolist()
    y_pred = []
    
    # Iterate over every row
    records = df.to_dict('records')
    for row in records:
        prediction = predict_row(bn, tables, row)
        y_pred.append(prediction)
        
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=bn.var_info[bn.target_col]['outcomes'])
    
    return acc, cm, y_pred

# ==========================================
# 3. Usage Example
# ==========================================

# Assuming you ran the optimization from the previous step:
# bn = DiscriminativeBN(...)
# best_betas = ... from optimizer
# final_tables = bn.betas_to_probabilities(best_betas)
# df = pd.read_csv("chess_data.csv")

def get_accuracy(bn, final_tables, df):
    print("\n--- Starting Evaluation ---")

    # Run prediction
    accuracy, conf_matrix, predictions = calculate_accuracy(bn, final_tables, df)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Example of a single prediction comparison
    print(f"\nRow 0 Actual: {df.iloc[0][bn.target_col]}")
    print(f"Row 0 Pred  : {predictions[0]}")