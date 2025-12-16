import numpy as np
import pandas as pd
from pgmpy.readwrite import XMLBIFReader

class DiscriminativeBN:
    def __init__(self, xml_path, target_col):
        """
        xml_path: Path to the XMLBIF file
        target_col: The name of the variable we want to predict (e.g., 'Class')
        """
        self.target_col = target_col
        
        # 1. Parse Structure
        reader = XMLBIFReader(xml_path)
        self.model = reader.get_model()
        self.nodes = list(self.model.nodes())
        
        # 2. Extract Cardinality (Outcomes) & Parents
        self.var_info = {} 
        self.node_offsets = {} # NEW: Maps node name to start index in beta vector
        
        self.total_params = 0
        
        for node in self.nodes:
            cpd = self.model.get_cpds(node)
            outcomes = cpd.state_names[node]
            parents = self.model.get_parents(node)
            
            # Dimension of parent configurations
            parent_cards = [len(self.model.get_cpds(p).state_names[p]) for p in parents]
            total_parent_configs = np.prod(parent_cards) if parent_cards else 1
            
            self.var_info[node] = {
                'parents': parents,
                'outcomes': outcomes,
                'cardinality': len(outcomes),
                'parent_cardinalities': parent_cards,
                'num_parent_configs': int(total_parent_configs)
            }
            
            # Save the starting index for this node's parameters
            self.node_offsets[node] = self.total_params
            
            # We need (Num_Parent_Configs * Num_Outcomes) weights per node
            self.total_params += int(total_parent_configs * len(outcomes))

    def _get_parent_config_index(self, row, node):
        """Helper to find the index of the parent configuration"""
        parents = self.var_info[node]['parents']
        if not parents:
            return 0
        
        idx = 0
        stride = 1
        for p in reversed(parents):
            p_val = row[p]
            p_outcomes = self.var_info[p]['outcomes']
            # Ensure safe lookup
            try:
                p_idx = p_outcomes.index(str(p_val))
            except ValueError:
                # Fallback if data format differs slightly
                p_idx = 0 
            idx += p_idx * stride
            stride *= len(p_outcomes)
        return idx

    def betas_to_probabilities(self, betas):
        """Greiner Transformation: Beta -> Softmax -> Theta"""
        tables = {}
        
        for node in self.nodes:
            info = self.var_info[node]
            start_idx = self.node_offsets[node]
            n_rows = info['num_parent_configs']
            n_cols = info['cardinality']
            
            # Slice beta vector using cached offsets
            node_betas = betas[start_idx : start_idx + (n_rows * n_cols)]
            
            # Reshape to (Parent_Configs, Outcomes)
            beta_matrix = node_betas.reshape((n_rows, n_cols))
            
            # Softmax row-wise (axis 1)
            max_b = np.max(beta_matrix, axis=1, keepdims=True) # Stability
            exp_b = np.exp(beta_matrix - max_b)
            probs = exp_b / np.sum(exp_b, axis=1, keepdims=True)
            
            tables[node] = probs
            
        return tables

    def calculate_cll(self, betas, data):
        """Calculates Sum of Conditional Log Likelihood: P(Target | Evidence)"""
        tables = self.betas_to_probabilities(betas)
        log_likelihood_sum = 0
        records = data.to_dict('records')
        target_outcomes = self.var_info[self.target_col]['outcomes']
        
        for row in records:
            # 1. Joint of ACTUAL data
            log_joint_actual = 0
            for node in self.nodes:
                p_idx = self._get_parent_config_index(row, node)
                val_idx = self.var_info[node]['outcomes'].index(str(row[node]))
                prob = tables[node][p_idx, val_idx]
                log_joint_actual += np.log(prob + 1e-10)
            
            # 2. Marginal of Evidence (Sum over Target)
            marginal_sum = 0
            temp_row = row.copy()
            
            # Log-Sum-Exp Trick for stability could be used here, but keeping standard for now
            for t_val in target_outcomes:
                temp_row[self.target_col] = t_val
                current_log_joint = 0
                for node in self.nodes:
                    p_idx = self._get_parent_config_index(temp_row, node)
                    val_idx = self.var_info[node]['outcomes'].index(str(temp_row[node]))
                    prob = tables[node][p_idx, val_idx]
                    current_log_joint += np.log(prob + 1e-10)
                marginal_sum += np.exp(current_log_joint)
            
            if marginal_sum > 0:
                log_prob_cond = log_joint_actual - np.log(marginal_sum)
                log_likelihood_sum += log_prob_cond
            else:
                log_likelihood_sum -= 100
                
        return log_likelihood_sum

    def calculate_gradient(self, betas, data):
        """
        Computes the Analytical Gradient of the Negative CLL.
        Formula per sample: sum_t' [ (Indicator(t'=Actual) - P(t'|E)) * Gradient_LogJoint(t') ]
        """
        # 1. Get current Thetas
        tables = self.betas_to_probabilities(betas)
        grad = np.zeros_like(betas)
        
        records = data.to_dict('records')
        target_outcomes = self.var_info[self.target_col]['outcomes']
        
        for row in records:
            # --- Step A: Forward Pass (Compute P(Target=t' | Evidence)) ---
            log_joints = []
            temp_row = row.copy()
            
            # Compute Joint P(T=t, E) for all possible t values
            for t_val in target_outcomes:
                temp_row[self.target_col] = t_val
                log_joint = 0
                for node in self.nodes:
                    p_idx = self._get_parent_config_index(temp_row, node)
                    val_idx = self.var_info[node]['outcomes'].index(str(temp_row[node]))
                    prob = tables[node][p_idx, val_idx]
                    log_joint += np.log(prob + 1e-10)
                log_joints.append(log_joint)
            
            # Normalize to get P(T|E) using Log-Sum-Exp for stability
            log_joints = np.array(log_joints)
            max_log = np.max(log_joints)
            exp_joints = np.exp(log_joints - max_log)
            sum_exp = np.sum(exp_joints)
            probs_given_e = exp_joints / sum_exp # These are P(Target=t' | Evidence)
            
            # --- Step B: Backward Pass (Accumulate Gradients) ---
            actual_target_val = str(row[self.target_col])
            
            for t_idx, t_val in enumerate(target_outcomes):
                # The "Error Signal": (Actual - Predicted)
                # If t_val is the actual target, we want to boost its prob (positive coeff)
                # If t_val is not actual, we want to suppress it (negative coeff)
                is_actual = (t_val == actual_target_val)
                coeff = (1.0 if is_actual else 0.0) - probs_given_e[t_idx]
                
                # Apply this signal to all nodes involved in this configuration (Target=t_val)
                temp_row[self.target_col] = t_val
                
                for node in self.nodes:
                    # Identify which row of the CPT is active
                    p_idx = self._get_parent_config_index(temp_row, node)
                    # Identify which outcome occurred
                    val_idx = self.var_info[node]['outcomes'].index(str(temp_row[node]))
                    
                    # Locate parameters in flat vector
                    start_idx = self.node_offsets[node]
                    n_cols = self.var_info[node]['cardinality']
                    row_start = start_idx + (p_idx * n_cols)
                    
                    # Update Rule for Softmax:
                    # Grad_beta = Coeff * (Indicator(Outcome=k) - Prob(Outcome=k))
                    # We vectorized this update for the whole row of outcomes:
                    
                    # 1. Subtract P(outcome) from all weights in this row
                    #    (coeff * -prob)
                    current_probs_row = tables[node][p_idx]
                    grad[row_start : row_start + n_cols] -= coeff * current_probs_row
                    
                    # 2. Add 1.0 only to the specific outcome that occurred
                    #    (coeff * 1)
                    grad[row_start + val_idx] += coeff
        
        # Return NEGATIVE gradient because L-BFGS minimizes, but CLL is maximized
        return -grad