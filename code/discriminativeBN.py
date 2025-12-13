import numpy as np
import pandas as pd
from pgmpy.readwrite import XMLBIFReader
import random
import copy

# ==========================================
# 1. The Bayesian Network & Objective Class
# ==========================================
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
        self.var_info = {} # Stores parents and outcome list for each node
        self.param_map = [] # Maps flat beta vector index to (node, parent_config, outcome_idx)
        
        self.total_params = 0
        
        for node in self.nodes:
            cpd = self.model.get_cpds(node)
            outcomes = cpd.state_names[node]
            parents = self.model.get_parents(node)
            
            # For each node, we need to know the dimension of its parent configurations
            parent_cards = [len(self.model.get_cpds(p).state_names[p]) for p in parents]
            total_parent_configs = np.prod(parent_cards) if parent_cards else 1
            
            self.var_info[node] = {
                'parents': parents,
                'outcomes': outcomes,
                'cardinality': len(outcomes),
                'parent_cardinalities': parent_cards,
                'num_parent_configs': int(total_parent_configs)
            }
            
            # Map parameters for optimization vector beta
            # We need (Num_Parent_Configs * Num_Outcomes) weights per node
            self.total_params += int(total_parent_configs * len(outcomes))

    def _get_parent_config_index(self, row, node):
        """Helper to find the index of the parent configuration in the CPD table"""
        parents = self.var_info[node]['parents']
        if not parents:
            return 0
        
        # Calculate flattened index for parent combination
        # This assumes standard pgmpy/BIF ordering logic
        idx = 0
        stride = 1
        for p in reversed(parents):
            p_val = row[p]
            p_outcomes = self.var_info[p]['outcomes']
            p_idx = p_outcomes.index(str(p_val)) # Ensure string matching
            idx += p_idx * stride
            stride *= len(p_outcomes)
        return idx

    def betas_to_probabilities(self, betas):
        """
        The Greiner Transformation: Beta (Real) -> Theta (Probabilities)
        Uses Softmax per parent-configuration row.
        """
        # Reconstruct CPD tables from flat beta vector
        current_idx = 0
        tables = {}
        
        for node in self.nodes:
            info = self.var_info[node]
            n_rows = info['num_parent_configs']
            n_cols = info['cardinality']
            
            # Slice beta vector
            node_betas = betas[current_idx : current_idx + (n_rows * n_cols)]
            current_idx += (n_rows * n_cols)
            
            # Reshape to (Parent_Configs, Outcomes)
            beta_matrix = node_betas.reshape((n_rows, n_cols))
            
            # Softmax row-wise (axis 1)
            # exp(b) / sum(exp(b))
            max_b = np.max(beta_matrix, axis=1, keepdims=True) # Stability trick
            exp_b = np.exp(beta_matrix - max_b)
            probs = exp_b / np.sum(exp_b, axis=1, keepdims=True)
            
            tables[node] = probs
            
        return tables

    def calculate_cll(self, betas, data):
        """
        Calculates Conditional Log Likelihood: P(Target | Evidence)
        CLL = Sum_data [ log( P(Target, Evidence) / Sum_target'(P(Target', Evidence)) ) ]
        """
        tables = self.betas_to_probabilities(betas)
        log_likelihood_sum = 0
        
        # Pre-process data to list of dicts for speed
        records = data.to_dict('records')
        
        target_outcomes = self.var_info[self.target_col]['outcomes']
        
        for row in records:
            # 1. Calculate Joint Probability of the ACTUAL row: P(Target=t_actual, Evidence)
            # We compute joint as Product(P(node|parents))
            log_joint_actual = 0
            for node in self.nodes:
                p_idx = self._get_parent_config_index(row, node)
                val_idx = self.var_info[node]['outcomes'].index(str(row[node]))
                prob = tables[node][p_idx, val_idx]
                log_joint_actual += np.log(prob + 1e-10) # Small epsilon
            
            joint_actual = np.exp(log_joint_actual)

            # 2. Calculate Marginal Probability of Evidence: Sum_over_Target(P(Target=t, Evidence))
            # We iterate over all possible values of the Target variable, keeping evidence fixed
            marginal_sum = 0
            
            temp_row = row.copy()
            for t_val in target_outcomes:
                temp_row[self.target_col] = t_val
                
                # Compute joint for this hypothetical target value
                current_log_joint = 0
                for node in self.nodes:
                    p_idx = self._get_parent_config_index(temp_row, node)
                    val_idx = self.var_info[node]['outcomes'].index(str(temp_row[node]))
                    prob = tables[node][p_idx, val_idx]
                    current_log_joint += np.log(prob + 1e-10)
                
                marginal_sum += np.exp(current_log_joint)
            
            # 3. Conditional Probability = Joint_Actual / Marginal_Evidence
            # Log(CP) = Log(Joint_Actual) - Log(Marginal)
            if marginal_sum > 0:
                log_prob_cond = log_joint_actual - np.log(marginal_sum)
                log_likelihood_sum += log_prob_cond
            else:
                log_likelihood_sum -= 100 # Penalty for zero probability
                
        return log_likelihood_sum