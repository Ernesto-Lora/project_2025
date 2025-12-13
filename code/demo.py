import numpy as np

# ==========================================
# 1. REPRESENTATION & HELPER FUNCTIONS
# ==========================================

class BayesianNetwork:
    def __init__(self, structure, cardinalities, target_node):
        """
        structure: dict {node: [parents]}
        cardinalities: dict {node: int_num_states}
        target_node: str (The 'Y' in your equation)
        """
        self.structure = structure
        self.cardinalities = cardinalities
        self.target = target_node
        self.nodes = list(structure.keys())
        self.parameters = {} # This will hold Theta

    def init_random_parameters(self):
        """
        Initializes Theta with random values. 
        Crucial: Probabilities in a CPT row must sum to 1.
        """
        for node in self.nodes:
            parents = self.structure[node]
            
            # Calculate size of CPT: (Car_Parent1 * Car_Parent2 * ...) x Car_Node
            parent_cardinalities = [self.cardinalities[p] for p in parents]
            num_parent_configs = np.prod(parent_cardinalities, dtype=int) if parent_cardinalities else 1
            node_cardinality = self.cardinalities[node]
            
            # Create random matrix
            rand_matrix = np.random.rand(num_parent_configs, node_cardinality)
            
            # Normalize rows to sum to 1
            row_sums = rand_matrix.sum(axis=1, keepdims=True)
            self.parameters[node] = rand_matrix / row_sums

    def get_probability(self, node, node_val, parent_vals):
        """
        Retrieves theta_{node | parents}
        """
        cpt = self.parameters[node]
        
        # Calculate row index based on parent values (Multi-index to flat index)
        # Assuming parents are ordered as in self.structure[node]
        row_idx = 0
        parents = self.structure[node]
        
        if not parents:
            row_idx = 0
        else:
            # Standard stride calculation for flat array indexing
            stride = 1
            for i in range(len(parents) - 1, -1, -1):
                p_val = parent_vals[i]
                row_idx += p_val * stride
                stride *= self.cardinalities[parents[i]]
        
        return cpt[row_idx, node_val]

# ==========================================
# 2. THE CLL CALCULATION (The Equation)
# ==========================================

def compute_log_joint_prob(bn, sample_dict, specific_y_val=None):
    """
    Computes log P(Y, X). 
    If specific_y_val is provided, we override the Y in the sample with it.
    """
    total_log_prob = 0.0
    
    # Create a context merging sample data and the specific Y being tested
    context = sample_dict.copy()
    if specific_y_val is not None:
        context[bn.target] = specific_y_val
        
    for node in bn.nodes:
        val = context[node]
        
        # Get parent values for this node
        parents = bn.structure[node]
        parent_vals = [context[p] for p in parents]
        
        # Get Theta
        prob = bn.get_probability(node, val, parent_vals)
        
        # Log stability clip (avoid log(0))
        prob = max(prob, 1e-10)
        total_log_prob += np.log(prob)
        
    return total_log_prob

def calculate_cll(bn, data):
    """
    Computes the Conditional Log-Likelihood as defined in the image.
    CLL = Sum_over_samples [ log P(Y_actual, X) - log( Sum_Y' P(Y', X) ) ]
    """
    total_cll = 0.0
    y_cardinality = bn.cardinalities[bn.target]
    
    for i, sample in enumerate(data):
        # 1. Compute term A: log P(Y_actual, X)
        log_prob_actual = compute_log_joint_prob(bn, sample)
        
        # 2. Compute term B: log Sum_{Y'} P(Y', X)
        # We must iterate over all possible values of the Class variable Y
        probs_all_y = []
        for y_prime in range(y_cardinality):
            log_p_y_prime = compute_log_joint_prob(bn, sample, specific_y_val=y_prime)
            probs_all_y.append(np.exp(log_p_y_prime))
            
        denom = np.sum(probs_all_y)
        log_denom = np.log(denom) if denom > 0 else -1e10
        
        # 3. CLL contribution for this sample
        sample_cll = log_prob_actual - log_denom
        total_cll += sample_cll
        
    return total_cll

# ==========================================
# 3. DEMO
# ==========================================

if __name__ == "__main__":
    # --- A. Define Structure (Naive Bayes: Y -> X1, Y -> X2) ---
    # Node names are integers or strings. Let's use strings for clarity.
    # Y is the Class. X1, X2 are features.
    structure = {
        'Y': [],          # Y has no parents
        'X1': ['Y'],      # X1 has parent Y
        'X2': ['Y']       # X2 has parent Y
    }
    
    # Cardinalities (How many states? e.g., 0 or 1)
    cards = {'Y': 2, 'X1': 3, 'X2': 2}
    
    # Instantiate BN
    my_bn = BayesianNetwork(structure, cards, target_node='Y')
    
    # --- B. Initialize Random Parameters (Theta) ---
    print("Initializing random parameters (Thetas)...")
    my_bn.init_random_parameters()
    
    # Show a parameter example (Theta for X1 | Y)
    print("\nParameter Shape for X1 (Rows=Parent configs, Cols=States):")
    print(my_bn.parameters['X1'].shape)
    print("Values (Rows sum to 1):")
    print(np.round(my_bn.parameters['X1'], 3))

    # --- C. Generate Dummy Data (N=5 Samples) ---
    # Data is a list of dictionaries
    data = [
        {'Y': 0, 'X1': 0, 'X2': 1},
        {'Y': 1, 'X1': 2, 'X2': 0},
        {'Y': 0, 'X1': 1, 'X2': 1},
        {'Y': 1, 'X1': 2, 'X2': 0},
        {'Y': 0, 'X1': 0, 'X2': 0},
    ]
    print(f"\nComputing CLL for {len(data)} samples...")

    # --- D. Compute CLL ---
    cll_value = calculate_cll(my_bn, data)
    
    print("-" * 30)
    print(f"Total Conditional Log-Likelihood: {cll_value:.4f}")
    print("-" * 30)
    
    print("Note: This value is negative. Maximizing CLL means bringing this closer to 0.")