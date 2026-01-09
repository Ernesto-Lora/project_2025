
---

# Discriminative BN Optimization Demo

This notebook demonstrates how to learn parameters (betas) for a Discriminative Bayesian Network (TAN structure) using numerical optimization to maximize Conditional Log Likelihood (CLL).

It Uses: **Hooke-Jeeves**, **Nelder-Mead**, and **L-BFGS**.


## Quick Start Tutorial

### Initialization

Load in order all the notebook lines. This will read files and and define the objective function for the methods, in this case, it is the Conditional log likelihood.

### 3. Run Optimizers

The notebook executes the following algorithms sequentially:

**A. Hooke-Jeeves (Pattern Search)**

```python
optimizer = HookeJeevesOptimizer(objective, dim=bn.total_params, max_iter=5)
best_betas = optimizer.optimize(start_betas)

```

**B. Nelder-Mead (Simplex)**

```python
optimizer_nel = NelderMeadOptimizer(objective, dim=bn.total_params, max_iter=5)
best_betas_nelder = optimizer_nel.optimize(start_betas)

```

**C. L-BFGS (Gradient-Based)**
Requires a gradient function.

```python
gradient_func = lambda betas: bn.calculate_gradient(betas, train_df)
op_lbfgs = LBFGSOptimizerWrapper(func=objective, gradient_func=gradient_func, dim=bn.total_params)
best_betas_lbfgs = op_lbfgs.optimize(start_betas)

```

### 4. Evaluate Results

For each method, the notebook outputs:

1. **Final CLL:** How well the model fits the training data.
2. **CPTs:** The learned probability tables.
3. **Accuracy & Confusion Matrix:** Performance on `test_df`.

---

# Batch K-Fold Experiment Runner

This script automates the comparison of different optimization algorithms using **K-Fold Cross-Validation**. It dynamically generates experiments by combining dataset configurations with optimizer settings.


### 1. Define Datasets

Add entries to `data_configs`. Each entry requires the CSV data, the XML network structure, and the target variable node name.

```python
data_configs = [
    {
        "name": "Australian",
        "csv": "datasets/AustralianDisc.csv",
        "xml": "models/australianTAN.xml",
        "target_var": "A15" 
    }
    # Add more datasets here...
]

```

### 2. Define Optimizers

Configure the algorithms in the `optimizers` list. You can map specific runner functions (e.g., standard vs. gradient-based) to each optimizer.

```python
optimizers = [
    {
        "name": "L-BFGS (Gradient)",
        "func": k_fold_validation_gradient,   # Uses gradient runner
        "Optimizer_fun": LBFGSOptimizerWrapper,
        "max_iter": 100,
        "file_suffix": "LBFGS"
    },
    # ... other optimizers (Nelder-Mead, Hooke-Jeeves)
]

```

### 3. Build Experiment Queue

The script automatically generates the Cartesian product of `data_configs`  `optimizers`.

* **Input:** `data_configs`, `optimizers`
* **Output:** A list of `experiments` dictionaries containing all parameters needed for execution.
* **File Naming:** Automatically generates filenames, e.g., `resultsDemo/results_AustralianNAIVE_simplex.txt`.

### 4. Run Loop

The script iterates through the queue and executes each experiment safely.

* **Error Handling:** If an experiment crashes, it logs the error and `continue`s to the next one without stopping the entire batch.
* **Timing:** Tracks and prints the elapsed time for each experiment.

```python
for exp in experiments:
    try:
        exp['func'](**exp['params']) # Execute K-Fold Validation
    except Exception as e:
        print(f"CRITICAL ERROR: {e}") # Log and continue

```

## Output

Results are saved as text files in the `resultsDemo/` directory. Each file contains the cross-validation metrics for that specific dataset-optimizer combination.