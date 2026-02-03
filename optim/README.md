# Optim Experiments: MESU vs Baselines under Concept Drift

This folder contains the concept-drift experiment that compares a MESU-based Bayesian anomaly detector against deterministic baselines on UNSW-NB15.

## What This Experiment Does
- Loads and preprocesses UNSW-NB15 CSVs from `data/`.
- Simulates concept drift on the training set (sudden, gradual, incremental).
- Trains models sequentially across concepts.
- Evaluates after each concept on both the current concept and the fixed test set.
- Produces plots and prints summary metrics plus a catastrophic forgetting score.

## Architecture Overview

### MESU Model
- `mesu_model.py` defines `MESUAnomalyDetector`, a Bayesian MLP with `Linear_MetaBayes` layers.
- Each `Linear_MetaBayes` layer (in `mesu_layers.py`) maintains Gaussian parameters (`mu`, `sigma`) and samples weights using the reparameterization trick.
- Forward pass uses Monte Carlo sampling to produce log-probabilities. Uncertainty is estimated from the variance across samples.
- The loss is negative log-likelihood; the MESU optimizer handles the Bayesian regularization through its update rule.
- The optimizer in `mesu_optimizer.py` implements Metaplasticity from Synaptic Uncertainty (MESU) with momentum and optional second-order terms to mitigate catastrophic forgetting.

### Baseline Model Used in the Experiment
- `baseline_model.py` provides a deterministic MLP with experience replay, `ExperienceReplay`.
- The baseline shares the same hidden sizes and dropout as the MESU model but uses standard `nn.Linear` layers and Adam.
- The replay buffer is implemented in the class, but in the current experiment loop the replay buffer is not actively populated or mixed during training.
- Other baselines also live in `baseline_model.py` (`DeterministicBayesianModel`, `EWCModel`) but are not used by `main_experiment.py`.

## Key Files
- `main_experiment.py`: Runs MESU vs Experience Replay across drift types and plots results.
- `mesu_model.py`: Bayesian anomaly detector definition.
- `mesu_layers.py`: Bayesian linear layer with Gaussian parameters.
- `mesu_optimizer.py`: MESU update rule.
- `baseline_model.py`: Deterministic baselines (Experience Replay used here).
- `data_utils.py`: UNSW-NB15 preprocessing and drift simulation.

## Data and Preprocessing
- Expected files:
  - `data/UNSW_NB15_training-set.csv`
  - `data/UNSW_NB15_testing-set.csv`
- Categorical columns (`proto`, `service`, `state`) are label-encoded using combined train+test fitting for consistency.
- Features are standardized using `StandardScaler` from scikit-learn.

## Concept Drift Simulation
Implemented in `data_utils.py`:
- `sudden`: data split into chunks, each chunk gets a stronger scale and more noise.
- `gradual`: shift factor linearly changes within each chunk.
- `incremental`: continuous drift across samples, then chunked for evaluation.

## Training and Evaluation Flow
`main_experiment.py`:
- For each drift type (`sudden`, `gradual`, `incremental`):
  - Train MESU for 3 drift points (4 concepts total).
  - Train baseline for the same drift points.
  - Evaluate after each concept on:
    - current concept data
    - fixed test set
  - Compute a catastrophic forgetting metric: initial test accuracy minus final test accuracy.
  - Save plots to `results/`.

Metrics reported: accuracy, precision, recall, F1, AUC.

## How to Run
From the repo root:

```bash
python optim/main_experiment.py
```

Outputs:
- `results/comparison_<drift>_drift.png`
- `results/loss_<drift>_drift.png`

## Configuration Notes
- Model config is defined in `main_experiment.py`:
  - Hidden sizes: `[128, 64, 32]`
  - Dropout: `0.3`
  - MESU sampling: 5 samples for training loss, 10 for prediction
- Adjust drift behavior in `data_utils.py` or pass different `num_drifts` in `main_experiment.py`.

## Dependencies
See `optim/requirements.txt` for the experiment-specific requirements.
