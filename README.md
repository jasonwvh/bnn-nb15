# UBCL: Uncertainty-Aware Bayesian Continual Learning for NIDS

Bayesian neural network for network intrusion detection on **UNSW-NB15**, with uncertainty quantification. Current scope: single-phase BNN training and prediction; full UBCL (continual learning under drift, three-way decision) is planned.

## Data

- **UNSW-NB15**: training-set and testing-set CSVs in `data/`.
- Use `data/UNSW_NB15_training-set.csv` and `data/UNSW_NB15_testing-set.csv`. Temporal order is preserved for future streaming/drift experiments.

## Architecture

```mermaid
flowchart LR
    subgraph input [Input]
        Data[UNSW-NB15 CSV]
    end
    subgraph prep [Preprocess]
        Scale[RobustScaler]
        Encode[LabelEncoder]
    end
    subgraph bnn [BNN]
        BL1[BayesianLinear 256]
        BL2[BayesianLinear 128]
        BL3[BayesianLinear 64]
        BL4[BayesianLinear 2]
    end
    subgraph out [Output]
        Pred[Predictions]
        UQ[Uncertainty]
        ThreeWay[Three-way decision]
    end
    Data --> Scale
    Scale --> Encode
    Encode --> BL1
    BL1 --> BL2 --> BL3 --> BL4
    BL4 --> Pred
    Pred --> UQ
    UQ --> ThreeWay
```

- **BNN**: Variational inference (VI), Gaussian priors, 4 layers (256 → 128 → 64 → 2). Log-softmax output.
- **Training**: Class-weighted NLL + KL (warm-up); i.e. class-weighted ELBO.
- **Uncertainty**: Monte Carlo sampling over weight posterior; predictive entropy (combined).
- **Three-way decision** (in prediction): Accept (benign), Reject (attack), or Defer to human, using thresholds τ_benign, τ_attack, η.

## Scripts

| Script | Role |
|--------|------|
| `bnn_train.py` | Reads training-set, trains BNN, saves model to `models/bnn.pth`. |
| `bnn_pred.py` | Loads `models/bnn.pth`, runs on testing-set; reports ML metrics (accuracy, precision, recall, F1, AUC), calibration (ECE, Brier), and three-way decision (accept/reject/defer). |
| `rf_train.py` | Trains Random Forest baseline on training-set; saves `models/rf.pkl`. |
| `rf_pred.py` | Loads `models/rf.pkl`, runs on testing-set; reports same ML and calibration metrics for comparison. Objective: BNN should surpass RF on the same dataset. |
| `drift.py` | Concept drift simulation: splits test set into sequential time windows; optional **gradual** or **sudden** drift injection (scale/shift) for later windows. |
| `eval_drift.py` | Prequential evaluation under drift: compares **RF** (no update), **BNN static** (no update), and **BNN-CL** (continual learning with epistemic/aleatoric-guided updates and VCL). Objective: show RF loses accuracy under drift while BNN-CL maintains. |

## Concept drift and continual learning

- **Drift simulation** ([drift.py](drift.py)): Test set is split into `n_windows` consecutive chunks. Optional injection: **gradual** (scale/shift increases with window index) or **sudden** (fixed shift from window 1 onward). Same preprocessing as training (train+test scaled together).
- **BNN continual learning** ([bnn_train.py](bnn_train.py)): **Epistemic** and **aleatoric** uncertainty from MC sampling; **aleatoric** used to down-weight noisy samples; **epistemic** used to increase plasticity (lower KL weight when model is uncertain). **VCL**: posterior at window t becomes prior for window t+1.
- **Evaluation** ([eval_drift.py](eval_drift.py)): For each window, evaluate RF, BNN static, and BNN-CL; then update BNN-CL on that window. Results (acc, F1 per window) saved to `models/drift_eval_results.csv`. Run after `bnn_train.py` and `rf_train.py`.

## Requirements

Install: `pip install -r requirements.txt`

- Python 3.x, numpy, pandas, scikit-learn, torch, joblib (see [requirements.txt](requirements.txt)).

## Quick start

1. Place UNSW-NB15 CSVs in `data/`:
   - `data/UNSW_NB15_training-set.csv`
   - `data/UNSW_NB15_testing-set.csv`
2. **BNN:** `python bnn_train.py` then `python bnn_pred.py` — predictions to `models/predictions.csv`.
3. **RF baseline:** `python rf_train.py` then `python rf_pred.py` — predictions to `models/rf_predictions.csv`. Compare metrics to ensure BNN surpasses RF.
4. **Concept drift:** After training both, run `python eval_drift.py` — compares RF vs BNN static vs BNN-CL over drifted windows; results in `models/drift_eval_results.csv`.

## Future work (full UBCL)

- CICIDS2017, baselines (EWC, static BNN), multi-seed evaluation (RO4).
- Expose epistemic/aleatoric in prediction script and three-way deferral by uncertainty type.

Detailed methodology: [docs/proposal.md](docs/proposal.md).
