"""
Prequential evaluation under concept drift: RF vs BNN (static vs continual learning).
Objective: show RF loses accuracy under drift while BNN with epistemic/aleatoric-guided
continual learning maintains performance.
"""
import numpy as np
import pandas as pd
import torch
import copy
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from drift import get_drift_stream
from rf_train import load_and_preprocess_data
from bnn_train import (
    NetworkBNN,
    validate,
    continual_update_bnn,
    HIDDEN_DIM,
    N_CLASSES,
    MODEL_PATH as BNN_MODEL_PATH,
)
from rf_train import MODEL_PATH as RF_MODEL_PATH

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_CSV = 'models/drift_eval_results.csv'

N_WINDOWS = 10
DRIFT_TYPE = 'gradual'   # 'none' | 'gradual' | 'sudden'
DRIFT_STRENGTH = 0.6

# BNN continual learning
CL_N_EPOCHS = 15
CL_BATCH_SIZE = 128
CL_LR = 1e-3
CL_KL_WEIGHT_MAX = 0.001
CL_ALEATORIC_SCALE = 1.0   # down-weight noisy samples
CL_EPISTEMIC_PLASTICITY = 0.3  # more plasticity when epistemic high
MC_SAMPLES_EVAL = 15


def eval_rf(rf_model, X, y):
    """Return accuracy and F1 for RF on (X, y)."""
    pred = rf_model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, zero_division=0)


def eval_bnn(model, X, y, device, batch_size=256, mc_samples=15):
    """Return accuracy and F1 for BNN on (X, y) with MC sampling."""
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    acc, _, _, f1, _, _ = validate(model, loader, device, mc_samples=mc_samples)
    return acc, f1


def main(train_path=TRAIN_PATH, test_path=TEST_PATH,
         n_windows=N_WINDOWS, drift_type=DRIFT_TYPE, drift_strength=DRIFT_STRENGTH,
         rf_path=RF_MODEL_PATH, bnn_path=BNN_MODEL_PATH,
         output_csv=OUTPUT_CSV):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    print("=" * 70)
    print("Concept drift evaluation: RF vs BNN (static vs continual learning)")
    print("=" * 70)
    print(f"Drift: type={drift_type}, strength={drift_strength}, n_windows={n_windows}\n")

    # Stream with drift
    X_train, y_train, scaler, feature_names, windows = get_drift_stream(
        train_path, test_path,
        n_windows=n_windows, drift_type=drift_type, drift_strength=drift_strength,
    )
    print(f"Train size: {len(X_train)}; {len(windows)} stream windows\n")

    # Load RF
    print(f"Loading RF from {rf_path}")
    rf_model = joblib.load(rf_path)

    # Load BNN (one for static, one for CL)
    print(f"Loading BNN from {bnn_path}")
    ckpt = torch.load(bnn_path, map_location=device)
    input_dim = ckpt.get('input_dim', X_train.shape[1])
    hidden_dim = ckpt.get('hidden_dim', HIDDEN_DIM)
    n_classes = ckpt.get('n_classes', N_CLASSES)
    bnn_static = NetworkBNN(input_dim, hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    bnn_static.load_state_dict(ckpt['model_state_dict'])
    bnn_cl = NetworkBNN(input_dim, hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    bnn_cl.load_state_dict(copy.deepcopy(ckpt['model_state_dict']))

    # Class weights for BNN continual update (same as initial training)
    n_samples = len(y_train)
    n_normal, n_attack = (y_train == 0).sum(), (y_train == 1).sum()
    class_weights = torch.FloatTensor([
        n_samples / (2 * max(1, n_normal)),
        n_samples / (2 * max(1, n_attack)),
    ])
    prior_params = bnn_cl.get_posterior_params()

    results = []
    for t, (X_w, y_w) in enumerate(windows):
        n_w = len(y_w)
        rf_acc, rf_f1 = eval_rf(rf_model, X_w, y_w)
        bnn_s_acc, bnn_s_f1 = eval_bnn(bnn_static, X_w, y_w, device, mc_samples=MC_SAMPLES_EVAL)
        bnn_cl_acc, bnn_cl_f1 = eval_bnn(bnn_cl, X_w, y_w, device, mc_samples=MC_SAMPLES_EVAL)

        results.append({
            'window': t,
            'n_samples': n_w,
            'rf_acc': rf_acc,
            'rf_f1': rf_f1,
            'bnn_static_acc': bnn_s_acc,
            'bnn_static_f1': bnn_s_f1,
            'bnn_cl_acc': bnn_cl_acc,
            'bnn_cl_f1': bnn_cl_f1,
        })
        print(f"Window {t} (n={n_w}): RF acc={rf_acc*100:.2f}% F1={rf_f1*100:.2f}% | "
              f"BNN static acc={bnn_s_acc*100:.2f}% F1={bnn_s_f1*100:.2f}% | "
              f"BNN-CL acc={bnn_cl_acc*100:.2f}% F1={bnn_cl_f1*100:.2f}%")

        # Update BNN-CL on this window (VCL + epistemic/aleatoric)
        bnn_cl, prior_params = continual_update_bnn(
            bnn_cl, X_w, y_w, device,
            prior_params=prior_params,
            class_weights=class_weights,
            n_epochs=CL_N_EPOCHS,
            batch_size=CL_BATCH_SIZE,
            lr=CL_LR,
            kl_weight_max=CL_KL_WEIGHT_MAX,
            aleatoric_scale=CL_ALEATORIC_SCALE,
            epistemic_plasticity=CL_EPISTEMIC_PLASTICITY,
            mc_samples_uq=MC_SAMPLES_EVAL,
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Summary: average over windows (optionally excluding window 0)
    print("\n" + "=" * 70)
    print("Summary (mean over all windows)")
    print("=" * 70)
    print(f"  RF:       acc={df['rf_acc'].mean()*100:.2f}%  F1={df['rf_f1'].mean()*100:.2f}%")
    print(f"  BNN static: acc={df['bnn_static_acc'].mean()*100:.2f}%  F1={df['bnn_static_f1'].mean()*100:.2f}%")
    print(f"  BNN-CL:   acc={df['bnn_cl_acc'].mean()*100:.2f}%  F1={df['bnn_cl_f1'].mean()*100:.2f}%")
    print("\nBNN-CL uses epistemic/aleatoric uncertainty to guide learning under drift.")
    print("=" * 70)
    return df


if __name__ == '__main__':
    main()
