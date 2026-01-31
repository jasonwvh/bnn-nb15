"""
Prequential evaluation with EXPERIENCE REPLAY.
Fixes catastrophic forgetting by mixing original training data with new stream windows.
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

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler  # ← ADD THIS IMPORT
from rf_train import load_and_preprocess_data

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

# Default paths (must match rf_train / bnn_train for same preprocessing)
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Drift defaults
DEFAULT_N_WINDOWS = 5
DEFAULT_DRIFT_STRENGTH = 0.4


def make_drift_windows(X_test, y_test, n_windows=5, drift_type='gradual', drift_strength=0.4, seed=42):
    """
    Split test set into sequential time windows and optionally inject concept drift.

    Args:
        X_test: Test features (already scaled).
        y_test: Test labels.
        n_windows: Number of consecutive stream windows.
        drift_type: 'none' | 'gradual' | 'sudden'
          - none: no injection; later windows are just later in time (natural drift if any).
          - gradual: feature scale/shift increases linearly with window index (simulates slow drift).
          - sudden: from window 1 onward, apply a fixed distribution shift (simulates change point).
        drift_strength: Magnitude of injected drift (e.g. 0.3 = 30% scale noise or shift).
        seed: Random seed for reproducibility.

    Returns:
        List of (X_w, y_w) for each window, in temporal order.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    if n < n_windows:
        n_windows = max(1, n // 100)
    indices = np.arange(n)
    # Consecutive chunks (temporal order)
    splits = np.linspace(0, n, n_windows + 1, dtype=int)
    windows = []
    for i in range(n_windows):
        start, end = splits[i], splits[i + 1]
        if start >= end:
            continue
        X_w = X_test[start:end].copy()
        y_w = y_test[start:end]

        if drift_type == 'none' or i == 0:
            windows.append((X_w, y_w))
            continue

        # Inject drift in later windows
        if drift_type == 'gradual':
            # Scale and shift increase with window index (simulates gradual drift)
            scale = 1.0 + drift_strength * (i / max(1, n_windows))
            shift = rng.standard_normal(X_w.shape[1]) * drift_strength * 0.5 * (i / max(1, n_windows))
            X_w = X_w * scale + shift
        elif drift_type == 'sudden':
            # From window 1 onward: fixed scale + noise (simulates sudden change)
            scale = 1.0 + drift_strength
            shift = rng.standard_normal(X_w.shape[1]) * drift_strength * 0.5
            X_w = X_w * scale + shift

        X_w = np.nan_to_num(X_w, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # ================================================================
        # FIX: RE-NORMALIZE AFTER DRIFT INJECTION
        # This preserves the input distribution that BNN expects
        # ================================================================
        # if i > 0 and drift_type != 'none':
        #     # Re-normalize the drifted window so BNN can process it
        #     scaler_window = RobustScaler()
        #     X_w = scaler_window.fit_transform(X_w)
        # ================================================================
        
        windows.append((X_w, y_w))

    return windows


def get_train_and_stream_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Load and preprocess train and test data (same pipeline as rf_train / bnn_train).
    Use this before making drift windows so scaling is consistent.

    Returns:
        X_train, y_train, X_test, y_test, scaler, feature_names
    """
    return load_and_preprocess_data(train_path, test_path)


def get_drift_stream(train_path=TRAIN_PATH, test_path=TEST_PATH,
                     n_windows=DEFAULT_N_WINDOWS, drift_type='gradual',
                     drift_strength=DEFAULT_DRIFT_STRENGTH, seed=42):
    """
    One-shot: load data, preprocess, split test into windows with optional drift.

    Returns:
        X_train, y_train, scaler, feature_names, windows
        where windows = [(X_1, y_1), (X_2, y_2), ...] in temporal order.
    """
    X_train, y_train, X_test, y_test, scaler, feature_names = get_train_and_stream_data(
        train_path, test_path
    )
    windows = make_drift_windows(X_test, y_test, n_windows=n_windows,
                                  drift_type=drift_type, drift_strength=drift_strength, seed=seed)
    return X_train, y_train, scaler, feature_names, windows
# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_CSV = 'models/drift_eval_results.csv'

N_WINDOWS = 10
DRIFT_TYPE = 'sudden'
DRIFT_STRENGTH = 0.4

# BNN CONTINUAL LEARNING SETTINGS (TUNED)
# Lower LR and Higher KL to prevent over-reacting to noise
CL_N_EPOCHS = 5
CL_BATCH_SIZE = 64
CL_LR = 1e-4             # Reduced from 1e-3 to prevent weights jumping too fast
CL_KL_WEIGHT_MAX = 0.1   # Increased from 0.001 to force model to respect Prior
CL_ALEATORIC_SCALE = 1.0
CL_EPISTEMIC_PLASTICITY = 0.5

# REPLAY SETTINGS (ARCHITECTURAL FIX)
REPLAY_SIZE = 2000       # Number of original samples to keep in memory
REPLAY_BATCH_RATIO = 0.5 # Percentage of training batch that is Replay data

MC_SAMPLES_EVAL = 15


def eval_rf(rf_model, X, y):
    pred = rf_model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, zero_division=0)


def eval_bnn(model, X, y, device, batch_size=256, mc_samples=15):
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    acc, _, _, f1, _, _ = validate(model, loader, device, mc_samples=mc_samples)
    return acc, f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)
    print("Concept drift evaluation: RF vs BNN (Static) vs BNN-CL (Replay)")
    print("=" * 70)

    # 1. Load Stream
    X_train, y_train, scaler, feature_names, windows = get_drift_stream(
        TRAIN_PATH, TEST_PATH,
        n_windows=N_WINDOWS, drift_type=DRIFT_TYPE, drift_strength=DRIFT_STRENGTH,
    )

    # 2. Initialize Replay Buffer (The Fix)
    print(f"\nInitializing Replay Buffer (Size: {REPLAY_SIZE})...")
    indices = np.random.choice(len(X_train), size=min(len(X_train), REPLAY_SIZE), replace=False)
    X_replay = X_train[indices]
    y_replay = y_train[indices]
    print(f"Replay buffer created with {len(X_replay)} samples.")

    # 3. Load Models
    print(f"Loading RF from {RF_MODEL_PATH}")
    rf_model = joblib.load(RF_MODEL_PATH)

    print(f"Loading BNN from {BNN_MODEL_PATH}")
    ckpt = torch.load(BNN_MODEL_PATH, map_location=device)
    input_dim = ckpt.get('input_dim', X_train.shape[1])
    
    bnn_static = NetworkBNN(input_dim).to(device)
    bnn_static.load_state_dict(ckpt['model_state_dict'])
    
    bnn_cl = NetworkBNN(input_dim).to(device)
    bnn_cl.load_state_dict(copy.deepcopy(ckpt['model_state_dict']))

    # Class weights
    n_samples = len(y_train)
    n_normal, n_attack = (y_train == 0).sum(), (y_train == 1).sum()
    class_weights = torch.FloatTensor([
        n_samples / (2 * max(1, n_normal)),
        n_samples / (2 * max(1, n_attack)),
    ])
    
    # Initialize Prior (VCL)
    prior_params = bnn_cl.get_posterior_params()

    results = []
    
    # 4. Stream Loop
    for t, (X_w, y_w) in enumerate(windows):
        n_w = len(y_w)
        
        # Evaluate BEFORE update (Test-Then-Train)
        rf_acc, rf_f1 = eval_rf(rf_model, X_w, y_w)
        bnn_s_acc, bnn_s_f1 = eval_bnn(bnn_static, X_w, y_w, device, mc_samples=MC_SAMPLES_EVAL)
        bnn_cl_acc, bnn_cl_f1 = eval_bnn(bnn_cl, X_w, y_w, device, mc_samples=MC_SAMPLES_EVAL)

        results.append({
            'window': t,
            'rf_f1': rf_f1,
            'bnn_static_f1': bnn_s_f1,
            'bnn_cl_f1': bnn_cl_f1,
        })
        
        print(f"Window {t}: RF F1={rf_f1:.3f} | Static F1={bnn_s_f1:.3f} | CL F1={bnn_cl_f1:.3f}")

        # 5. The Overhaul: Mix Window + Replay for Update
        # We concatenate replay data to the window data
        X_update = np.concatenate([X_w, X_replay], axis=0)
        y_update = np.concatenate([y_w, y_replay], axis=0)

        # Update BNN-CL
        bnn_cl, prior_params = continual_update_bnn(
            bnn_cl, X_update, y_update, device,
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
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal Avg F1 - RF: {df['rf_f1'].mean():.3f}, Static: {df['bnn_static_f1'].mean():.3f}, CL: {df['bnn_cl_f1'].mean():.3f}")

if __name__ == '__main__':
    main()