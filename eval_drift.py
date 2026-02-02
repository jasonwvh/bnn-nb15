"""
Prequential evaluation with EXPERIENCE REPLAY for Anomaly Detection.
Compares AE (static) vs VAE (static) vs VAE-CL vs BVAE (static) vs BVAE-CL.
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import copy
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler

from ae_train import (
    Autoencoder,
    compute_reconstruction_error,
    MODEL_PATH as AE_MODEL_PATH,
    THRESHOLD_PATH as AE_THRESHOLD_PATH,
)
from vae_train import (
    VAE,
    compute_reconstruction_metrics as vae_compute_metrics,
    vae_loss,
    MODEL_PATH as VAE_MODEL_PATH,
    THRESHOLD_PATH as VAE_THRESHOLD_PATH,
    WEIGHT_DECAY as VAE_WEIGHT_DECAY,
)
from bvae_train import (
    BayesianVAE,
    compute_reconstruction_metrics as bvae_compute_metrics,
    continual_update_bvae,
    load_and_preprocess_data,
    vae_loss as bvae_vae_loss,
    LATENT_DIM,
    HIDDEN_DIMS,
    PRIOR_SIGMA,
    MODEL_PATH as BVAE_MODEL_PATH,
    THRESHOLD_PATH as BVAE_THRESHOLD_PATH,
    WEIGHT_DECAY as BVAE_WEIGHT_DECAY,
)

# Default paths
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Drift defaults
DEFAULT_N_WINDOWS = 100
DEFAULT_DRIFT_STRENGTH = 0.4

# Output
OUTPUT_CSV = 'models/drift_eval_results.csv'

# VAE Continual Learning Settings
VAE_CL_N_EPOCHS = 5
VAE_CL_BATCH_SIZE = 64
VAE_CL_LR = 1e-4
VAE_CL_BETA = 0.001

# BVAE Continual Learning Settings
BVAE_CL_N_EPOCHS = 5
BVAE_CL_BATCH_SIZE = 64
BVAE_CL_LR = 1e-4
BVAE_CL_KL_WEIGHT_MAX = 0.001
BVAE_CL_WEIGHT_KL_SCALE = 0.0001

# Replay settings
REPLAY_SIZE = 2000
REPLAY_BATCH_RATIO = 0.5

# MC samples for evaluation
MC_SAMPLES_EVAL = 10


def continual_update_vae(model, X, y, device, n_epochs=5, batch_size=64, lr=1e-4, beta=0.001):
    """
    Continual learning update for VAE (no VCL, just fine-tuning)
    
    Args:
        model: VAE model
        X: New data features
        y: New data labels
        device: torch device
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        beta: KL weight
    
    Returns:
        Updated model
    """
    # Train only on normal data
    X_normal = X[y == 0] if len(X[y == 0]) > 0 else X
    
    if len(X_normal) == 0:
        return model
    
    dataset = TensorDataset(torch.FloatTensor(X_normal))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=VAE_WEIGHT_DECAY)
    
    model.train()
    for epoch in range(n_epochs):
        for data in loader:
            data = data[0].to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data)
            
            # Check for NaN
            if torch.isnan(x_recon).any() or torch.isnan(mu).any():
                continue
            
            # VAE loss
            loss, recon_loss, kl_latent = vae_loss(x_recon, data, mu, logvar, beta)
            
            if torch.isnan(loss):
                continue
            
            (loss / len(data)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    return model


def make_drift_windows(X_test, y_test, n_windows=10, drift_type='gradual', drift_strength=0.4, seed=42):
    """
    Split test set into sequential time windows and optionally inject concept drift.
    
    Args:
        X_test: Test features (already scaled)
        y_test: Test labels
        n_windows: Number of consecutive stream windows
        drift_type: 'none' | 'gradual' | 'sudden'
        drift_strength: Magnitude of injected drift
        seed: Random seed
    
    Returns:
        List of (X_w, y_w) for each window in temporal order
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    if n < n_windows:
        n_windows = max(1, n // 100)
    
    # Consecutive chunks
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
            # Scale and shift increase with window index
            scale = 1.0 + drift_strength * (i / max(1, n_windows))
            shift = rng.standard_normal(X_w.shape[1]) * drift_strength * 0.5 * (i / max(1, n_windows))
            X_w = X_w * scale + shift
        elif drift_type == 'sudden':
            # Fixed scale + noise from window 1 onward
            scale = 1.0 + drift_strength
            shift = rng.standard_normal(X_w.shape[1]) * drift_strength * 0.5
            X_w = X_w * scale + shift
        
        X_w = np.nan_to_num(X_w, nan=0.0, posinf=1e6, neginf=-1e6)
        windows.append((X_w, y_w))
    
    return windows


def get_drift_stream(train_path=TRAIN_PATH, test_path=TEST_PATH,
                     n_windows=DEFAULT_N_WINDOWS, drift_type='gradual',
                     drift_strength=DEFAULT_DRIFT_STRENGTH, seed=42):
    """
    Load data, preprocess, split test into windows with optional drift.
    
    Returns:
        X_train_normal, X_test, y_test, scaler, feature_names, windows
    """
    X_train_normal, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        train_path, test_path
    )
    windows = make_drift_windows(X_test, y_test, n_windows=n_windows,
                                  drift_type=drift_type, drift_strength=drift_strength, seed=seed)
    return X_train_normal, X_test, y_test, scaler, feature_names, windows


def eval_ae(ae_model, X, y, threshold, device, batch_size=256):
    """Evaluate Autoencoder"""
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    errors = compute_reconstruction_error(ae_model, loader, device)
    y_pred = (errors > threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, errors)
    except:
        auc = 0.0
    
    return acc, prec, rec, f1, auc


def eval_vae(vae_model, X, y, threshold, device, batch_size=256, mc_samples=10):
    """Evaluate VAE"""
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    errors, epi_unc, ale_unc = vae_compute_metrics(vae_model, loader, device, mc_samples)
    y_pred = (errors > threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, errors)
    except:
        auc = 0.0
    
    return acc, prec, rec, f1, auc, epi_unc.mean(), ale_unc.mean()


def eval_bvae(bvae_model, X, y, threshold, device, batch_size=256, mc_samples=10):
    """Evaluate BVAE"""
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    errors, epi_unc, ale_unc = bvae_compute_metrics(bvae_model, loader, device, mc_samples)
    y_pred = (errors > threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, errors)
    except:
        auc = 0.0
    
    return acc, prec, rec, f1, auc, epi_unc.mean(), ale_unc.mean()


def main(n_windows=DEFAULT_N_WINDOWS, drift_type='sudden', drift_strength=DEFAULT_DRIFT_STRENGTH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)
    print("Drift Evaluation: AE vs VAE vs VAE-CL vs BVAE vs BVAE-CL")
    print("=" * 70)
    print(f"Drift Type: {drift_type}, Strength: {drift_strength}, Windows: {n_windows}\n")
    
    # 1. Load Stream
    X_train_normal, X_test, y_test, scaler, feature_names, windows = get_drift_stream(
        TRAIN_PATH, TEST_PATH,
        n_windows=n_windows, drift_type=drift_type, drift_strength=drift_strength,
    )
    
    # 2. Initialize Replay Buffer (normal data only)
    print(f"Initializing Replay Buffer (Size: {REPLAY_SIZE})...")
    indices = np.random.choice(len(X_train_normal), size=min(len(X_train_normal), REPLAY_SIZE), replace=False)
    X_replay = X_train_normal[indices]
    y_replay = np.zeros(len(X_replay))  # All normal
    print(f"Replay buffer created with {len(X_replay)} normal samples.\n")
    
    # 3. Load AE Model
    print(f"Loading AE from {AE_MODEL_PATH}")
    ae_ckpt = torch.load(AE_MODEL_PATH, map_location=device)
    ae_model = Autoencoder(ae_ckpt['input_dim'], ae_ckpt['latent_dim'], ae_ckpt['hidden_dims']).to(device)
    ae_model.load_state_dict(ae_ckpt['model_state_dict'])
    ae_threshold = np.load(AE_THRESHOLD_PATH)
    print(f"AE Threshold: {ae_threshold:.6f}\n")
    
    # 4. Load VAE Models
    print(f"Loading VAE from {VAE_MODEL_PATH}")
    vae_ckpt = torch.load(VAE_MODEL_PATH, map_location=device)
    
    # VAE Static (no updates)
    vae_static = VAE(vae_ckpt['input_dim'], vae_ckpt['latent_dim'], 
                     vae_ckpt['hidden_dims'], vae_ckpt['dropout_rate']).to(device)
    vae_static.load_state_dict(vae_ckpt['model_state_dict'])
    
    # VAE-CL (continual learning)
    vae_cl = VAE(vae_ckpt['input_dim'], vae_ckpt['latent_dim'],
                 vae_ckpt['hidden_dims'], vae_ckpt['dropout_rate']).to(device)
    vae_cl.load_state_dict(copy.deepcopy(vae_ckpt['model_state_dict']))
    
    vae_threshold = np.load(VAE_THRESHOLD_PATH)
    print(f"VAE Threshold: {vae_threshold:.6f}\n")
    
    # 5. Load BVAE Models
    print(f"Loading BVAE from {BVAE_MODEL_PATH}")
    bvae_ckpt = torch.load(BVAE_MODEL_PATH, map_location=device)
    
    # BVAE Static (no updates)
    bvae_static = BayesianVAE(bvae_ckpt['input_dim'], bvae_ckpt['latent_dim'],
                              bvae_ckpt['hidden_dims'], bvae_ckpt['prior_sigma']).to(device)
    bvae_static.load_state_dict(bvae_ckpt['model_state_dict'])
    
    # BVAE-CL (continual learning)
    bvae_cl = BayesianVAE(bvae_ckpt['input_dim'], bvae_ckpt['latent_dim'],
                          bvae_ckpt['hidden_dims'], bvae_ckpt['prior_sigma']).to(device)
    bvae_cl.load_state_dict(copy.deepcopy(bvae_ckpt['model_state_dict']))
    
    bvae_threshold = np.load(BVAE_THRESHOLD_PATH)
    print(f"BVAE Threshold: {bvae_threshold:.6f}\n")
    
    # Initialize prior for VCL (BVAE-CL only)
    prior_params = bvae_cl.get_posterior_params()
    
    results = []
    
    # 6. Stream Loop
    print("=" * 70)
    print("Evaluating on drift windows...")
    print("=" * 70)
    
    for t, (X_w, y_w) in enumerate(windows):
        n_w = len(y_w)
        n_normal = (y_w == 0).sum()
        n_attack = (y_w == 1).sum()
        
        print(f"\nWindow {t}: {n_w} samples (Normal: {n_normal}, Attack: {n_attack})")
        
        # Evaluate BEFORE update (Test-Then-Train)
        ae_acc, ae_prec, ae_rec, ae_f1, ae_auc = eval_ae(
            ae_model, X_w, y_w, ae_threshold, device
        )
        
        vae_s_acc, vae_s_prec, vae_s_rec, vae_s_f1, vae_s_auc, vae_s_epi, vae_s_ale = eval_vae(
            vae_static, X_w, y_w, vae_threshold, device, mc_samples=MC_SAMPLES_EVAL
        )
        
        vae_cl_acc, vae_cl_prec, vae_cl_rec, vae_cl_f1, vae_cl_auc, vae_cl_epi, vae_cl_ale = eval_vae(
            vae_cl, X_w, y_w, vae_threshold, device, mc_samples=MC_SAMPLES_EVAL
        )
        
        bvae_s_acc, bvae_s_prec, bvae_s_rec, bvae_s_f1, bvae_s_auc, bvae_s_epi, bvae_s_ale = eval_bvae(
            bvae_static, X_w, y_w, bvae_threshold, device, mc_samples=MC_SAMPLES_EVAL
        )
        
        bvae_cl_acc, bvae_cl_prec, bvae_cl_rec, bvae_cl_f1, bvae_cl_auc, bvae_cl_epi, bvae_cl_ale = eval_bvae(
            bvae_cl, X_w, y_w, bvae_threshold, device, mc_samples=MC_SAMPLES_EVAL
        )
        
        results.append({
            'window': t,
            'n_samples': n_w,
            'n_normal': n_normal,
            'n_attack': n_attack,
            # AE
            'ae_acc': ae_acc,
            'ae_prec': ae_prec,
            'ae_rec': ae_rec,
            'ae_f1': ae_f1,
            'ae_auc': ae_auc,
            # VAE Static
            'vae_static_acc': vae_s_acc,
            'vae_static_prec': vae_s_prec,
            'vae_static_rec': vae_s_rec,
            'vae_static_f1': vae_s_f1,
            'vae_static_auc': vae_s_auc,
            'vae_static_epistemic': vae_s_epi,
            'vae_static_aleatoric': vae_s_ale,
            # VAE CL
            'vae_cl_acc': vae_cl_acc,
            'vae_cl_prec': vae_cl_prec,
            'vae_cl_rec': vae_cl_rec,
            'vae_cl_f1': vae_cl_f1,
            'vae_cl_auc': vae_cl_auc,
            'vae_cl_epistemic': vae_cl_epi,
            'vae_cl_aleatoric': vae_cl_ale,
            # BVAE Static
            'bvae_static_acc': bvae_s_acc,
            'bvae_static_prec': bvae_s_prec,
            'bvae_static_rec': bvae_s_rec,
            'bvae_static_f1': bvae_s_f1,
            'bvae_static_auc': bvae_s_auc,
            'bvae_static_epistemic': bvae_s_epi,
            'bvae_static_aleatoric': bvae_s_ale,
            # BVAE CL
            'bvae_cl_acc': bvae_cl_acc,
            'bvae_cl_prec': bvae_cl_prec,
            'bvae_cl_rec': bvae_cl_rec,
            'bvae_cl_f1': bvae_cl_f1,
            'bvae_cl_auc': bvae_cl_auc,
            'bvae_cl_epistemic': bvae_cl_epi,
            'bvae_cl_aleatoric': bvae_cl_ale,
        })
        
        print(f"  AE       | F1: {ae_f1:.3f} | AUC: {ae_auc:.3f}")
        print(f"  VAE-S    | F1: {vae_s_f1:.3f} | AUC: {vae_s_auc:.3f} | Epi: {vae_s_epi:.4f}")
        print(f"  VAE-CL   | F1: {vae_cl_f1:.3f} | AUC: {vae_cl_auc:.3f} | Epi: {vae_cl_epi:.4f}")
        print(f"  BVAE-S   | F1: {bvae_s_f1:.3f} | AUC: {bvae_s_auc:.3f} | Epi: {bvae_s_epi:.4f}")
        print(f"  BVAE-CL  | F1: {bvae_cl_f1:.3f} | AUC: {bvae_cl_auc:.3f} | Epi: {bvae_cl_epi:.4f}")
        
        # 7. Update VAE-CL with Replay
        X_w_normal = X_w[y_w == 0] if (y_w == 0).sum() > 0 else X_w[:0]
        y_w_normal = np.zeros(len(X_w_normal))
        
        if len(X_w_normal) > 0:
            X_update = np.concatenate([X_w_normal, X_replay], axis=0)
            y_update = np.concatenate([y_w_normal, y_replay], axis=0)
        else:
            X_update = X_replay
            y_update = y_replay
        
        print(f"  Updating VAE-CL with {len(X_update)} samples ({len(X_w_normal)} new + {len(X_replay)} replay)...")
        vae_cl = continual_update_vae(
            vae_cl, X_update, y_update, device,
            n_epochs=VAE_CL_N_EPOCHS,
            batch_size=VAE_CL_BATCH_SIZE,
            lr=VAE_CL_LR,
            beta=VAE_CL_BETA,
        )
        
        # 8. Update BVAE-CL with Replay (VCL)
        print(f"  Updating BVAE-CL with {len(X_update)} samples ({len(X_w_normal)} new + {len(X_replay)} replay)...")
        bvae_cl, prior_params = continual_update_bvae(
            bvae_cl, X_update, y_update, device,
            prior_params=prior_params,
            n_epochs=BVAE_CL_N_EPOCHS,
            batch_size=BVAE_CL_BATCH_SIZE,
            lr=BVAE_CL_LR,
            kl_weight_max=BVAE_CL_KL_WEIGHT_MAX,
            weight_kl_scale=BVAE_CL_WEIGHT_KL_SCALE,
            mc_samples_uq=MC_SAMPLES_EVAL,
        )
    
    # 9. Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Results saved to {OUTPUT_CSV}\n")
    
    print("Average F1-Score across all windows:")
    print(f"  AE (Static):      {df['ae_f1'].mean():.3f} ± {df['ae_f1'].std():.3f}")
    print(f"  VAE (Static):     {df['vae_static_f1'].mean():.3f} ± {df['vae_static_f1'].std():.3f}")
    print(f"  VAE-CL (Replay):  {df['vae_cl_f1'].mean():.3f} ± {df['vae_cl_f1'].std():.3f}")
    print(f"  BVAE (Static):    {df['bvae_static_f1'].mean():.3f} ± {df['bvae_static_f1'].std():.3f}")
    print(f"  BVAE-CL (VCL):    {df['bvae_cl_f1'].mean():.3f} ± {df['bvae_cl_f1'].std():.3f}")
    
    print("\nAverage AUC across all windows:")
    print(f"  AE (Static):      {df['ae_auc'].mean():.3f} ± {df['ae_auc'].std():.3f}")
    print(f"  VAE (Static):     {df['vae_static_auc'].mean():.3f} ± {df['vae_static_auc'].std():.3f}")
    print(f"  VAE-CL (Replay):  {df['vae_cl_auc'].mean():.3f} ± {df['vae_cl_auc'].std():.3f}")
    print(f"  BVAE (Static):    {df['bvae_static_auc'].mean():.3f} ± {df['bvae_static_auc'].std():.3f}")
    print(f"  BVAE-CL (VCL):    {df['bvae_cl_auc'].mean():.3f} ± {df['bvae_cl_auc'].std():.3f}")
    
    print("\nAverage Epistemic Uncertainty:")
    print(f"  VAE (Static):     {df['vae_static_epistemic'].mean():.4f}")
    print(f"  VAE-CL (Replay):  {df['vae_cl_epistemic'].mean():.4f}")
    print(f"  BVAE (Static):    {df['bvae_static_epistemic'].mean():.4f}")
    print(f"  BVAE-CL (VCL):    {df['bvae_cl_epistemic'].mean():.4f}")
    
    print("=" * 70)
    
    return df


if __name__ == '__main__':
    main()