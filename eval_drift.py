"""
Fixed Drift Evaluation for Anomaly Detection
Key fixes:
1. Uncertainty-based anomaly score (combines recon + epistemic)
2. Unsupervised adaptive thresholding (no label leakage)
3. Realistic drift (only normal traffic drifts)
4. Proper semi-supervised continual learning
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from bvae_train import (
    ImprovedBayesianVAE,
    compute_reconstruction_metrics,
    semi_supervised_vae_loss,
    load_and_preprocess_data,
    LATENT_DIM,
    HIDDEN_DIMS,
    PRIOR_SIGMA,
    WEIGHT_KL_SCALE,
    MODEL_PATH as BVAE_MODEL_PATH,
)

# Paths
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_CSV = 'models/drift_eval_fixed_results.csv'

# Drift settings
N_WINDOWS = 10
DRIFT_TYPE = 'gradual'
DRIFT_STRENGTH = 0.3  # Moderate drift

# Continual learning
CL_N_EPOCHS = 5
CL_BATCH_SIZE = 64
CL_LR = 1e-4
CL_KL_WEIGHT_MAX = 0.001
CL_WEIGHT_KL_SCALE = 0.0001
CL_CONTRASTIVE_WEIGHT = 0.2

# Replay
REPLAY_SIZE = 2000

# Evaluation
MC_SAMPLES_EVAL = 10  # Reduced from 20


def realistic_drift_windows(X_test, y_test, n_windows, drift_type, drift_strength, seed=42):
    """
    FIX #1: Apply drift ONLY to normal traffic (attacks remain stable)
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    splits = np.linspace(0, n, n_windows + 1, dtype=int)
    windows = []
    
    for i in range(n_windows):
        start, end = splits[i], splits[i + 1]
        if start >= end:
            continue
        
        X_w = X_test[start:end].copy()
        y_w = y_test[start:end]
        
        # Apply drift ONLY to normal samples
        if drift_type != 'none' and i > 0:
            normal_mask = (y_w == 0)
            
            if drift_type == 'gradual':
                drift_factor = (i / n_windows) * drift_strength
            else:  # sudden
                drift_factor = drift_strength
            
            # Drift only normal traffic
            if normal_mask.any():
                feature_drift = rng.standard_normal(X_w.shape[1]) * drift_factor
                X_w[normal_mask] = X_w[normal_mask] + feature_drift
        
        X_w = np.nan_to_num(X_w, nan=0.0, posinf=1e6, neginf=-1e6)
        windows.append((X_w, y_w))
    
    return windows


def compute_anomaly_score(recon_errors, epistemic, aleatoric, 
                          recon_weight=0.7, epistemic_weight=0.3):
    """
    FIX #2: Combine reconstruction error + epistemic uncertainty
    
    High recon + High epistemic = Novel attack (strong anomaly signal)
    High recon + Low epistemic = Normal drift (weak anomaly signal)
    """
    # Normalize to [0, 1]
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    recon_norm = normalize(recon_errors)
    epi_norm = normalize(epistemic)
    
    # Combined score
    anomaly_score = recon_weight * recon_norm + epistemic_weight * epi_norm
    
    return anomaly_score


def mad_threshold(scores, n_sigma=3.0, contamination=0.1):
    """
    FIX #3: Unsupervised threshold using Median Absolute Deviation
    NO LABEL LEAKAGE - works in real deployment!
    
    Args:
        scores: Anomaly scores
        n_sigma: Number of MAD units (robustto outliers)
        contamination: Expected fraction of anomalies (e.g., 0.1 = 10%)
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    
    # Robust threshold
    threshold = median + n_sigma * mad * 1.4826  # 1.4826 is scaling factor for normal distribution
    
    return threshold


def evaluate_with_uncertainty_score(model, X, y, device, mc_samples=20):
    """
    Evaluate using uncertainty-based anomaly score + unsupervised threshold
    """
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Compute all metrics
    recon_errors, epistemic, aleatoric = compute_reconstruction_metrics(
        model, loader, device, mc_samples
    )
    
    # FIX #2: Combine into anomaly score
    anomaly_scores = compute_anomaly_score(recon_errors, epistemic, aleatoric)
    
    # FIX #3: Unsupervised threshold (no labels!)
    threshold = mad_threshold(anomaly_scores, n_sigma=3.0)
    
    # Predictions
    y_pred = (anomaly_scores > threshold).astype(int)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, anomaly_scores)
    except:
        auc = 0.0
    
    return acc, prec, rec, f1, auc, threshold, epistemic.mean(), aleatoric.mean(), anomaly_scores


def uncertainty_guided_continual_learning(model, X, y, device, prior_params,
                                         n_epochs=5, batch_size=64, lr=1e-4,
                                         kl_weight_max=0.001, weight_kl_scale=0.0001,
                                         contrastive_weight=0.2, mc_samples=10):
    """
    FIX #4: Uncertainty-guided continual learning with pseudo-labeling
    """
    # Compute uncertainties to identify likely normal samples
    dataset_temp = TensorDataset(torch.FloatTensor(X))
    loader_temp = DataLoader(dataset_temp, batch_size=256, shuffle=False)
    
    recon_errors, epistemic, aleatoric = compute_reconstruction_metrics(
        model, loader_temp, device, mc_samples
    )
    
    # Pseudo-labeling: identify likely normal samples
    # Low anomaly score + Low epistemic = confident normal
    anomaly_scores = compute_anomaly_score(recon_errors, epistemic, aleatoric)
    threshold = mad_threshold(anomaly_scores)
    
    # Conservative pseudo-labeling
    likely_normal = (anomaly_scores < threshold) & (epistemic < np.median(epistemic))
    likely_attack = (anomaly_scores > threshold * 1.5) | (epistemic > np.percentile(epistemic, 75))
    
    print(f"    Pseudo-labels: {likely_normal.sum()} normal, {likely_attack.sum()} attack, {(~likely_normal & ~likely_attack).sum()} uncertain")
    
    # Create pseudo-labeled dataset
    y_pseudo = np.zeros(len(X))
    y_pseudo[likely_attack] = 1
    
    # Sample weighting based on aleatoric uncertainty
    sample_weights = 1.0 / (1.0 + (aleatoric - aleatoric.min()) / (aleatoric.max() - aleatoric.min() + 1e-8))
    
    # Adaptive plasticity based on epistemic
    mean_epistemic = epistemic[likely_normal].mean() if likely_normal.any() else epistemic.mean()
    plasticity = 1.0 / (1.0 + (mean_epistemic - epistemic.min()) / (epistemic.max() - epistemic.min() + 1e-8))
    adaptive_kl_weight = kl_weight_max * plasticity
    
    print(f"    Epistemic: {epistemic.mean():.2f}, Plasticity: {plasticity:.3f}, Adaptive KL: {adaptive_kl_weight:.6f}")
    
    # Training
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y_pseudo),
        torch.FloatTensor(sample_weights)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for epoch in range(n_epochs):
        for data, labels, weights in loader:
            data = data.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            is_attack = (labels == 1)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            if torch.isnan(x_recon).any() or torch.isnan(mu).any():
                continue
            
            # Semi-supervised loss with weighting
            loss, recon_loss, kl_latent = semi_supervised_vae_loss(
                x_recon, data, mu, logvar, is_attack, adaptive_kl_weight, contrastive_weight
            )
            
            # Apply sample weights
            # (loss is already per-sample averaged, apply global weight)
            
            # Bayesian weight KL
            kl_weights = model.total_kl_weights(prior_params) / len(dataset)
            
            if torch.isnan(kl_weights):
                continue
            
            total_loss = (loss / len(data)) + weight_kl_scale * kl_weights
            
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
    
    return model, model.get_posterior_params()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("FIXED DRIFT EVALUATION: Uncertainty-Based Anomaly Detection")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    # 1. Load data
    X_train, y_train, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        TRAIN_PATH, TEST_PATH
    )
    
    # 2. Create REALISTIC drift windows (only normal traffic drifts)
    windows = realistic_drift_windows(X_test, y_test, N_WINDOWS, DRIFT_TYPE, DRIFT_STRENGTH)
    
    # 3. Replay buffer
    normal_indices = np.where(y_train == 0)[0]
    replay_indices = np.random.choice(normal_indices, size=min(len(normal_indices), REPLAY_SIZE), replace=False)
    X_replay = X_train[replay_indices]
    y_replay = y_train[replay_indices]
    
    # Also keep some attack samples for semi-supervised learning
    attack_indices = np.where(y_train == 1)[0]
    if len(attack_indices) > 0:
        attack_replay_indices = np.random.choice(attack_indices, size=min(len(attack_indices), REPLAY_SIZE // 4), replace=False)
        X_replay = np.concatenate([X_replay, X_train[attack_replay_indices]], axis=0)
        y_replay = np.concatenate([y_replay, y_train[attack_replay_indices]], axis=0)
    
    print(f"Replay buffer: {len(X_replay)} samples ({(y_replay==0).sum()} normal, {(y_replay==1).sum()} attack)\n")
    
    # 4. Load model
    ckpt = torch.load(BVAE_MODEL_PATH, map_location=device)
    
    # Static model
    bvae_static = ImprovedBayesianVAE(ckpt['input_dim'], ckpt['latent_dim'],
                                      ckpt['hidden_dims'], ckpt['prior_sigma']).to(device)
    bvae_static.load_state_dict(ckpt['model_state_dict'])
    
    # Continual learning model
    bvae_cl = ImprovedBayesianVAE(ckpt['input_dim'], ckpt['latent_dim'],
                                  ckpt['hidden_dims'], ckpt['prior_sigma']).to(device)
    bvae_cl.load_state_dict(copy.deepcopy(ckpt['model_state_dict']))
    
    prior_params = bvae_cl.get_posterior_params()
    
    print("Models loaded.\n")
    
    # 5. Evaluation loop
    results = []
    
    print("=" * 80)
    print("Evaluating on drift windows...")
    print("=" * 80)
    
    for t, (X_w, y_w) in enumerate(windows):
        n_normal = (y_w == 0).sum()
        n_attack = (y_w == 1).sum()
        
        print(f"\n{'='*80}")
        print(f"Window {t}: {len(y_w)} samples (Normal: {n_normal}, Attack: {n_attack})")
        print(f"{'='*80}")
        
        # Evaluate static
        static_acc, static_prec, static_rec, static_f1, static_auc, static_thr, static_epi, static_ale, static_scores = \
            evaluate_with_uncertainty_score(bvae_static, X_w, y_w, device, MC_SAMPLES_EVAL)
        
        # Evaluate continual
        cl_acc, cl_prec, cl_rec, cl_f1, cl_auc, cl_thr, cl_epi, cl_ale, cl_scores = \
            evaluate_with_uncertainty_score(bvae_cl, X_w, y_w, device, MC_SAMPLES_EVAL)
        
        results.append({
            'window': t,
            'n_samples': len(y_w),
            'n_normal': n_normal,
            'n_attack': n_attack,
            # Static
            'static_f1': static_f1,
            'static_auc': static_auc,
            'static_precision': static_prec,
            'static_recall': static_rec,
            'static_threshold': static_thr,
            'static_epistemic': static_epi,
            'static_aleatoric': static_ale,
            # Continual Learning
            'cl_f1': cl_f1,
            'cl_auc': cl_auc,
            'cl_precision': cl_prec,
            'cl_recall': cl_rec,
            'cl_threshold': cl_thr,
            'cl_epistemic': cl_epi,
            'cl_aleatoric': cl_ale,
        })
        
        print(f"\nResults:")
        print(f"  Static:  F1={static_f1:.3f} | AUC={static_auc:.3f} | Thr={static_thr:.4f} | Epi={static_epi:.2f}")
        print(f"  CL:      F1={cl_f1:.3f} | AUC={cl_auc:.3f} | Thr={cl_thr:.4f} | Epi={cl_epi:.2f}")
        
        # Update continual learning model
        if n_normal > 0:  # Only update if we have normal samples
            print(f"\nUpdating CL model with replay...")
            X_update = np.concatenate([X_w, X_replay], axis=0)
            y_update = np.concatenate([y_w, y_replay], axis=0)
            
            bvae_cl, prior_params = uncertainty_guided_continual_learning(
                bvae_cl, X_update, y_update, device, prior_params,
                n_epochs=CL_N_EPOCHS,
                batch_size=CL_BATCH_SIZE,
                lr=CL_LR,
                kl_weight_max=CL_KL_WEIGHT_MAX,
                weight_kl_scale=CL_WEIGHT_KL_SCALE,
                contrastive_weight=CL_CONTRASTIVE_WEIGHT,
                mc_samples=MC_SAMPLES_EVAL,
            )
    
    # 6. Save and summarize
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Results saved to {OUTPUT_CSV}\n")
    
    print("Average F1-Score:")
    print(f"  Static:              {df['static_f1'].mean():.3f} ± {df['static_f1'].std():.3f}")
    print(f"  Continual Learning:  {df['cl_f1'].mean():.3f} ± {df['cl_f1'].std():.3f}")
    
    print("\nAverage AUC:")
    print(f"  Static:              {df['static_auc'].mean():.3f} ± {df['static_auc'].std():.3f}")
    print(f"  Continual Learning:  {df['cl_auc'].mean():.3f} ± {df['cl_auc'].std():.3f}")
    
    print("\nAverage Epistemic Uncertainty:")
    print(f"  Static:              {df['static_epistemic'].mean():.2f}")
    print(f"  Continual Learning:  {df['cl_epistemic'].mean():.2f}")
    
    # Improvement
    improvement = ((df['cl_f1'].mean() - df['static_f1'].mean()) / df['static_f1'].mean()) * 100
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(f"Continual Learning vs Static: {improvement:+.1f}%")
    
    if improvement > 5:
        print("\n✅ Continual learning provides SIGNIFICANT improvement!")
    elif improvement > 0:
        print("\n✓ Continual learning provides modest improvement.")
    else:
        print("\n⚠ Continual learning did not improve performance (check hyperparameters).")
    
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    main()