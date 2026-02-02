"""
Comparison: Standard BVAE-CL vs Uncertainty-Guided BVAE-CL
Shows the benefit of using uncertainty to guide continual learning
"""
import numpy as np
import pandas as pd
import torch
import copy
from torch.utils.data import TensorDataset, DataLoader

from bvae_train import (
    BayesianVAE,
    compute_reconstruction_metrics,
    load_and_preprocess_data,
    MODEL_PATH as BVAE_MODEL_PATH,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Paths
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_CSV = 'models/uncertainty_guided_comparison.csv'

# Settings
N_WINDOWS = 100
DRIFT_TYPE = 'gradual'
DRIFT_STRENGTH = 0.5
REPLAY_SIZE = 2000
MC_SAMPLES = 10

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def compute_sample_uncertainties(model, X, device, mc_samples=10):
    """
    Compute per-sample epistemic and aleatoric uncertainties
    
    Returns:
        epistemic: np.array of shape (n_samples,) - variance across MC samples
        aleatoric: np.array of shape (n_samples,) - mean latent variance
    """
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for data in loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            
            # MC sampling
            recons = []
            logvars = []
            for _ in range(mc_samples):
                x_recon, mu, logvar, z = model(data, sample=True)
                recons.append(x_recon)
                logvars.append(logvar)
            
            recons = torch.stack(recons)  # [mc_samples, batch_size, input_dim]
            logvars = torch.stack(logvars)  # [mc_samples, batch_size, latent_dim]
            
            # Epistemic: variance of reconstructions across MC samples
            epistemic = torch.mean(torch.var(recons, dim=0), dim=1)
            
            # Aleatoric: mean variance in latent space
            aleatoric = torch.mean(torch.exp(logvars.mean(dim=0)), dim=1)
            
            all_epistemic.append(epistemic.cpu().numpy())
            all_aleatoric.append(aleatoric.cpu().numpy())
    
    return np.concatenate(all_epistemic), np.concatenate(all_aleatoric)


def uncertainty_guided_continual_learning(
    model, X, y, device,
    prior_params=None,
    n_epochs=5,
    batch_size=64,
    lr=1e-4,
    base_kl_weight=0.001,
    weight_kl_scale=0.0001,
    mc_samples_uq=10,
    epistemic_threshold=None,
    aleatoric_threshold=None,
):
    """
    Continual learning with uncertainty-guided sample weighting and plasticity.
    
    Key innovations:
    1. **Aleatoric-based sample weighting**: Down-weight noisy samples in loss
    2. **Epistemic-based plasticity**: Adjust KL weight based on model uncertainty
    3. **Selective learning**: Focus on uncertain regions
    
    Args:
        model: BVAE model
        X: New data features
        y: New data labels
        device: torch device
        prior_params: Previous posterior for VCL
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        base_kl_weight: Base KL weight for VAE latent
        weight_kl_scale: KL weight for Bayesian weights
        mc_samples_uq: MC samples for uncertainty quantification
        epistemic_threshold: Threshold for high epistemic uncertainty (auto if None)
        aleatoric_threshold: Threshold for high aleatoric uncertainty (auto if None)
    
    Returns:
        Updated model and posterior parameters
    """
    # Train only on normal data
    X_normal = X[y == 0] if len(X[y == 0]) > 0 else X
    
    if len(X_normal) == 0:
        return model, model.get_posterior_params()
    
    # ========================================================================
    # STEP 1: COMPUTE UNCERTAINTIES FOR ALL SAMPLES
    # ========================================================================
    print("    Computing uncertainties for adaptive learning...")
    epistemic, aleatoric = compute_sample_uncertainties(model, X_normal, device, mc_samples_uq)
    
    # Normalize uncertainties to [0, 1] range
    epistemic_norm = (epistemic - epistemic.min()) / (epistemic.max() - epistemic.min() + 1e-8)
    aleatoric_norm = (aleatoric - aleatoric.min()) / (aleatoric.max() - aleatoric.min() + 1e-8)
    
    # Auto-compute thresholds if not provided (use median)
    if epistemic_threshold is None:
        epistemic_threshold = np.median(epistemic_norm)
    if aleatoric_threshold is None:
        aleatoric_threshold = np.median(aleatoric_norm)
    
    # ========================================================================
    # STEP 2: SAMPLE WEIGHTING BASED ON ALEATORIC UNCERTAINTY
    # High aleatoric = noisy data = down-weight in loss
    # ========================================================================
    # Weight = 1 / (1 + aleatoric_norm)
    # High aleatoric → low weight
    sample_weights = 1.0 / (1.0 + aleatoric_norm)
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)  # Normalize
    
    # ========================================================================
    # STEP 3: ADAPTIVE PLASTICITY BASED ON EPISTEMIC UNCERTAINTY
    # High epistemic = model uncertain = need to learn more = lower KL weight
    # ========================================================================
    mean_epistemic = epistemic_norm.mean()
    
    # Plasticity factor: higher when model is uncertain
    # epistemic high → plasticity high → KL weight low → more learning
    plasticity = 1.0 / (1.0 + mean_epistemic)
    adaptive_kl_weight = base_kl_weight * plasticity
    
    print(f"    Mean Epistemic: {epistemic.mean():.2f}, Plasticity: {plasticity:.3f}")
    print(f"    Mean Aleatoric: {aleatoric.mean():.2f}")
    print(f"    Adaptive KL weight: {adaptive_kl_weight:.6f} (base: {base_kl_weight:.6f})")
    print(f"    High uncertainty samples: {(epistemic_norm > epistemic_threshold).sum()}/{len(epistemic_norm)}")
    print(f"    High noise samples: {(aleatoric_norm > aleatoric_threshold).sum()}/{len(aleatoric_norm)}")
    
    # ========================================================================
    # STEP 4: TRAINING WITH UNCERTAINTY-GUIDED LOSS
    # ========================================================================
    dataset = TensorDataset(
        torch.FloatTensor(X_normal),
        torch.FloatTensor(sample_weights)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        
        for data, weights in loader:
            data = data.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            # Check for NaN
            if torch.isnan(x_recon).any() or torch.isnan(mu).any():
                continue
            
            # ================================================================
            # WEIGHTED VAE LOSS (using aleatoric-based sample weights)
            # ================================================================
            # Reconstruction loss (per sample)
            recon_loss_per_sample = F.mse_loss(x_recon, data, reduction='none').mean(dim=1)
            
            # Weighted reconstruction loss
            weighted_recon_loss = (recon_loss_per_sample * weights).sum()
            
            # KL divergence for latent (not weighted by samples)
            kl_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_latent = torch.clamp(kl_latent, min=0, max=1e6)
            
            # ================================================================
            # BAYESIAN WEIGHT KL WITH VCL
            # ================================================================
            kl_weights = model.total_kl_weights(prior_params) / len(dataset)
            
            if torch.isnan(kl_weights):
                continue
            
            # ================================================================
            # TOTAL LOSS with ADAPTIVE KL WEIGHT
            # ================================================================
            vae_loss = weighted_recon_loss + adaptive_kl_weight * kl_latent
            total_loss = (vae_loss / len(data)) + weight_kl_scale * kl_weights
            
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        if n_batches > 0 and (epoch + 1) % 2 == 0:
            print(f"      Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss/n_batches:.4f}")
    
    return model, model.get_posterior_params()


def standard_continual_learning(model, X, y, device, prior_params=None,
                                n_epochs=5, batch_size=64, lr=1e-4,
                                kl_weight_max=0.001, weight_kl_scale=0.0001):
    """
    Standard continual learning WITHOUT uncertainty guidance (for comparison)
    """
    X_normal = X[y == 0] if len(X[y == 0]) > 0 else X
    
    if len(X_normal) == 0:
        return model, model.get_posterior_params()
    
    dataset = TensorDataset(torch.FloatTensor(X_normal))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for epoch in range(n_epochs):
        for data in loader:
            data = data[0].to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            if torch.isnan(x_recon).any() or torch.isnan(mu).any():
                continue
            
            # Standard VAE loss
            recon_loss = F.mse_loss(x_recon, data, reduction='sum')
            kl_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_latent = torch.clamp(kl_latent, min=0, max=1e6)
            
            # Bayesian weight KL
            kl_weights = model.total_kl_weights(prior_params) / len(dataset)
            
            if torch.isnan(kl_weights):
                continue
            
            total_loss = (recon_loss + kl_weight_max * kl_latent) / len(data) + weight_kl_scale * kl_weights
            
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
    
    return model, model.get_posterior_params()

def make_drift_windows(X_test, y_test, n_windows, drift_type, drift_strength, seed=42):
    """Create drift windows"""
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
        
        if drift_type != 'none' and i > 0:
            if drift_type == 'gradual':
                drift_factor = (i / n_windows) * drift_strength
            else:  # sudden
                drift_factor = drift_strength
            
            feature_drift = rng.standard_normal(X_w.shape[1]) * drift_factor
            X_w = X_w + feature_drift
        
        X_w = np.nan_to_num(X_w, nan=0.0, posinf=1e6, neginf=-1e6)
        windows.append((X_w, y_w))
    
    return windows


def adaptive_threshold(errors, y, percentile=95):
    """Compute adaptive threshold from normal samples"""
    normal_errors = errors[y == 0]
    if len(normal_errors) < 10:
        return np.percentile(errors, percentile)
    return np.percentile(normal_errors, percentile)


def evaluate_bvae(model, X, y, device, mc_samples=10):
    """Evaluate BVAE with adaptive threshold"""
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    errors, epistemic, aleatoric = compute_reconstruction_metrics(model, loader, device, mc_samples)
    
    # Adaptive threshold
    threshold = adaptive_threshold(errors, y, percentile=95)
    y_pred = (errors > threshold).astype(int)
    
    # Metrics
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, errors)
    except:
        auc = 0.0
    
    return f1, auc, threshold, epistemic.mean(), aleatoric.mean()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("UNCERTAINTY-GUIDED CONTINUAL LEARNING COMPARISON")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    # 1. Load data
    X_train_normal, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        TRAIN_PATH, TEST_PATH
    )
    
    # 2. Create drift windows
    windows = make_drift_windows(X_test, y_test, N_WINDOWS, DRIFT_TYPE, DRIFT_STRENGTH)
    
    # 3. Replay buffer
    indices = np.random.choice(len(X_train_normal), size=min(len(X_train_normal), REPLAY_SIZE), replace=False)
    X_replay = X_train_normal[indices]
    y_replay = np.zeros(len(X_replay))
    
    # 4. Load BVAE
    ckpt = torch.load(BVAE_MODEL_PATH, map_location=device)
    
    # BVAE with STANDARD continual learning
    bvae_standard = BayesianVAE(ckpt['input_dim'], ckpt['latent_dim'],
                                ckpt['hidden_dims'], ckpt['prior_sigma']).to(device)
    bvae_standard.load_state_dict(ckpt['model_state_dict'])
    prior_standard = bvae_standard.get_posterior_params()
    
    # BVAE with UNCERTAINTY-GUIDED continual learning
    bvae_guided = BayesianVAE(ckpt['input_dim'], ckpt['latent_dim'],
                              ckpt['hidden_dims'], ckpt['prior_sigma']).to(device)
    bvae_guided.load_state_dict(copy.deepcopy(ckpt['model_state_dict']))
    prior_guided = bvae_guided.get_posterior_params()
    
    # BVAE Static (no updates)
    bvae_static = BayesianVAE(ckpt['input_dim'], ckpt['latent_dim'],
                             ckpt['hidden_dims'], ckpt['prior_sigma']).to(device)
    bvae_static.load_state_dict(copy.deepcopy(ckpt['model_state_dict']))
    
    print("Models initialized.\n")
    
    # 5. Evaluation loop
    results = []
    
    print("=" * 80)
    print("Window-by-Window Comparison")
    print("=" * 80)
    
    for t, (X_w, y_w) in enumerate(windows):
        n_normal = (y_w == 0).sum()
        n_attack = (y_w == 1).sum()
        
        print(f"\n{'='*80}")
        print(f"Window {t}: {len(y_w)} samples (Normal: {n_normal}, Attack: {n_attack})")
        print(f"{'='*80}")
        
        # Evaluate all three variants
        static_f1, static_auc, static_thr, static_epi, static_ale = evaluate_bvae(
            bvae_static, X_w, y_w, device, MC_SAMPLES
        )
        
        standard_f1, standard_auc, standard_thr, standard_epi, standard_ale = evaluate_bvae(
            bvae_standard, X_w, y_w, device, MC_SAMPLES
        )
        
        guided_f1, guided_auc, guided_thr, guided_epi, guided_ale = evaluate_bvae(
            bvae_guided, X_w, y_w, device, MC_SAMPLES
        )
        
        results.append({
            'window': t,
            'n_samples': len(y_w),
            'n_normal': n_normal,
            'n_attack': n_attack,
            # Static
            'static_f1': static_f1,
            'static_auc': static_auc,
            'static_epistemic': static_epi,
            # Standard CL
            'standard_cl_f1': standard_f1,
            'standard_cl_auc': standard_auc,
            'standard_cl_epistemic': standard_epi,
            # Uncertainty-Guided CL
            'guided_cl_f1': guided_f1,
            'guided_cl_auc': guided_auc,
            'guided_cl_epistemic': guided_epi,
        })
        
        print(f"\nResults:")
        print(f"  Static:       F1={static_f1:.3f} | AUC={static_auc:.3f} | Epi={static_epi:.2f}")
        print(f"  Standard-CL:  F1={standard_f1:.3f} | AUC={standard_auc:.3f} | Epi={standard_epi:.2f}")
        print(f"  Guided-CL:    F1={guided_f1:.3f} | AUC={guided_auc:.3f} | Epi={guided_epi:.2f}")
        
        # Prepare update data
        X_w_normal = X_w[y_w == 0] if n_normal > 0 else X_w[:0]
        y_w_normal = np.zeros(len(X_w_normal))
        
        if len(X_w_normal) > 0:
            X_update = np.concatenate([X_w_normal, X_replay], axis=0)
            y_update = np.concatenate([y_w_normal, y_replay], axis=0)
        else:
            X_update = X_replay
            y_update = y_replay
        
        # Update Standard CL (no uncertainty guidance)
        print(f"\nUpdating Standard-CL...")
        bvae_standard, prior_standard = standard_continual_learning(
            bvae_standard, X_update, y_update, device,
            prior_params=prior_standard,
            n_epochs=5,
            batch_size=64,
            lr=1e-4,
            kl_weight_max=0.001,
            weight_kl_scale=0.0001,
        )
        
        # Update Uncertainty-Guided CL
        print(f"\nUpdating Uncertainty-Guided-CL...")
        bvae_guided, prior_guided = uncertainty_guided_continual_learning(
            bvae_guided, X_update, y_update, device,
            prior_params=prior_guided,
            n_epochs=5,
            batch_size=64,
            lr=1e-4,
            base_kl_weight=0.001,
            weight_kl_scale=0.0001,
            mc_samples_uq=MC_SAMPLES,
        )
    
    # 6. Save and summarize
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\nAverage F1-Score:")
    print(f"  Static (No CL):           {df['static_f1'].mean():.3f} ± {df['static_f1'].std():.3f}")
    print(f"  Standard CL:              {df['standard_cl_f1'].mean():.3f} ± {df['standard_cl_f1'].std():.3f}")
    print(f"  Uncertainty-Guided CL:    {df['guided_cl_f1'].mean():.3f} ± {df['guided_cl_f1'].std():.3f}")
    
    print("\nAverage AUC:")
    print(f"  Static (No CL):           {df['static_auc'].mean():.3f} ± {df['static_auc'].std():.3f}")
    print(f"  Standard CL:              {df['standard_cl_auc'].mean():.3f} ± {df['standard_cl_auc'].std():.3f}")
    print(f"  Uncertainty-Guided CL:    {df['guided_cl_auc'].mean():.3f} ± {df['guided_cl_auc'].std():.3f}")
    
    print("\nAverage Epistemic Uncertainty:")
    print(f"  Static (No CL):           {df['static_epistemic'].mean():.2f}")
    print(f"  Standard CL:              {df['standard_cl_epistemic'].mean():.2f}")
    print(f"  Uncertainty-Guided CL:    {df['guided_cl_epistemic'].mean():.2f}")
    
    # Improvement analysis
    improvement_std = ((df['standard_cl_f1'].mean() - df['static_f1'].mean()) / df['static_f1'].mean()) * 100
    improvement_guided = ((df['guided_cl_f1'].mean() - df['static_f1'].mean()) / df['static_f1'].mean()) * 100
    relative_improvement = ((df['guided_cl_f1'].mean() - df['standard_cl_f1'].mean()) / df['standard_cl_f1'].mean()) * 100
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(f"Standard CL vs Static:              {improvement_std:+.1f}%")
    print(f"Uncertainty-Guided CL vs Static:    {improvement_guided:+.1f}%")
    print(f"Uncertainty-Guided vs Standard CL:  {relative_improvement:+.1f}%")
    
    if relative_improvement > 5:
        print("\n✅ Uncertainty guidance provides SIGNIFICANT improvement!")
    elif relative_improvement > 0:
        print("\n✓ Uncertainty guidance provides modest improvement.")
    else:
        print("\n⚠ Uncertainty guidance did not improve performance.")
    
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    main()