"""
QUICK TEST VERSION - For Fast Experimentation
Minimal settings to verify the approach works before full training
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import joblib
warnings.filterwarnings('ignore')

# Import the fixed model class
import sys
sys.path.insert(0, '.')
from bvae_train import (
    ImprovedBayesianVAE,
    semi_supervised_vae_loss,
    compute_reconstruction_metrics,
    load_and_preprocess_data
)

# ============================================================================
# QUICK TEST CONFIGURATION - MINIMAL FOR SPEED
# ============================================================================
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Smaller model
LATENT_DIM = 20
HIDDEN_DIMS = [96, 48]  # Small but not tiny
PRIOR_SIGMA = 1.0

# Minimal training
EPOCHS = 20  # Very few epochs
BATCH_SIZE = 512  # Large batches
LEARNING_RATE = 0.002  # Slightly higher LR for faster convergence
WEIGHT_DECAY = 1e-5

# Semi-supervised
CONTRASTIVE_WEIGHT = 0.8

# Minimal optimization
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 5  # Stop early

# Minimal KL
KL_WEIGHT_MAX = 0.001
KL_WARMUP_EPOCHS = 10  # Fast warmup
WEIGHT_KL_SCALE = 0.0001

# Minimal MC sampling
MC_SAMPLES_TRAIN = 1  # Just 1!
MC_SAMPLES_TEST = 5  # Just 5!

# Model saving
MODEL_PATH = 'models/bvae_quick.pth'
SCALER_PATH = 'models/bvae_quick_scaler.pkl'

VAL_FREQUENCY = 2


def train_quick_bvae():
    """Quick training for testing the approach"""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print("\n" + "=" * 70)
    print("QUICK TEST MODE - Minimal Configuration")
    print("=" * 70)
    print(f"Model: {HIDDEN_DIMS} → {LATENT_DIM}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"MC Samples: {MC_SAMPLES_TEST}")
    print("=" * 70)
    print()
    
    # Load data
    X_train, y_train, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        TRAIN_PATH, TEST_PATH
    )
    
    # OPTIONAL: Use subset for even faster testing
    USE_SUBSET = True
    if USE_SUBSET:
        subset_size = min(50000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Using subset: {len(X_train)} samples\n")
    
    joblib.dump(scaler, SCALER_PATH)
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation
    val_size = min(5000, len(X_train) // 5)
    val_indices = np.random.choice(len(X_train), val_size, replace=False)
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = ImprovedBayesianVAE(input_dim, latent_dim=LATENT_DIM, 
                                hidden_dims=HIDDEN_DIMS, prior_sigma=PRIOR_SIGMA).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    print("Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        kl_weight = min(KL_WEIGHT_MAX, (epoch + 1) / KL_WARMUP_EPOCHS * KL_WEIGHT_MAX)
        
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            is_attack = (labels == 1)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            if torch.isnan(x_recon).any():
                continue
            
            loss, recon_loss, kl_latent = semi_supervised_vae_loss(
                x_recon, data, mu, logvar, is_attack, kl_weight, CONTRASTIVE_WEIGHT
            )
            
            kl_weights = model.total_kl_weights() / len(train_dataset)
            total_loss = (loss / len(data)) + WEIGHT_KL_SCALE * kl_weights
            
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            
            train_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if (epoch + 1) % VAL_FREQUENCY == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    is_attack = (labels == 1)
                    
                    x_recon, mu, logvar, z = model(data, sample=True)
                    loss, _, _ = semi_supervised_vae_loss(
                        x_recon, data, mu, logvar, is_attack, kl_weight, CONTRASTIVE_WEIGHT
                    )
                    kl_weights = model.total_kl_weights() / len(val_dataset)
                    total_loss = (loss / len(data)) + WEIGHT_KL_SCALE * kl_weights
                    val_loss += total_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'input_dim': input_dim,
                    'latent_dim': LATENT_DIM,
                    'hidden_dims': HIDDEN_DIMS,
                    'prior_sigma': PRIOR_SIGMA,
                }, MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Quick Evaluation")
    print("=" * 70)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_errors, test_epi, test_ale = compute_reconstruction_metrics(
        model, test_loader, device, mc_samples=MC_SAMPLES_TEST
    )
    
    print(f"\nReconstruction Error:")
    print(f"  Normal: {test_errors[y_test == 0].mean():.6f}")
    print(f"  Attack: {test_errors[y_test == 1].mean():.6f}")
    sep = (test_errors[y_test == 1].mean() - test_errors[y_test == 0].mean()) / test_errors[y_test == 0].std()
    print(f"  Separation: {sep:.2f}σ")
    
    if sep > 2.0:
        print("\n✅ Good separation! Ready for full training.")
    elif sep > 1.0:
        print("\n✓ Moderate separation. Consider tuning CONTRASTIVE_WEIGHT.")
    else:
        print("\n⚠ Poor separation. Check model or increase CONTRASTIVE_WEIGHT.")
    
    print(f"\nEpistemic Uncertainty:")
    print(f"  Normal: {test_epi[y_test == 0].mean():.2f}")
    print(f"  Attack: {test_epi[y_test == 1].mean():.2f}")
    ratio = test_epi[y_test == 1].mean() / (test_epi[y_test == 0].mean() + 1e-8)
    print(f"  Ratio: {ratio:.2f}x")
    
    if ratio > 3.0:
        print("\n✅ Attacks have high epistemic uncertainty!")
    elif ratio > 1.5:
        print("\n✓ Some epistemic separation.")
    else:
        print("\n⚠ Low epistemic separation.")
    
    print("\n" + "=" * 70)
    print(f"Model saved to: {MODEL_PATH}")
    print("If results look good, run full training with bvae_train.py")
    print("=" * 70)
    
    return model


if __name__ == '__main__':
    train_quick_bvae()