"""
Simplified Variational Autoencoder for Anomaly Detection
More stable alternative to full BVAE - uses standard VAE with dropout for uncertainty
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

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Model architecture
LATENT_DIM = 16
HIDDEN_DIMS = [128, 64]
DROPOUT_RATE = 0.2

# Training parameters
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Optimization
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 20

# KL divergence weight
BETA_MAX = 0.001
BETA_WARMUP_EPOCHS = 40

# Monte Carlo samples (using dropout)
MC_SAMPLES_TRAIN = 1
MC_SAMPLES_TEST = 20

# Model saving
MODEL_PATH = 'models/vae.pth'
SCALER_PATH = 'models/vae_scaler.pkl'
THRESHOLD_PATH = 'models/vae_threshold.npy'

VAL_FREQUENCY = 2


# ============================================================================
# SIMPLE VAE MODEL
# ============================================================================
class VAE(nn.Module):
    """Standard Variational Autoencoder with dropout for uncertainty"""
    
    def __init__(self, input_dim, latent_dim=16, hidden_dims=[128, 64], dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess UNSW-NB15 dataset for anomaly detection"""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = train_df.drop('id', axis=1)
    test_df = test_df.drop('id', axis=1)
    
    categorical_cols = ['proto', 'service', 'state']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        le.fit(combined.astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    
    train_df = train_df.drop('attack_cat', axis=1)
    test_df = test_df.drop('attack_cat', axis=1)
    
    X_train_full = train_df.drop('label', axis=1).values
    y_train_full = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Train only on normal data
    X_train_normal = X_train_full[y_train_full == 0]
    
    scaler = RobustScaler()
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_test = scaler.transform(X_test)
    
    feature_names = [col for col in train_df.columns if col != 'label']
    
    print(f"Normal training samples: {len(X_train_normal)}")
    print(f"Test samples: {len(X_test)} (Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")
    print(f"Feature dimension: {X_train_normal.shape[1]}\n")
    
    return X_train_normal, X_test, y_test, scaler, feature_names


# ============================================================================
# LOSS FUNCTION
# ============================================================================
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """VAE loss with beta-VAE formulation"""
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_loss, min=0, max=1e6)
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================
def compute_reconstruction_metrics(model, data_loader, device, mc_samples=20):
    """Compute reconstruction error and uncertainty using MC dropout"""
    model.train()  # Keep dropout active
    
    all_recon_errors = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            
            # MC sampling with dropout
            recons = []
            logvars = []
            for _ in range(mc_samples):
                x_recon, mu, logvar, z = model(data)
                recons.append(x_recon)
                logvars.append(logvar)
            
            recons = torch.stack(recons)
            logvars = torch.stack(logvars)
            
            # Mean reconstruction error
            recon_error = torch.mean((data - recons.mean(dim=0)) ** 2, dim=1)
            
            # Epistemic: variance of reconstructions
            epistemic = torch.mean(torch.var(recons, dim=0), dim=1)
            
            # Aleatoric: mean latent variance
            aleatoric = torch.mean(torch.exp(logvars.mean(dim=0)), dim=1)
            
            all_recon_errors.append(recon_error.cpu().numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_aleatoric.append(aleatoric.cpu().numpy())
    
    return (np.concatenate(all_recon_errors), 
            np.concatenate(all_epistemic), 
            np.concatenate(all_aleatoric))


# ============================================================================
# TRAINING
# ============================================================================
def train_vae(train_path=TRAIN_PATH, test_path=TEST_PATH,
              latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              learning_rate=LEARNING_RATE, model_path=MODEL_PATH):
    """Train VAE on normal data only"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    X_train_normal, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        train_path, test_path
    )
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}\n")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_normal))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_size = min(5000, len(X_train_normal) // 5)
    val_indices = np.random.choice(len(X_train_normal), val_size, replace=False)
    X_val = X_train_normal[val_indices]
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train_normal.shape[1]
    model = VAE(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, 
                dropout_rate=DROPOUT_RATE).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )
    
    print("=" * 70)
    print("Training VAE on Normal Data Only")
    print("=" * 70)
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        # Beta warmup
        beta = min(BETA_MAX, (epoch + 1) / BETA_WARMUP_EPOCHS * BETA_MAX)
        
        for data in train_loader:
            data = data[0].to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data)
            
            loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, logvar, beta)
            
            (loss / len(data)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            
            train_loss += loss.item() / len(data)
            train_recon += recon_loss.item() / len(data)
            train_kl += kl_loss.item() / len(data)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_recon = train_recon / len(train_loader)
        avg_kl = train_kl / len(train_loader)
        
        # Validation
        if (epoch + 1) % VAL_FREQUENCY == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data[0].to(device)
                    x_recon, mu, logvar, z = model(data)
                    loss, _, _ = vae_loss(x_recon, data, mu, logvar, beta)
                    val_loss += loss.item() / len(data)
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Beta: {beta:.6f}\n")
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'hidden_dims': hidden_dims,
                    'dropout_rate': DROPOUT_RATE,
                }, model_path)
                patience_counter = 0
                print(f"  ✓ New best model saved\n")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Load best model
    print("\n" + "=" * 70)
    print("Loading best model...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Compute threshold
    print("Computing optimal threshold...")
    train_errors, _, _ = compute_reconstruction_metrics(
        model, train_loader, device, mc_samples=MC_SAMPLES_TEST
    )
    
    threshold = np.percentile(train_errors, 95)
    np.save(THRESHOLD_PATH, threshold)
    print(f"Threshold (95th percentile): {threshold:.6f}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_errors, test_epi, test_ale = compute_reconstruction_metrics(
        model, test_loader, device, mc_samples=MC_SAMPLES_TEST
    )
    y_pred = (test_errors > threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, test_errors)
    
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  ROC-AUC:   {auc:.4f}")
    print("=" * 70)
    
    return model, threshold, f1


if __name__ == '__main__':
    train_vae()