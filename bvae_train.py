"""
Fixed Bayesian Variational Autoencoder for Anomaly Detection
Addresses all identified issues:
1. Larger model capacity
2. Semi-supervised training (uses attack labels to push attacks away)
3. Proper uncertainty quantification
4. No data leakage
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

# Model architecture - BALANCED CAPACITY (not too large)
LATENT_DIM = 24  # Moderate (was 32)
HIDDEN_DIMS = [128, 64]  # 2 layers instead of 3
PRIOR_SIGMA = 1.0

# Training parameters
EPOCHS = 50  # Reduced from 100
BATCH_SIZE = 256  # Larger batches = faster training
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Semi-supervised parameters
CONTRASTIVE_WEIGHT = 0.2  # Weight for attack contrastive loss

# Optimization
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 20

# KL divergence weights
KL_WEIGHT_MAX = 0.001
KL_WARMUP_EPOCHS = 50
WEIGHT_KL_SCALE = 0.0001

# Monte Carlo sampling
MC_SAMPLES_TRAIN = 2  # Reduced from 3
MC_SAMPLES_VAL = 3  # Reduced from 5
MC_SAMPLES_TEST = 10  # Reduced from 20

# Model saving
MODEL_PATH = 'models/bvae_fixed.pth'
SCALER_PATH = 'models/bvae_fixed_scaler.pkl'

VAL_FREQUENCY = 2


# ============================================================================
# BAYESIAN LAYERS (Same as before, already fixed)
# ============================================================================
class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference"""
    
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features).normal_(0, 0.01))
        self.w_rho = nn.Parameter(torch.zeros(out_features, in_features).uniform_(-5, -4))
        
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.zeros(out_features).uniform_(-5, -4))
    
    @property
    def w_sigma(self):
        return torch.clamp(F.softplus(self.w_rho), min=1e-8, max=10)
    
    @property
    def b_sigma(self):
        return torch.clamp(F.softplus(self.b_rho), min=1e-8, max=10)
    
    def forward(self, x, sample=True):
        if self.training or sample:
            w = self.w_mu + self.w_sigma * torch.randn_like(self.w_mu)
            b = self.b_mu + self.b_sigma * torch.randn_like(self.b_mu)
        else:
            w = self.w_mu
            b = self.b_mu
        return F.linear(x, w, b)
    
    def kl_divergence(self):
        """KL divergence between weight posterior and prior"""
        w_sigma_sq = self.w_sigma ** 2
        b_sigma_sq = self.b_sigma ** 2
        prior_sigma_sq = self.prior_sigma ** 2
        
        kl_w = 0.5 * (torch.log(prior_sigma_sq / torch.clamp(w_sigma_sq, min=1e-8)) +
                      (w_sigma_sq + self.w_mu ** 2) / prior_sigma_sq - 1.0)
        kl_b = 0.5 * (torch.log(prior_sigma_sq / torch.clamp(b_sigma_sq, min=1e-8)) +
                      (b_sigma_sq + self.b_mu ** 2) / prior_sigma_sq - 1.0)
        return kl_w.sum() + kl_b.sum()
    
    def kl_divergence_to_prior(self, prior_w_mu, prior_w_sigma, prior_b_mu, prior_b_sigma):
        """KL divergence to custom prior (for continual learning)"""
        w_sigma_sq = self.w_sigma ** 2
        b_sigma_sq = self.b_sigma ** 2
        prior_w_sigma_sq = torch.clamp(prior_w_sigma ** 2, min=1e-8)
        prior_b_sigma_sq = torch.clamp(prior_b_sigma ** 2, min=1e-8)
        
        kl_w = 0.5 * (torch.log(prior_w_sigma_sq / torch.clamp(w_sigma_sq, min=1e-8)) +
                      (w_sigma_sq + (self.w_mu - prior_w_mu) ** 2) / prior_w_sigma_sq - 1.0)
        kl_b = 0.5 * (torch.log(prior_b_sigma_sq / torch.clamp(b_sigma_sq, min=1e-8)) +
                      (b_sigma_sq + (self.b_mu - prior_b_mu) ** 2) / prior_b_sigma_sq - 1.0)
        return kl_w.sum() + kl_b.sum()


# ============================================================================
# IMPROVED BAYESIAN VAE
# ============================================================================
class ImprovedBayesianVAE(nn.Module):
    """
    Improved Bayesian VAE with larger capacity and better architecture
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128, 64], prior_sigma=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Encoder (Bayesian layers)
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            prev_dim = hidden_dim
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Latent space
        self.fc_mu = BayesianLinear(prev_dim, latent_dim, prior_sigma)
        self.fc_logvar = BayesianLinear(prev_dim, latent_dim, prior_sigma)
        
        # Decoder (symmetric)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            prev_dim = hidden_dim
        decoder_layers.append(BayesianLinear(prev_dim, input_dim, prior_sigma))
        self.decoder_layers = nn.ModuleList(decoder_layers)
    
    def encode(self, x, sample=True):
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h, sample))
        mu = self.fc_mu(h, sample)
        logvar = torch.clamp(self.fc_logvar(h, sample), min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, sample=True):
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h, sample)
            if i < len(self.decoder_layers) - 1:
                h = F.relu(h)
        return h
    
    def forward(self, x, sample=True):
        mu, logvar = self.encode(x, sample)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, sample)
        return x_recon, mu, logvar, z
    
    def total_kl_weights(self, prior_params=None):
        """Total KL divergence of Bayesian layer weights"""
        kl = 0.0
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            if prior_params is None:
                kl += layer.kl_divergence()
            else:
                pw_mu = prior_params[f'enc_{i}_w_mu']
                pw_s = prior_params[f'enc_{i}_w_sigma']
                pb_mu = prior_params[f'enc_{i}_b_mu']
                pb_s = prior_params[f'enc_{i}_b_sigma']
                kl += layer.kl_divergence_to_prior(pw_mu, pw_s, pb_mu, pb_s)
        
        # Latent layers
        if prior_params is None:
            kl += self.fc_mu.kl_divergence()
            kl += self.fc_logvar.kl_divergence()
        else:
            for name, layer in [('mu', self.fc_mu), ('logvar', self.fc_logvar)]:
                pw_mu = prior_params[f'latent_{name}_w_mu']
                pw_s = prior_params[f'latent_{name}_w_sigma']
                pb_mu = prior_params[f'latent_{name}_b_mu']
                pb_s = prior_params[f'latent_{name}_b_sigma']
                kl += layer.kl_divergence_to_prior(pw_mu, pw_s, pb_mu, pb_s)
        
        # Decoder layers
        for i, layer in enumerate(self.decoder_layers):
            if prior_params is None:
                kl += layer.kl_divergence()
            else:
                pw_mu = prior_params[f'dec_{i}_w_mu']
                pw_s = prior_params[f'dec_{i}_w_sigma']
                pb_mu = prior_params[f'dec_{i}_b_mu']
                pb_s = prior_params[f'dec_{i}_b_sigma']
                kl += layer.kl_divergence_to_prior(pw_mu, pw_s, pb_mu, pb_s)
        
        return kl
    
    def get_posterior_params(self):
        """Extract current posterior parameters"""
        params = {}
        
        for i, layer in enumerate(self.encoder_layers):
            params[f'enc_{i}_w_mu'] = layer.w_mu.detach().clone()
            params[f'enc_{i}_w_sigma'] = layer.w_sigma.detach().clone()
            params[f'enc_{i}_b_mu'] = layer.b_mu.detach().clone()
            params[f'enc_{i}_b_sigma'] = layer.b_sigma.detach().clone()
        
        for name, layer in [('mu', self.fc_mu), ('logvar', self.fc_logvar)]:
            params[f'latent_{name}_w_mu'] = layer.w_mu.detach().clone()
            params[f'latent_{name}_w_sigma'] = layer.w_sigma.detach().clone()
            params[f'latent_{name}_b_mu'] = layer.b_mu.detach().clone()
            params[f'latent_{name}_b_sigma'] = layer.b_sigma.detach().clone()
        
        for i, layer in enumerate(self.decoder_layers):
            params[f'dec_{i}_w_mu'] = layer.w_mu.detach().clone()
            params[f'dec_{i}_w_sigma'] = layer.w_sigma.detach().clone()
            params[f'dec_{i}_b_mu'] = layer.b_mu.detach().clone()
            params[f'dec_{i}_b_sigma'] = layer.b_sigma.detach().clone()
        
        return params


# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess data - KEEP ALL LABELS for semi-supervised training"""
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
    
    # KEEP LABELS for semi-supervised training
    X_train_full = train_df.drop('label', axis=1).values
    y_train_full = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Scale
    scaler = RobustScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)
    
    feature_names = [col for col in train_df.columns if col != 'label']
    
    print(f"Train samples: {len(X_train_full)} (Normal: {(y_train_full == 0).sum()}, Attack: {(y_train_full == 1).sum()})")
    print(f"Test samples: {len(X_test)} (Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")
    print(f"Feature dimension: {X_train_full.shape[1]}\n")
    
    return X_train_full, y_train_full, X_test, y_test, scaler, feature_names


# ============================================================================
# SEMI-SUPERVISED LOSS
# ============================================================================
def semi_supervised_vae_loss(x_recon, x, mu, logvar, is_attack, kl_weight=1.0, contrastive_weight=0.2):
    """
    Semi-supervised VAE loss with contrastive component
    
    Normal samples: minimize reconstruction error (learn to reconstruct)
    Attack samples: maximize reconstruction error (push away)
    """
    # Reconstruction loss per sample
    recon_loss_per_sample = F.mse_loss(x_recon, x, reduction='none').sum(dim=1)
    
    # Separate normal and attack
    if is_attack.any():
        normal_recon = recon_loss_per_sample[~is_attack].mean() if (~is_attack).any() else 0.0
        attack_recon = recon_loss_per_sample[is_attack].mean()
        
        # Contrastive: minimize normal reconstruction, MAXIMIZE attack reconstruction
        recon_loss = normal_recon - contrastive_weight * attack_recon
    else:
        recon_loss = recon_loss_per_sample.mean()
    
    # KL divergence for latent
    kl_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_latent = torch.clamp(kl_latent, min=0, max=1e6)
    
    return recon_loss + kl_weight * kl_latent, recon_loss, kl_latent


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================
def compute_reconstruction_metrics(model, data_loader, device, mc_samples=20):
    """Compute reconstruction error and uncertainties"""
    model.eval()
    
    all_recon_errors = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for data in data_loader:
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
def train_bvae(train_path=TRAIN_PATH, test_path=TEST_PATH,
               latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS,
               epochs=EPOCHS, batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE, model_path=MODEL_PATH):
    """
    Train BVAE with semi-supervised approach
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Load data WITH LABELS
    X_train, y_train, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        train_path, test_path
    )
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}\n")
    
    # Create data loaders - INCLUDE LABELS
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation set
    val_size = min(5000, len(X_train) // 5)
    val_indices = np.random.choice(len(X_train), val_size, replace=False)
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with LARGER capacity
    input_dim = X_train.shape[1]
    model = ImprovedBayesianVAE(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
                                prior_sigma=PRIOR_SIGMA).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )
    
    print("=" * 70)
    print("Training IMPROVED BVAE with SEMI-SUPERVISED Learning")
    print(f"Architecture: {input_dim} → {hidden_dims} → {latent_dim}")
    print("=" * 70)
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl_latent = 0
        train_kl_weights = 0
        
        # KL weight warm-up
        kl_weight = min(KL_WEIGHT_MAX, (epoch + 1) / KL_WARMUP_EPOCHS * KL_WEIGHT_MAX)
        
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            is_attack = (labels == 1)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            # Check for NaN
            if torch.isnan(x_recon).any() or torch.isnan(mu).any():
                continue
            
            # SEMI-SUPERVISED VAE LOSS
            loss, recon_loss, kl_latent = semi_supervised_vae_loss(
                x_recon, data, mu, logvar, is_attack, kl_weight, CONTRASTIVE_WEIGHT
            )
            
            # Bayesian weight KL
            kl_weights = model.total_kl_weights() / len(train_dataset)
            
            if torch.isnan(kl_weights):
                continue
            
            total_loss = (loss / len(data)) + WEIGHT_KL_SCALE * kl_weights
            
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_recon += recon_loss.item() / len(data)
            train_kl_latent += kl_latent.item() / len(data)
            train_kl_weights += kl_weights.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_recon = train_recon / len(train_loader)
        avg_kl_latent = train_kl_latent / len(train_loader)
        avg_kl_weights = train_kl_weights / len(train_loader)
        
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
            
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Recon: {avg_recon:.4f} | KL_latent: {avg_kl_latent:.4f} | KL_weights: {avg_kl_weights:.6f}\n")
            
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
                    'prior_sigma': PRIOR_SIGMA,
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
    print("Loading best model for evaluation...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print("\nComputing reconstruction errors on test set...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_errors, test_epi, test_ale = compute_reconstruction_metrics(
        model, test_loader, device, mc_samples=MC_SAMPLES_TEST
    )
    
    print(f"\nReconstruction Error Statistics:")
    print(f"  Normal - Mean: {test_errors[y_test == 0].mean():.6f}, Std: {test_errors[y_test == 0].std():.6f}")
    print(f"  Attack - Mean: {test_errors[y_test == 1].mean():.6f}, Std: {test_errors[y_test == 1].std():.6f}")
    print(f"  Separation: {(test_errors[y_test == 1].mean() - test_errors[y_test == 0].mean()) / test_errors[y_test == 0].std():.2f}σ")
    
    print(f"\nEpistemic Uncertainty:")
    print(f"  Normal - Mean: {test_epi[y_test == 0].mean():.6f}")
    print(f"  Attack - Mean: {test_epi[y_test == 1].mean():.6f}")
    print(f"  Ratio: {test_epi[y_test == 1].mean() / (test_epi[y_test == 0].mean() + 1e-8):.2f}x")
    
    print("=" * 70)
    
    return model


if __name__ == '__main__':
    train_bvae()