"""
Bayesian Variational Autoencoder for Anomaly Detection
Combines VAE with Bayesian neural networks for uncertainty-aware anomaly detection
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import joblib
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data paths
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Model architecture
LATENT_DIM = 16
HIDDEN_DIMS = [64, 32]
PRIOR_SIGMA = 1.0

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

# KL divergence weights
KL_WEIGHT_MAX = 0.001  # For VAE latent KL (reduced from 0.01)
KL_WARMUP_EPOCHS = 50  # Longer warmup
WEIGHT_KL_SCALE = 0.0001  # For Bayesian layer weight KL (reduced from 0.001)

# Monte Carlo sampling
MC_SAMPLES_TRAIN = 3
MC_SAMPLES_VAL = 5
MC_SAMPLES_TEST = 10

# Model saving
MODEL_PATH = 'models/bvae.pth'
SCALER_PATH = 'models/bvae_scaler.pkl'
THRESHOLD_PATH = 'models/bvae_threshold.npy'

# Validation frequency
VAL_FREQUENCY = 2


# ============================================================================
# BAYESIAN LAYERS
# ============================================================================
class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference"""
    
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        
        # Weight parameters - better initialization
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features).normal_(0, 0.01))
        self.w_rho = nn.Parameter(torch.zeros(out_features, in_features).uniform_(-5, -4))
        
        # Bias parameters
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
# BAYESIAN VARIATIONAL AUTOENCODER
# ============================================================================
class BayesianVAE(nn.Module):
    """
    Bayesian Variational Autoencoder
    Combines VAE (for learning latent representation) with Bayesian layers (for epistemic uncertainty)
    """
    
    def __init__(self, input_dim, latent_dim=16, hidden_dims=[64, 32], prior_sigma=1.0):
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
        
        # Latent space (mean and log-variance)
        self.fc_mu = BayesianLinear(prev_dim, latent_dim, prior_sigma)
        self.fc_logvar = BayesianLinear(prev_dim, latent_dim, prior_sigma)
        
        # Decoder (Bayesian layers, symmetric)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            prev_dim = hidden_dim
        decoder_layers.append(BayesianLinear(prev_dim, input_dim, prior_sigma))
        self.decoder_layers = nn.ModuleList(decoder_layers)
    
    def encode(self, x, sample=True):
        """Encode input to latent distribution parameters"""
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h, sample))
        mu = self.fc_mu(h, sample)
        logvar = torch.clamp(self.fc_logvar(h, sample), min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, sample=True):
        """Decode latent vector to reconstruction"""
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h, sample)
            if i < len(self.decoder_layers) - 1:  # No activation on last layer
                h = F.relu(h)
        return h
    
    def forward(self, x, sample=True):
        """Forward pass"""
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
        """Extract current posterior parameters for continual learning"""
        params = {}
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            params[f'enc_{i}_w_mu'] = layer.w_mu.detach().clone()
            params[f'enc_{i}_w_sigma'] = layer.w_sigma.detach().clone()
            params[f'enc_{i}_b_mu'] = layer.b_mu.detach().clone()
            params[f'enc_{i}_b_sigma'] = layer.b_sigma.detach().clone()
        
        # Latent
        for name, layer in [('mu', self.fc_mu), ('logvar', self.fc_logvar)]:
            params[f'latent_{name}_w_mu'] = layer.w_mu.detach().clone()
            params[f'latent_{name}_w_sigma'] = layer.w_sigma.detach().clone()
            params[f'latent_{name}_b_mu'] = layer.b_mu.detach().clone()
            params[f'latent_{name}_b_sigma'] = layer.b_sigma.detach().clone()
        
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            params[f'dec_{i}_w_mu'] = layer.w_mu.detach().clone()
            params[f'dec_{i}_w_sigma'] = layer.w_sigma.detach().clone()
            params[f'dec_{i}_b_mu'] = layer.b_mu.detach().clone()
            params[f'dec_{i}_b_sigma'] = layer.b_sigma.detach().clone()
        
        return params


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess UNSW-NB15 dataset for anomaly detection"""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop ID column
    train_df = train_df.drop('id', axis=1)
    test_df = test_df.drop('id', axis=1)
    
    # Handle categorical features
    categorical_cols = ['proto', 'service', 'state']
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        le.fit(combined.astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    
    # Drop attack_cat
    train_df = train_df.drop('attack_cat', axis=1)
    test_df = test_df.drop('attack_cat', axis=1)
    
    # Separate features and labels
    X_train_full = train_df.drop('label', axis=1).values
    y_train_full = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Handle infinite values and NaNs
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # *** ANOMALY DETECTION: Train only on NORMAL data ***
    X_train_normal = X_train_full[y_train_full == 0]
    
    # Robust scaling
    scaler = RobustScaler()
    X_train_normal = scaler.fit_transform(X_train_normal)
    X_test = scaler.transform(X_test)
    
    # Get feature names
    feature_names = [col for col in train_df.columns if col != 'label']
    
    print(f"Normal training samples: {len(X_train_normal)}")
    print(f"Test samples: {len(X_test)} (Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")
    print(f"Feature dimension: {X_train_normal.shape[1]}\n")
    
    return X_train_normal, X_test, y_test, scaler, feature_names


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence (latent)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence for latent variables (more stable formulation)
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Clamp KL to prevent extreme values
    kl_latent = torch.clamp(kl_latent, min=0, max=1e6)
    
    return recon_loss + kl_weight * kl_latent, recon_loss, kl_latent


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================
def compute_reconstruction_metrics(model, data_loader, device, mc_samples=10):
    """
    Compute reconstruction error and uncertainty using MC sampling
    
    Returns:
        recon_errors: Mean reconstruction error per sample
        epistemic_unc: Epistemic uncertainty (variance across MC samples)
        aleatoric_unc: Aleatoric uncertainty (mean variance in latent space)
    """
    model.eval()
    
    all_recon_errors = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            batch_size = data.size(0)
            
            # MC sampling
            recons = []
            logvars = []
            for _ in range(mc_samples):
                x_recon, mu, logvar, z = model(data, sample=True)
                recons.append(x_recon)
                logvars.append(logvar)
            
            recons = torch.stack(recons)  # [mc_samples, batch_size, input_dim]
            logvars = torch.stack(logvars)  # [mc_samples, batch_size, latent_dim]
            
            # Mean reconstruction error
            recon_error = torch.mean((data - recons.mean(dim=0)) ** 2, dim=1)
            
            # Epistemic uncertainty: variance of reconstructions across MC samples
            epistemic = torch.mean(torch.var(recons, dim=0), dim=1)
            
            # Aleatoric uncertainty: mean variance in latent space
            aleatoric = torch.mean(torch.exp(logvars.mean(dim=0)), dim=1)
            
            all_recon_errors.append(recon_error.cpu().numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_aleatoric.append(aleatoric.cpu().numpy())
    
    recon_errors = np.concatenate(all_recon_errors)
    epistemic_unc = np.concatenate(all_epistemic)
    aleatoric_unc = np.concatenate(all_aleatoric)
    
    return recon_errors, epistemic_unc, aleatoric_unc


# ============================================================================
# CONTINUAL LEARNING UPDATE
# ============================================================================
def continual_update_bvae(model, X, y, device, prior_params=None,
                          n_epochs=5, batch_size=64, lr=1e-4,
                          kl_weight_max=0.01, weight_kl_scale=0.001,
                          mc_samples_uq=10):
    """
    Continual learning update with VCL and experience replay
    
    Args:
        model: BVAE model
        X: New data features
        y: New data labels (for weighting, though we train on all data)
        device: torch device
        prior_params: Previous posterior (for VCL), None = use fixed prior
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        kl_weight_max: KL weight for VAE latent
        weight_kl_scale: KL weight for Bayesian weights
        mc_samples_uq: MC samples for uncertainty quantification
    
    Returns:
        Updated model and posterior parameters
    """
    # Train only on normal data for anomaly detection
    X_normal = X[y == 0] if len(X[y == 0]) > 0 else X
    
    dataset = TensorDataset(torch.FloatTensor(X_normal))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    model.train()
    for epoch in range(n_epochs):
        for data in loader:
            data = data[0].to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            # VAE loss
            total_loss, recon_loss, kl_latent = vae_loss(x_recon, data, mu, logvar, kl_weight_max)
            
            # Bayesian weight KL
            kl_weights = model.total_kl_weights(prior_params) / len(dataset)
            
            loss = (total_loss / len(data)) + weight_kl_scale * kl_weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
    
    return model, model.get_posterior_params()


# ============================================================================
# TRAINING
# ============================================================================
def train_bvae(train_path=TRAIN_PATH, test_path=TEST_PATH,
               latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS,
               epochs=EPOCHS, batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE, model_path=MODEL_PATH):
    """
    Train Bayesian Variational Autoencoder on normal data only
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Load and preprocess data
    X_train_normal, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        train_path, test_path
    )
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}\n")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_normal))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation set
    val_size = min(5000, len(X_train_normal) // 5)
    val_indices = np.random.choice(len(X_train_normal), val_size, replace=False)
    X_val = X_train_normal[val_indices]
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train_normal.shape[1]
    model = BayesianVAE(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
                        prior_sigma=PRIOR_SIGMA).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )
    
    print("=" * 70)
    print("Training Bayesian Variational Autoencoder on Normal Data Only")
    print("=" * 70)
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0

    # timer = time.time()
    
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1} / {epochs} - Time: {time.time() - timer:.2f}s")
        # timer = time.time()

        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl_latent = 0
        train_kl_weights = 0
        
        # KL weight warm-up
        kl_weight = min(KL_WEIGHT_MAX, (epoch + 1) / KL_WARMUP_EPOCHS * KL_WEIGHT_MAX)
        
        for data in train_loader:
            data = data[0].to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data, sample=True)
            
            # Check for NaN in forward pass
            if torch.isnan(x_recon).any() or torch.isnan(mu).any() or torch.isnan(logvar).any():
                print(f"Warning: NaN detected in forward pass at epoch {epoch+1}, skipping batch")
                continue
            
            # VAE loss
            loss, recon_loss, kl_latent = vae_loss(x_recon, data, mu, logvar, kl_weight)
            
            # Bayesian weight KL
            kl_weights = model.total_kl_weights() / len(train_dataset)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isnan(kl_weights):
                print(f"Warning: NaN detected in loss at epoch {epoch+1}, skipping batch")
                continue
            
            total_loss = (loss / len(data)) + WEIGHT_KL_SCALE * kl_weights
            total_loss.backward()
            
            # Clip gradients more aggressively
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
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
                for data in val_loader:
                    data = data[0].to(device)
                    x_recon, mu, logvar, z = model(data, sample=True)
                    loss, _, _ = vae_loss(x_recon, data, mu, logvar, kl_weight)
                    kl_weights = model.total_kl_weights() / len(val_dataset)
                    total_loss = (loss / len(data)) + WEIGHT_KL_SCALE * kl_weights
                    val_loss += total_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Recon: {avg_recon:.4f} | KL_latent: {avg_kl_latent:.4f} | KL_weights: {avg_kl_weights:.6f}\n")
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
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
                print(f"  ✓ New best model saved (Val Loss: {avg_val_loss:.4f})\n")
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
    print("Computing optimal threshold on training (normal) data...")
    train_errors, train_epi, train_ale = compute_reconstruction_metrics(
        model, train_loader, device, mc_samples=MC_SAMPLES_TEST
    )
    
    # Threshold at 95th percentile
    threshold = np.percentile(train_errors, 95)
    np.save(THRESHOLD_PATH, threshold)
    
    print(f"Threshold (95th percentile): {threshold:.6f}")
    print(f"Threshold saved to {THRESHOLD_PATH}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_errors, test_epi, test_ale = compute_reconstruction_metrics(
        model, test_loader, device, mc_samples=MC_SAMPLES_TEST
    )
    y_pred = (test_errors > threshold).astype(int)
    
    # Metrics
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
    
    print(f"\nReconstruction Error Statistics:")
    print(f"  Normal - Mean: {test_errors[y_test == 0].mean():.6f}, Std: {test_errors[y_test == 0].std():.6f}")
    print(f"  Attack - Mean: {test_errors[y_test == 1].mean():.6f}, Std: {test_errors[y_test == 1].std():.6f}")
    
    print(f"\nEpistemic Uncertainty Statistics:")
    print(f"  Normal - Mean: {test_epi[y_test == 0].mean():.6f}, Std: {test_epi[y_test == 0].std():.6f}")
    print(f"  Attack - Mean: {test_epi[y_test == 1].mean():.6f}, Std: {test_epi[y_test == 1].std():.6f}")
    
    print(f"\nAleatoric Uncertainty Statistics:")
    print(f"  Normal - Mean: {test_ale[y_test == 0].mean():.6f}, Std: {test_ale[y_test == 0].std():.6f}")
    print(f"  Attack - Mean: {test_ale[y_test == 1].mean():.6f}, Std: {test_ale[y_test == 1].std():.6f}")
    print("=" * 70)
    
    return model, threshold, f1


if __name__ == '__main__':
    train_bvae()