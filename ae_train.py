"""
Autoencoder Baseline for Anomaly Detection
Trains on normal data only, uses reconstruction error for anomaly detection
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# Data paths
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Model architecture
LATENT_DIM = 16
HIDDEN_DIMS = [64, 32]  # Encoder: input -> 64 -> 32 -> latent

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

# Model saving
MODEL_PATH = 'models/ae.pth'
SCALER_PATH = 'models/ae_scaler.pkl'
THRESHOLD_PATH = 'models/ae_threshold.npy'

# Validation frequency
VAL_FREQUENCY = 2


# ============================================================================
# MODEL DEFINITION
# ============================================================================
class Autoencoder(nn.Module):
    """Vanilla Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim, latent_dim=16, hidden_dims=[64, 32]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (symmetric)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


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
    
    print(f"Normal training samples: {len(X_train_normal)}")
    print(f"Test samples: {len(X_test)} (Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")
    print(f"Feature dimension: {X_train_normal.shape[1]}\n")
    
    return X_train_normal, X_test, y_test, scaler


# ============================================================================
# TRAINING
# ============================================================================
def compute_reconstruction_error(model, data_loader, device):
    """Compute reconstruction error (MSE) for each sample"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            x_recon, _ = model(data)
            error = torch.mean((data - x_recon) ** 2, dim=1)
            errors.append(error.cpu().numpy())
    
    return np.concatenate(errors)


def train_autoencoder(train_path=TRAIN_PATH, test_path=TEST_PATH,
                      latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS,
                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                      learning_rate=LEARNING_RATE, model_path=MODEL_PATH):
    """
    Train Autoencoder on normal data only
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_path: Path to save model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load and preprocess data (only normal samples for training)
    X_train_normal, X_test, y_test, scaler = load_and_preprocess_data(train_path, test_path)
    
    # Save scaler for later use
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}\n")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_normal))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation loader (also normal data, for early stopping)
    val_size = min(5000, len(X_train_normal) // 5)
    val_indices = np.random.choice(len(X_train_normal), val_size, replace=False)
    X_val = X_train_normal[val_indices]
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train_normal.shape[1]
    model = Autoencoder(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )
    
    criterion = nn.MSELoss()
    
    print("=" * 70)
    print("Training Autoencoder on Normal Data Only")
    print("=" * 70)
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data[0].to(device)
            
            optimizer.zero_grad()
            x_recon, _ = model(data)
            loss = criterion(x_recon, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if (epoch + 1) % VAL_FREQUENCY == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data[0].to(device)
                    x_recon, _ = model(data)
                    loss = criterion(x_recon, data)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
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
                }, model_path)
                patience_counter = 0
                print(f"  ✓ New best model saved (Val Loss: {avg_val_loss:.6f})\n")
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
    
    # Compute reconstruction errors on training data to find threshold
    print("Computing optimal threshold on training (normal) data...")
    train_errors = compute_reconstruction_error(model, train_loader, device)
    
    # Use 95th percentile as threshold (allows 5% false positive on normal data)
    threshold = np.percentile(train_errors, 95)
    np.save(THRESHOLD_PATH, threshold)
    
    print(f"Threshold (95th percentile): {threshold:.6f}")
    print(f"Threshold saved to {THRESHOLD_PATH}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_errors = compute_reconstruction_error(model, test_loader, device)
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
    print(f"  Normal samples - Mean: {test_errors[y_test == 0].mean():.6f}, Std: {test_errors[y_test == 0].std():.6f}")
    print(f"  Attack samples - Mean: {test_errors[y_test == 1].mean():.6f}, Std: {test_errors[y_test == 1].std():.6f}")
    print("=" * 70)
    
    return model, threshold, f1


if __name__ == '__main__':
    train_autoencoder()