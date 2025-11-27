"""
Refactored Bayesian Neural Network Training
Original architecture preserved, just cleaned up and parameterized
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data paths
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Model architecture
HIDDEN_DIM = 256
N_CLASSES = 2
PRIOR_SIGMA = 1.0
DROPOUT_RATE = 0.3

# Training parameters
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = True  # Balance classes in loss function

# Optimization
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
GRADIENT_CLIP_NORM = 1.0

# Early stopping
EARLY_STOP_PATIENCE = 15

# KL divergence
KL_WEIGHT_MAX = 0.01
KL_WARMUP_EPOCHS = 50

# Monte Carlo sampling
MC_SAMPLES_TRAIN = 5
MC_SAMPLES_VAL = 10
MC_SAMPLES_TEST = 20

# Model saving
MODEL_PATH = 'models/bnn.pth'

# Validation frequency
VAL_FREQUENCY = 2


# ============================================================================
# MODEL DEFINITION (ORIGINAL ARCHITECTURE)
# ============================================================================
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.zeros(out_features, in_features).normal_(-3, 0.1))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.zeros(out_features).normal_(-3, 0.1))

    @property
    def w_sigma(self):
        return torch.log1p(torch.exp(self.w_rho))

    @property
    def b_sigma(self):
        return torch.log1p(torch.exp(self.b_rho))

    def forward(self, x, sample=True):
        if self.training or sample:
            w = self.w_mu + self.w_sigma * torch.randn_like(self.w_mu)
            b = self.b_mu + self.b_sigma * torch.randn_like(self.b_mu)
            return F.linear(x, w, b)
        return F.linear(x, self.w_mu, self.b_mu)

    def kl_divergence(self):
        kl_w = (torch.log(self.prior_sigma / self.w_sigma) +
                (self.w_sigma ** 2 + self.w_mu ** 2) / (2 * self.prior_sigma ** 2) - 0.5)
        kl_b = (torch.log(self.prior_sigma / self.b_sigma) +
                (self.b_sigma ** 2 + self.b_mu ** 2) / (2 * self.prior_sigma ** 2) - 0.5)
        return kl_w.sum() + kl_b.sum()


class NetworkBNN(nn.Module):
    """Bayesian Neural Network for Network Intrusion Detection"""

    def __init__(self, input_dim, hidden_dim=256, n_classes=2):
        super().__init__()
        self.l1 = BayesianLinear(input_dim, hidden_dim)
        self.l2 = BayesianLinear(hidden_dim, hidden_dim // 2)
        self.l3 = BayesianLinear(hidden_dim // 2, hidden_dim // 4)
        self.l4 = BayesianLinear(hidden_dim // 4, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, sample=True):
        x = F.relu(self.l1(x, sample))
        x = self.dropout(x)
        x = F.relu(self.l2(x, sample))
        x = self.dropout(x)
        x = F.relu(self.l3(x, sample))
        x = self.dropout(x)
        x = self.l4(x, sample)
        return F.log_softmax(x, dim=1)

    def total_kl(self):
        return (self.l1.kl_divergence() + self.l2.kl_divergence() +
                self.l3.kl_divergence() + self.l4.kl_divergence())


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess UNSW-NB15 dataset"""
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

    # Drop attack_cat (we use label for binary classification)
    train_df = train_df.drop('attack_cat', axis=1)
    test_df = test_df.drop('attack_cat', axis=1)

    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    # Handle infinite values and NaNs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Robust scaling (handles outliers better than StandardScaler)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train samples: {len(X_train)} (Normal: {(y_train == 0).sum()}, Attack: {(y_train == 1).sum()})")
    print(f"Test samples: {len(X_test)} (Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")
    print(f"Feature dimension: {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test, scaler


# ============================================================================
# VALIDATION
# ============================================================================
def validate(model, loader, device, mc_samples=10):
    """Validation with Monte Carlo sampling for uncertainty estimation"""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # MC sampling for uncertainty quantification
            probs = []
            for _ in range(mc_samples):
                probs.append(torch.exp(model(data, sample=True)))

            mean_probs = torch.stack(probs).mean(dim=0)
            preds = mean_probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(mean_probs.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return acc, prec, rec, f1, all_preds, all_probs


# ============================================================================
# TRAINING
# ============================================================================
def calculate_class_weights(y_train):
    """Calculate balanced class weights for loss function"""
    n_samples = len(y_train)
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()

    # Calculate weights inversely proportional to class frequencies
    weight_normal = n_samples / (2 * n_normal)
    weight_attack = n_samples / (2 * n_attack)

    class_weights = torch.FloatTensor([weight_normal, weight_attack])

    print(f"Class weights: Normal={weight_normal:.3f}, Attack={weight_attack:.3f}")

    return class_weights


def train_bnn(train_path=TRAIN_PATH, test_path=TEST_PATH,
              hidden_dim=HIDDEN_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE,
              learning_rate=LEARNING_RATE, model_path=MODEL_PATH,
              use_class_weights=USE_CLASS_WEIGHTS):
    """
    Train Bayesian Neural Network

    Args:
        train_path: Path to training data
        test_path: Path to test data
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        model_path: Path to save best model
        use_class_weights: Whether to use class weights in loss function
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(
        train_path, test_path
    )

    # Calculate class weights if enabled
    class_weights = None
    if use_class_weights:
        print("\nCalculating class weights for balanced training...")
        class_weights = calculate_class_weights(y_train).to(device)
        print()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NetworkBNN(X_train.shape[1], hidden_dim=hidden_dim, n_classes=N_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )

    print("Starting Bayesian Neural Network training...")
    if use_class_weights:
        print("Using balanced class weights for loss function")
    print()
    best_test_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # KL weight warm-up for stable training
        kl_weight = min(KL_WEIGHT_MAX, (epoch + 1) / KL_WARMUP_EPOCHS * KL_WEIGHT_MAX)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data, sample=True)

            # Use class weights in loss if enabled
            if use_class_weights:
                nll_loss = F.nll_loss(output, target, weight=class_weights)
            else:
                nll_loss = F.nll_loss(output, target)

            kl_loss = model.total_kl() / len(train_dataset)
            loss = nll_loss + kl_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation every VAL_FREQUENCY epochs
        if (epoch + 1) % VAL_FREQUENCY == 0:
            train_acc, train_prec, train_rec, train_f1, _, _ = validate(
                model, train_loader, device, mc_samples=MC_SAMPLES_TRAIN
            )
            test_acc, test_prec, test_rec, test_f1, test_preds, test_probs = validate(
                model, test_loader, device, mc_samples=MC_SAMPLES_VAL
            )

            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f}")
            print(f"  Train | Acc: {train_acc * 100:.2f}% | Prec: {train_prec * 100:.2f}% | "
                  f"Rec: {train_rec * 100:.2f}% | F1: {train_f1 * 100:.2f}%")
            print(f"  Test  | Acc: {test_acc * 100:.2f}% | Prec: {test_prec * 100:.2f}% | "
                  f"Rec: {test_rec * 100:.2f}% | F1: {test_f1 * 100:.2f}%\n")

            scheduler.step(avg_loss)

            # Save best model based on F1-score
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_f1': test_f1,
                    'test_acc': test_acc,
                }, model_path)
                patience_counter = 0
                print(f"  ✓ New best model saved (F1: {test_f1 * 100:.2f}%)\n")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    # Load best model and evaluate
    print("\n" + "=" * 60)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, test_prec, test_rec, test_f1, test_preds, test_probs = validate(
        model, test_loader, device, mc_samples=MC_SAMPLES_TEST
    )

    print("\nFinal Test Performance:")
    print(f"  Accuracy:  {test_acc * 100:.2f}%")
    print(f"  Precision: {test_prec * 100:.2f}%")
    print(f"  Recall:    {test_rec * 100:.2f}%")
    print(f"  F1-Score:  {test_f1 * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Attack")
    print(f"Actual Normal  {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"       Attack  {cm[1, 0]:6d}  {cm[1, 1]:6d}")
    print("=" * 60)

    return model, test_f1


if __name__ == '__main__':
    train_bnn()