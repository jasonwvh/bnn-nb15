"""
Refactored BNN Prediction with Uncertainty Quantification
Original logic preserved, just cleaned up and parameterized
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import warnings
warnings.filterwarnings('ignore')
from bnn_train import NetworkBNN, load_and_preprocess_data

# ============================================================================
# CONFIGURATION
# ============================================================================
# Paths
MODEL_PATH = 'models/bnn.pth'
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_PATH = 'models/predictions.csv'

# Prediction parameters
MC_SAMPLES = 50
BATCH_SIZE = 128

# Three-way decision thresholds (proposal: accept/reject when confident, else defer)
TAU_BENIGN = 0.9   # p(benign) > τ_benign to accept as benign
TAU_ATTACK = 0.9   # p(attack) > τ_attack to reject as attack
ETA = 0.1          # entropy H < η for low uncertainty (normalized entropy in [0,1])

# Uncertainty analysis
UNCERTAINTY_THRESHOLD_PERCENTILE = 90
HIGH_CONFIDENCE_THRESHOLD = 0.95

# Display settings
N_SAMPLE_PREDICTIONS = 5
N_ECE_BINS = 10


# ============================================================================
# PREDICTION WITH UNCERTAINTY
# ============================================================================
def predict_with_uncertainty(model, data_loader, device, mc_samples=50):
    """
    Predict with uncertainty quantification using Monte Carlo sampling

    Returns:
        predictions: Predicted class (0=Normal, 1=Attack)
        confidences: Prediction confidence (0-1)
        uncertainties: Prediction uncertainty (entropy-based)
        all_probs: Probability distributions for both classes
    """
    model.eval()

    all_preds = []
    all_confidences = []
    all_uncertainties = []
    all_probs = []
    all_targets = []

    print(f"Running inference with {mc_samples} Monte Carlo samples...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # Collect MC samples
            mc_probs = []
            for _ in range(mc_samples):
                output = model(data, sample=True)
                probs = torch.exp(output)  # Convert log_softmax to probabilities
                mc_probs.append(probs)

            # Stack all MC samples: [mc_samples, batch_size, n_classes]
            mc_probs = torch.stack(mc_probs)

            # Mean probability across MC samples
            mean_probs = mc_probs.mean(dim=0)

            # Predicted class
            preds = mean_probs.argmax(dim=1)

            # Confidence: probability of predicted class
            confidence = mean_probs.max(dim=1)[0]

            # Uncertainty: Predictive entropy
            # H = -sum(p * log(p))
            entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

            # Normalize entropy to [0, 1] (max entropy for 2 classes is log(2))
            max_entropy = np.log(2)
            normalized_uncertainty = entropy / max_entropy

            all_preds.extend(preds.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_uncertainties.extend(normalized_uncertainty.cpu().numpy())
            all_probs.extend(mean_probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * data_loader.batch_size} samples...")

    return (np.array(all_preds),
            np.array(all_confidences),
            np.array(all_uncertainties),
            np.array(all_probs),
            np.array(all_targets))


# ============================================================================
# CALIBRATION METRICS
# ============================================================================
def expected_calibration_error(probs, targets, n_bins=10):
    """Expected Calibration Error: weighted average of |accuracy - confidence| per bin."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            acc = (preds[in_bin] == targets[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += prop * np.abs(acc - avg_conf)
    return float(ece)


def brier_multi(probs, targets, n_classes=2):
    """Brier score for multi-class: mean squared error between prob and one-hot target."""
    n = len(targets)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), targets] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ============================================================================
# THREE-WAY DECISION (accept / reject / defer)
# ============================================================================
def three_way_decision(probs, uncertainties, tau_benign=0.9, tau_attack=0.9, eta=0.1):
    """
    Accept (benign) if p(benign) > τ_benign and H < η;
    Reject (attack) if p(attack) > τ_attack and H < η;
    else Defer to human.
    probs: (N, 2) with [prob_normal, prob_attack]; uncertainties: normalized entropy (0-1).
    Returns: array of 'accept', 'reject', 'defer'.
    """
    p_benign = probs[:, 0]
    p_attack = probs[:, 1]
    low_entropy = uncertainties < eta
    accept = (p_benign > tau_benign) & low_entropy
    reject = (p_attack > tau_attack) & low_entropy & ~accept
    decisions = np.where(accept, 'accept', np.where(reject, 'reject', 'defer'))
    return decisions


def report_three_way(decisions, targets, predictions):
    """Report counts/rates for accept, reject, defer; accuracy on non-deferred; deferral rate."""
    n = len(decisions)
    n_accept = (decisions == 'accept').sum()
    n_reject = (decisions == 'reject').sum()
    n_defer = (decisions == 'defer').sum()
    deferral_rate = n_defer / n if n else 0

    non_deferred = decisions != 'defer'
    n_non_def = non_deferred.sum()
    if n_non_def > 0:
        acc_non_deferred = (predictions[non_deferred] == targets[non_deferred]).mean()
    else:
        acc_non_deferred = float('nan')

    print("\n" + "=" * 70)
    print("THREE-WAY DECISION (accept / reject / defer)")
    print("=" * 70)
    print(f"\n  Accept (benign): {n_accept} ({100 * n_accept / n:.1f}%)")
    print(f"  Reject (attack): {n_reject} ({100 * n_reject / n:.1f}%)")
    print(f"  Defer:          {n_defer} ({100 * deferral_rate:.1f}%)")
    print(f"\n  Deferral rate:        {100 * deferral_rate:.2f}%")
    print(f"  Accuracy (non-deferred): {100 * acc_non_deferred:.2f}%" if not np.isnan(acc_non_deferred) else "  Accuracy (non-deferred): N/A (all deferred)")
    return deferral_rate, acc_non_deferred


# ============================================================================
# UNCERTAINTY ANALYSIS
# ============================================================================
def analyze_uncertainty(predictions, uncertainties, confidences, targets,
                       uncertainty_threshold_percentile=90):
    """Analyze uncertainty metrics"""
    correct = predictions == targets

    print("\n" + "=" * 70)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Mean Uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
    print(f"  Mean Confidence:  {confidences.mean():.4f} ± {confidences.std():.4f}")

    print(f"\nCorrect Predictions:")
    print(f"  Mean Uncertainty: {uncertainties[correct].mean():.4f}")
    print(f"  Mean Confidence:  {confidences[correct].mean():.4f}")

    print(f"\nIncorrect Predictions:")
    print(f"  Mean Uncertainty: {uncertainties[~correct].mean():.4f}")
    print(f"  Mean Confidence:  {confidences[~correct].mean():.4f}")

    # Uncertainty quartiles
    print(f"\nUncertainty Distribution:")
    for percentile in [25, 50, 75, 90, 95, 99]:
        value = np.percentile(uncertainties, percentile)
        print(f"  {percentile}th percentile: {value:.4f}")

    # High uncertainty samples
    high_uncertainty_threshold = np.percentile(uncertainties, uncertainty_threshold_percentile)
    high_uncertainty_mask = uncertainties > high_uncertainty_threshold
    high_unc_accuracy = correct[high_uncertainty_mask].mean()

    print(f"\nHigh Uncertainty Samples (top {100-uncertainty_threshold_percentile}%):")
    print(f"  Count: {high_uncertainty_mask.sum()}")
    print(f"  Accuracy: {high_unc_accuracy * 100:.2f}%")
    print(f"  These samples may require human review or additional investigation")


# ============================================================================
# SAVE AND DISPLAY PREDICTIONS
# ============================================================================
def save_predictions(predictions, confidences, uncertainties, probs,
                     targets, decisions=None, output_path=OUTPUT_PATH):
    """Save predictions with uncertainty metrics and three-way decision to CSV"""
    data = {
        'true_label': targets,
        'predicted_label': predictions,
        'prediction': ['Attack' if p == 1 else 'Normal' for p in predictions],
        'confidence': confidences,
        'uncertainty': uncertainties,
        'prob_normal': probs[:, 0],
        'prob_attack': probs[:, 1],
        'correct': predictions == targets,
    }
    if decisions is not None:
        data['decision'] = decisions
    results_df = pd.DataFrame(data)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    return results_df


def show_sample_predictions(results_df, n_samples=5, high_conf_threshold=0.95):
    """Display sample predictions"""
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    print(f"\nHigh Confidence Correct Predictions (top {n_samples}):")
    high_conf_correct = results_df[(results_df['correct']) &
                                   (results_df['confidence'] > high_conf_threshold)].head(n_samples)
    print(high_conf_correct[['prediction', 'confidence', 'uncertainty',
                             'prob_normal', 'prob_attack']].to_string(index=False))

    print(f"\nHigh Uncertainty Predictions (top {n_samples}):")
    high_unc = results_df.nlargest(n_samples, 'uncertainty')
    print(high_unc[['prediction', 'confidence', 'uncertainty', 'correct']].to_string(index=False))

    print(f"\nMisclassifications with High Confidence (top {n_samples}):")
    high_conf_wrong = results_df[(~results_df['correct']) &
                                 (results_df['confidence'] > 0.8)].head(n_samples)
    if len(high_conf_wrong) > 0:
        print(high_conf_wrong[['true_label', 'predicted_label', 'confidence',
                               'uncertainty']].to_string(index=False))
    else:
        print("  None found (good!)")


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================
def main_classify(model_path=MODEL_PATH,
                  train_path=TRAIN_PATH,
                  test_path=TEST_PATH,
                  mc_samples=MC_SAMPLES,
                  batch_size=BATCH_SIZE,
                  output_path=OUTPUT_PATH,
                  tau_benign=TAU_BENIGN,
                  tau_attack=TAU_ATTACK,
                  eta=ETA):
    """
    Main classification function with uncertainty quantification and three-way decision.

    Args:
        model_path: Path to trained model checkpoint
        train_path: Path to training data (needed for preprocessing)
        test_path: Path to test data
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
        batch_size: Batch size for inference
        output_path: Path to save predictions CSV
        tau_benign: Threshold for accept (benign); p(benign) > tau_benign
        tau_attack: Threshold for reject (attack); p(attack) > tau_attack
        eta: Entropy threshold; H < eta for low uncertainty (normalized [0,1])
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(
        train_path, test_path
    )

    # Create test data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and load model (use checkpoint metadata if present)
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint.get('input_dim', X_train.shape[1])
    hidden_dim = checkpoint.get('hidden_dim', 256)
    n_classes = checkpoint.get('n_classes', 2)
    model = NetworkBNN(input_dim, hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully (trained F1: {checkpoint['test_f1'] * 100:.2f}%)\n")

    # Predict with uncertainty quantification
    predictions, confidences, uncertainties, probs, targets = predict_with_uncertainty(
        model, test_loader, device, mc_samples=mc_samples
    )

    # Classification metrics
    print("\n" + "=" * 70)
    print("CLASSIFICATION PERFORMANCE")
    print("=" * 70)

    acc = accuracy_score(targets, predictions)
    prec = precision_score(targets, predictions, zero_division=0)
    rec = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")

    # AUC-ROC and AUC-PR
    try:
        auc_roc = roc_auc_score(targets, probs[:, 1])
        auc_pr = average_precision_score(targets, probs[:, 1])
        print(f"  AUC-ROC:   {auc_roc:.4f}")
        print(f"  AUC-PR:    {auc_pr:.4f}")
    except Exception:
        print(f"  AUC-ROC:   N/A")
        print(f"  AUC-PR:    N/A")

    # Calibration: ECE and Brier
    ece = expected_calibration_error(probs, targets, n_bins=N_ECE_BINS)
    brier = brier_multi(probs, targets, n_classes=n_classes)
    print(f"\nCalibration:")
    print(f"  ECE (expected calibration error): {ece:.4f}")
    print(f"  Brier score: {brier:.4f}")

    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Attack")
    print(f"Actual Normal  {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"       Attack  {cm[1, 0]:6d}  {cm[1, 1]:6d}")

    # Per-class metrics
    print(f"\nPer-Class Breakdown:")
    print(f"  Normal Traffic:")
    print(f"    True Positives:  {cm[0, 0]} (correctly identified as normal)")
    print(f"    False Negatives: {cm[0, 1]} (normal misclassified as attack)")
    print(f"  Attack Traffic:")
    print(f"    True Positives:  {cm[1, 1]} (correctly identified as attack)")
    print(f"    False Negatives: {cm[1, 0]} (attack misclassified as normal)")

    # Three-way decision (accept / reject / defer)
    decisions = three_way_decision(probs, uncertainties, tau_benign, tau_attack, eta)
    report_three_way(decisions, targets, predictions)

    # Uncertainty analysis
    analyze_uncertainty(predictions, uncertainties, confidences, targets,
                       UNCERTAINTY_THRESHOLD_PERCENTILE)

    # Save predictions (including decision column)
    results_df = save_predictions(predictions, confidences, uncertainties,
                                  probs, targets, decisions=decisions, output_path=output_path)

    # Sample predictions
    show_sample_predictions(results_df, N_SAMPLE_PREDICTIONS, HIGH_CONFIDENCE_THRESHOLD)

    print("\n" + "=" * 70)
    print("Classification complete!")
    print("=" * 70)

    return results_df


if __name__ == '__main__':
    # Run classification with uncertainty quantification
    results = main_classify()