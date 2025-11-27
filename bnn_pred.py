"""
Refactored BNN Prediction with Uncertainty Quantification
Original logic preserved, just cleaned up and parameterized
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from bnn_train import NetworkBNN, load_and_preprocess_data

# ============================================================================
# CONFIGURATION
# ============================================================================
# Paths
MODEL_PATH = 'bnn_unsw_nb15_best.pth'
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_PATH = 'predictions_with_uncertainty.csv'

# Prediction parameters
MC_SAMPLES = 50
BATCH_SIZE = 128

# Uncertainty thresholds
UNCERTAINTY_THRESHOLD_PERCENTILE = 90
HIGH_CONFIDENCE_THRESHOLD = 0.95

# Display settings
N_SAMPLE_PREDICTIONS = 5


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
                     targets, output_path=OUTPUT_PATH):
    """Save predictions with uncertainty metrics to CSV"""
    results_df = pd.DataFrame({
        'true_label': targets,
        'predicted_label': predictions,
        'prediction': ['Attack' if p == 1 else 'Normal' for p in predictions],
        'confidence': confidences,
        'uncertainty': uncertainties,
        'prob_normal': probs[:, 0],
        'prob_attack': probs[:, 1],
        'correct': predictions == targets
    })

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
                  output_path=OUTPUT_PATH):
    """
    Main classification function with uncertainty quantification

    Args:
        model_path: Path to trained model checkpoint
        train_path: Path to training data (needed for preprocessing)
        test_path: Path to test data
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
        batch_size: Batch size for inference
        output_path: Path to save predictions CSV
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

    # Initialize and load model
    print(f"Loading model from: {model_path}")
    model = NetworkBNN(X_train.shape[1], hidden_dim=256, n_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully (trained F1: {checkpoint['test_f1'] * 100:.2f}%)\n")

    # Predict with uncertainty quantification
    predictions, confidences, uncertainties, probs, targets = predict_with_uncertainty(
        model, test_loader, device, mc_samples=mc_samples
    )

    # Calculate metrics
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

    # Analyze uncertainty
    analyze_uncertainty(predictions, uncertainties, confidences, targets,
                       UNCERTAINTY_THRESHOLD_PERCENTILE)

    # Save predictions
    results_df = save_predictions(predictions, confidences, uncertainties,
                                  probs, targets, output_path)

    # Show some example predictions
    show_sample_predictions(results_df, N_SAMPLE_PREDICTIONS, HIGH_CONFIDENCE_THRESHOLD)

    print("\n" + "=" * 70)
    print("Classification complete!")
    print("=" * 70)

    return results_df


if __name__ == '__main__':
    # Run classification with uncertainty quantification
    results = main_classify()