import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from train_bnn import NetworkBNN, load_and_preprocess_data

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


def analyze_uncertainty(predictions, uncertainties, confidences, targets):
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
    high_uncertainty_threshold = np.percentile(uncertainties, 90)
    high_uncertainty_mask = uncertainties > high_uncertainty_threshold
    high_unc_accuracy = correct[high_uncertainty_mask].mean()

    print(f"\nHigh Uncertainty Samples (top 10%):")
    print(f"  Count: {high_uncertainty_mask.sum()}")
    print(f"  Accuracy: {high_unc_accuracy * 100:.2f}%")
    print(f"  These samples may require human review or additional investigation")


def save_predictions(predictions, confidences, uncertainties, probs,
                     targets, output_path='predictions_with_uncertainty.csv'):
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


def main_classify(model_path='bnn_unsw_nb15_best.pth',
                  train_path='data/UNSW_NB15_training-set.csv',
                  test_path='data/UNSW_NB15_testing-set.csv',
                  mc_samples=50):
    """
    Main classification function with uncertainty quantification

    Args:
        model_path: Path to trained model checkpoint
        train_path: Path to training data (needed for preprocessing)
        test_path: Path to test data
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
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
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

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
    analyze_uncertainty(predictions, uncertainties, confidences, targets)

    # Save predictions
    results_df = save_predictions(predictions, confidences, uncertainties,
                                  probs, targets)

    # Show some example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    print("\nHigh Confidence Correct Predictions:")
    high_conf_correct = results_df[(results_df['correct']) & (results_df['confidence'] > 0.95)].head(5)
    print(high_conf_correct[['prediction', 'confidence', 'uncertainty', 'prob_normal', 'prob_attack']].to_string(
        index=False))

    print("\nHigh Uncertainty Predictions:")
    high_unc = results_df.nlargest(5, 'uncertainty')
    print(high_unc[['prediction', 'confidence', 'uncertainty', 'correct']].to_string(index=False))

    print("\nMisclassifications with High Confidence:")
    high_conf_wrong = results_df[(~results_df['correct']) & (results_df['confidence'] > 0.8)].head(5)
    if len(high_conf_wrong) > 0:
        print(high_conf_wrong[['true_label', 'predicted_label', 'confidence', 'uncertainty']].to_string(index=False))
    else:
        print("  None found (good!)")

    print("\n" + "=" * 70)
    print("Classification complete!")
    print("=" * 70)

    return results_df


if __name__ == '__main__':
    # Run classification with uncertainty quantification
    results = main_classify(
        model_path='bnn_unsw_nb15_best.pth',
        train_path='data/UNSW_NB15_training-set.csv',
        test_path='data/UNSW_NB15_testing-set.csv',
        mc_samples=50  # Higher = more accurate uncertainty but slower
    )