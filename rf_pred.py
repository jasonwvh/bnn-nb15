"""
Random Forest prediction on UNSW-NB15 testing-set.
Loads model from rf_train pipeline; reports ML metrics for comparison with BNN.
Objective: BNN should surpass RF performance on the same dataset.
"""
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
warnings.filterwarnings('ignore')
from rf_train import load_and_preprocess_data
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = 'models/rf.pkl'
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_PATH = 'models/rf_predictions.csv'
N_ECE_BINS = 10


# ============================================================================
# CALIBRATION METRICS (for comparison with BNN)
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
    """Brier score: mean squared error between prob and one-hot target."""
    n = len(targets)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), targets] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ============================================================================
# MAIN PREDICTION
# ============================================================================
def main_predict(model_path=MODEL_PATH,
                 train_path=TRAIN_PATH,
                 test_path=TEST_PATH,
                 output_path=OUTPUT_PATH):
    """
    Load RF model, run on testing-set, report metrics.
    Use same preprocessing as rf_train (requires train + test paths).
    """
    print("\n" + "=" * 70)
    print("Random Forest — Test-set evaluation (UNSW-NB15)")
    print("=" * 70)

    # Preprocess (same as rf_train: needs train+test for encoders/scaler)
    X_train, y_train, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        train_path, test_path
    )

    # Load model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded.\n")

    # Predict
    print("Running predictions on testing-set...")
    start = time.time()
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)  # (N, 2) [prob_0, prob_1]
    elapsed = time.time() - start
    n_test = len(y_test)
    print(f"  Done in {elapsed:.4f}s ({n_test / elapsed:.0f} samples/s)\n")

    targets = y_test

    # Classification metrics
    print("=" * 70)
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

    try:
        auc_roc = roc_auc_score(targets, probs[:, 1])
        auc_pr = average_precision_score(targets, probs[:, 1])
        print(f"  AUC-ROC:   {auc_roc:.4f}")
        print(f"  AUC-PR:    {auc_pr:.4f}")
    except Exception:
        print(f"  AUC-ROC:   N/A")
        print(f"  AUC-PR:    N/A")

    ece = expected_calibration_error(probs, targets, n_bins=N_ECE_BINS)
    brier = brier_multi(probs, targets, n_classes=2)
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

    print(f"\nPer-Class Breakdown:")
    print(f"  Normal: TP={cm[0, 0]}, FN (misclassified as attack)={cm[0, 1]}")
    print(f"  Attack: TP={cm[1, 1]}, FN (misclassified as normal)={cm[1, 0]}")

    print(f"\nClassification Report:")
    print(classification_report(targets, predictions,
                                 target_names=['Normal', 'Attack'],
                                 digits=4))

    # Save predictions for comparison with BNN
    results_df = pd.DataFrame({
        'true_label': targets,
        'predicted_label': predictions,
        'prediction': ['Attack' if p == 1 else 'Normal' for p in predictions],
        'prob_normal': probs[:, 0],
        'prob_attack': probs[:, 1],
        'confidence': probs.max(axis=1),
        'correct': predictions == targets,
    })
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Run bnn_pred.py on the same test set to compare BNN vs RF.")
    print("=" * 70)

    return results_df


if __name__ == '__main__':
    main_predict()
