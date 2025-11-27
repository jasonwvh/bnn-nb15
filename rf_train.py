import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import joblib
import time
import warnings

warnings.filterwarnings('ignore')


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
        # Fit on combined data to handle unseen categories
        combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        le.fit(combined.astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le

    # Drop attack_cat (we use label for binary classification)
    train_df = train_df.drop('attack_cat', axis=1)
    test_df = test_df.drop('attack_cat', axis=1)

    # Get feature names before converting to numpy
    feature_names = [col for col in train_df.columns if col != 'label']

    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    # Handle infinite values and NaNs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Robust scaling
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train samples: {len(X_train)} (Normal: {(y_train == 0).sum()}, Attack: {(y_train == 1).sum()})")
    print(f"Test samples: {len(X_test)} (Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Class distribution (train): {(y_train == 0).sum() / len(y_train) * 100:.1f}% Normal, "
          f"{(y_train == 1).sum() / len(y_train) * 100:.1f}% Attack\n")

    return X_train, y_train, X_test, y_test, scaler, feature_names


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n{'=' * 60}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 60}")

    # Predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time

    # Probability predictions for ROC-AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Prediction Time: {pred_time:.4f}s ({len(X_test) / pred_time:.0f} samples/s)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Attack")
    print(f"Actual Normal  {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"       Attack  {cm[1, 0]:6d}  {cm[1, 1]:6d}")

    # Calculate false positive rate and false negative rate
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    print(f"\nError Analysis:")
    print(f"  False Positive Rate: {fpr * 100:.2f}% (Normal classified as Attack)")
    print(f"  False Negative Rate: {fnr * 100:.2f}% (Attack classified as Normal)")

    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Attack'],
                                digits=4))

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train Random Forest classifier with hyperparameter tuning"""

    print("\n" + "=" * 60)
    print("Training Random Forest Baseline")
    print("=" * 60)

    # Class weights to handle imbalance
    n_samples = len(y_train)
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()

    class_weight = {
        0: n_samples / (2 * n_normal),
        1: n_samples / (2 * n_attack)
    }

    print(f"\nClass weights: Normal={class_weight[0]:.3f}, Attack={class_weight[1]:.3f}")

    # Initialize Random Forest with good hyperparameters
    print("\nTraining Random Forest...")
    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=200,  # Number of trees
        max_depth=30,  # Maximum depth of trees
        min_samples_split=10,  # Minimum samples to split a node
        min_samples_leaf=4,  # Minimum samples in a leaf
        max_features='sqrt',  # Number of features for best split
        class_weight=class_weight,  # Handle class imbalance
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )

    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.2f}s")

    # Feature importance analysis
    print("\n" + "-" * 60)
    print("Top 20 Most Important Features:")
    print("-" * 60)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # Evaluate on training set
    print("\n" + "=" * 60)
    print("Training Set Performance")
    print("=" * 60)
    train_pred = rf_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)
    print(f"  Training Accuracy: {train_acc * 100:.2f}%")
    print(f"  Training F1-Score: {train_f1 * 100:.2f}%")

    # Evaluate on test set
    results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Save model
    joblib.dump(rf_model, 'rf_unsw_nb15_baseline.pkl')
    print(f"\n✓ Model saved to 'rf_unsw_nb15_baseline.pkl'")

    # Save feature importance
    feature_importance.to_csv('rf_feature_importance.csv', index=False)
    print(f"✓ Feature importance saved to 'rf_feature_importance.csv'")

    return rf_model, results, feature_importance


def compare_with_simple_rf(X_train, y_train, X_test, y_test):
    """Train a simpler RF for comparison"""
    print("\n" + "=" * 60)
    print("Training Simple Random Forest (for comparison)")
    print("=" * 60)

    simple_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start_time = time.time()
    simple_rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    results = evaluate_model(simple_rf, X_test, y_test, "Simple Random Forest")

    return simple_rf, results


def main(train_path='data/UNSW_NB15_training-set.csv',
         test_path='data/UNSW_NB15_testing-set.csv'):
    print("\n" + "=" * 60)
    print("Random Forest Baseline for UNSW-NB15 Dataset")
    print("=" * 60)

    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler, feature_names = load_and_preprocess_data(
        train_path, test_path
    )

    # Train optimized Random Forest
    rf_model, rf_results, feature_importance = train_random_forest(
        X_train, y_train, X_test, y_test, feature_names
    )

    # Train simple RF for comparison
    simple_rf, simple_results = compare_with_simple_rf(
        X_train, y_train, X_test, y_test
    )

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\nOptimized Random Forest:")
    print(f"  Accuracy: {rf_results['accuracy'] * 100:.2f}%")
    print(f"  F1-Score: {rf_results['f1'] * 100:.2f}%")
    print(f"  ROC-AUC:  {rf_results['auc']:.4f}")

    print(f"\nSimple Random Forest:")
    print(f"  Accuracy: {simple_results['accuracy'] * 100:.2f}%")
    print(f"  F1-Score: {simple_results['f1'] * 100:.2f}%")
    print(f"  ROC-AUC:  {simple_results['auc']:.4f}")

    improvement = (rf_results['f1'] - simple_results['f1']) * 100
    print(f"\nImprovement: {improvement:+.2f}% F1-Score")
    print("=" * 60)

    return rf_model, rf_results


if __name__ == '__main__':
    main()