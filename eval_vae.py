"""
Comprehensive Evaluation: AE vs VAE vs BVAE
Compares all three models on the test set with detailed metrics
"""
import numpy as np
import pandas as pd
import torch
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from ae_train import (
    Autoencoder,
    compute_reconstruction_error,
    MODEL_PATH as AE_MODEL_PATH,
    THRESHOLD_PATH as AE_THRESHOLD_PATH,
    SCALER_PATH as AE_SCALER_PATH,
)
from vae_train import (
    VAE,
    compute_reconstruction_metrics as vae_compute_metrics,
    MODEL_PATH as VAE_MODEL_PATH,
    THRESHOLD_PATH as VAE_THRESHOLD_PATH,
    SCALER_PATH as VAE_SCALER_PATH,
)
from bvae_train import (
    BayesianVAE,
    compute_reconstruction_metrics as bvae_compute_metrics,
    MODEL_PATH as BVAE_MODEL_PATH,
    THRESHOLD_PATH as BVAE_THRESHOLD_PATH,
    SCALER_PATH as BVAE_SCALER_PATH,
)

# Paths
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
OUTPUT_DIR = 'models/'
RESULTS_CSV = OUTPUT_DIR + 'model_comparison_results.csv'
DETAILED_CSV = OUTPUT_DIR + 'model_comparison_detailed.csv'

# Evaluation settings
MC_SAMPLES = 20
BATCH_SIZE = 256


def load_test_data(test_path, scaler_path):
    """Load and preprocess test data"""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    test_df = pd.read_csv(test_path)
    test_df = test_df.drop('id', axis=1)
    
    # Encode categorical
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        le = LabelEncoder()
        test_df[col] = le.fit_transform(test_df[col].astype(str))
    
    test_df = test_df.drop('attack_cat', axis=1)
    
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Handle NaN/Inf
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Scale
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)
    
    return X_test, y_test


def evaluate_ae(device):
    """Evaluate Autoencoder"""
    print("\n" + "=" * 70)
    print("EVALUATING AUTOENCODER (AE)")
    print("=" * 70)
    
    try:
        # Load model
        checkpoint = torch.load(AE_MODEL_PATH, map_location=device)
        model = Autoencoder(
            checkpoint['input_dim'],
            checkpoint['latent_dim'],
            checkpoint['hidden_dims']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load threshold
        threshold = np.load(AE_THRESHOLD_PATH)
        
        # Load test data
        X_test, y_test = load_test_data(TEST_PATH, AE_SCALER_PATH)
        
        # Compute reconstruction errors
        dataset = TensorDataset(torch.FloatTensor(X_test))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        errors = compute_reconstruction_error(model, loader, device)
        
        # Predictions
        y_pred = (errors > threshold).astype(int)
        
        # Metrics
        results = compute_metrics(y_test, y_pred, errors, model_name="AE")
        results['threshold'] = threshold
        results['has_uncertainty'] = False
        
        print(f"\nThreshold: {threshold:.6f}")
        print_metrics(results)
        
        return results, errors, None, None
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not load AE model - {e}")
        print("Please run: python ae_train.py")
        return None, None, None, None


def evaluate_vae(device):
    """Evaluate VAE"""
    print("\n" + "=" * 70)
    print("EVALUATING VARIATIONAL AUTOENCODER (VAE)")
    print("=" * 70)
    
    try:
        # Load model
        checkpoint = torch.load(VAE_MODEL_PATH, map_location=device)
        model = VAE(
            checkpoint['input_dim'],
            checkpoint['latent_dim'],
            checkpoint['hidden_dims'],
            checkpoint['dropout_rate']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load threshold
        threshold = np.load(VAE_THRESHOLD_PATH)
        
        # Load test data
        X_test, y_test = load_test_data(TEST_PATH, VAE_SCALER_PATH)
        
        # Compute reconstruction errors with uncertainty
        dataset = TensorDataset(torch.FloatTensor(X_test))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        errors, epistemic, aleatoric = vae_compute_metrics(model, loader, device, mc_samples=MC_SAMPLES)
        
        # Predictions
        y_pred = (errors > threshold).astype(int)
        
        # Metrics
        results = compute_metrics(y_test, y_pred, errors, model_name="VAE")
        results['threshold'] = threshold
        results['has_uncertainty'] = True
        results['mean_epistemic_normal'] = epistemic[y_test == 0].mean()
        results['mean_epistemic_attack'] = epistemic[y_test == 1].mean()
        results['mean_aleatoric_normal'] = aleatoric[y_test == 0].mean()
        results['mean_aleatoric_attack'] = aleatoric[y_test == 1].mean()
        
        print(f"\nThreshold: {threshold:.6f}")
        print_metrics(results)
        print_uncertainty(epistemic, aleatoric, y_test)
        
        return results, errors, epistemic, aleatoric
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not load VAE model - {e}")
        print("Please run: python vae_train.py")
        return None, None, None, None


def evaluate_bvae(device):
    """Evaluate Bayesian VAE"""
    print("\n" + "=" * 70)
    print("EVALUATING BAYESIAN VARIATIONAL AUTOENCODER (BVAE)")
    print("=" * 70)
    
    try:
        # Load model
        checkpoint = torch.load(BVAE_MODEL_PATH, map_location=device)
        model = BayesianVAE(
            checkpoint['input_dim'],
            checkpoint['latent_dim'],
            checkpoint['hidden_dims'],
            checkpoint['prior_sigma']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load threshold
        threshold = np.load(BVAE_THRESHOLD_PATH)
        
        # Load test data
        X_test, y_test = load_test_data(TEST_PATH, BVAE_SCALER_PATH)
        
        # Compute reconstruction errors with uncertainty
        dataset = TensorDataset(torch.FloatTensor(X_test))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        errors, epistemic, aleatoric = bvae_compute_metrics(model, loader, device, mc_samples=MC_SAMPLES)
        
        # Predictions
        y_pred = (errors > threshold).astype(int)
        
        # Metrics
        results = compute_metrics(y_test, y_pred, errors, model_name="BVAE")
        results['threshold'] = threshold
        results['has_uncertainty'] = True
        results['mean_epistemic_normal'] = epistemic[y_test == 0].mean()
        results['mean_epistemic_attack'] = epistemic[y_test == 1].mean()
        results['mean_aleatoric_normal'] = aleatoric[y_test == 0].mean()
        results['mean_aleatoric_attack'] = aleatoric[y_test == 1].mean()
        
        print(f"\nThreshold: {threshold:.6f}")
        print_metrics(results)
        print_uncertainty(epistemic, aleatoric, y_test)
        
        return results, errors, epistemic, aleatoric
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not load BVAE model - {e}")
        print("Please run: python bvae_train.py")
        return None, None, None, None


def compute_metrics(y_true, y_pred, scores, model_name="Model"):
    """Compute comprehensive metrics"""
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, scores),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results['true_negative'] = tn
    results['false_positive'] = fp
    results['false_negative'] = fn
    results['true_positive'] = tp
    results['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    results['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Reconstruction error statistics
    results['error_mean_normal'] = scores[y_true == 0].mean()
    results['error_std_normal'] = scores[y_true == 0].std()
    results['error_mean_attack'] = scores[y_true == 1].mean()
    results['error_std_attack'] = scores[y_true == 1].std()
    results['error_separation'] = (scores[y_true == 1].mean() - scores[y_true == 0].mean()) / scores[y_true == 0].std()
    
    return results


def print_metrics(results):
    """Print metrics in a nice format"""
    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy:  {results['accuracy'] * 100:6.2f}%")
    print(f"  Precision: {results['precision'] * 100:6.2f}%")
    print(f"  Recall:    {results['recall'] * 100:6.2f}%")
    print(f"  F1-Score:  {results['f1'] * 100:6.2f}%")
    print(f"  ROC-AUC:   {results['auc']:6.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Attack")
    print(f"Actual Normal  {results['true_negative']:6d}  {results['false_positive']:6d}")
    print(f"       Attack  {results['false_negative']:6d}  {results['true_positive']:6d}")
    
    print(f"\n❌ Error Rates:")
    print(f"  False Positive Rate: {results['fpr'] * 100:6.2f}%")
    print(f"  False Negative Rate: {results['fnr'] * 100:6.2f}%")
    
    print(f"\n📏 Reconstruction Error:")
    print(f"  Normal - Mean: {results['error_mean_normal']:.6f}, Std: {results['error_std_normal']:.6f}")
    print(f"  Attack - Mean: {results['error_mean_attack']:.6f}, Std: {results['error_std_attack']:.6f}")
    print(f"  Separation (σ): {results['error_separation']:.2f}")


def print_uncertainty(epistemic, aleatoric, y_true):
    """Print uncertainty statistics"""
    print(f"\n🔮 Epistemic Uncertainty (Model Uncertainty):")
    print(f"  Normal - Mean: {epistemic[y_true == 0].mean():.6f}, Std: {epistemic[y_true == 0].std():.6f}")
    print(f"  Attack - Mean: {epistemic[y_true == 1].mean():.6f}, Std: {epistemic[y_true == 1].std():.6f}")
    print(f"  Ratio (Attack/Normal): {epistemic[y_true == 1].mean() / epistemic[y_true == 0].mean():.2f}x")
    
    print(f"\n📊 Aleatoric Uncertainty (Data Noise):")
    print(f"  Normal - Mean: {aleatoric[y_true == 0].mean():.6f}, Std: {aleatoric[y_true == 0].std():.6f}")
    print(f"  Attack - Mean: {aleatoric[y_true == 1].mean():.6f}, Std: {aleatoric[y_true == 1].std():.6f}")
    print(f"  Ratio (Attack/Normal): {aleatoric[y_true == 1].mean() / aleatoric[y_true == 0].mean():.2f}x")


def create_comparison_table(results_list):
    """Create comparison table"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    if len(results_list) == 0:
        print("No models successfully evaluated!")
        return None
    
    df = pd.DataFrame(results_list)
    
    # Main metrics table
    print("\n📈 Performance Metrics:")
    print("-" * 70)
    print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['accuracy']*100:>9.2f}% {row['precision']*100:>9.2f}% "
              f"{row['recall']*100:>9.2f}% {row['f1']*100:>9.2f}% {row['auc']:>9.4f}")
    print("-" * 70)
    
    # Best model
    best_f1 = df.loc[df['f1'].idxmax()]
    print(f"\n🏆 Best F1-Score: {best_f1['model']} ({best_f1['f1']*100:.2f}%)")
    
    best_auc = df.loc[df['auc'].idxmax()]
    print(f"🏆 Best ROC-AUC: {best_auc['model']} ({best_auc['auc']:.4f})")
    
    # Error rates
    print("\n⚠️  Error Rates:")
    print("-" * 70)
    print(f"{'Model':<10} {'FPR':>10} {'FNR':>10} {'Separation':>12}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['fpr']*100:>9.2f}% {row['fnr']*100:>9.2f}% {row['error_separation']:>11.2f}σ")
    print("-" * 70)
    
    # Uncertainty (if available)
    if 'mean_epistemic_attack' in df.columns:
        uncertainty_df = df[df['has_uncertainty'] == True]
        if len(uncertainty_df) > 0:
            print("\n🔮 Uncertainty Metrics (Attack Samples):")
            print("-" * 70)
            print(f"{'Model':<10} {'Epistemic':>12} {'Aleatoric':>12}")
            print("-" * 70)
            for _, row in uncertainty_df.iterrows():
                print(f"{row['model']:<10} {row['mean_epistemic_attack']:>12.6f} {row['mean_aleatoric_attack']:>12.6f}")
            print("-" * 70)
    
    return df


def save_results(results_list, output_csv):
    """Save results to CSV"""
    results_list = [r for r in results_list if r is not None]
    if len(results_list) > 0:
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")


def main():
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION: AE vs VAE vs BVAE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"MC Samples: {MC_SAMPLES}")
    print(f"Test Data: {TEST_PATH}")
    
    # Evaluate all models
    ae_results, ae_errors, _, _ = evaluate_ae(device)
    vae_results, vae_errors, vae_epi, vae_ale = evaluate_vae(device)
    bvae_results, bvae_errors, bvae_epi, bvae_ale = evaluate_bvae(device)
    
    # Compare
    all_results = [ae_results, vae_results, bvae_results]
    comparison_df = create_comparison_table(all_results)
    
    # Save
    if comparison_df is not None:
        save_results(all_results, RESULTS_CSV)
        
        # Save detailed errors for plotting
        if all([r is not None for r in [ae_results, vae_results, bvae_results]]):
            print("\n📊 All models evaluated successfully!")
            print(f"\nTo visualize results, you can load {RESULTS_CSV}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return comparison_df


if __name__ == '__main__':
    main()