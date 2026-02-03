#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive analysis of MESU vs Baseline for anomaly detection
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from mesu_model import MESUAnomalyDetector
from baseline_model import BaselineMLP
from mesu_optimizer import MESU
from data_utils import load_and_preprocess_unsw, simulate_concept_drift, create_dataloaders


def analyze_model_performance(train_path='data/UNSW_NB15_training-set.csv',
                              test_path='data/UNSW_NB15_testing-set.csv'):
    """Comprehensive analysis of MESU model."""
    
    print("MESU Anomaly Detection - Comprehensive Analysis")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    (X_train, y_train), (X_test, y_test), features, _ = \
        load_and_preprocess_unsw(train_path, test_path)
    
    # Create drift datasets
    drift_datasets, _ = simulate_concept_drift(X_train, y_train, 'sudden', 3)
    
    # Model config
    config = {
        'input_dim': X_train.shape[1],
        'hidden_dims': [128, 64, 32],
        'output_dim': 2,
        'sigma_init': 0.1,
        'sigma_prior': 0.1,
        'activation': 'Relu',
        'dropout': 0.3
    }
    
    # Initialize models
    mesu_model = MESUAnomalyDetector(config).to(device)
    baseline_model = BaselineMLP(config).to(device)
    
    mesu_opt = MESU(mesu_model, {
        'N': 100, 'c_sigma': 0.001, 'c_mu': 0.001,
        'sigma_prior': 0.1, 'mu_prior': 0.0,
        'second_order': True, 'clamp_sigma': [0.001, 1.0],
        'clamp_mu': [0, 0], 'ratio_max': 0.1,
        'moment_sigma': 0.9, 'moment_mu': 0.9
    })
    baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    
    test_loader = create_dataloaders(X_test, y_test, batch_size=256, shuffle=False)
    
    # Train on all concepts
    print("\nTraining models on concept sequence...")
    for idx, (X_c, y_c) in enumerate(drift_datasets):
        print(f"  Concept {idx+1}/{len(drift_datasets)}...")
        loader = create_dataloaders(X_c, y_c, batch_size=64, shuffle=True)
        
        # Train MESU
        for _ in range(5):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                mesu_opt.zero_grad()
                loss = mesu_model.loss(batch_x, batch_y, samples=5)
                loss.backward()
                mesu_opt.step()
        
        # Train Baseline
        for _ in range(5):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                baseline_opt.zero_grad()
                loss = baseline_model.loss(batch_x, batch_y)
                loss.backward()
                baseline_opt.step()
    
    # Evaluate
    print("\nEvaluating models...")
    mesu_model.eval()
    baseline_model.eval()
    
    mesu_preds = []
    baseline_preds = []
    mesu_probs = []
    mesu_uncertainties = []
    labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # MESU
            preds, unc = mesu_model.predict(batch_x, samples=10)
            probs, _ = mesu_model.get_anomaly_score(batch_x, samples=10)
            mesu_preds.extend(preds.cpu().numpy())
            mesu_probs.extend(probs.cpu().numpy())
            mesu_uncertainties.extend(unc.cpu().numpy())
            
            # Baseline
            preds = baseline_model.predict(batch_x)
            baseline_preds.extend(preds.cpu().numpy())
            
            labels.extend(batch_y.cpu().numpy())
    
    mesu_preds = np.array(mesu_preds)
    baseline_preds = np.array(baseline_preds)
    labels = np.array(labels)
    mesu_probs = np.array(mesu_probs)
    mesu_uncertainties = np.array(mesu_uncertainties)
    
    # Generate plots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Confusion matrices
    ax1 = plt.subplot(2, 3, 1)
    cm_mesu = confusion_matrix(labels, mesu_preds)
    sns.heatmap(cm_mesu, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('MESU Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    ax2 = plt.subplot(2, 3, 2)
    cm_baseline = confusion_matrix(labels, baseline_preds)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Oranges', ax=ax2)
    ax2.set_title('Baseline Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 2. ROC-style plot
    ax3 = plt.subplot(2, 3, 3)
    thresholds = np.linspace(0, 1, 100)
    tpr_mesu = []
    fpr_mesu = []
    for thresh in thresholds:
        pred = (mesu_probs > thresh).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tpr_mesu.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_mesu.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    ax3.plot(fpr_mesu, tpr_mesu, 'b-', linewidth=2, label='MESU')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 3. Uncertainty histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(mesu_uncertainties[labels == 0], bins=30, alpha=0.5, label='Normal', color='blue')
    ax4.hist(mesu_uncertainties[labels == 1], bins=30, alpha=0.5, label='Attack', color='red')
    ax4.set_xlabel('Uncertainty')
    ax4.set_ylabel('Count')
    ax4.set_title('MESU Uncertainty Distribution')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 4. Accuracy by uncertainty level
    ax5 = plt.subplot(2, 3, 5)
    unc_bins = np.percentile(mesu_uncertainties, [0, 25, 50, 75, 100])
    accuracies = []
    bin_labels = []
    for i in range(len(unc_bins) - 1):
        mask = (mesu_uncertainties >= unc_bins[i]) & (mesu_uncertainties < unc_bins[i+1])
        if mask.sum() > 0:
            acc = (mesu_preds[mask] == labels[mask]).mean()
            accuracies.append(acc)
            bin_labels.append(f'{unc_bins[i]:.2f}-{unc_bins[i+1]:.2f}')
    
    ax5.bar(range(len(accuracies)), accuracies, color='green', alpha=0.7)
    ax5.set_xticks(range(len(accuracies)))
    ax5.set_xticklabels(bin_labels, rotation=45)
    ax5.set_ylabel('Accuracy')
    ax5.set_xlabel('Uncertainty Range')
    ax5.set_title('Accuracy vs Uncertainty Level')
    ax5.grid(alpha=0.3, axis='y')
    ax5.set_ylim([0, 1])
    
    # 5. Model comparison metrics
    ax6 = plt.subplot(2, 3, 6)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics_data = {
        'Accuracy': [accuracy_score(labels, mesu_preds), accuracy_score(labels, baseline_preds)],
        'Precision': [precision_score(labels, mesu_preds), precision_score(labels, baseline_preds)],
        'Recall': [recall_score(labels, mesu_preds), recall_score(labels, baseline_preds)],
        'F1': [f1_score(labels, mesu_preds), f1_score(labels, baseline_preds)]
    }
    
    x = np.arange(len(metrics_data))
    width = 0.35
    
    mesu_vals = [v[0] for v in metrics_data.values()]
    base_vals = [v[1] for v in metrics_data.values()]
    
    ax6.bar(x - width/2, mesu_vals, width, label='MESU', color='blue', alpha=0.7)
    ax6.bar(x + width/2, base_vals, width, label='Baseline', color='orange', alpha=0.7)
    ax6.set_ylabel('Score')
    ax6.set_title('Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_data.keys())
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')
    ax6.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("\nAnalysis plot saved: comprehensive_analysis.png")
    
    # Print detailed reports
    print("\n" + "=" * 60)
    print("MESU Classification Report:")
    print("=" * 60)
    print(classification_report(labels, mesu_preds, target_names=['Normal', 'Attack']))
    
    print("\n" + "=" * 60)
    print("Baseline Classification Report:")
    print("=" * 60)
    print(classification_report(labels, baseline_preds, target_names=['Normal', 'Attack']))
    
    # Uncertainty analysis
    print("\n" + "=" * 60)
    print("Uncertainty Analysis:")
    print("=" * 60)
    print(f"Mean uncertainty (Normal): {mesu_uncertainties[labels == 0].mean():.4f}")
    print(f"Mean uncertainty (Attack): {mesu_uncertainties[labels == 1].mean():.4f}")
    print(f"High uncertainty samples (>90th percentile): {(mesu_uncertainties > np.percentile(mesu_uncertainties, 90)).sum()}")
    
    high_unc_mask = mesu_uncertainties > np.percentile(mesu_uncertainties, 90)
    print(f"Accuracy on high uncertainty samples: {(mesu_preds[high_unc_mask] == labels[high_unc_mask]).mean():.4f}")
    print(f"Accuracy on low uncertainty samples: {(mesu_preds[~high_unc_mask] == labels[~high_unc_mask]).mean():.4f}")


if __name__ == '__main__':
    analyze_model_performance()
