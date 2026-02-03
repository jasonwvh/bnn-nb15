#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize uncertainty estimates from MESU model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from mesu_model import MESUAnomalyDetector
from mesu_optimizer import MESU
from data_utils import create_dataloaders


def visualize_uncertainty():
    """Demonstrate MESU's uncertainty quantification."""
    
    print("Visualizing MESU Uncertainty Estimates")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=42, n_informative=30,
        n_classes=2, weights=[0.7, 0.3], random_state=42
    )
    X = StandardScaler().fit_transform(X)
    
    # Split train/test
    split = 800
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # Initialize model
    config = {
        'input_dim': 42, 'hidden_dims': [128, 64, 32],
        'output_dim': 2, 'sigma_init': 0.1, 'sigma_prior': 0.1,
        'activation': 'Relu', 'dropout': 0.3
    }
    
    model = MESUAnomalyDetector(config).to(device)
    optimizer = MESU(model, {
        'N': 100, 'c_sigma': 0.001, 'c_mu': 0.001,
        'sigma_prior': 0.1, 'mu_prior': 0.0,
        'second_order': True, 'clamp_sigma': [0.001, 1.0],
        'clamp_mu': [0, 0], 'ratio_max': 0.1,
        'moment_sigma': 0.9, 'moment_mu': 0.9
    })
    
    # Train
    print("\nTraining model...")
    train_loader = create_dataloaders(X_train, y_train, batch_size=64, shuffle=True)
    
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch_x, batch_y, samples=5)
            loss.backward()
            optimizer.step()
    
    # Evaluate with uncertainty
    print("Computing predictions with uncertainty...")
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    attack_probs, uncertainties = model.get_anomaly_score(X_test_tensor, samples=20)
    
    attack_probs = attack_probs.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Prediction distribution
    ax = axes[0, 0]
    ax.hist(attack_probs[y_test == 0], bins=30, alpha=0.5, label='Normal', color='blue')
    ax.hist(attack_probs[y_test == 1], bins=30, alpha=0.5, label='Attack', color='red')
    ax.set_xlabel('Attack Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Prediction Distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Uncertainty distribution
    ax = axes[0, 1]
    ax.hist(uncertainties[y_test == 0], bins=30, alpha=0.5, label='Normal', color='blue')
    ax.hist(uncertainties[y_test == 1], bins=30, alpha=0.5, label='Attack', color='red')
    ax.set_xlabel('Uncertainty', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Uncertainty Distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Prediction vs Uncertainty (colored by true label)
    ax = axes[1, 0]
    scatter_normal = ax.scatter(attack_probs[y_test == 0], uncertainties[y_test == 0],
                                alpha=0.6, s=30, c='blue', label='Normal')
    scatter_attack = ax.scatter(attack_probs[y_test == 1], uncertainties[y_test == 1],
                                alpha=0.6, s=30, c='red', label='Attack')
    ax.set_xlabel('Attack Probability', fontsize=11)
    ax.set_ylabel('Uncertainty', fontsize=11)
    ax.set_title('Predictions vs Uncertainty', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Confidence-based detection
    ax = axes[1, 1]
    # Define high-confidence regions
    high_conf_normal = (attack_probs < 0.3) & (uncertainties < 0.2)
    high_conf_attack = (attack_probs > 0.7) & (uncertainties < 0.2)
    uncertain = uncertainties > 0.2
    
    correct_normal = high_conf_normal & (y_test == 0)
    correct_attack = high_conf_attack & (y_test == 1)
    incorrect = (high_conf_normal & (y_test == 1)) | (high_conf_attack & (y_test == 0))
    
    bars = ax.bar(['High Conf\nNormal', 'High Conf\nAttack', 'Uncertain', 'Errors'],
                  [correct_normal.sum(), correct_attack.sum(), uncertain.sum(), incorrect.sum()],
                  color=['lightblue', 'lightcoral', 'gold', 'gray'])
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Confidence-Based Classification', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: uncertainty_analysis.png")
    
    # Statistics
    print("\nStatistics:")
    print(f"  Mean uncertainty (Normal): {uncertainties[y_test == 0].mean():.4f}")
    print(f"  Mean uncertainty (Attack): {uncertainties[y_test == 1].mean():.4f}")
    print(f"  High confidence predictions: {(high_conf_normal | high_conf_attack).sum()}/{len(y_test)}")
    print(f"  Uncertain predictions: {uncertain.sum()}/{len(y_test)}")
    print(f"  Errors in high-confidence: {incorrect.sum()}/{(high_conf_normal | high_conf_attack).sum()}")


if __name__ == '__main__':
    visualize_uncertainty()
