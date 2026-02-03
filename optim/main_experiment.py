#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for MESU anomaly detection with concept drift evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from pathlib import Path

from mesu_model import MESUAnomalyDetector
from baseline_model import BaselineMLP, ExperienceReplay
from mesu_optimizer import MESU
from data_utils import (load_and_preprocess_unsw, simulate_concept_drift,
                        create_dataloaders, balance_dataset)


def train_epoch(model, dataloader, optimizer, device, is_mesu=True, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        if is_mesu:
            optimizer.zero_grad()
            loss = model.loss(batch_x, batch_y, samples=10, class_weights=class_weights)
        else:
            optimizer.zero_grad()
            loss = model.loss(batch_x, batch_y, class_weights=class_weights)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device, is_mesu=True):
    """Evaluate model performance."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if is_mesu:
                preds, _ = model.predict(batch_x, samples=10)
                log_probs = model.forward(batch_x, samples=10).mean(0)
                probs = torch.exp(log_probs)[:, 1]
            else:
                preds = model.predict(batch_x)
                log_probs = model.forward(batch_x)
                probs = torch.exp(log_probs)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    attack_pred_rate = (all_preds == 1).mean() if len(all_preds) > 0 else 0.0
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'attack_pred_rate': attack_pred_rate
    }
    
    return metrics


def train_with_concept_drift(model_type='mesu', drift_type='sudden', num_drifts=3,
                             train_path='data/UNSW_NB15_training-set.csv',
                             test_path='data/UNSW_NB15_testing-set.csv',
                             device='cpu'):
    """
    Train and evaluate model with concept drift.
    
    Args:
        model_type: 'mesu' or 'baseline'
        drift_type: 'sudden', 'gradual', or 'incremental'
        num_drifts: Number of drift points
        
    Returns:
        results: Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} with {drift_type} concept drift")
    print(f"{'='*60}\n")
    
    # Load data
    (X_train, y_train), (X_test, y_test), feature_names, scalers = \
        load_and_preprocess_unsw(train_path, test_path)
    
    input_dim = X_train.shape[1]
    
    # Simulate concept drift on training data
    drift_datasets, drift_points = simulate_concept_drift(
        X_train, y_train, drift_type=drift_type, num_drifts=num_drifts
    )
    
    # Model configuration
    model_config = {
        'input_dim': input_dim,
        'hidden_dims': [128, 64, 32],
        'output_dim': 2,
        # Use paper-aligned sigma init (layer-dependent) when None.
        'sigma_init': None,
        'sigma_prior': 0.1,
        'activation': 'Relu',
        'dropout': 0.3,
        'coeff_likeli': 1.0,
        # Use mean reduction for stable scaling with minibatches.
        'reduction': 'mean'
    }
    
    # Initialize model
    if model_type == 'mesu':
        model = MESUAnomalyDetector(model_config).to(device)
        
        optimizer_config = {
            'mu_prior': 0.0,
            'sigma_prior': 0.1,
            # Use dynamic N per concept and moderate learning rates for tabular data.
            'N': 100,
            'c_sigma': 1.0,
            'c_mu': 5.0,
            'second_order': True,
            'clamp_sigma': [1e-4, 0.1],
            'clamp_mu': [0, 0],
            'ratio_max': 0.5,
            'moment_sigma': 0.9,
            'moment_mu': 0.9
        }
        optimizer = MESU(model, optimizer_config)
        is_mesu = True
        
    else:  # baseline
        model = BaselineMLP(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        is_mesu = False
    
    # Test dataloader (remains constant)
    test_loader = create_dataloaders(X_test, y_test, batch_size=256, shuffle=False)
    
    # Track results
    results = {
        'drift_points': drift_points,
        'concept_metrics': [],  # Metrics for each concept
        'test_metrics': [],     # Test metrics after each concept
        'training_times': [],
        'losses': []
    }
    
    # Train on each concept sequentially
    for concept_idx, (X_concept, y_concept) in enumerate(drift_datasets):
        print(f"\n--- Concept {concept_idx + 1}/{len(drift_datasets)} ---")
        normal_count = int((y_concept == 0).sum())
        attack_count = int((y_concept == 1).sum())
        print(f"Samples: {len(X_concept)}, Normal: {normal_count}, Attack: {attack_count}")
        
        # Create dataloader for this concept
        concept_loader = create_dataloaders(X_concept, y_concept, batch_size=64, shuffle=True)
        
        # Train on this concept
        epochs_per_concept = 20 if is_mesu else 10
        concept_losses = []
        start_time = time.time()

        # Align MESU's N with the current concept data size (number of batches).
        if is_mesu:
            optimizer.N = max(1, len(concept_loader))
        
        # Compute class weights for this concept (balanced loss)
        total = max(1, normal_count + attack_count)
        w0 = total / (2 * max(1, normal_count))
        w1 = total / (2 * max(1, attack_count))
        class_weights = torch.tensor([w0, w1], device=device, dtype=torch.float32)
        print(f"Class weights: normal={w0:.3f}, attack={w1:.3f}")

        for epoch in range(epochs_per_concept):
            loss = train_epoch(model, concept_loader, optimizer, device, is_mesu, class_weights=class_weights)
            concept_losses.append(loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs_per_concept}, Loss: {loss:.4f}")
        
        training_time = time.time() - start_time
        results['training_times'].append(training_time)
        results['losses'].extend(concept_losses)
        
        # Evaluate on current concept (train data)
        concept_metrics = evaluate_model(model, concept_loader, device, is_mesu)
        results['concept_metrics'].append(concept_metrics)
        
        print(f"Concept {concept_idx + 1} Training Metrics:")
        print(f"  Accuracy: {concept_metrics['accuracy']:.4f}")
        print(f"  F1: {concept_metrics['f1']:.4f}")
        print(f"  AUC: {concept_metrics['auc']:.4f}")
        print(f"  Attack Prediction Rate: {concept_metrics['attack_pred_rate']:.4f}")
        
        # Evaluate on test set (to check forgetting)
        test_metrics = evaluate_model(model, test_loader, device, is_mesu)
        results['test_metrics'].append(test_metrics)
        
        print(f"Test Metrics after Concept {concept_idx + 1}:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  Attack Prediction Rate: {test_metrics['attack_pred_rate']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_test_metrics = results['test_metrics'][-1]
    print(f"\nFinal Test Performance:")
    print(f"  Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_test_metrics['precision']:.4f}")
    print(f"  Recall: {final_test_metrics['recall']:.4f}")
    print(f"  F1: {final_test_metrics['f1']:.4f}")
    print(f"  AUC: {final_test_metrics['auc']:.4f}")
    
    # Calculate catastrophic forgetting metric
    # Compare first concept performance to final
    if len(results['test_metrics']) > 1:
        initial_acc = results['test_metrics'][0]['accuracy']
        final_acc = results['test_metrics'][-1]['accuracy']
        forgetting = initial_acc - final_acc
        print(f"\nCatastrophic Forgetting Metric:")
        print(f"  Initial accuracy: {initial_acc:.4f}")
        print(f"  Final accuracy: {final_acc:.4f}")
        print(f"  Forgetting (Δ): {forgetting:.4f} ({'better' if forgetting < 0 else 'worse'})")
        results['forgetting'] = forgetting
    
    return results


def plot_results(mesu_results, baseline_results, drift_type, output_dir='results'):
    """Plot comparison of MESU vs Baseline."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract metrics over concepts
    mesu_test_acc = [m['accuracy'] for m in mesu_results['test_metrics']]
    mesu_test_f1 = [m['f1'] for m in mesu_results['test_metrics']]
    baseline_test_acc = [m['accuracy'] for m in baseline_results['test_metrics']]
    baseline_test_f1 = [m['f1'] for m in baseline_results['test_metrics']]
    
    concepts = list(range(1, len(mesu_test_acc) + 1))
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(concepts, mesu_test_acc, 'o-', label='MESU', linewidth=2, markersize=8)
    plt.plot(concepts, baseline_test_acc, 's-', label='Baseline EP', linewidth=2, markersize=8)
    plt.xlabel('Concept Number', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f'Accuracy with {drift_type.capitalize()} Drift', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(concepts, mesu_test_f1, 'o-', label='MESU', linewidth=2, markersize=8)
    plt.plot(concepts, baseline_test_f1, 's-', label='Baseline EP', linewidth=2, markersize=8)
    plt.xlabel('Concept Number', fontsize=12)
    plt.ylabel('Test F1 Score', fontsize=12)
    plt.title(f'F1 Score with {drift_type.capitalize()} Drift', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_{drift_type}_drift.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_dir}/comparison_{drift_type}_drift.png")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(mesu_results['losses'], label='MESU', alpha=0.7)
    plt.plot(baseline_results['losses'], label='Baseline', alpha=0.7)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_dir}/loss_{drift_type}_drift.png', dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {output_dir}/loss_{drift_type}_drift.png")


def main():
    """Main experiment runner."""
    
    # Configuration
    train_path = 'data/UNSW_NB15_training-set.csv'
    test_path = 'data/UNSW_NB15_testing-set.csv'
    drift_types = ['sudden', 'gradual', 'incremental']
    
    # Run experiments for each drift type
    for drift_type in drift_types:
        print(f"\n{'#'*60}")
        print(f"EXPERIMENT: {drift_type.upper()} DRIFT")
        print(f"{'#'*60}")
        
        # Train MESU
        mesu_results = train_with_concept_drift(
            model_type='mesu',
            drift_type=drift_type,
            num_drifts=3,
            train_path=train_path,
            test_path=test_path
        )
        
        # Train Baseline
        baseline_results = train_with_concept_drift(
            model_type='baseline',
            drift_type=drift_type,
            num_drifts=3,
            train_path=train_path,
            test_path=test_path
        )
        
        # Plot results
        plot_results(mesu_results, baseline_results, drift_type)
        
        # Summary comparison
        print(f"\n{'='*60}")
        print(f"SUMMARY: {drift_type.upper()} DRIFT")
        print(f"{'='*60}")
        print("\nMESU:")
        print(f"  Final Accuracy: {mesu_results['test_metrics'][-1]['accuracy']:.4f}")
        print(f"  Final F1: {mesu_results['test_metrics'][-1]['f1']:.4f}")
        print(f"  Forgetting: {mesu_results.get('forgetting', 0):.4f}")
        
        print("\nBaseline:")
        print(f"  Final Accuracy: {baseline_results['test_metrics'][-1]['accuracy']:.4f}")
        print(f"  Final F1: {baseline_results['test_metrics'][-1]['f1']:.4f}")
        print(f"  Forgetting: {baseline_results.get('forgetting', 0):.4f}")
        
        print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
