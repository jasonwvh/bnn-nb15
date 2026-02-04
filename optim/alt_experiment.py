#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternate experiment: compare Bayes VCL vs Bayes VCL-U.
"""

import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from baseline_model import VCLBayesianModel, UncertaintyVCLBayesianModel, BaselineMLP
from data_utils import (load_and_preprocess_unsw, simulate_concept_drift,
                        create_dataloaders)

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Compute ECE for binary classification."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        acc = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)

def brier_score(y_true, y_prob):
    """Brier score for binary classification."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def train_epoch(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = model.loss(batch_x, batch_y, class_weights=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device, is_bayes=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if is_bayes:
                preds, _ = model.predict(batch_x)
                log_probs = model.forward(batch_x, samples=model.eval_samples).mean(0)
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

    attack_pred_rate = (all_preds == 1).mean() if len(all_preds) > 0 else 0.0
    ece = expected_calibration_error(all_labels, all_probs)
    brier = brier_score(all_labels, all_probs)
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'ece': ece,
        'brier': brier,
        'attack_pred_rate': attack_pred_rate
    }

    return metrics


def train_with_concept_drift(model_type='vcl', drift_type='sudden', num_drifts=3,
                             train_path='data/UNSW_NB15_training-set.csv',
                             test_path='data/UNSW_NB15_testing-set.csv',
                             device='cuda'):
    print(f"\n{'='*60}")
    if model_type == 'vcl':
        label = 'BAYES VCL'
    elif model_type == 'vcl_u':
        label = 'BAYES VCL-U'
    elif model_type == 'mlp':
        label = 'BASELINE MLP'
    else:
        label = f'UNKNOWN ({model_type})'

    print(f"Training {label} with {drift_type} concept drift")
    print(f"{'='*60}\n")

    (X_train, y_train), (X_test, y_test), feature_names, scalers = \
        load_and_preprocess_unsw(train_path, test_path)

    input_dim = X_train.shape[1]

    drift_datasets, drift_points = simulate_concept_drift(
        X_train, y_train, drift_type=drift_type, num_drifts=num_drifts
    )

    model_config = {
        'input_dim': input_dim,
        'hidden_dims': [128, 64, 32],
        'output_dim': 2,
        'sigma_init': 0.001,
        'sigma_prior': 0.01,
        'activation': 'Relu',
        'dropout': 0.3,
        'kl_weight': 1.0,
        'train_samples': 5,
        'eval_samples': 10
    }

    if model_type == 'vcl':
        model = VCLBayesianModel(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        is_bayes = True
    elif model_type == 'vcl_u':
        model = UncertaintyVCLBayesianModel(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        is_bayes = True
    elif model_type == 'mlp':
        model = BaselineMLP(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        is_bayes = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create an overall test loader (kept for compatibility) and per-concept test sets
    test_loader = create_dataloaders(X_test, y_test, batch_size=256, shuffle=False)

    test_concept_datasets, test_drift_points = simulate_concept_drift(
        X_test, y_test, drift_type=drift_type, num_drifts=num_drifts, shuffle=False
    )
    test_concept_loaders = [create_dataloaders(Xc, yc, batch_size=256, shuffle=False)
                            for (Xc, yc) in test_concept_datasets]

    results = {
        'drift_points': drift_points,
        'concept_metrics': [],            # training-time metrics per concept
        'per_concept_test_metrics': [],   # after each concept, metrics on each concept's test set
        'test_metrics': [],               # aggregated (mean) test metrics after each concept (for plotting/backwards-compat)
        'training_times': [],
        'losses': [],
        'initial_accs': [None] * len(test_concept_loaders)
    }

    for concept_idx, (X_concept, y_concept) in enumerate(drift_datasets):
        print(f"\n--- Concept {concept_idx + 1}/{len(drift_datasets)} ---")
        normal_count = int((y_concept == 0).sum())
        attack_count = int((y_concept == 1).sum())
        print(f"Samples: {len(X_concept)}, Normal: {normal_count}, Attack: {attack_count}")

        concept_loader = create_dataloaders(X_concept, y_concept, batch_size=64, shuffle=True)

        epochs_per_concept = 10
        concept_losses = []
        start_time = time.time()

        total = max(1, normal_count + attack_count)
        w0 = total / (2 * max(1, normal_count))
        w1 = total / (2 * max(1, attack_count))
        class_weights = torch.tensor([w0, w1], device=device, dtype=torch.float32)
        print(f"Class weights: normal={w0:.3f}, attack={w1:.3f}")

        for epoch in range(epochs_per_concept):
            loss = train_epoch(model, concept_loader, optimizer, device, class_weights=class_weights)
            concept_losses.append(loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs_per_concept}, Loss: {loss:.4f}")

        training_time = time.time() - start_time
        results['training_times'].append(training_time)
        results['losses'].extend(concept_losses)

        concept_metrics = evaluate_model(model, concept_loader, device, is_bayes=is_bayes)
        results['concept_metrics'].append(concept_metrics)

        print(f"Concept {concept_idx + 1} Training Metrics:")
        print(f"  Accuracy: {concept_metrics['accuracy']:.4f}")
        print(f"  F1: {concept_metrics['f1']:.4f}")
        print(f"  AUC: {concept_metrics['auc']:.4f}")
        print(f"  ECE: {concept_metrics['ece']:.4f}")
        print(f"  Brier: {concept_metrics['brier']:.4f}")
        print(f"  Attack Prediction Rate: {concept_metrics['attack_pred_rate']:.4f}")

        # Evaluate on each concept-specific test set to measure forgetting properly
        per_concept_metrics = []
        for t_idx, t_loader in enumerate(test_concept_loaders):
            m = evaluate_model(model, t_loader, device, is_bayes=is_bayes)
            per_concept_metrics.append(m)

            # Record initial accuracy for a concept right after it was learned
            if results['initial_accs'][t_idx] is None and t_idx == concept_idx:
                results['initial_accs'][t_idx] = m['accuracy']

        # Aggregate test metrics (mean over concepts) for backward compatibility / plotting
        agg_metrics = {}
        keys = per_concept_metrics[0].keys()
        for k in keys:
            agg_metrics[k] = float(np.mean([pm[k] for pm in per_concept_metrics]))

        results['per_concept_test_metrics'].append(per_concept_metrics)
        results['test_metrics'].append(agg_metrics)

        print(f"Test Metrics (aggregated) after Concept {concept_idx + 1}:")
        print(f"  Accuracy: {agg_metrics['accuracy']:.4f}")
        print(f"  F1: {agg_metrics['f1']:.4f}")
        print(f"  AUC: {agg_metrics['auc']:.4f}")
        print(f"  ECE: {agg_metrics['ece']:.4f}")
        print(f"  Brier: {agg_metrics['brier']:.4f}")
        print(f"  Attack Prediction Rate: {agg_metrics['attack_pred_rate']:.4f}")
        print(f"  Training time: {training_time:.2f}s")

        if is_bayes and hasattr(model, 'update_prior'):
            # Some implementations accept (dataloader, device), others no-arg
            try:
                model.update_prior(concept_loader, device)
            except TypeError:
                model.update_prior()

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    # After all concepts, evaluate final model on each concept's test set
    final_per_concept = [evaluate_model(model, tl, device, is_bayes=is_bayes)
                         for tl in test_concept_loaders]

    final_accs = [m['accuracy'] for m in final_per_concept]
    initial_accs = results.get('initial_accs', [None] * len(final_accs))

    final_test_metrics = {}
    keys = final_per_concept[0].keys()
    for k in keys:
        final_test_metrics[k] = float(np.mean([pm[k] for pm in final_per_concept]))

    print(f"\nFinal Test Performance (aggregated over concepts):")
    print(f"  Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_test_metrics['precision']:.4f}")
    print(f"  Recall: {final_test_metrics['recall']:.4f}")
    print(f"  F1: {final_test_metrics['f1']:.4f}")
    print(f"  AUC: {final_test_metrics['auc']:.4f}")
    print(f"  ECE: {final_test_metrics['ece']:.4f}")
    print(f"  Brier: {final_test_metrics['brier']:.4f}")

    # Compute Backward Transfer (BWT) / Forgetting per concept
    # initial_accs: accuracy on a concept right after learning it
    # final_accs: accuracy on same concept after training on all concepts
    initial_accs = np.array([a if a is not None else 0.0 for a in initial_accs])
    final_accs = np.array(final_accs)
    bwt_per_concept = final_accs - initial_accs
    bwt = float(np.mean(bwt_per_concept))
    results['initial_accs'] = initial_accs.tolist()
    results['final_accs'] = final_accs.tolist()
    results['bwt'] = bwt

    if len(initial_accs) > 0:
        print(f"\nCatastrophic Forgetting / Backward Transfer:")
        for i, (ia, fa) in enumerate(zip(initial_accs, final_accs)):
            print(f"  Concept {i+1}: initial={ia:.4f}, final={fa:.4f}, delta={fa-ia:+.4f}")
        print(f"  Mean BWT: {bwt:+.4f} ({'positive (improved)' if bwt>0 else 'negative (forgotten)'} )")

    return results


def plot_results(vcl_results, vcl_u_results, mlp_results, drift_type, output_dir='results'):
    Path(output_dir).mkdir(exist_ok=True)

    vcl_test_acc = [m['accuracy'] for m in vcl_results['test_metrics']]
    vcl_test_f1 = [m['f1'] for m in vcl_results['test_metrics']]
    vcl_u_test_acc = [m['accuracy'] for m in vcl_u_results['test_metrics']]
    vcl_u_test_f1 = [m['f1'] for m in vcl_u_results['test_metrics']]

    mlp_test_acc = [m['accuracy'] for m in mlp_results['test_metrics']] if mlp_results is not None else []
    mlp_test_f1 = [m['f1'] for m in mlp_results['test_metrics']] if mlp_results is not None else []

    concepts = list(range(1, len(vcl_test_acc) + 1))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(concepts, vcl_test_acc, 'o-', label='Bayes VCL', linewidth=2, markersize=8)
    plt.plot(concepts, vcl_u_test_acc, 's-', label='Bayes VCL-U', linewidth=2, markersize=8)
    if mlp_test_acc:
        plt.plot(concepts, mlp_test_acc, '^-', label='Baseline MLP', linewidth=2, markersize=8)
    plt.xlabel('Concept Number', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f'Accuracy with {drift_type.capitalize()} Drift', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(concepts, vcl_test_f1, 'o-', label='Bayes VCL', linewidth=2, markersize=8)
    plt.plot(concepts, vcl_u_test_f1, 's-', label='Bayes VCL-U', linewidth=2, markersize=8)
    if mlp_test_f1:
        plt.plot(concepts, mlp_test_f1, '^-', label='Baseline MLP', linewidth=2, markersize=8)
    plt.xlabel('Concept Number', fontsize=12)
    plt.ylabel('Test F1 Score', fontsize=12)
    plt.title(f'F1 Score with {drift_type.capitalize()} Drift', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_vcl_vs_vclu_vs_mlp_{drift_type}.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_dir}/comparison_vcl_vs_vclu_vs_mlp_{drift_type}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(vcl_results['losses'], label='Bayes VCL', alpha=0.7)
    plt.plot(vcl_u_results['losses'], label='Bayes VCL-U', alpha=0.7)
    if mlp_results is not None:
        plt.plot(mlp_results['losses'], label='Baseline MLP', alpha=0.7)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_dir}/loss_vcl_vs_vclu_vs_mlp_{drift_type}.png', dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {output_dir}/loss_vcl_vs_vclu_vs_mlp_{drift_type}.png")


def main():
    train_path = 'data/UNSW_NB15_training-set.csv'
    test_path = 'data/UNSW_NB15_testing-set.csv'
    drift_types = ['sudden', 'gradual', 'incremental']

    for drift_type in drift_types:
        print(f"\n{'#'*60}")
        print(f"EXPERIMENT: {drift_type.upper()} DRIFT")
        print(f"{'#'*60}")

        vcl_u_results = train_with_concept_drift(
            model_type='vcl_u',
            drift_type=drift_type,
            num_drifts=3,
            train_path=train_path,
            test_path=test_path
        )

        vcl_results = train_with_concept_drift(
            model_type='vcl',
            drift_type=drift_type,
            num_drifts=3,
            train_path=train_path,
            test_path=test_path
        )

        mlp_results = train_with_concept_drift(
            model_type='mlp',
            drift_type=drift_type,
            num_drifts=3,
            train_path=train_path,
            test_path=test_path
        )
        plot_results(vcl_results, vcl_u_results, mlp_results, drift_type)

        print(f"\n{'='*60}")
        print(f"SUMMARY: {drift_type.upper()} DRIFT")
        print(f"{'='*60}")
        print("\nBayes VCL:")
        print(f"  Final Accuracy: {vcl_results['test_metrics'][-1]['accuracy']:.4f}")
        print(f"  Final F1: {vcl_results['test_metrics'][-1]['f1']:.4f}")
        print(f"  Final ECE: {vcl_results['test_metrics'][-1]['ece']:.4f}")
        print(f"  Final Brier: {vcl_results['test_metrics'][-1]['brier']:.4f}")
        print(f"  Forgetting: {vcl_results.get('forgetting', 0):.4f}")

        print("\nBayes VCL-U:")
        print(f"  Final Accuracy: {vcl_u_results['test_metrics'][-1]['accuracy']:.4f}")
        print(f"  Final F1: {vcl_u_results['test_metrics'][-1]['f1']:.4f}")
        print(f"  Final ECE: {vcl_u_results['test_metrics'][-1]['ece']:.4f}")
        print(f"  Final Brier: {vcl_u_results['test_metrics'][-1]['brier']:.4f}")
        print(f"  Forgetting: {vcl_u_results.get('forgetting', 0):.4f}")

        print("\nBaseline MLP:")
        print(f"  Final Accuracy: {mlp_results['test_metrics'][-1]['accuracy']:.4f}")
        print(f"  Final F1: {mlp_results['test_metrics'][-1]['f1']:.4f}")
        print(f"  Final ECE: {mlp_results['test_metrics'][-1]['ece']:.4f}")
        print(f"  Final Brier: {mlp_results['test_metrics'][-1]['brier']:.4f}")
        print(f"  Forgetting: {mlp_results.get('forgetting', 0):.4f}")

        print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
