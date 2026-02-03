#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing and concept drift simulation for UNSW-NB15
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class UNSWNB15Dataset(Dataset):
    """PyTorch dataset for UNSW-NB15."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_unsw(train_path, test_path):
    """
    Load and preprocess UNSW-NB15 dataset.
    
    Returns:
        train_data: Training features and labels
        test_data: Test features and labels
        feature_names: List of feature names
        scalers: Dict of preprocessing objects
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Features to exclude
    exclude_cols = ['id', 'attack_cat', 'label']
    
    # Categorical features (UNSW-NB15 known categoricals)
    categorical_features = ['proto', 'service', 'state']
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Extract labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Split feature columns into numeric / categorical
    cat_cols = [c for c in categorical_features if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Preprocess: impute + scale numeric; impute + one-hot categorical
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # Fit on train only (avoid leakage), transform train/test
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    # Convert to dense arrays for PyTorch dataset
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training labels - Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
    print(f"Test labels - Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
    
    # Feature names after preprocessing (best-effort)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = feature_cols

    return (X_train, y_train), (X_test, y_test), feature_names, {
        'preprocessor': preprocessor
    }


def simulate_concept_drift(X, y, drift_type='sudden', num_drifts=3, shuffle=True, random_state=42):
    """
    Simulate concept drift in the data.
    
    Args:
        X: Features
        y: Labels
        drift_type: 'sudden', 'gradual', or 'incremental'
        num_drifts: Number of drift points
        
    Returns:
        drift_datasets: List of (X, y) tuples representing different concepts
        drift_points: Indices where drifts occur
    """
    n_samples = len(X)

    # UNSW-NB15 is ordered by label in the raw CSV. If we split by index without
    # shuffling, concepts can become single-class, which collapses learning and
    # makes metrics meaningless. Shuffle once by default for fair concepts.
    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    # Stratified split so each concept has a similar class balance.
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Cannot create stratified concepts: one class is missing.")

    rng = np.random.default_rng(random_state)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    idx0_chunks = np.array_split(idx0, num_drifts)
    idx1_chunks = np.array_split(idx1, num_drifts)
    
    if drift_type == 'sudden':
        # Sudden drift: stratified chunks with distinct shifts
        datasets = []
        drift_points = []
        
        for i in range(num_drifts):
            concept_idx = np.concatenate([idx0_chunks[i], idx1_chunks[i]])
            rng.shuffle(concept_idx)
            X_chunk = X[concept_idx]
            y_chunk = y[concept_idx]
            
            # Apply distribution shift: scale features differently for each concept
            shift_factor = 1.0 + 0.3 * i  # Increasing shift
            noise_scale = 0.1 * (i + 1)  # Increasing noise
            
            X_shifted = X_chunk * shift_factor + np.random.normal(0, noise_scale, X_chunk.shape)
            datasets.append((X_shifted, y_chunk))
            drift_points.append(len(X_chunk))
        
        return datasets, np.cumsum(drift_points)[:-1]
    
    elif drift_type == 'gradual':
        # Gradual drift: Smooth transition within each stratified concept
        datasets = []
        drift_points = []
        
        for i in range(num_drifts):
            concept_idx = np.concatenate([idx0_chunks[i], idx1_chunks[i]])
            rng.shuffle(concept_idx)
            X_chunk = X[concept_idx]
            y_chunk = y[concept_idx]
            chunk_size = len(X_chunk)
            
            # Gradual shift: linearly interpolate between old and new concept
            start_shift = 1.0 + 0.3 * i
            end_shift = 1.0 + 0.3 * (i + 1)
            
            shifts = np.linspace(start_shift, end_shift, chunk_size)
            X_shifted = X_chunk * shifts[:, np.newaxis]
            
            datasets.append((X_shifted, y_chunk))
            drift_points.append(chunk_size)
        
        return datasets, np.cumsum(drift_points)[:-1]
    
    elif drift_type == 'incremental':
        # Incremental drift: Continuous changes over the stratified sequence
        order = np.concatenate([np.concatenate([idx0_chunks[i], idx1_chunks[i]]) for i in range(num_drifts)])
        X_ordered = X[order].copy()
        y_ordered = y[order]
        
        # Add cumulative drift
        for i in range(n_samples):
            drift_amount = (i / n_samples) * 0.5
            noise = np.random.normal(0, 0.05, X.shape[1])
            X_ordered[i] = X_ordered[i] * (1 + drift_amount) + noise
        
        # Split into chunks for evaluation
        drift_points = np.linspace(0, n_samples, num_drifts + 1, dtype=int)
        datasets = []
        
        for i in range(num_drifts):
            start_idx = drift_points[i]
            end_idx = drift_points[i + 1]
            datasets.append((X_ordered[start_idx:end_idx], y_ordered[start_idx:end_idx]))
        
        return datasets, drift_points[1:-1]
    
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")


def create_dataloaders(X, y, batch_size=64, shuffle=True):
    """Create PyTorch DataLoader from numpy arrays."""
    dataset = UNSWNB15Dataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def balance_dataset(X, y, method='undersample'):
    """
    Balance the dataset to handle class imbalance.
    
    Args:
        X: Features
        y: Labels
        method: 'undersample' or 'oversample'
        
    Returns:
        X_balanced, y_balanced
    """
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    
    if method == 'undersample':
        # Undersample majority class
        min_samples = min(len(normal_idx), len(attack_idx))
        normal_idx = np.random.choice(normal_idx, min_samples, replace=False)
        attack_idx = np.random.choice(attack_idx, min_samples, replace=False)
    
    elif method == 'oversample':
        # Oversample minority class
        max_samples = max(len(normal_idx), len(attack_idx))
        normal_idx = np.random.choice(normal_idx, max_samples, replace=True)
        attack_idx = np.random.choice(attack_idx, max_samples, replace=True)
    
    balanced_idx = np.concatenate([normal_idx, attack_idx])
    np.random.shuffle(balanced_idx)
    
    return X[balanced_idx], y[balanced_idx]
