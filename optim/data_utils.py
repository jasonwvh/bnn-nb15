#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing and concept drift simulation for UNSW-NB15
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
    # Categorical features that need encoding
    categorical_features = ['proto', 'service', 'state']
    
    # Process categorical features
    label_encoders = {}
    for col in categorical_features:
        if col in train_df.columns:
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            le.fit(combined.fillna('missing'))
            
            train_df[col] = le.transform(train_df[col].fillna('missing'))
            test_df[col] = le.transform(test_df[col].fillna('missing'))
            label_encoders[col] = le
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Handle missing values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training labels - Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
    print(f"Test labels - Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
    
    return (X_train, y_train), (X_test, y_test), feature_cols, {
        'scaler': scaler,
        'label_encoders': label_encoders
    }


def simulate_concept_drift(X, y, drift_type='sudden', num_drifts=3):
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
    
    if drift_type == 'sudden':
        # Sudden drift: Split data into distinct chunks
        drift_points = np.linspace(0, n_samples, num_drifts + 1, dtype=int)
        datasets = []
        
        for i in range(num_drifts):
            start_idx = drift_points[i]
            end_idx = drift_points[i + 1]
            
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            
            # Apply distribution shift: scale features differently for each concept
            shift_factor = 1.0 + 0.3 * i  # Increasing shift
            noise_scale = 0.1 * (i + 1)  # Increasing noise
            
            X_shifted = X_chunk * shift_factor + np.random.normal(0, noise_scale, X_chunk.shape)
            
            datasets.append((X_shifted, y_chunk))
        
        return datasets, drift_points[1:-1]
    
    elif drift_type == 'gradual':
        # Gradual drift: Smooth transition between concepts
        drift_points = np.linspace(0, n_samples, num_drifts + 1, dtype=int)
        datasets = []
        
        for i in range(num_drifts):
            start_idx = drift_points[i]
            end_idx = drift_points[i + 1]
            chunk_size = end_idx - start_idx
            
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            
            # Gradual shift: linearly interpolate between old and new concept
            start_shift = 1.0 + 0.3 * i
            end_shift = 1.0 + 0.3 * (i + 1)
            
            shifts = np.linspace(start_shift, end_shift, chunk_size)
            X_shifted = X_chunk * shifts[:, np.newaxis]
            
            datasets.append((X_shifted, y_chunk))
        
        return datasets, drift_points[1:-1]
    
    elif drift_type == 'incremental':
        # Incremental drift: Small continuous changes
        X_shifted = X.copy()
        
        # Add cumulative drift
        for i in range(n_samples):
            drift_amount = (i / n_samples) * 0.5
            noise = np.random.normal(0, 0.05, X.shape[1])
            X_shifted[i] = X[i] * (1 + drift_amount) + noise
        
        # Split into chunks for evaluation
        drift_points = np.linspace(0, n_samples, num_drifts + 1, dtype=int)
        datasets = []
        
        for i in range(num_drifts):
            start_idx = drift_points[i]
            end_idx = drift_points[i + 1]
            datasets.append((X_shifted[start_idx:end_idx], y[start_idx:end_idx]))
        
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
