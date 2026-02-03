#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline MLP model for anomaly detection comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mesu_layers import Linear_MetaBayes
import numpy as np

class BaselineMLP(nn.Module):
    """
    Standard MLP for anomaly detection (baseline comparison).
    """
    
    def __init__(self, args_dict):
        super(BaselineMLP, self).__init__()
        
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)
        
        activation = args_dict.get('activation', 'Relu')
        if activation == 'Relu':
            self.act = nn.ReLU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.act)
            layers.append(nn.Dropout(args_dict.get('dropout', 0.3)))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass."""
        logits = self.network(x)
        return F.log_softmax(logits, dim=1)
    
    def loss(self, x, target):
        """Compute loss."""
        outputs = self.forward(x)
        return F.nll_loss(outputs, target)
    
    def predict(self, x):
        """Predict class labels."""
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)
            predictions = log_probs.argmax(dim=1)
        return predictions


class DeterministicBayesianModel(nn.Module):
    """
    Bayesian model with standard optimizer (no MESU).
    Tests if metaplasticity is necessary or if Bayesian framework alone helps.
    """
    def __init__(self, args_dict):
        super().__init__()
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)
        sigma_init = args_dict.get('sigma_init', 0.1)
        sigma_prior = args_dict.get('sigma_prior', 0.1)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args_dict.get('dropout', 0.3))
        
        # Same Bayesian layers as MESU
        self.layers = nn.ModuleList()
        self.layers.append(Linear_MetaBayes(self.input_dim, self.hidden_dims[0],
                                           sigma_init=sigma_init, sigma_prior=sigma_prior))
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(Linear_MetaBayes(self.hidden_dims[i], self.hidden_dims[i+1],
                                               sigma_init=sigma_init, sigma_prior=sigma_prior))
        self.layers.append(Linear_MetaBayes(self.hidden_dims[-1], self.output_dim,
                                           sigma_init=sigma_init, sigma_prior=sigma_prior))
    
    def forward(self, x, samples=1):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, samples)
            x = self.act(x)
            if i == len(self.layers) - 2:
                x = self.dropout(x)
        x = self.layers[-1](x, samples)
        return F.log_softmax(x, dim=2)
    
    def loss(self, x, target, samples=1):
        outputs = self.forward(x, samples)
        return F.nll_loss(outputs.mean(0), target)
    
    def predict(self, x, samples=10):
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples)
            probs = torch.exp(log_probs)
            mean_probs = probs.mean(0)
            predictions = mean_probs.argmax(dim=1)
            uncertainties = probs.std(0).mean(1)
        return predictions, uncertainties


class EWCModel(nn.Module):
    """
    Elastic Weight Consolidation baseline.
    Standard continual learning approach - deterministic.
    """
    def __init__(self, args_dict):
        super().__init__()
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args_dict.get('dropout', 0.3))
        
        # Standard deterministic layers
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.act)
            layers.append(nn.Dropout(args_dict.get('dropout', 0.3)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)
        
        # EWC parameters
        self.fisher = {}
        self.optimal_params = {}
        self.ewc_lambda = args_dict.get('ewc_lambda', 1000)
        
    def forward(self, x):
        return F.log_softmax(self.network(x), dim=1)
    
    def loss(self, x, target):
        nll = F.nll_loss(self.forward(x), target)
        
        # Add EWC penalty
        ewc_loss = 0
        if len(self.fisher) > 0:
            for name, param in self.named_parameters():
                if name in self.fisher:
                    ewc_loss += (self.fisher[name] * 
                                (param - self.optimal_params[name])**2).sum()
        
        return nll + self.ewc_lambda * ewc_loss
    
    def compute_fisher(self, dataloader, device):
        """Compute Fisher Information Matrix after training on a concept."""
        self.eval()
        fisher = {name: torch.zeros_like(param) 
                 for name, param in self.named_parameters()}
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            self.zero_grad()
            log_probs = self.forward(batch_x)
            loss = F.nll_loss(log_probs, batch_y)
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Average over batches
        for name in fisher:
            fisher[name] /= len(dataloader)
        
        # Store Fisher and optimal params
        self.fisher = fisher
        self.optimal_params = {name: param.clone().detach() 
                              for name, param in self.named_parameters()}
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


class ExperienceReplay(nn.Module):
    """
    Experience Replay baseline - deterministic.
    Stores samples from old concepts and retrains on mixed data.
    """
    def __init__(self, args_dict):
        super().__init__()
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)
        
        self.act = nn.ReLU()
        
        # Standard deterministic layers
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.act)
            layers.append(nn.Dropout(args_dict.get('dropout', 0.3)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)
        
        # Replay buffer
        self.buffer_size = args_dict.get('buffer_size', 1000)
        self.replay_buffer_x = []
        self.replay_buffer_y = []
        
    def forward(self, x):
        return F.log_softmax(self.network(x), dim=1)
    
    def loss(self, x, target):
        return F.nll_loss(self.forward(x), target)
    
    def add_to_buffer(self, X, y):
        """Add samples to replay buffer."""
        n_samples = min(self.buffer_size // 3, len(X))  # Reserve space for multiple concepts
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        self.replay_buffer_x.append(X[indices])
        self.replay_buffer_y.append(y[indices])
        
        # Limit total buffer size
        total_size = sum(len(x) for x in self.replay_buffer_x)
        while total_size > self.buffer_size and len(self.replay_buffer_x) > 1:
            self.replay_buffer_x.pop(0)
            self.replay_buffer_y.pop(0)
            total_size = sum(len(x) for x in self.replay_buffer_x)
    
    def get_replay_data(self):
        """Get mixed replay data."""
        if not self.replay_buffer_x:
            return None, None
        X = np.vstack(self.replay_buffer_x)
        y = np.concatenate(self.replay_buffer_y)
        return X, y
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)
