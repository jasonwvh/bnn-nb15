#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESU-based anomaly detection model for UNSW-NB15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mesu_layers import Linear_MetaBayes


class MESUAnomalyDetector(nn.Module):
    """
    Bayesian neural network for anomaly detection using MESU.
    
    Architecture designed for tabular network traffic data with ability
    to handle concept drift through Bayesian continual learning.
    """
    
    def __init__(self, args_dict):
        super(MESUAnomalyDetector, self).__init__()
        
        # Model hyperparameters
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)  # Binary: normal/attack
        
        # Bayesian parameters
        sigma_init = args_dict.get('sigma_init', 0.1)
        self.sigma_prior = args_dict.get('sigma_prior', 0.1)
        self.coeff_likeli = args_dict.get('coeff_likeli', 1.0)
        self.reduction = args_dict.get('reduction', 'mean')
        
        # Activation function
        activation = args_dict.get('activation', 'Relu')
        if activation == 'Relu':
            self.act = nn.ReLU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Hardtanh':
            self.act = nn.Hardtanh(min_val=-1, max_val=1)
        else:
            self.act = nn.ReLU()
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            Linear_MetaBayes(
                self.input_dim, 
                self.hidden_dims[0],
                sigma_init=sigma_init,
                sigma_prior=self.sigma_prior,
                bias=True
            )
        )
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(
                Linear_MetaBayes(
                    self.hidden_dims[i],
                    self.hidden_dims[i + 1],
                    sigma_init=sigma_init,
                    sigma_prior=self.sigma_prior,
                    bias=True
                )
            )
        
        # Output layer
        self.layers.append(
            Linear_MetaBayes(
                self.hidden_dims[-1],
                self.output_dim,
                sigma_init=sigma_init,
                sigma_prior=self.sigma_prior,
                bias=True
            )
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=args_dict.get('dropout', 0.3))
        
    def forward(self, x, samples=1):
        """
        Forward pass through the network.
        
        Args:
            x: Input features [batch_size, input_dim]
            samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Log probabilities [samples, batch_size, output_dim]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, samples)
            x = self.act(x)
            if i == len(self.layers) - 2:  # Apply dropout before output layer
                x = self.dropout(x)
        
        x = self.layers[-1](x, samples)
        pred = F.log_softmax(x, dim=2)
        
        return pred
    
    def loss(self, x, target, samples=1):
        """
        Compute negative log likelihood loss.
        
        The KL divergence term is handled by the MESU optimizer update rule.
        """
        outputs = self.forward(x, samples)
        # Average over Monte Carlo samples
        nll = F.nll_loss(outputs.mean(0), target, reduction=self.reduction)
        return nll * self.coeff_likeli
    
    def predict(self, x, samples=10):
        """
        Predict with uncertainty estimation.
        
        Returns:
            predictions: Class predictions
            uncertainties: Predictive uncertainty (std of probabilities)
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples)  # [samples, batch, classes]
            probs = torch.exp(log_probs)  # Convert to probabilities
            
            # Mean prediction across samples
            mean_probs = probs.mean(0)  # [batch, classes]
            predictions = mean_probs.argmax(dim=1)
            
            # Uncertainty: std of predicted probabilities
            uncertainties = probs.std(0).mean(1)  # [batch]
            
        return predictions, uncertainties
    
    def get_anomaly_score(self, x, samples=10):
        """
        Get anomaly score for each sample.
        
        Higher score = more likely to be anomalous.
        Returns both the probability of being an attack and the uncertainty.
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples)
            probs = torch.exp(log_probs)
            
            # Probability of attack (class 1)
            attack_prob = probs[:, :, 1].mean(0)
            
            # Epistemic uncertainty
            uncertainty = probs.std(0).mean(1)
            
        return attack_prob, uncertainty
