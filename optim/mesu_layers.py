#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESU layers for anomaly detection on tabular data
Adapted from Bonnet et al. for network intrusion detection
"""

import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.init as init
import numpy as np


class Gaussian_MetaBayes(object):
    """Gaussian distribution with reparameterization trick for sampling."""
    
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self, samples=1):
        if samples == 0:
            return self.mu.unsqueeze(0)
        else:
            sigma = self.sigma.unsqueeze(0).repeat(samples, *([1]*len(self.mu.shape)))
            epsilon = torch.empty_like(sigma).normal_()
            mu = self.mu.unsqueeze(0).repeat(samples, *([1]*len(self.mu.shape)))
            return mu + sigma * epsilon


class Linear_MetaBayes(Module):
    """Bayesian linear layer with Gaussian weights for MESU optimizer."""
    
    __constants__ = ['in_features', 'out_features']
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, sigma_init=None, sigma_prior=0.1,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_MetaBayes, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define weight parameters (sigma before mu for optimizer compatibility)
        self.weight_sigma = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_mu = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bound = math.sqrt(2/in_features)
        if sigma_init is None:
            # Paper-derived init with a floor to avoid vanishing updates on tabular data.
            self.sigma_init = max(0.1, 0.5 * math.sqrt(1 / out_features))
        else:
            self.sigma_init = sigma_init
        self.sigma_prior = sigma_prior
        self.weight = Gaussian_MetaBayes(self.weight_mu, self.weight_sigma)
        
        # Define bias if applicable
        if bias:
            self.bias_sigma = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_mu = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias = Gaussian_MetaBayes(self.bias_mu, self.bias_sigma)
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        init.uniform_(self.weight_mu, -self.bound, self.bound)
        init.constant_(self.weight_sigma, self.sigma_init)

        if self.bias is not None:
            init.uniform_(self.bias_mu, -self.bound, self.bound)
            init.constant_(self.bias_sigma, self.sigma_init)

    def forward(self, x: Tensor, samples: int) -> Tensor:
        """Forward pass using sampled weights and biases."""
        samples_dim = np.maximum(samples, 1)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            x = x.repeat(samples_dim, 1, 1)
        
        W = self.weight.sample(samples)
        
        if self.bias:
            B = self.bias.sample(samples)
            return torch.einsum('soi, sbi -> sbo', W, x) + B[:, None]
        else:
            return torch.einsum('soi, sbi -> sbo', W, x)
    
    def extra_repr(self) -> str:
        """Representation for debugging."""
        return 'in_features={}, out_features={}, sigma_init={}, sigma_prior={}, bias={}'.format(
            self.in_features, self.out_features, self.sigma_init, self.sigma_prior, self.bias is not None)
