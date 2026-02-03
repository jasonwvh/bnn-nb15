#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESU optimizer for anomaly detection
Adapted from Bonnet et al. for continual learning with concept drift
"""

import torch
from torch import Tensor
from torch.nn import Module
import numpy as np


class MESU(object):
    """
    Metaplasticity from Synaptic Uncertainty optimizer.
    
    Adapted for anomaly detection with concept drift handling.
    """

    def __init__(self, model, args_dict):
        super().__init__()
        self.model = model
        self.mu_prior = args_dict.get('mu_prior', 0.0)
        self.sigma_prior = args_dict.get('sigma_prior', 0.1)
        self.N = args_dict.get('N', 100)  # Number of batches for memory
        self.c_sigma = args_dict.get('c_sigma', 0.001)
        self.c_mu = args_dict.get('c_mu', 0.001)
        self.second_order = args_dict.get('second_order', True)
        self.clamp_sigma = args_dict.get('clamp_sigma', [0.001, 1.0])
        self.clamp_mu = args_dict.get('clamp_mu', [0, 0])
        self.ratio_max = args_dict.get('ratio_max', 0.1)
        self.moment_sigma = args_dict.get('moment_sigma', 0.9)
        self.moment_mu = args_dict.get('moment_mu', 0.9)

        num_params = len(list(model.parameters())) 
        print(f'MESU initialized with {num_params} parameters.')
        
        self.grad_eff_sigma = {}
        self.grad_eff_mu = {}
        
        for name, param in model.named_parameters(recurse=True):
            if name.endswith('sigma'):
                self.grad_eff_sigma[name] = torch.zeros_like(param)
            if name.endswith('mu'):
                self.grad_eff_mu[name] = torch.zeros_like(param)
                
    def step(self):
        """Performs a single optimization step."""
        mesu_step(
            model=self.model,
            grad_eff_sigma=self.grad_eff_sigma,
            grad_eff_mu=self.grad_eff_mu,
            moment_sigma=self.moment_sigma,
            moment_mu=self.moment_mu,
            mu_prior=self.mu_prior,
            sigma_prior=self.sigma_prior,
            N=self.N,
            c_sigma=self.c_sigma,
            c_mu=self.c_mu,
            second_order=self.second_order,
            clamp_sigma=self.clamp_sigma,
            clamp_mu=self.clamp_mu,
            ratio_max=self.ratio_max,
        )
    
    def zero_grad(self):
        """Zero gradients of all parameters."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()


def mesu_step(model: Module, *, grad_eff_sigma: dict, grad_eff_mu: dict,
              moment_sigma: float, moment_mu: float, mu_prior: float,
              sigma_prior: float, N: int, c_sigma: float, c_mu: float,
              second_order: bool, clamp_sigma: list, clamp_mu: list,
              ratio_max: float):
    """MESU update rule."""
    
    for name, param in model.named_parameters(recurse=True):
        if name.endswith('sigma'):
            sigma = param
            variance = param.data ** 2
            grad_sigma = param.grad
            name_sigma = name
            
        if name.endswith('mu'):
            mu = param
            grad_mu = param.grad
            name_mu = name
            
            if grad_sigma is not None and grad_mu is not None:
                # Update effective gradients with momentum
                grad_eff_sigma[name_sigma].mul_(moment_sigma)
                grad_eff_sigma[name_sigma].add_((1 - moment_sigma) * grad_sigma)
                grad_eff_mu[name_mu].mul_(moment_mu)
                grad_eff_mu[name_mu].add_((1 - moment_mu) * grad_mu)
                
                # Compute denominator with second-order term
                denominator = 1 + second_order * sigma * grad_eff_sigma[name_sigma].abs()
                
                # Compute updates
                delta_sigma = -c_sigma * (
                    variance * grad_eff_sigma[name_sigma] + 
                    sigma.data * (variance - sigma_prior ** 2) / (N * (sigma_prior ** 2))
                ) / denominator
                
                delta_mu = -c_mu * (
                    variance * grad_eff_mu[name_mu] + 
                    variance * (mu.data - mu_prior) / (N * sigma_prior ** 2)
                ) / denominator
                
                # Clamp updates to prevent instability
                delta_sigma = torch.clamp(delta_sigma, -ratio_max * sigma.data, ratio_max * sigma.data)
                delta_mu = torch.clamp(delta_mu, -ratio_max * sigma.data, ratio_max * sigma.data)
                
                # Apply updates
                sigma.data.add_(delta_sigma)
                mu.data.add_(delta_mu)
                
                # Clamp parameters if specified
                if clamp_sigma[0] != 0:
                    sigma.data = torch.clamp(sigma.data, clamp_sigma[0], clamp_sigma[1])
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(mu.data, clamp_mu[0], clamp_mu[1])
            
            elif grad_sigma is None and grad_mu is not None:
                # Deterministic layer update
                mu.data.add_(-(variance * grad_mu))
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(mu.data, clamp_mu[0], clamp_mu[1])
