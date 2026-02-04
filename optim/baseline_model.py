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
    
    def loss(self, x, target, class_weights=None):
        """Compute loss."""
        outputs = self.forward(x)
        return F.nll_loss(outputs, target, weight=class_weights)
    
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
    
    def loss(self, x, target, samples=1, class_weights=None):
        outputs = self.forward(x, samples)
        return F.nll_loss(outputs.mean(0), target, weight=class_weights)
    
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
    
    def loss(self, x, target, class_weights=None):
        nll = F.nll_loss(self.forward(x), target, weight=class_weights)
        
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
    
    def loss(self, x, target, class_weights=None):
        return F.nll_loss(self.forward(x), target, weight=class_weights)
    
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


class VCLBayesianModel(nn.Module):
    """
    Variational Continual Learning (VCL) baseline.
    Bayesian model that updates its prior to the previous posterior per concept.
    """
    def __init__(self, args_dict):
        super().__init__()
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)
        self.kl_weight = args_dict.get('kl_weight', 1.0)
        self.train_samples = args_dict.get('train_samples', 5)
        self.eval_samples = args_dict.get('eval_samples', 10)
        sigma_init = args_dict.get('sigma_init', 0.1)
        sigma_prior = args_dict.get('sigma_prior', 0.1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args_dict.get('dropout', 0.3))

        self.layers = nn.ModuleList()
        self.layers.append(Linear_MetaBayes(self.input_dim, self.hidden_dims[0],
                                            sigma_init=sigma_init, sigma_prior=sigma_prior))
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(Linear_MetaBayes(self.hidden_dims[i], self.hidden_dims[i + 1],
                                                sigma_init=sigma_init, sigma_prior=sigma_prior))
        self.layers.append(Linear_MetaBayes(self.hidden_dims[-1], self.output_dim,
                                            sigma_init=sigma_init, sigma_prior=sigma_prior))

        self._register_priors()

    def _register_priors(self):
        """Initialize prior buffers from the current posterior."""
        for i, layer in enumerate(self.layers):
            self.register_buffer(f"w_mu_prior_{i}", layer.weight_mu.detach().clone())
            self.register_buffer(f"w_sigma_prior_{i}", layer.weight_sigma.detach().clone())
            if layer.bias is not None:
                self.register_buffer(f"b_mu_prior_{i}", layer.bias_mu.detach().clone())
                self.register_buffer(f"b_sigma_prior_{i}", layer.bias_sigma.detach().clone())

    @staticmethod
    def _kl_gaussian(mu_q, sigma_q, mu_p, sigma_p):
        sigma_q = torch.clamp(sigma_q, min=1e-6)
        sigma_p = torch.clamp(sigma_p, min=1e-6)
        term = torch.log(sigma_p / sigma_q)
        term += (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p.pow(2))
        return term - 0.5

    def kl_divergence(self):
        """KL(q || p) between current posterior and stored prior."""
        kl = 0.0
        for i, layer in enumerate(self.layers):
            w_mu_prior = getattr(self, f"w_mu_prior_{i}")
            w_sigma_prior = getattr(self, f"w_sigma_prior_{i}")
            kl += self._kl_gaussian(layer.weight_mu, layer.weight_sigma,
                                    w_mu_prior, w_sigma_prior).sum()
            if layer.bias is not None:
                b_mu_prior = getattr(self, f"b_mu_prior_{i}")
                b_sigma_prior = getattr(self, f"b_sigma_prior_{i}")
                kl += self._kl_gaussian(layer.bias_mu, layer.bias_sigma,
                                        b_mu_prior, b_sigma_prior).sum()
        return kl

    def update_prior(self):
        """Set prior to current posterior (call after finishing a concept)."""
        for i, layer in enumerate(self.layers):
            getattr(self, f"w_mu_prior_{i}").copy_(layer.weight_mu.detach())
            getattr(self, f"w_sigma_prior_{i}").copy_(layer.weight_sigma.detach())
            if layer.bias is not None:
                getattr(self, f"b_mu_prior_{i}").copy_(layer.bias_mu.detach())
                getattr(self, f"b_sigma_prior_{i}").copy_(layer.bias_sigma.detach())

    def forward(self, x, samples=1):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, samples)
            x = self.act(x)
            if i == len(self.layers) - 2:
                x = self.dropout(x)
        x = self.layers[-1](x, samples)
        return F.log_softmax(x, dim=2)

    def loss(self, x, target, class_weights=None):
        outputs = self.forward(x, samples=self.train_samples)
        nll = F.nll_loss(outputs.mean(0), target, weight=class_weights)
        kl = self.kl_divergence()
        kl = kl / max(1, x.size(0))
        return nll + (self.kl_weight * kl)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples=self.eval_samples)
            probs = torch.exp(log_probs)
            mean_probs = probs.mean(0)
            predictions = mean_probs.argmax(dim=1)
            uncertainties = probs.std(0).mean(1)
        return predictions, uncertainties

class UncertaintyVCLBayesianModel(nn.Module):
    """
    VCL with uncertainty-guided regularization.
    
    Key ideas:
    1. Epistemic uncertainty: Model uncertainty about parameters
       - High epistemic → Parameter not important → Allow changes
       - Low epistemic → Parameter well-determined → Protect it
    
    2. Aleatoric uncertainty: Data noise (irreducible)
       - High aleatoric → Noisy region → Don't overtrust gradients
       - Low aleatoric → Clean region → Trust gradients more
    
    3. Adaptive KL weight: Scale regularization per parameter based on uncertainty
    """
    
    def __init__(self, args_dict):
        super().__init__()
        self.input_dim = args_dict.get('input_dim', 42)
        self.hidden_dims = args_dict.get('hidden_dims', [128, 64, 32])
        self.output_dim = args_dict.get('output_dim', 2)
        self.base_kl_weight = args_dict.get('kl_weight', 1.0)
        self.train_samples = args_dict.get('train_samples', 5)
        self.eval_samples = args_dict.get('eval_samples', 10)
        sigma_init = args_dict.get('sigma_init', 0.01)
        sigma_prior = args_dict.get('sigma_prior', 0.1)
        
        # Uncertainty-guided hyperparameters
        self.uncertainty_mode = args_dict.get('uncertainty_mode', 'epistemic')  # 'epistemic', 'aleatoric', 'both'
        self.epistemic_threshold = args_dict.get('epistemic_threshold', 0.1)
        self.adaptive_kl = args_dict.get('adaptive_kl', True)
        self.uncertainty_ema = args_dict.get('uncertainty_ema', 0.9)  # Exponential moving average

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args_dict.get('dropout', 0.3))

        # Build network
        from mesu_layers import Linear_MetaBayes
        self.layers = nn.ModuleList()
        self.layers.append(Linear_MetaBayes(self.input_dim, self.hidden_dims[0],
                                            sigma_init=sigma_init, sigma_prior=sigma_prior))
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(Linear_MetaBayes(self.hidden_dims[i], self.hidden_dims[i + 1],
                                                sigma_init=sigma_init, sigma_prior=sigma_prior))
        self.layers.append(Linear_MetaBayes(self.hidden_dims[-1], self.output_dim,
                                            sigma_init=sigma_init, sigma_prior=sigma_prior))

        self._register_priors()
        self._register_importance()
        
    def _register_priors(self):
        """Initialize prior buffers from current posterior."""
        for i, layer in enumerate(self.layers):
            self.register_buffer(f"w_mu_prior_{i}", layer.weight_mu.detach().clone())
            self.register_buffer(f"w_sigma_prior_{i}", layer.weight_sigma.detach().clone())
            if layer.bias is not None:
                self.register_buffer(f"b_mu_prior_{i}", layer.bias_mu.detach().clone())
                self.register_buffer(f"b_sigma_prior_{i}", layer.bias_sigma.detach().clone())
    
    def _register_importance(self):
        """Initialize parameter importance tracking (like Fisher but uncertainty-based)."""
        for i, layer in enumerate(self.layers):
            # Track epistemic uncertainty history
            self.register_buffer(f"w_epistemic_{i}", torch.zeros_like(layer.weight_mu))
            if layer.bias is not None:
                self.register_buffer(f"b_epistemic_{i}", torch.zeros_like(layer.bias_mu))
            
            # Track gradient variance (aleatoric proxy)
            self.register_buffer(f"w_grad_var_{i}", torch.zeros_like(layer.weight_mu))
            if layer.bias is not None:
                self.register_buffer(f"b_grad_var_{i}", torch.zeros_like(layer.bias_mu))

    @staticmethod
    def _kl_gaussian(mu_q, sigma_q, mu_p, sigma_p):
        """KL divergence between two Gaussians."""
        sigma_q = torch.clamp(sigma_q, min=1e-6)
        sigma_p = torch.clamp(sigma_p, min=1e-6)
        term = torch.log(sigma_p / sigma_q)
        term += (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p.pow(2))
        return term - 0.5

    def compute_epistemic_uncertainty(self, x, samples=10):
        """
        Compute epistemic uncertainty: variance in predictions across weight samples.
        High epistemic = model unsure about what function to use.
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples=samples)  # [samples, batch, classes]
            probs = torch.exp(log_probs)
            
            # Epistemic: variance across samples (parameter uncertainty)
            epistemic = probs.var(dim=0).mean(dim=1)  # [batch]
            
        return epistemic
    
    def compute_aleatoric_uncertainty(self, x, samples=10):
        """
        Compute aleatoric uncertainty: expected entropy (data noise).
        High aleatoric = inherently noisy/ambiguous data.
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples=samples)  # [samples, batch, classes]
            probs = torch.exp(log_probs)
            
            # Aleatoric: average entropy (irreducible uncertainty)
            entropy = -(probs * log_probs).sum(dim=2)  # [samples, batch]
            aleatoric = entropy.mean(dim=0)  # [batch]
            
        return aleatoric
    
    def update_importance_from_uncertainty(self, dataloader, device):
        """
        Update parameter importance based on uncertainty measurements.
        
        Key insight: Parameters that produce low epistemic uncertainty are important
        (well-determined by data) and should be protected from forgetting.
        """
        self.eval()
        
        # Accumulate epistemic uncertainty per parameter
        epistemic_maps = {i: [] for i in range(len(self.layers))}
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                
                # Forward with multiple samples to get epistemic uncertainty
                log_probs = self.forward(batch_x, samples=self.train_samples)
                probs = torch.exp(log_probs)
                
                # Epistemic uncertainty per sample
                epistemic_per_sample = probs.var(dim=0).mean(dim=1)  # [batch]
                
                # Propagate epistemic uncertainty back to parameters
                # Low epistemic → parameter is important → high importance score
                importance = 1.0 / (epistemic_per_sample.mean() + 1e-6)
                
                # Store (simplified: use output epistemic as proxy for all layers)
                for i in range(len(self.layers)):
                    epistemic_maps[i].append(importance)
        
        # Update importance buffers with exponential moving average
        for i, layer in enumerate(self.layers):
            avg_importance = torch.tensor(epistemic_maps[i]).mean()
            
            # Low epistemic → high importance → strong protection
            w_epistemic = getattr(self, f"w_epistemic_{i}")
            w_epistemic.mul_(self.uncertainty_ema).add_(
                (1 - self.uncertainty_ema) * avg_importance
            )
            
            if layer.bias is not None:
                b_epistemic = getattr(self, f"b_epistemic_{i}")
                b_epistemic.mul_(self.uncertainty_ema).add_(
                    (1 - self.uncertainty_ema) * avg_importance
                )
    
    def update_gradient_variance(self):
        """
        Track gradient variance during training (aleatoric proxy).
        High gradient variance → noisy/ambiguous region → reduce trust in gradients.
        """
        for i, layer in enumerate(self.layers):
            if layer.weight_mu.grad is not None:
                grad_var = layer.weight_mu.grad.var()
                w_grad_var = getattr(self, f"w_grad_var_{i}")
                w_grad_var.mul_(self.uncertainty_ema).add_(
                    (1 - self.uncertainty_ema) * grad_var
                )
            
            if layer.bias is not None and layer.bias_mu.grad is not None:
                grad_var = layer.bias_mu.grad.var()
                b_grad_var = getattr(self, f"b_grad_var_{i}")
                b_grad_var.mul_(self.uncertainty_ema).add_(
                    (1 - self.uncertainty_ema) * grad_var
                )

    def uncertainty_weighted_kl_divergence(self):
        """
        Compute KL divergence with uncertainty-based weighting.
        
        Strategy:
        - Parameters with low epistemic uncertainty are important → high KL weight
        - Parameters with high aleatoric-related gradient variance → lower KL weight
        """
        total_kl = 0.0
        
        for i, layer in enumerate(self.layers):
            # Get priors and epistemic importance
            w_mu_prior = getattr(self, f"w_mu_prior_{i}")
            w_sigma_prior = getattr(self, f"w_sigma_prior_{i}")
            w_epistemic = getattr(self, f"w_epistemic_{i}")
            w_grad_var = getattr(self, f"w_grad_var_{i}")
            
            # Base KL divergence
            kl_w = self._kl_gaussian(
                layer.weight_mu, layer.weight_sigma,
                w_mu_prior, w_sigma_prior
            )
            
            if self.adaptive_kl:
                # Adaptive weighting based on uncertainty mode
                if self.uncertainty_mode == 'epistemic':
                    # Low epistemic → important → high weight
                    weight = torch.clamp(w_epistemic, min=0.1, max=10.0)
                    
                elif self.uncertainty_mode == 'aleatoric':
                    # High gradient variance → noisy → lower weight
                    weight = 1.0 / (1.0 + w_grad_var)
                    
                elif self.uncertainty_mode == 'both':
                    # Combine: protect low-epistemic, discount high-variance
                    epistemic_weight = torch.clamp(w_epistemic, min=0.1, max=10.0)
                    aleatoric_weight = 1.0 / (1.0 + w_grad_var)
                    weight = epistemic_weight * aleatoric_weight
                else:
                    weight = 1.0
                
                kl_w = kl_w * weight
            
            total_kl += kl_w.sum()
            
            # Same for bias
            if layer.bias is not None:
                b_mu_prior = getattr(self, f"b_mu_prior_{i}")
                b_sigma_prior = getattr(self, f"b_sigma_prior_{i}")
                b_epistemic = getattr(self, f"b_epistemic_{i}")
                b_grad_var = getattr(self, f"b_grad_var_{i}")
                
                kl_b = self._kl_gaussian(
                    layer.bias_mu, layer.bias_sigma,
                    b_mu_prior, b_sigma_prior
                )
                
                if self.adaptive_kl:
                    if self.uncertainty_mode == 'epistemic':
                        weight = torch.clamp(b_epistemic, min=0.1, max=10.0)
                    elif self.uncertainty_mode == 'aleatoric':
                        weight = 1.0 / (1.0 + b_grad_var)
                    elif self.uncertainty_mode == 'both':
                        weight = torch.clamp(b_epistemic, min=0.1, max=10.0) * \
                                (1.0 / (1.0 + b_grad_var))
                    else:
                        weight = 1.0
                    
                    kl_b = kl_b * weight
                
                total_kl += kl_b.sum()
        
        return total_kl

    def update_prior(self, dataloader=None, device=None):
        """
        Set prior to current posterior after finishing a concept.
        Optionally update importance from uncertainty.
        """
        # Update prior distribution
        for i, layer in enumerate(self.layers):
            getattr(self, f"w_mu_prior_{i}").copy_(layer.weight_mu.detach())
            getattr(self, f"w_sigma_prior_{i}").copy_(layer.weight_sigma.detach())
            if layer.bias is not None:
                getattr(self, f"b_mu_prior_{i}").copy_(layer.bias_mu.detach())
                getattr(self, f"b_sigma_prior_{i}").copy_(layer.bias_sigma.detach())
        
        # Update importance based on epistemic uncertainty
        if dataloader is not None and device is not None:
            self.update_importance_from_uncertainty(dataloader, device)

    def forward(self, x, samples=1):
        """Forward pass with Monte Carlo sampling."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, samples)
            x = self.act(x)
            if i == len(self.layers) - 2:
                x = self.dropout(x)
        x = self.layers[-1](x, samples)
        return F.log_softmax(x, dim=2)

    def loss(self, x, target, class_weights=None):
        """
        Loss with uncertainty-weighted KL divergence.
        """
        # Negative log likelihood
        outputs = self.forward(x, samples=self.train_samples)
        nll = F.nll_loss(outputs.mean(0), target, weight=class_weights)
        
        # Uncertainty-weighted KL divergence
        kl = self.uncertainty_weighted_kl_divergence()
        kl_normalized = kl / max(1, x.size(0))  # Normalize by batch size
        
        return nll + (self.base_kl_weight * kl_normalized)

    def predict(self, x):
        """Predict with uncertainty quantification."""
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, samples=self.eval_samples)
            probs = torch.exp(log_probs)
            
            # Predictions
            mean_probs = probs.mean(0)
            predictions = mean_probs.argmax(dim=1)
            
            # Epistemic uncertainty (variance across samples)
            epistemic = probs.var(dim=0).mean(dim=1)
            
            # Aleatoric uncertainty (average entropy)
            entropy = -(probs * log_probs).sum(dim=2)
            aleatoric = entropy.mean(dim=0)
            
            # Total uncertainty
            total_uncertainty = epistemic + aleatoric
            
        return predictions, {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total_uncertainty
        }
    
    def get_parameter_importance_stats(self):
        """Get statistics on parameter importance for debugging."""
        stats = {}
        for i in range(len(self.layers)):
            w_epistemic = getattr(self, f"w_epistemic_{i}")
            w_grad_var = getattr(self, f"w_grad_var_{i}")
            
            stats[f'layer_{i}'] = {
                'epistemic_mean': w_epistemic.mean().item(),
                'epistemic_std': w_epistemic.std().item(),
                'grad_var_mean': w_grad_var.mean().item(),
                'grad_var_std': w_grad_var.std().item()
            }
        
        return stats

# class UncertaintyVCLBayesianModel(VCLBayesianModel):
#     """
#     VCL variant that uses predictive uncertainty to mildly guide the KL weight.
#     When uncertain (std across MC samples is high), apply slightly more regularization.
#     When confident, apply slightly less regularization.
#     """
#     def __init__(self, args_dict):
#         super().__init__(args_dict)
#         # Small scaling factor to keep uncertainty influence mild
#         self.uncertainty_scale = args_dict.get('uncertainty_scale', 0.1)

#     def loss(self, x, target, class_weights=None):
#         outputs = self.forward(x, samples=self.train_samples)
#         nll = F.nll_loss(outputs.mean(0), target, weight=class_weights)

#         # Compute predictive uncertainty from MC samples
#         probs = torch.exp(outputs)  # Shape: [samples, batch_size, num_classes]
#         # Uncertainty: std of predicted probabilities across samples, averaged over classes
#         uncertainty = probs.std(0).mean(dim=1)  # Shape: [batch_size]
        
#         # Normalize to roughly [0, 1] by clamping
#         uncertainty_normalized = torch.clamp(uncertainty / 0.5, 0.0, 1.0)
        
#         # Mild adjustment: higher uncertainty -> slightly higher KL penalty
#         # The scale factor (0.1 by default) keeps this effect conservative
#         kl_adjustment = 1.0 + self.uncertainty_scale * uncertainty_normalized.mean()

#         kl = self.kl_divergence()
#         kl = kl / max(1, x.size(0))
#         return nll + (self.kl_weight * kl_adjustment * kl)
