"""
Autoencoder and Energy Wrapper for SHAP Gradient Explainer Feature Selection
"""

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ReconstructionEnergyWrapper(nn.Module):
    """
    Computes total reconstruction error energy:
    E(x) = sum_i (x_i - dec(enc(x))_i)^2
    Used by SHAP GradientExplainer to compute exact feature gradients.
    """
    def __init__(self, autoencoder_model):
        super().__init__()
        self.autoencoder = autoencoder_model

    def forward(self, x):
        recon = self.autoencoder(x)
        return torch.sum((recon - x) ** 2, dim=1, keepdim=True)
