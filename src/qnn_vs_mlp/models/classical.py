"""Classical baseline models (MLP and CompactMLP)."""
from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Conventional reference MLP (193 trainable parameters for 2D inputs)."""
    def __init__(self, in_features: int = 2):
        super().__init__()
        self.in_features = in_features
        self.network = nn.Sequential(
            nn.Linear(in_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, X):
        return self.network(X).squeeze(-1)


class CompactMLPClassifier(nn.Module):
    """Small MLP capacity control (9 parameters for 2D inputs, 13 for 4D)."""
    def __init__(self, in_features: int = 2):
        super().__init__()
        self.in_features = in_features
        self.network = nn.Sequential(nn.Linear(in_features, 2), nn.ReLU(), nn.Linear(2, 1))

    def forward(self, X):
        return self.network(X).squeeze(-1)
