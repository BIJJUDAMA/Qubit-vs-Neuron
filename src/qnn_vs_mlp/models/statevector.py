"""Differentiable two-qubit statevector backend (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _apply_ry(state: torch.Tensor, theta: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a real-valued RY rotation using pairwise amplitude updates.

    State-vector order is |q0 q1>: [|00>, |01>, |10>, |11>]. This optimized
    real-amplitude implementation avoids unnecessary complex matrix operations
    because the circuit uses only RY rotations and CNOT gates.
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    a0, a1, a2, a3 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    if wire == 0:  # pairs (|00>, |10>) and (|01>, |11>)
        o0 = c * a0 - s * a2
        o2 = s * a0 + c * a2
        o1 = c * a1 - s * a3
        o3 = s * a1 + c * a3
        return torch.stack([o0, o1, o2, o3], dim=1)
    # wire 1: pairs (|00>, |01>) and (|10>, |11>)
    o0 = c * a0 - s * a1
    o1 = s * a0 + c * a1
    o2 = c * a2 - s * a3
    o3 = s * a2 + c * a3
    return torch.stack([o0, o1, o2, o3], dim=1)


def _apply_cnot_01(state: torch.Tensor) -> torch.Tensor:
    # q0 is the control and q1 is the target: |10> <-> |11>.
    return state[:, [0, 1, 3, 2]]


class TorchStatevectorQuantumClassifier(nn.Module):
    """Two-qubit data-reuploading variational classifier.

    Each layer applies angle encoding RY(x_i) on both wires, a trainable RY
    rotation per wire, then optionally CNOT(q0 -> q1). The two measured features
    <Z0> and <Z1> feed a one-layer classical readout. The no-entanglement
    ablation keeps the same circuit parameters and readout but omits CNOT.
    """
    def __init__(self, depth: int = 2, entangle: bool = True):
        super().__init__()
        self.depth = depth
        self.entangle = entangle
        self.weights = nn.Parameter(0.15 * torch.randn(depth, 2))
        self.readout = nn.Linear(2, 1)

    def statevector(self, X: torch.Tensor) -> torch.Tensor:
        batch = X.shape[0]
        state = torch.zeros((batch, 4), dtype=X.dtype, device=X.device)
        state[:, 0] = 1.0
        for layer in range(self.depth):
            state = _apply_ry(state, X[:, 0], 0)
            state = _apply_ry(state, X[:, 1], 1)
            state = _apply_ry(state, self.weights[layer, 0], 0)
            state = _apply_ry(state, self.weights[layer, 1], 1)
            if self.entangle:
                state = _apply_cnot_01(state)
        return state

    def expectation_features(self, X: torch.Tensor) -> torch.Tensor:
        state = self.statevector(X)
        p = state.square()
        z0 = p[:, 0] + p[:, 1] - p[:, 2] - p[:, 3]
        z1 = p[:, 0] + p[:, 2] - p[:, 1] - p[:, 3]
        return torch.stack([z0, z1], dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.readout(self.expectation_features(X)).squeeze(-1)

    @torch.no_grad()
    def basis_probabilities(self, X: torch.Tensor) -> np.ndarray:
        return self.statevector(X).square().cpu().numpy()
