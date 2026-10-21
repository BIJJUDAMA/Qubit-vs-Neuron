"""Differentiable multi-qubit statevector backend (PyTorch).

Provides an independent PyTorch statevector implementation matching PennyLane default.qubit
for n-qubit data-reuploading variational circuits.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _apply_ry_tensor(state: torch.Tensor, theta: torch.Tensor, wire: int, n_qubits: int) -> torch.Tensor:
    """Apply real-valued RY rotation to target wire on tensor of shape [batch, 2, ..., 2]."""
    if theta.ndim == 1:
        c = torch.cos(theta / 2).view(-1, *([1] * (n_qubits - 1)))
        s = torch.sin(theta / 2).view(-1, *([1] * (n_qubits - 1)))
    else:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)

    idx0 = [slice(None)] * (n_qubits + 1)
    idx1 = [slice(None)] * (n_qubits + 1)
    idx0[wire + 1] = 0
    idx1[wire + 1] = 1

    a0 = state[tuple(idx0)]
    a1 = state[tuple(idx1)]

    new_state = torch.empty_like(state)
    new_state[tuple(idx0)] = c * a0 - s * a1
    new_state[tuple(idx1)] = s * a0 + c * a1
    return new_state


def _apply_cnot_tensor(state: torch.Tensor, q_ctrl: int, q_tgt: int, n_qubits: int) -> torch.Tensor:
    """Apply CNOT between control and target wires on state [batch, 2, ..., 2]."""
    idx_ctrl1_tgt0 = [slice(None)] * (n_qubits + 1)
    idx_ctrl1_tgt1 = [slice(None)] * (n_qubits + 1)
    idx_ctrl1_tgt0[q_ctrl + 1] = 1
    idx_ctrl1_tgt0[q_tgt + 1] = 0
    idx_ctrl1_tgt1[q_ctrl + 1] = 1
    idx_ctrl1_tgt1[q_tgt + 1] = 1

    res = state.clone()
    res[tuple(idx_ctrl1_tgt0)] = state[tuple(idx_ctrl1_tgt1)]
    res[tuple(idx_ctrl1_tgt1)] = state[tuple(idx_ctrl1_tgt0)]
    return res


def _apply_cnot_01(state: torch.Tensor) -> torch.Tensor:
    """Legacy 2-qubit CNOT(0 -> 1) on flat statevector [batch, 4]."""
    return state[:, [0, 1, 3, 2]]


def _apply_ry(state: torch.Tensor, theta: torch.Tensor, wire: int) -> torch.Tensor:
    """Legacy 2-qubit RY on flat statevector [batch, 4]."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    a0, a1, a2, a3 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    if wire == 0:
        o0 = c * a0 - s * a2
        o2 = s * a0 + c * a2
        o1 = c * a1 - s * a3
        o3 = s * a1 + c * a3
        return torch.stack([o0, o1, o2, o3], dim=1)
    o0 = c * a0 - s * a1
    o1 = s * a0 + c * a1
    o2 = c * a2 - s * a3
    o3 = s * a2 + c * a3
    return torch.stack([o0, o1, o2, o3], dim=1)


class TorchStatevectorQuantumClassifier(nn.Module):
    """N-qubit data-reuploading variational classifier implemented in pure PyTorch."""

    def __init__(self, depth: int = 2, entangle: bool = True, n_qubits: int = 2):
        super().__init__()
        self.depth = depth
        self.entangle = entangle
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(0.15 * torch.randn(depth, n_qubits))
        self.readout = nn.Linear(n_qubits, 1)

    def state_tensor(self, X: torch.Tensor) -> torch.Tensor:
        batch = X.shape[0]
        n = self.n_qubits
        state = torch.zeros((batch, *([2] * n)), dtype=X.dtype, device=X.device)
        # Initialize ground state |0...0>
        init_idx = [slice(None)] + [0] * n
        state[tuple(init_idx)] = 1.0

        for layer in range(self.depth):
            for q in range(n):
                state = _apply_ry_tensor(state, X[:, q], q, n)
                state = _apply_ry_tensor(state, self.weights[layer, q], q, n)
            if self.entangle:
                if n == 2:
                    state = _apply_cnot_tensor(state, 0, 1, n)
                else:
                    for q in range(n):
                        state = _apply_cnot_tensor(state, q, (q + 1) % n, n)
        return state

    def expectation_features(self, X: torch.Tensor) -> torch.Tensor:
        state = self.state_tensor(X)
        batch = X.shape[0]
        n = self.n_qubits
        exp_vals = []
        for q in range(n):
            idx0 = [slice(None)] * (n + 1)
            idx1 = [slice(None)] * (n + 1)
            idx0[q + 1] = 0
            idx1[q + 1] = 1
            z0 = state[tuple(idx0)].reshape(batch, -1).square().sum(dim=1)
            z1 = state[tuple(idx1)].reshape(batch, -1).square().sum(dim=1)
            exp_vals.append(z0 - z1)
        return torch.stack(exp_vals, dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.readout(self.expectation_features(X)).squeeze(-1)

    @torch.no_grad()
    def basis_probabilities(self, X: torch.Tensor) -> np.ndarray:
        st = self.state_tensor(X)
        probs = st.reshape(X.shape[0], -1).square().cpu().numpy()
        return np.asarray(probs, dtype=np.float64)
