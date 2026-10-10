"""PennyLane-based two-qubit data-reuploading classifier."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PennyLaneQuantumClassifier(nn.Module):
    """Data-reuploading classifier implemented with PennyLane.

    The QNode is batch-broadcasted: each RY data angle varies by example, while
    shared variational angles are applied to all examples. Analytic, differentiable
    default.qubit execution uses simulator backpropagation, not a hardware gradient.
    """
    def __init__(self, depth: int = 2, entangle: bool = True, n_qubits: int = 2):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as exc:
            raise RuntimeError(
                "PennyLane backend requested but PennyLane is not installed. Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc
        self.depth = depth
        self.entangle = entangle
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(0.15 * torch.randn(depth, n_qubits))
        self.readout = nn.Linear(n_qubits, 1)
        dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        dev_probs = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            # inputs has shape [batch, n_qubits]; PennyLane broadcasts these rotations.
            for layer in range(depth):
                for q in range(n_qubits):
                    qml.RY(inputs[..., q], wires=q)
                    qml.RY(weights[layer, q], wires=q)
                if entangle:
                    if n_qubits == 2:
                        qml.CNOT(wires=[0, 1])  # control q0 -> target q1
                    else:
                        for q in range(n_qubits):
                            qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return tuple(qml.expval(qml.PauliZ(q)) for q in range(n_qubits))

        @qml.qnode(dev_probs, interface="torch", diff_method="backprop")
        def prob_qnode(inputs, weights):
            for layer in range(depth):
                for q in range(n_qubits):
                    qml.RY(inputs[..., q], wires=q)
                    qml.RY(weights[layer, q], wires=q)
                if entangle:
                    if n_qubits == 2:
                        qml.CNOT(wires=[0, 1])
                    else:
                        for q in range(n_qubits):
                            qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return qml.probs(wires=list(range(n_qubits)))

        self._qnode = qnode
        self._prob_qnode = prob_qnode

    def expectation_features(self, X: torch.Tensor) -> torch.Tensor:
        res = self._qnode(X, self.weights)
        if isinstance(res, (tuple, list)):
            features = torch.stack(res, dim=-1)
        else:
            features = res
        return features.to(dtype=X.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.readout(self.expectation_features(X)).squeeze(-1)

    @torch.no_grad()
    def basis_probabilities(self, X: torch.Tensor) -> np.ndarray:
        probabilities = self._prob_qnode(X, self.weights).detach().cpu().numpy()
        probabilities = np.asarray(probabilities, dtype=np.float64)
        n_basis = 2 ** self.n_qubits
        if probabilities.ndim == 2 and probabilities.shape == (n_basis, len(X)):
            probabilities = probabilities.T
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        return probabilities
