"""Model architectures and factory helpers."""
from __future__ import annotations

import torch
from torch import nn

from .classical import MLPClassifier, CompactMLPClassifier
from .statevector import (
    TorchStatevectorQuantumClassifier,
    _apply_cnot_01,
    _apply_ry,
)
from .pennylane import PennyLaneQuantumClassifier


def build_qnn(backend: str, depth: int, entangle: bool, n_qubits: int = 2):
    if backend == "pennylane":
        return PennyLaneQuantumClassifier(depth=depth, entangle=entangle, n_qubits=n_qubits)
    elif backend == "torch":
        return TorchStatevectorQuantumClassifier(depth=depth, entangle=entangle, n_qubits=n_qubits)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'pennylane' or 'torch'.")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


__all__ = [
    "build_qnn",
    "count_parameters",
    "MLPClassifier",
    "CompactMLPClassifier",
    "TorchStatevectorQuantumClassifier",
    "PennyLaneQuantumClassifier",
    "_apply_cnot_01",
    "_apply_ry",
]
