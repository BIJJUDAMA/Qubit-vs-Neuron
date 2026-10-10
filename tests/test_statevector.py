import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from main import TorchStatevectorQuantumClassifier, _apply_cnot_01


def test_cnot_control_direction():
    state = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # |10>
    out = _apply_cnot_01(state)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))  # -> |11>


def test_probability_normalization_and_finite_values():
    X = torch.tensor([[0.1, -0.2], [1.2, 0.8], [-2.0, 0.5]], dtype=torch.float32)
    for depth in (1, 2, 3):
        for entangle in (True, False):
            model = TorchStatevectorQuantumClassifier(depth=depth, entangle=entangle)
            probs = model.basis_probabilities(X)
            assert probs.shape == (3, 4)
            assert np.all(probs >= -1e-7)
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            out = model(X)
            assert out.shape == (3,)
            assert torch.isfinite(out).all()


def test_gradient_flows_through_quantum_circuit():
    X = torch.tensor([[0.2, -0.3], [0.8, 1.2], [-1.4, 0.5]], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 0.0])
    model = TorchStatevectorQuantumClassifier(depth=2, entangle=True)
    before = model.weights.detach().clone()
    loss = torch.nn.BCEWithLogitsLoss()(model(X), y)
    loss.backward()
    assert model.weights.grad is not None
    assert torch.isfinite(model.weights.grad).all()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    optimizer.step()
    assert not torch.allclose(before, model.weights.detach())
