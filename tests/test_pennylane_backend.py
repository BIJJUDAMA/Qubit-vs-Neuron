"""PennyLane-specific tests. They skip cleanly when PennyLane is not installed."""
import importlib.util
import numpy as np
import pytest
import torch

if importlib.util.find_spec("pennylane") is None:
    pytest.skip("PennyLane is not installed in this environment", allow_module_level=True)

from main import PennyLaneQuantumClassifier, TorchStatevectorQuantumClassifier


def _copy_state(source, target):
    target.load_state_dict(source.state_dict())


@pytest.mark.parametrize("depth,entangle", [(1, True), (2, True), (2, False), (3, True)])
def test_pennylane_matches_reference_statevector(depth, entangle):
    torch.manual_seed(42)
    reference = TorchStatevectorQuantumClassifier(depth=depth, entangle=entangle)
    pennylane_model = PennyLaneQuantumClassifier(depth=depth, entangle=entangle)
    _copy_state(reference, pennylane_model)
    X = torch.tensor([[0.1, -0.2], [0.7, 1.2], [-1.1, 0.4]], dtype=torch.float32)
    with torch.no_grad():
        a = reference.expectation_features(X)
        b = pennylane_model.expectation_features(X)
        p = pennylane_model.basis_probabilities(X)
    assert a.shape == b.shape == (3, 2)
    assert np.allclose(a.numpy(), b.numpy(), atol=2e-5, rtol=2e-5)
    assert p.shape == (3, 4)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_pennylane_gradients_are_finite():
    model = PennyLaneQuantumClassifier(depth=2, entangle=True)
    X = torch.tensor([[0.2, -0.1], [0.5, 0.9], [-0.6, 0.3]], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X), y)
    loss.backward()
    assert model.weights.grad is not None
    assert torch.isfinite(model.weights.grad).all()
    assert model.readout.weight.grad is not None
    assert torch.isfinite(model.readout.weight.grad).all()
