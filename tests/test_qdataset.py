"""Tests for the QDataSet 2-qubit Hamiltonian dynamics generator."""
import numpy as np
import pytest

from qnn_vs_mlp.qdataset import generate_qdataset_hamiltonian


def test_qdataset_hamiltonian_shapes_and_balance():
    X, y, labels, desc = generate_qdataset_hamiltonian(n_samples=100, seed=42)
    assert X.shape == (100, 4)
    assert y.shape == (100,)
    assert set(np.unique(y)) == {0, 1}
    assert np.bincount(y)[0] == 50
    assert np.bincount(y)[1] == 50
    assert len(labels) == 2
    assert "QDataSet" in desc
    assert np.all(np.isfinite(X))


def test_qdataset_hamiltonian_deterministic_seed():
    X1, y1, _, _ = generate_qdataset_hamiltonian(n_samples=50, seed=123)
    X2, y2, _, _ = generate_qdataset_hamiltonian(n_samples=50, seed=123)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)
