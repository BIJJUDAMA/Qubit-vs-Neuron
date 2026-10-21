"""Comprehensive unit tests for the 5 quantum and classical benchmark datasets and EDA."""
import numpy as np
import pytest

from qnn_vs_mlp.data import load_dataset, split_and_scale
from qnn_vs_mlp.eda import compute_dataset_eda


DATASET_NAMES = [
    "pennylane_tfim",
    "qiskit_phase_of_matter",
    "havlicek_adhoc",
    "breast_cancer",
    "mnist_digits",
]


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_dataset_loading_and_properties(name):
    ds = load_dataset(name, moons_samples=60, data_seed=42)
    # Unpacks as 4 elements for backwards compatibility
    X, y, labels, desc = ds
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert len(X) == len(y)
    assert len(X) > 0
    assert set(np.unique(y)).issubset({0, 1})
    assert len(labels) == 2
    assert len(desc) > 0
    assert np.all(np.isfinite(X))
    # Number of qubits / features is 2 or 4
    assert X.shape[1] in (2, 4)

    # Verify purposeful column metadata & pandas DataFrame
    assert hasattr(ds, "df")
    assert hasattr(ds, "feature_names")
    assert hasattr(ds, "target_name")
    assert len(ds.feature_names) == X.shape[1]
    assert ds.target_name in ds.df.columns
    for fname in ds.feature_names:
        assert fname in ds.df.columns
        assert fname in ds.column_info
        assert ds.column_info[fname]["role"] == "Feature"
        assert len(ds.column_info[fname]["meaning"]) > 0


def test_concentric_circles_removed():
    """Verify that concentric_circles has been completely removed."""
    with pytest.raises(ValueError, match="deprecated and removed"):
        load_dataset("concentric_circles")


def test_dataset_deterministic_seed():
    """Verify deterministic dataset loading with random seeds."""
    X1, y1, _, _ = load_dataset("havlicek_adhoc", moons_samples=60, data_seed=123)
    X2, y2, _, _ = load_dataset("havlicek_adhoc", moons_samples=60, data_seed=123)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_dataset_split_and_scale(name):
    X, y, _, _ = load_dataset(name, moons_samples=60, data_seed=42)
    train_idx, test_idx, scaler, Xtr, Xte, Atr, Ate = split_and_scale(X, y, seed=42, test_size=0.25)
    assert len(train_idx) + len(test_idx) == len(X)
    assert Xtr.shape[1] == X.shape[1]
    assert Xte.shape[1] == X.shape[1]
    assert Atr.shape == Xtr.shape
    assert Ate.shape == Xte.shape
    # Check that angles are strictly in [-pi, pi]
    assert np.all(Atr >= -np.pi - 1e-5)
    assert np.all(Atr <= np.pi + 1e-5)


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_compute_dataset_eda(name):
    ds = load_dataset(name, moons_samples=60, data_seed=42)
    X, y, labels, desc = ds
    eda = compute_dataset_eda(
        name,
        X,
        y,
        labels,
        desc,
        feature_names=ds.feature_names,
        target_name=ds.target_name,
        column_info=ds.column_info,
    )
    assert eda["name"] == name
    assert eda["n_samples"] == len(X)
    assert eda["n_features"] == X.shape[1]
    assert eda["n_qubits"] == X.shape[1]
    assert 0.0 <= eda["balance_ratio"] <= 1.0
    assert eda["centroid_distance"] >= 0.0
    assert len(eda["feature_statistics"]) == X.shape[1]

    # Verify column schemas and explicit target metadata
    assert "columns" in eda
    assert len(eda["columns"]) == X.shape[1]
    for col in eda["columns"]:
        assert col["role"] == "Feature"
        assert col["dtype"] == "float32"
        assert col["non_null_count"] == len(X)
        assert len(col["meaning"]) > 0

    assert "target_column" in eda
    assert eda["target_column"]["role"] == "Target"
    assert eda["target_column"]["dtype"] == "int64"
    assert eda["target_column"]["column_name"] == ds.target_name
    assert len(eda["target_column"]["class_mapping"]) == 2
