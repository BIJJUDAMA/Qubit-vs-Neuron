"""Dataset loading, splitting, and scaling utilities."""
from __future__ import annotations

from typing import Tuple, List, Any
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(
    name: str, moons_samples: int, moons_noise: float, data_seed: int = 2026
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Return fixed data. Iris is Setosa-versus-Versicolor, petal length/width."""
    if name == "iris":
        data = load_iris()
        mask = data.target > 0
        X = data.data[mask][:, [2, 3]].astype(np.float32)
        y = (data.target[mask] - 1).astype(np.int64)  # versicolor=0, virginica=1
        labels = ["Versicolor", "Virginica"]
        description = "Iris versicolor vs virginica; petal length and petal width (overlapping non-trivial task)"
    elif name == "two_moons":
        X, y = make_moons(n_samples=moons_samples, noise=moons_noise,
                          random_state=data_seed)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        labels = ["Moon 0", "Moon 1"]
        description = f"Two Moons; n={moons_samples}, noise={moons_noise}"
    elif name == "concentric_circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=moons_samples, factor=0.5, noise=0.08,
                            random_state=data_seed)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        labels = ["Inner", "Outer"]
        description = f"Concentric Circles; n={moons_samples}, factor=0.5, noise=0.08"
    elif name == "periodic_parity":
        rng = np.random.default_rng(data_seed)
        X = rng.uniform(-np.pi, np.pi, size=(moons_samples, 4)).astype(np.float32)
        # y = sign(cos(x1 + x2) * cos(x3 + x4)) mapped to binary labels 0 and 1
        val = np.cos(X[:, 0] + X[:, 1]) * np.cos(X[:, 2] + X[:, 3])
        y = (val >= 0.0).astype(np.int64)
        labels = ["Parity -", "Parity +"]
        description = "4-qubit Periodic Parity: y = sign(cos(x1+x2)*cos(x3+x4))"
    elif name == "qdataset_hamiltonian":
        from .qdataset import generate_qdataset_hamiltonian
        X, y, labels, description = generate_qdataset_hamiltonian(
            n_samples=moons_samples, seed=data_seed
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, labels, description


def split_and_scale(
    X: np.ndarray, y: np.ndarray, seed: int, test_size: float
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    # The QNN uses the standardized features as rotation angles, clipped to
    # [-pi, pi]. The scaler is fitted only on training data.
    angle_train = (np.pi * np.clip(X_train / 2.5, -1.0, 1.0)).astype(np.float32)
    angle_test = (np.pi * np.clip(X_test / 2.5, -1.0, 1.0)).astype(np.float32)
    return train_idx, test_idx, scaler, X_train, X_test, angle_train, angle_test
