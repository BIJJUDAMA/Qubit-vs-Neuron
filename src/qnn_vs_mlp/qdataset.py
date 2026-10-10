"""QDataSet 2-Qubit Quantum Dynamics Generator.

Adapted from the QDataSet protocol:
    Perrier, E., Youssry, A. & Ferrie, C.
    "QDataSet, quantum datasets for machine learning."
    Sci Data 9, 582 (2022). https://doi.org/10.1038/s41597-022-01639-1
    GitHub: https://github.com/eperrier/QDataSet

This module implements the physical time-dependent Hamiltonian dynamics of a
two-qubit system evolving under single-qubit drift and control fields with versus
without a bipartite interaction term (H_int = g * X \otimes X).
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import scipy.linalg


def generate_qdataset_hamiltonian(
    n_samples: int = 500, seed: int = 2026
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Generate 2-qubit Hamiltonian dynamics classification benchmark.

    Classes:
      0: Separable Local Dynamics (no XX interaction, g_int = 0).
      1: Entangled Interaction Dynamics (g_int in [0.8, 3.0]).

    Features:
      Four Pauli expectation values evaluated at evolution time T=1.0:
      [<X \otimes I>, <I \otimes X>, <X \otimes X>, <Z \otimes Z>].
    """
    rng = np.random.default_rng(seed)

    # 2-qubit Pauli basis operators
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    XI = np.kron(X, I)
    IX = np.kron(I, X)
    XX = np.kron(X, X)
    ZZ = np.kron(Z, Z)
    ZI = np.kron(Z, I)
    IZ = np.kron(I, Z)

    # Ground state initial condition |00>
    psi0 = np.array([1, 0, 0, 0], dtype=complex)

    X_data = []
    y_data = []

    half = n_samples // 2
    for i in range(n_samples):
        is_entangled = (i >= half)
        # Random physical drift frequencies and control pulse amplitudes
        omega1 = float(rng.uniform(0.5, 3.0))
        omega2 = float(rng.uniform(0.5, 3.0))
        c1 = float(rng.uniform(-1.5, 1.5))
        c2 = float(rng.uniform(-1.5, 1.5))

        # Drift H0 + local controls H_c
        H = (omega1 * ZI + omega2 * IZ) + c1 * XI + c2 * IX
        if is_entangled:
            g_int = float(rng.uniform(0.8, 3.0))
            H += g_int * XX
            y = 1
        else:
            y = 0

        # Coherent time evolution U = exp(-i * H * T) for T = 1.0
        U = scipy.linalg.expm(-1j * H * 1.0)
        psi_t = U @ psi0
        rho = np.outer(psi_t, psi_t.conj())

        # Exact Pauli expectation values
        exp_XI = float(np.real(np.trace(rho @ XI)))
        exp_IX = float(np.real(np.trace(rho @ IX)))
        exp_XX = float(np.real(np.trace(rho @ XX)))
        exp_ZZ = float(np.real(np.trace(rho @ ZZ)))

        X_data.append([exp_XI, exp_IX, exp_XX, exp_ZZ])
        y_data.append(y)

    X_arr = np.array(X_data, dtype=np.float32)
    y_arr = np.array(y_data, dtype=np.int64)

    # Deterministic shuffle
    perm = rng.permutation(n_samples)
    X_arr = X_arr[perm]
    y_arr = y_arr[perm]

    labels = ["Separable", "Entangled"]
    description = (
        "QDataSet 2-Qubit Dynamics: Separable local fields vs. "
        "entangling XX Hamiltonian (Perrier et al., 2022)"
    )
    return X_arr, y_arr, labels, description
