"""Dataset loading, splitting, and scaling utilities for QNN vs MLP benchmarks.

Supports three distinct, domain-purposeful benchmark categories:
1. Quantum State & Phase Classification Datasets:
   - pennylane_tfim: Transverse-Field Ising Model 1D spin chain ground states (PennyLane Datasets).
     Features: 4 Pauli expectation observables (<Z0Z1>, <Z1Z2>, <Z2Z3>, <X_mean>).
     Target: magnetic_phase (0: Ferromagnetic, 1: Paramagnetic).
   - qiskit_phase_of_matter: Bond-alternating Heisenberg model ground-state phases (Qiskit ML).
     Features: 4 Heisenberg bond correlation & dimer order observables.
     Target: topological_phase (0: Trivial Dimerized, 1: Topological SPT).
2. High-Dimensional / Non-Linear Quantum Kernel Datasets:
   - havlicek_adhoc: Havlicek ZZ-feature map quantum kernel benchmark (Nature 2019, Qiskit ML).
     Features: 2 continuous phase angles x1, x2 in [0, 2pi].
     Target: quantum_kernel_class (0: Class A, 1: Class B).
3. Classical Small-Data & Low-Sample Benchmarks:
   - breast_cancer: Wisconsin Diagnostic clinical measurements (UCI / Scikit-Learn).
     Features: 4 direct clinical cell nucleus morphological features.
     Target: diagnosis (0: Malignant, 1: Benign).
   - mnist_digits: Binary downsampled handwritten digits 0 vs 1 (Scikit-Learn / Yann LeCun).
     Features: 4 morphological spatial image features (core, ring, spine, cross).
     Target: digit_label (0: Digit 0, 1: Digit 1).
"""
from __future__ import annotations

import warnings
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetResult(tuple):
    """Container for benchmark dataset arrays and column-level domain schema.

    Inherits from tuple to unpack seamlessly as (X, y, labels, description)
    for 100% backwards compatibility with existing pipeline callers, while
    exposing rich column schemas, domain meanings, and a pandas DataFrame.
    """

    def __new__(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        labels: List[str],
        description: str,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        column_info: Optional[Dict[str, Dict[str, Any]]] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        return super().__new__(cls, (X, y, labels, description))

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labels: List[str],
        description: str,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        column_info: Optional[Dict[str, Dict[str, Any]]] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        self.X = X
        self.y = y
        self.labels = labels
        self.description = description
        self.feature_names = feature_names or [f"dim_{i+1}" for i in range(X.shape[1])]
        self.target_name = target_name
        self.column_info = column_info or {}
        self.target_classes = {i: lbl for i, lbl in enumerate(labels)}

        if df is not None:
            self.df = df
        else:
            data_dict = {
                name: X[:, i] for i, name in enumerate(self.feature_names)
            }
            data_dict[self.target_name] = y
            self.df = pd.DataFrame(data_dict)


def _load_pennylane_tfim(
    n_samples: int = 100, seed: int = 2026
) -> DatasetResult:
    """Load PennyLane Transverse-Field Ising Model (TFIM) ground states.

    System: 4-spin 1D chain with open boundary conditions.
    Hamiltonian: H(h) = -J sum_{i=0}^2 Z_i Z_{i+1} - h sum_{i=0}^3 X_i (J = -1).
    Phases:
      - Ferromagnetic ordered phase: h < 1.0 (label 0)
      - Paramagnetic disordered phase: h > 1.0 (label 1)

    Features: 4 physical ground-state Pauli expectation observables:
      [corr_Z0Z1, corr_Z1Z2, corr_Z2Z3, mag_transverse_X]
    """
    import pennylane as qml

    try:
        datasets = qml.data.load("qspin", sysname="Ising", periodicity="open", lattice="chain", layout="1x4")
    except Exception:
        datasets = qml.data.load("qspin", sysname="TFIM", periodicity="open", lattice="chain", layout="1x4")

    ds = datasets[0]
    h_values = np.asarray(ds.parameters["h"], dtype=np.float64)
    ground_states = ds.ground_states

    # Pauli operators for 4-qubit chain
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    Z0Z1 = np.kron(np.kron(Z, Z), np.kron(I, I))
    Z1Z2 = np.kron(np.kron(I, Z), np.kron(Z, I))
    Z2Z3 = np.kron(np.kron(I, I), np.kron(Z, Z))
    X_mean = sum(
        np.kron(
            np.kron(X if i == 0 else I, X if i == 1 else I),
            np.kron(X if i == 2 else I, X if i == 3 else I),
        )
        for i in range(4)
    ) / 4.0

    features_list = []
    labels_list = []
    for i, (h, psi) in enumerate(zip(h_values, ground_states)):
        # Exclude immediate boundary point h=1.0 for well-defined phase margin
        if np.isclose(h, 1.0, atol=0.015):
            continue
        rho = np.outer(psi, np.conj(psi))
        exp_z0z1 = float(np.real(np.trace(rho @ Z0Z1)))
        exp_z1z2 = float(np.real(np.trace(rho @ Z1Z2)))
        exp_z2z3 = float(np.real(np.trace(rho @ Z2Z3)))
        exp_x = float(np.real(np.trace(rho @ X_mean)))
        features_list.append([exp_z0z1, exp_z1z2, exp_z2z3, exp_x])
        labels_list.append(0 if h < 1.0 else 1)

    X_arr = np.array(features_list, dtype=np.float32)
    y_arr = np.array(labels_list, dtype=np.int64)

    # Balance classes deterministically
    rng = np.random.default_rng(seed)
    idx0 = np.where(y_arr == 0)[0]
    idx1 = np.where(y_arr == 1)[0]
    n_per_class = min(len(idx0), len(idx1), n_samples // 2 if n_samples > 0 else 34)
    sel0 = rng.choice(idx0, size=n_per_class, replace=False)
    sel1 = rng.choice(idx1, size=n_per_class, replace=False)
    selected = rng.permutation(np.concatenate([sel0, sel1]))

    X_arr = X_arr[selected]
    y_arr = y_arr[selected]

    feature_names = ["corr_Z0Z1", "corr_Z1Z2", "corr_Z2Z3", "mag_transverse_X"]
    target_name = "magnetic_phase"
    labels = ["Ferromagnetic", "Paramagnetic"]
    description = (
        "PennyLane TFIM: Transverse-Field Ising Model 4-spin chain ground states "
        "(Ferromagnetic h<1.0 vs Paramagnetic h>1.0, 4 Pauli expectation features)"
    )

    column_info = {
        "corr_Z0Z1": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Longitudinal spin correlation <Z0 Z1> between sites 0 and 1",
            "purpose": "Measures nearest-neighbor magnetic order along the Z axis",
        },
        "corr_Z1Z2": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Longitudinal spin correlation <Z1 Z2> between sites 1 and 2",
            "purpose": "Measures central nearest-neighbor magnetic exchange order",
        },
        "corr_Z2Z3": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Longitudinal spin correlation <Z2 Z3> between sites 2 and 3",
            "purpose": "Measures nearest-neighbor magnetic order at the chain edge",
        },
        "mag_transverse_X": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean transverse field magnetization (1/4) sum_{i=0}^3 <X_i>",
            "purpose": "Measures quantum spin fluctuations driven by transverse field h",
        },
        "magnetic_phase": {
            "role": "Target",
            "dtype": "int64",
            "meaning": "Ground state magnetic phase (0: Ferromagnetic, 1: Paramagnetic)",
            "purpose": "Ground-truth phase label across quantum critical point h_c = 1.0",
        },
    }

    df = pd.DataFrame(
        {
            "corr_Z0Z1": X_arr[:, 0],
            "corr_Z1Z2": X_arr[:, 1],
            "corr_Z2Z3": X_arr[:, 2],
            "mag_transverse_X": X_arr[:, 3],
            "magnetic_phase": y_arr,
        }
    )

    return DatasetResult(
        X_arr, y_arr, labels, description,
        feature_names=feature_names,
        target_name=target_name,
        column_info=column_info,
        df=df,
    )


def _load_qiskit_phase_of_matter(
    n_samples: int = 120, seed: int = 2026
) -> DatasetResult:
    """Load Qiskit ML Phase of Matter dataset.

    Model: Bond-alternating XXX Heisenberg spin chain (n=4 qubits).
    Phases: Trivial (label 0) vs Symmetry-Protected Topological / SPT (label 1).
    Features: 4 physical ground-state Heisenberg bond correlation observables
      and dimerization order parameter:
      [bond_spin_corr_01, bond_spin_corr_12, bond_spin_corr_23, dimer_order_param]
    """
    from qiskit_machine_learning.datasets import phase_of_matter_data

    half = max(10, n_samples // 2)
    tr_f, tr_l, te_f, te_l = phase_of_matter_data(
        training_size=half,
        test_size=half,
        n=4,
        model="heisenberg",
        one_hot=False,
        seed=seed,
    )
    states = np.vstack([tr_f, te_f])
    phase_names = np.concatenate([tr_l, te_l])

    # Construct 4-qubit Heisenberg bond spin operators S_i . S_{i+1}
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron4(A, B, C, D):
        return np.kron(np.kron(A, B), np.kron(C, D))

    # S_i . S_j = (X_i X_j + Y_i Y_j + Z_i Z_j) / 4.0
    SS_01 = (kron4(X, X, I, I) + kron4(Y, Y, I, I) + kron4(Z, Z, I, I)) / 4.0
    SS_12 = (kron4(I, X, X, I) + kron4(I, Y, Y, I) + kron4(I, Z, Z, I)) / 4.0
    SS_23 = (kron4(I, I, X, X) + kron4(I, I, Y, Y) + kron4(I, I, Z, Z)) / 4.0
    # Dimerization topological order parameter
    dimer_op = (SS_01 - SS_12 + SS_23) / 2.0

    operators = [SS_01, SS_12, SS_23, dimer_op]
    features_list = []
    for psi in states:
        rho = np.outer(psi, np.conj(psi))
        obs_vals = [float(np.real(np.trace(rho @ op))) for op in operators]
        features_list.append(obs_vals)

    X_arr = np.array(features_list, dtype=np.float32)
    y_arr = np.array([1 if str(p).lower() == "topological" else 0 for p in phase_names], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_arr))
    X_arr = X_arr[perm]
    y_arr = y_arr[perm]

    feature_names = ["bond_spin_corr_01", "bond_spin_corr_12", "bond_spin_corr_23", "dimer_order_param"]
    target_name = "topological_phase"
    labels = ["Trivial", "Topological"]
    description = (
        "Qiskit Phase of Matter: 4-qubit Heisenberg bond-alternating spin chain "
        "(Trivial dimerized phase vs Topological SPT phase, 4 bond correlation observables)"
    )

    column_info = {
        "bond_spin_corr_01": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Heisenberg exchange correlation <S0 . S1> on bond 0-1",
            "purpose": "Measures intra-dimer singlet pairing strength on the first bond",
        },
        "bond_spin_corr_12": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Heisenberg exchange correlation <S1 . S2> on central bond 1-2",
            "purpose": "Detects inter-dimer coupling across the central lattice junction",
        },
        "bond_spin_corr_23": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Heisenberg exchange correlation <S2 . S3> on bond 2-3",
            "purpose": "Measures intra-dimer singlet pairing strength on the terminating bond",
        },
        "dimer_order_param": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Topological dimerization order parameter (1/2)(<S0.S1> - <S1.S2> + <S2.S3>)",
            "purpose": "Order parameter explicitly separating trivial and topological SPT ground states",
        },
        "topological_phase": {
            "role": "Target",
            "dtype": "int64",
            "meaning": "Topological quantum phase (0: Trivial Dimerized, 1: Topological SPT)",
            "purpose": "Ground-truth quantum many-body topological order label",
        },
    }

    df = pd.DataFrame(
        {
            "bond_spin_corr_01": X_arr[:, 0],
            "bond_spin_corr_12": X_arr[:, 1],
            "bond_spin_corr_23": X_arr[:, 2],
            "dimer_order_param": X_arr[:, 3],
            "topological_phase": y_arr,
        }
    )

    return DatasetResult(
        X_arr, y_arr, labels, description,
        feature_names=feature_names,
        target_name=target_name,
        column_info=column_info,
        df=df,
    )


def _load_havlicek_adhoc(
    n_samples: int = 160, seed: int = 2026
) -> DatasetResult:
    """Load Havlicek Quantum Feature Map Benchmark dataset (ad_hoc_data).

    Reference:
      Havlicek et al., "Supervised learning with quantum-enhanced feature spaces,"
      Nature 567, 209-212 (2019).
    Features: 2 continuous phase coordinates in [0, 2pi] with separation gap=0.3.
    """
    from qiskit_machine_learning.datasets import ad_hoc_data
    try:
        from qiskit_machine_learning.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    except Exception:
        pass

    n_per_class = max(10, n_samples // 4)
    tr_f, tr_l, te_f, te_l = ad_hoc_data(
        training_size=n_per_class,
        test_size=n_per_class,
        n=2,
        gap=0.3,
        one_hot=False,
    )
    X_arr = np.vstack([tr_f, te_f]).astype(np.float32)
    y_arr = np.concatenate([tr_l, te_l]).astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_arr))
    X_arr = X_arr[perm]
    y_arr = y_arr[perm]

    feature_names = ["feature_angle_x1", "feature_angle_x2"]
    target_name = "quantum_kernel_class"
    labels = ["Class A", "Class B"]
    description = (
        "Havlicek Quantum Kernel Benchmark: ZZFeatureMap non-linearly separable "
        "ad-hoc dataset (Nature 2019, 2 qubits, gap=0.3)"
    )

    column_info = {
        "feature_angle_x1": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "First coordinate angle x1 in [0, 2pi] (radians)",
            "purpose": "Inputs to single-qubit Z rotation and ZZ interaction feature map",
        },
        "feature_angle_x2": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Second coordinate angle x2 in [0, 2pi] (radians)",
            "purpose": "Inputs to single-qubit Z rotation and non-linear ZZ phase coupling",
        },
        "quantum_kernel_class": {
            "role": "Target",
            "dtype": "int64",
            "meaning": "Quantum feature kernel class (0: Class A, 1: Class B)",
            "purpose": "Kernel separation label generated via quantum state inner-product gap",
        },
    }

    df = pd.DataFrame(
        {
            "feature_angle_x1": X_arr[:, 0],
            "feature_angle_x2": X_arr[:, 1],
            "quantum_kernel_class": y_arr,
        }
    )

    return DatasetResult(
        X_arr, y_arr, labels, description,
        feature_names=feature_names,
        target_name=target_name,
        column_info=column_info,
        df=df,
    )


def _load_breast_cancer(
    n_samples: int = 200, seed: int = 2026
) -> DatasetResult:
    """Load Wisconsin Breast Cancer Diagnostic benchmark.

    Extracts the 4 primary clinical cell nucleus morphological features:
    [mean_radius, mean_texture, mean_perimeter, mean_concave_points]
    """
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer()
    feature_keys = ["mean radius", "mean texture", "mean perimeter", "mean concave points"]
    indices = [list(raw.feature_names).index(k) for k in feature_keys]

    X_sel = raw.data[:, indices].astype(np.float32)
    y_raw = raw.target.astype(np.int64)  # 0=malignant, 1=benign

    # Select balanced stratified subset
    rng = np.random.default_rng(seed)
    idx0 = np.where(y_raw == 0)[0]
    idx1 = np.where(y_raw == 1)[0]
    half = min(len(idx0), len(idx1), n_samples // 2 if n_samples > 0 else len(idx0))
    sel0 = rng.choice(idx0, size=half, replace=False)
    sel1 = rng.choice(idx1, size=half, replace=False)
    perm = rng.permutation(np.concatenate([sel0, sel1]))

    X_arr = X_sel[perm]
    y_arr = y_raw[perm]

    feature_names = ["mean_radius", "mean_texture", "mean_perimeter", "mean_concave_points"]
    target_name = "diagnosis"
    labels = ["Malignant", "Benign"]
    description = (
        "Breast Cancer Wisconsin: 4 clinical cell nucleus morphological features "
        "(UCI Diagnostic / Scikit-Learn: radius, texture, perimeter, concave points)"
    )

    column_info = {
        "mean_radius": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean of distances from cell nucleus center to points on perimeter (micrometers)",
            "purpose": "Primary indicator of cell nuclear enlargement characteristic of malignancy",
        },
        "mean_texture": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Standard deviation of gray-scale pixel values across the cell nucleus",
            "purpose": "Quantifies chromatin texture heterogeneity inside the cell nucleus",
        },
        "mean_perimeter": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean perimeter length of the cell nucleus contour (micrometers)",
            "purpose": "Captures overall morphological enlargement and boundary irregularity",
        },
        "mean_concave_points": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean number of concave indentations along the cell nucleus contour",
            "purpose": "Clinically proven biomarker for aggressive nuclear invagination and cancer invasion",
        },
        "diagnosis": {
            "role": "Target",
            "dtype": "int64",
            "meaning": "Diagnostic biopsy outcome (0: Malignant, 1: Benign)",
            "purpose": "Ground-truth histopathological cancer diagnosis",
        },
    }

    df = pd.DataFrame(
        {
            "mean_radius": X_arr[:, 0],
            "mean_texture": X_arr[:, 1],
            "mean_perimeter": X_arr[:, 2],
            "mean_concave_points": X_arr[:, 3],
            "diagnosis": y_arr,
        }
    )

    return DatasetResult(
        X_arr, y_arr, labels, description,
        feature_names=feature_names,
        target_name=target_name,
        column_info=column_info,
        df=df,
    )


def _load_mnist_digits(
    n_samples: int = 200, seed: int = 2026, use_openml: bool = False
) -> DatasetResult:
    """Load binary downsampled / subsetted MNIST digits (0 vs 1).

    Extracts 4 morphological spatial image features from 8x8 grayscale handwritten digits:
    1. center_core_intensity: 4x4 central core intensity (rows 2-5, cols 2-5)
    2. outer_perimeter_intensity: Outer boundary ring intensity
    3. vertical_spine_intensity: Central vertical stroke intensity (cols 3-4)
    4. horizontal_cross_intensity: Central horizontal stroke intensity (rows 3-4)
    """
    X_sub: Optional[np.ndarray] = None
    y_sub: Optional[np.ndarray] = None

    if use_openml:
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            mask = (mnist.target == "0") | (mnist.target == "1")
            X_sub = mnist.data[mask]
            y_sub = (mnist.target[mask] == "1").astype(np.int64)
        except Exception:
            pass

    if X_sub is None:
        with warnings.catch_warnings():
            # Catch upstream sklearn DeprecationWarning on numpy 2.5
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn\.datasets\._base")
            from sklearn.datasets import load_digits
            digits = load_digits()
        mask = (digits.target == 0) | (digits.target == 1)
        X_sub = digits.data[mask]
        y_sub = digits.target[mask].astype(np.int64)

    # Reshape 64 flattened pixels into 8x8 images
    imgs = X_sub.reshape(-1, 8, 8)

    # Extract 4 purposeful morphological spatial features
    # Feature 1: Central core intensity (rows 2..6, cols 2..6)
    center = imgs[:, 2:6, 2:6].mean(axis=(1, 2))
    # Feature 2: Outer perimeter ring intensity
    outer = (imgs.sum(axis=(1, 2)) - imgs[:, 2:6, 2:6].sum(axis=(1, 2))) / (64.0 - 16.0)
    # Feature 3: Vertical spine stroke intensity (cols 3..5)
    spine = imgs[:, :, 3:5].mean(axis=(1, 2))
    # Feature 4: Horizontal mid-line cross intensity (rows 3..5)
    cross = imgs[:, 3:5, :].mean(axis=(1, 2))

    X_morph = np.column_stack([center, outer, spine, cross]).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx0 = np.where(y_sub == 0)[0]
    idx1 = np.where(y_sub == 1)[0]
    half = min(len(idx0), len(idx1), n_samples // 2 if n_samples > 0 else 100)
    sel0 = rng.choice(idx0, size=half, replace=False)
    sel1 = rng.choice(idx1, size=half, replace=False)
    perm = rng.permutation(np.concatenate([sel0, sel1]))

    X_arr = X_morph[perm]
    y_arr = y_sub[perm]

    feature_names = ["center_core_intensity", "outer_perimeter_intensity", "vertical_spine_intensity", "horizontal_cross_intensity"]
    target_name = "digit_label"
    labels = ["Digit 0", "Digit 1"]
    description = (
        f"MNIST Digits (0 vs 1): 4 purposeful morphological spatial features "
        f"(central core, outer ring, vertical spine, horizontal cross; n={len(y_arr)})"
    )

    column_info = {
        "center_core_intensity": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean pixel intensity in 4x4 central core (rows 2-5, cols 2-5)",
            "purpose": "Distinguishes hollow ring of digit 0 from solid center of digit 1",
        },
        "outer_perimeter_intensity": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean pixel intensity along outer boundary ring (border pixels)",
            "purpose": "Measures annular boundary stroke density characteristic of digit 0",
        },
        "vertical_spine_intensity": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean pixel intensity along central vertical spine (cols 3-4)",
            "purpose": "Directly detects the primary vertical stroke of handwritten digit 1",
        },
        "horizontal_cross_intensity": {
            "role": "Feature",
            "dtype": "float32",
            "meaning": "Mean pixel intensity along horizontal mid-line (rows 3-4)",
            "purpose": "Measures horizontal stroke presence across the middle of the image",
        },
        "digit_label": {
            "role": "Target",
            "dtype": "int64",
            "meaning": "Handwritten digit identity (0: Digit 0, 1: Digit 1)",
            "purpose": "Ground-truth classification target for binary digit recognition",
        },
    }

    df = pd.DataFrame(
        {
            "center_core_intensity": X_arr[:, 0],
            "outer_perimeter_intensity": X_arr[:, 1],
            "vertical_spine_intensity": X_arr[:, 2],
            "horizontal_cross_intensity": X_arr[:, 3],
            "digit_label": y_arr,
        }
    )

    return DatasetResult(
        X_arr, y_arr, labels, description,
        feature_names=feature_names,
        target_name=target_name,
        column_info=column_info,
        df=df,
    )


def load_dataset(
    name: str, moons_samples: int = 200, moons_noise: float = 0.05, data_seed: int = 2026
) -> DatasetResult:
    """Load specified benchmark dataset.

    Active, purposeful datasets:
      1. pennylane_tfim: PennyLane Transverse-Field Ising Model ground states (4 qubits)
      2. qiskit_phase_of_matter: Qiskit Phase of Matter Heisenberg chain (4 qubits)
      3. havlicek_adhoc: Havlicek Quantum Kernel Benchmark (2 qubits)
      4. breast_cancer: Wisconsin Diagnostic clinical features (4 qubits)
      5. mnist_digits: Binary MNIST digits 0 vs 1 morphological features (4 qubits)
    """
    clean_name = name.lower().strip()

    if clean_name in ("pennylane_tfim", "tfim", "pennylane_ising", "ising"):
        return _load_pennylane_tfim(n_samples=moons_samples, seed=data_seed)
    elif clean_name in ("qiskit_phase_of_matter", "phase_of_matter", "phase_of_matter_data"):
        return _load_qiskit_phase_of_matter(n_samples=moons_samples, seed=data_seed)
    elif clean_name in ("havlicek_adhoc", "ad_hoc", "ad_hoc_data", "havlicek"):
        return _load_havlicek_adhoc(n_samples=moons_samples, seed=data_seed)
    elif clean_name in ("breast_cancer", "cancer", "wisconsin_breast_cancer"):
        return _load_breast_cancer(n_samples=moons_samples, seed=data_seed)
    elif clean_name in ("mnist_digits", "mnist", "mnist_subset"):
        return _load_mnist_digits(n_samples=moons_samples, seed=data_seed)
    elif clean_name in ("concentric_circles", "circles"):
        raise ValueError(
            f"Dataset '{name}' has been deprecated and removed in favor of real, purposeful datasets. "
            "Supported benchmarks: ['pennylane_tfim', 'qiskit_phase_of_matter', 'havlicek_adhoc', "
            "'breast_cancer', 'mnist_digits']"
        )
    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. Supported datasets: "
            "['pennylane_tfim', 'qiskit_phase_of_matter', 'havlicek_adhoc', "
            "'breast_cancer', 'mnist_digits']"
        )


def split_and_scale(
    X: np.ndarray, y: np.ndarray, seed: int, test_size: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split and standard scaling with trigonometric angle bounding."""
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    # The QNN maps standardized continuous features to rotation angles in [-pi, pi]
    angle_train = (np.pi * np.clip(X_train / 2.5, -1.0, 1.0)).astype(np.float32)
    angle_test = (np.pi * np.clip(X_test / 2.5, -1.0, 1.0)).astype(np.float32)
    return train_idx, test_idx, scaler, X_train, X_test, angle_train, angle_test
