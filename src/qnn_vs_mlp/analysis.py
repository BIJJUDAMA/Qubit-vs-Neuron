"""Statistical analysis, finite-shot simulation, and paired comparisons."""
from __future__ import annotations

from typing import Dict, Any, List, Sequence
import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score, f1_score, log_loss
import torch
from torch import nn


def finite_shot_features(probabilities: np.ndarray, shots: int, seed: int) -> np.ndarray:
    """Sample computational-basis bitstrings and estimate <Z_q> for each wire.

    Basis ordering is standard computational basis |0...0> to |1...1>.
    Dynamically supports 2-qubit (4 basis states) and n-qubit (2^n states) systems.
    """
    rng = np.random.default_rng(seed)
    features = []
    n_basis = probabilities.shape[1]
    n_qubits = int(round(np.log2(n_basis)))

    # Precompute eigenvalue matrix: shape (n_qubits, n_basis)
    # Bit (k >> (n_qubits - 1 - q)) & 1: 0 -> +1, 1 -> -1
    eig_matrix = np.array([
        [1.0 - 2.0 * ((k >> (n_qubits - 1 - q)) & 1) for k in range(n_basis)]
        for q in range(n_qubits)
    ], dtype=np.float64)

    for p in probabilities:
        p = np.asarray(np.maximum(p, 0), dtype=np.float64)
        p = p / p.sum(dtype=np.float64)
        subtotal = float(p[:-1].sum(dtype=np.float64))
        if subtotal >= 1.0:
            p[:-1] *= (1.0 - 1e-12) / subtotal
            p[-1] = 1.0 - float(p[:-1].sum(dtype=np.float64))
        else:
            p[-1] = 1.0 - subtotal
        p = p / p.sum(dtype=np.float64)
        counts = rng.multinomial(shots, p)
        z_vals = [float(np.dot(counts, eig_matrix[q]) / shots) for q in range(n_qubits)]
        features.append(z_vals)
    return np.asarray(features, dtype=np.float32)


def eval_shot_readout(
    model: nn.Module, basis_probs: np.ndarray, y: torch.Tensor, shots: int, seed: int
) -> Dict[str, float]:
    features = finite_shot_features(basis_probs, shots, seed)
    features_t = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model.readout(features_t).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    pred = (probs >= 0.5).astype(int)
    yy = y.cpu().numpy().astype(int)
    return {
        "accuracy": float(accuracy_score(yy, pred)),
        "f1": float(f1_score(yy, pred, zero_division=0)),
        "log_loss": float(log_loss(yy, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def mean_std(values: Sequence[float] | np.ndarray) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0,
    }


def paired_stats(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> Dict[str, Any]:
    """Paired metric comparison; a-b and 95% t interval for mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return {
            "n": 0,
            "mean_difference": None,
            "ci95": [None, None],
            "paired_t_p": None,
            "wilcoxon_p": None,
        }
    md = float(np.mean(diff))
    if n > 1:
        se = stats.sem(diff)
        tcrit = stats.t.ppf(0.975, n - 1)
        ci = [float(md - tcrit * se), float(md + tcrit * se)]
        if np.allclose(diff, 0):
            # t-test is degenerate (zero variance); Wilcoxon provides no evidence
            # of a difference when every paired difference is zero.
            tp = None
            wp = 1.0
        else:
            ttest = stats.ttest_rel(a, b, nan_policy="omit")
            tp = float(ttest.pvalue)
            try:
                wp = float(
                    stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue
                )
            except ValueError:
                wp = 1.0
    else:
        ci = [md, md]
        tp = wp = None
    return {"n": n, "mean_difference": md, "ci95": ci, "paired_t_p": tp, "wilcoxon_p": wp}
