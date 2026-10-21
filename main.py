#!/usr/bin/env python3
"""Reproducible QNN-vs-MLP case study across three benchmark categories:

1. Quantum State & Phase Classification Datasets:
   - PennyLane Transverse-Field Ising Model (TFIM)
   - IBM / Qiskit Phase of Matter (Heisenberg spin chain)
2. High-Dimensional / Non-Linear Quantum Kernel Datasets:
   - Havlicek Quantum Feature Map Benchmark (ad_hoc_data)
3. Classical Small-Data & Low-Sample Benchmarks:
   - Breast Cancer Wisconsin (Diagnostic)
   - MNIST Digits (0 vs 1 downsampled/subsetted)

Outputs: eda_summary.json, run_metrics.csv, summary.json, and figures under --output-dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src is on sys.path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from qnn_vs_mlp import (
    CompactMLPClassifier,
    MLPClassifier,
    PennyLaneQuantumClassifier,
    RunConfig,
    TorchStatevectorQuantumClassifier,
    _apply_cnot_01,
    _apply_ry,
    build_qnn,
    count_parameters,
    eval_shot_readout,
    evaluate_model,
    load_dataset,
    mean_std,
    paired_stats,
    save_aggregate_outputs,
    set_seed,
    split_and_scale,
    train_model,
)
from qnn_vs_mlp.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["pennylane", "torch"],
        default="pennylane",
        help=(
            "pennylane (default): PennyLane default.qubit QNode; "
            "torch: independent PyTorch statevector reference backend"
        ),
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--qnn-epochs", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "pennylane_tfim",
            "qiskit_phase_of_matter",
            "havlicek_adhoc",
            "breast_cancer",
            "mnist_digits",
        ],
        help=(
            "Datasets to evaluate (choices: pennylane_tfim, qiskit_phase_of_matter, "
            "havlicek_adhoc, breast_cancer, mnist_digits)"
        ),
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    cfg = RunConfig(
        seeds=args.seeds,
        qnn_epochs=args.qnn_epochs,
        mlp_epochs=args.mlp_epochs,
        datasets=tuple(args.datasets),
    )
    run_experiment(cfg, Path(args.output_dir), args.backend)


if __name__ == "__main__":
    main()
