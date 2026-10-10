#!/usr/bin/env python3
"""Reproducible QNN-vs-MLP case study.

The primary backend is PennyLane default.qubit with a Torch-interface QNode
(`--backend pennylane`). A separate PyTorch statevector reference backend
(`--backend torch`) is available for independent circuit-parity checks and for
reproducing the included results in offline environments. Outputs are always
tagged with the backend so reference data cannot be mistaken for a PennyLane
execution.

Outputs: run_metrics.csv, summary.json, and figures under --output-dir.
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
    finite_shot_features,
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
        default=["iris", "two_moons", "concentric_circles", "qdataset_hamiltonian"],
        help="Datasets to evaluate (choices: iris, two_moons, concentric_circles, qdataset_hamiltonian)",
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
