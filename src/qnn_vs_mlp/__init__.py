"""QNN vs MLP Case Study Package."""
from __future__ import annotations

__version__ = "0.1.0"

from .configs import RunConfig, set_seed
from .models import (
    MLPClassifier,
    CompactMLPClassifier,
    TorchStatevectorQuantumClassifier,
    PennyLaneQuantumClassifier,
    build_qnn,
    count_parameters,
    _apply_cnot_01,
    _apply_ry,
)
from .data import load_dataset, split_and_scale
from .training import train_model, evaluate_model
from .analysis import finite_shot_features, eval_shot_readout, mean_std, paired_stats
from .reporting import save_aggregate_outputs
from .pipeline import run_experiment

__all__ = [
    "RunConfig",
    "set_seed",
    "MLPClassifier",
    "CompactMLPClassifier",
    "TorchStatevectorQuantumClassifier",
    "PennyLaneQuantumClassifier",
    "build_qnn",
    "count_parameters",
    "_apply_cnot_01",
    "_apply_ry",
    "load_dataset",
    "split_and_scale",
    "train_model",
    "evaluate_model",
    "finite_shot_features",
    "eval_shot_readout",
    "mean_std",
    "paired_stats",
    "save_aggregate_outputs",
    "run_experiment",
]
