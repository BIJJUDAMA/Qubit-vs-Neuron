"""Reporting, summary generation, and LaTeX macro output utilities."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import sklearn
import torch

from .analysis import mean_std, paired_stats
from .configs import RunConfig


def save_aggregate_outputs(
    results_df: pd.DataFrame,
    shot_df: pd.DataFrame,
    histories: Dict[Tuple[str, str], List[List[float]]],
    confusions: Dict[Tuple[str, str], np.ndarray],
    outdir: Path,
    backend: str,
    cfg: RunConfig,
) -> None:
    summary: Dict[str, Any] = {}
    datasets = list(results_df["dataset"].unique())
    for dataset in datasets:
        ddf = results_df[results_df.dataset == dataset]
        summary[dataset] = {}
        for config_name, part in ddf.groupby("config"):
            summary[dataset][config_name] = {}
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "log_loss",
                "train_time_s",
            ]:
                summary[dataset][config_name][metric] = mean_std(part[metric].values)
        mlp = ddf[ddf.config == "MLP"].set_index("seed")
        qnn = ddf[ddf.config == "QNN_depth2_entangled"].set_index("seed")
        summary[dataset]["paired_primary_comparison"] = {
            metric: paired_stats(qnn[metric].values, mlp[metric].values)
            for metric in ["accuracy", "f1", "roc_auc"]
        }
        compact = ddf[ddf.config == "MLP_small"].set_index("seed")
        summary[dataset]["paired_compact_mlp_comparison"] = {
            metric: paired_stats(qnn[metric].values, compact[metric].values)
            for metric in ["accuracy", "f1", "roc_auc"]
        }
    try:
        pl_version = importlib.metadata.version("pennylane")
    except importlib.metadata.PackageNotFoundError:
        pl_version = None
    summary["run_metadata"] = {
        "backend_used": backend,
        "pennylane_version": pl_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "seeds": list(range(cfg.seeds)),
        "qnn_epochs": cfg.qnn_epochs,
        "mlp_epochs": cfg.mlp_epochs,
        "dataset_splits": "Stratified train/test splits; same split reused for all models for each seed",
        "metric_provenance": (
            "PennyLane default.qubit QNode executed"
            if backend == "pennylane"
            else "Reference metrics from the differentiable PyTorch statevector; not a PennyLane execution"
        ),
        "device": (
            "PennyLane default.qubit, shots=None"
            if backend == "pennylane"
            else "Exact two-qubit statevector implemented with PyTorch tensor operations"
        ),
        "gradient_method": (
            "PennyLane simulator backpropagation"
            if backend == "pennylane"
            else "PyTorch autograd through explicit statevector tensor operations"
        ),
    }

    shot_summary: Dict[str, Any] = {}
    for dataset in datasets:
        shot_summary[dataset] = {}
        if not shot_df.empty and dataset in shot_df.dataset.values:
            for setting, group in shot_df[shot_df.dataset == dataset].groupby("setting"):
                shot_summary[dataset][setting] = {
                    m: mean_std(group[m].values) for m in ["accuracy", "f1", "log_loss"]
                }
    summary["shot_sensitivity"] = shot_summary
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown-free LaTeX data fragments facilitate regeneration of the paper.
    with open(outdir / "results_macros.tex", "w", encoding="utf-8") as f:
        for dataset in datasets:
            for config_name in [
                "MLP",
                "MLP_small",
                "QNN_depth1_entangled",
                "QNN_depth2_entangled",
                "QNN_depth3_entangled",
                "QNN_depth2_no_entanglement",
            ]:
                row = summary[dataset].get(config_name)
                if not row:
                    continue
                dataset_clean = "".join(w.capitalize() for w in dataset.split("_"))
                dataset_clean = dataset_clean[0].lower() + dataset_clean[1:]
                cfg_clean = (
                    config_name.replace("QNN_", "QNN")
                    .replace("depth", "d")
                    .replace("_entangled", "Ent")
                    .replace("_no_entanglement", "NoEnt")
                    .replace("MLP_small", "MLPSmall")
                )
                prefix = f"{dataset_clean}{cfg_clean}"
                p_clean = (
                    prefix.replace("d1", "dOne")
                    .replace("d2", "dTwo")
                    .replace("d3", "dThree")
                )
                for metric in ["accuracy", "f1", "roc_auc", "train_time_s"]:
                    m = row[metric]
                    # In LaTeX, command names can only contain ASCII letters A-Za-z
                    metric_clean = (
                        "Accuracy"
                        if metric == "accuracy"
                        else "FOne"
                        if metric == "f1"
                        else "RocAuc"
                        if metric == "roc_auc"
                        else "TrainTimeS"
                    )
                    key = f"{p_clean}{metric_clean}"
                    f.write(f"\\newcommand{{\\{key}}}{{{m['mean']:.3f} $\\pm$ {m['std']:.3f}}}\n")
    print(f"Saved summary: {outdir / 'summary.json'}")
