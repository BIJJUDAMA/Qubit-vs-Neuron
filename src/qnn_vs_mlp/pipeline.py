"""End-to-end experiment pipeline orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch

from .analysis import eval_shot_readout, paired_stats
from .configs import RunConfig, set_seed
from .data import load_dataset, split_and_scale
from .models import (
    CompactMLPClassifier,
    MLPClassifier,
    build_qnn,
    count_parameters,
)
from .reporting import save_aggregate_outputs
from .training import evaluate_model, train_model
from .visualization import (
    plot_ablation,
    plot_capacity_comparison,
    plot_confusion_matrices,
    plot_decision_boundaries,
    plot_loss_curves,
    plot_metrics,
    plot_shot_sensitivity,
)

# Reproducibility and predictable CPU timings for this small experiment.
torch.set_num_threads(1)


def run_experiment(
    cfg: RunConfig, outdir: Path, backend: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("QNN_depth1_entangled", "QNN", 1, True),
        ("QNN_depth2_entangled", "QNN", 2, True),
        ("QNN_depth3_entangled", "QNN", 3, True),
        ("QNN_depth2_no_entanglement", "QNN", 2, False),
    ]
    results, shot_results = [], []
    histories: Dict[Tuple[str, str], List[List[float]]] = {}
    confusions: Dict[Tuple[str, str], np.ndarray] = {}
    boundary_models = {}
    plot_data = {}

    # PennyLane availability is checked; never silently claim it is the active backend.
    if backend == "pennylane":
        try:
            import pennylane  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PennyLane is not installed. Install dependencies with `python -m pip install -r requirements.txt`. "
                "The reference backend is available with --backend torch."
            ) from exc

    datasets_to_run = getattr(cfg, "datasets", ("iris", "two_moons", "concentric_circles"))
    for dataset_name in datasets_to_run:
        plot_data[dataset_name] = {}
        for seed in range(cfg.seeds):
            print(f"[{dataset_name}] seed {seed + 1}/{cfg.seeds}", flush=True)
            X, y, labels, description = load_dataset(
                dataset_name, cfg.moons_samples, cfg.moons_noise, data_seed=2026 + seed
            )
            n_features = X.shape[1]
            if seed == 0:
                plot_data[dataset_name].update({"description": description, "labels": labels})
            train_idx, test_idx, scaler, Xtr, Xte, Atr, Ate = split_and_scale(
                X, y, seed, cfg.test_size
            )
            ytr = torch.tensor(y[train_idx], dtype=torch.float32)
            yte = torch.tensor(y[test_idx], dtype=torch.float32)
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
            Xte_t = torch.tensor(Xte, dtype=torch.float32)
            Atr_t = torch.tensor(Atr, dtype=torch.float32)
            Ate_t = torch.tensor(Ate, dtype=torch.float32)

            # Same split and scaler are stored for a seed-0 visualization.
            if seed == 0:
                # Scale all data only for plotting; scaler is fitted on the training partition.
                all_scaled = scaler.transform(X).astype(np.float32)
                plot_data[dataset_name].update(
                    {"X_all_scaled": all_scaled, "y_all": y, "scaler": scaler}
                )

            # MLP trained on standardized classical features.
            set_seed(10000 + seed + (0 if dataset_name == "iris" else 1000))
            mlp = MLPClassifier(in_features=n_features)
            mlp_loss, mlp_acc, mlp_time = train_model(
                mlp, Xtr_t, ytr, cfg.mlp_epochs, cfg.learning_rate_mlp
            )
            mlp_metrics, mlp_prob, mlp_pred = evaluate_model(mlp, Xte_t, yte)
            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "config": "MLP",
                "model_type": "MLP",
                "depth": 0,
                "entanglement": "not_applicable",
                "parameters": count_parameters(mlp),
                "train_time_s": mlp_time,
                **mlp_metrics,
            })
            histories.setdefault((dataset_name, "MLP"), []).append(mlp_loss)
            confusions[(dataset_name, "MLP")] = confusions.get(
                (dataset_name, "MLP"), np.zeros((2, 2), dtype=int)
            ) + confusion_matrix(y[test_idx], mlp_pred, labels=[0, 1])
            if seed == 0:
                boundary_models[(dataset_name, "MLP")] = mlp

            # Compact MLP: parameter-count control close to the depth-2 QNN.
            set_seed(15000 + seed + (0 if dataset_name == "iris" else 1000))
            compact_mlp = CompactMLPClassifier(in_features=n_features)
            _, _, compact_time = train_model(
                compact_mlp, Xtr_t, ytr, cfg.mlp_epochs, cfg.learning_rate_mlp
            )
            compact_metrics, _, _ = evaluate_model(compact_mlp, Xte_t, yte)
            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "config": "MLP_small",
                "model_type": "MLP_small",
                "depth": 1,
                "entanglement": "not_applicable",
                "parameters": count_parameters(compact_mlp),
                "train_time_s": compact_time,
                **compact_metrics,
            })

            # QNN configurations: depth and entanglement ablations.
            for conf_name, model_type, depth, entangle in configs:
                set_seed(
                    20000
                    + seed
                    + (0 if dataset_name == "iris" else 1000)
                    + depth * 10
                    + int(entangle)
                )
                qnn = build_qnn(backend, depth=depth, entangle=entangle, n_qubits=n_features)
                qnn_loss, qnn_acc, qnn_time = train_model(
                    qnn, Atr_t, ytr, cfg.qnn_epochs, cfg.learning_rate_qnn
                )
                qnn_metrics, qnn_prob, qnn_pred = evaluate_model(qnn, Ate_t, yte)
                results.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "config": conf_name,
                    "model_type": model_type,
                    "depth": depth,
                    "entanglement": "yes" if entangle else "no",
                    "parameters": count_parameters(qnn),
                    "train_time_s": qnn_time,
                    **qnn_metrics,
                })
                if conf_name == "QNN_depth2_entangled":
                    histories.setdefault((dataset_name, conf_name), []).append(qnn_loss)
                    confusions[(dataset_name, conf_name)] = confusions.get(
                        (dataset_name, conf_name), np.zeros((2, 2), dtype=int)
                    ) + confusion_matrix(y[test_idx], qnn_pred, labels=[0, 1])
                    if seed == 0:
                        boundary_models[(dataset_name, "QNN")] = qnn
                    probs = qnn.basis_probabilities(Ate_t)
                    # Five independent measurement resamplings per split and shot count.
                    # Duplicate the analytic baseline five times to keep group sizes balanced.
                    for replicate in range(5):
                        shot_results.append({
                            "dataset": dataset_name,
                            "seed": seed,
                            "replicate": replicate,
                            "setting": "analytic",
                            **qnn_metrics,
                        })
                    for setting, shot_n in [("256 shots", 256), ("1024 shots", 1024)]:
                        for replicate in range(5):
                            shotm = eval_shot_readout(
                                qnn,
                                probs,
                                yte,
                                shot_n,
                                seed=5000
                                + seed * 100
                                + shot_n
                                + replicate * 10000
                                + (20000 if dataset_name == "two_moons" else 0),
                            )
                            shot_results.append({
                                "dataset": dataset_name,
                                "seed": seed,
                                "replicate": replicate,
                                "setting": setting,
                                **shotm,
                            })

    results_df = pd.DataFrame(results)
    results_df.insert(0, "backend", backend)
    results_df.to_csv(outdir / "run_metrics.csv", index=False)
    shot_df = pd.DataFrame(shot_results)
    if not shot_df.empty:
        shot_df.insert(0, "backend", backend)
    shot_df.to_csv(outdir / "shot_sensitivity.csv", index=False)
    plot_metrics(results_df, figures_dir)
    plot_capacity_comparison(results_df, figures_dir)
    plot_ablation(results_df, figures_dir)
    plot_loss_curves(histories, figures_dir)
    plot_confusion_matrices(confusions, figures_dir)
    plot_shot_sensitivity(shot_results, figures_dir)
    plot_decision_boundaries(plot_data, boundary_models, figures_dir)
    save_aggregate_outputs(results_df, shot_df, histories, confusions, outdir, backend, cfg)
    print("\nPRIMARY SUMMARY (mean +/- SD):")
    for dataset in datasets_to_run:
        print(f"\n{dataset}")
        part = results_df[results_df.dataset == dataset]
        for conf in [
            "MLP",
            "QNN_depth1_entangled",
            "QNN_depth2_entangled",
            "QNN_depth3_entangled",
            "QNN_depth2_no_entanglement",
        ]:
            p = part[part.config == conf]

            def fmt(metric, p=p):
                return f"{p[metric].mean():.3f} +/- {p[metric].std(ddof=1):.3f}"

            print(
                f"{conf:32s} accuracy={fmt('accuracy')} F1={fmt('f1')} train_s={fmt('train_time_s')}"
            )
        a = part[part.config == "QNN_depth2_entangled"].sort_values("seed")
        b = part[part.config == "MLP"].sort_values("seed")
        ps = paired_stats(a.accuracy, b.accuracy)
        print("QNN depth2 - MLP accuracy difference:", json.dumps(ps))
        compact = part[part.config == "MLP_small"].sort_values("seed")
        print(
            f"MLP_small                     accuracy={compact.accuracy.mean():.3f} +/- {compact.accuracy.std(ddof=1):.3f} F1={compact.f1.mean():.3f} +/- {compact.f1.std(ddof=1):.3f}"
        )
        print(
            "QNN depth2 - compact MLP accuracy difference:",
            json.dumps(paired_stats(a.accuracy, compact.accuracy)),
        )
        print("shot sensitivity:")
        for setting, g in shot_df[shot_df.dataset == dataset].groupby("setting"):
            print(
                f"  {setting:12s}: accuracy={g.accuracy.mean():.3f} +/- {g.accuracy.std(ddof=1):.3f}; F1={g.f1.mean():.3f} +/- {g.f1.std(ddof=1):.3f}"
            )
    return results_df, shot_df
