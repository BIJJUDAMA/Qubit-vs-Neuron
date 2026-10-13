"""Visualization and plotting routines for experimental figures."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_decision_boundaries(
    dataset_data: Dict[str, Any], models: Dict[Tuple[str, str], Any], output_dir: Path
) -> None:
    datasets = list(dataset_data.keys())
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, 4.4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    for row, dataset_name in enumerate(datasets):
        entry = dataset_data[dataset_name]
        X_plot = entry["X_all_scaled"]
        y_all = entry["y_all"]
        scaler = entry["scaler"]
        x_min, x_max = X_plot[:, 0].min() - 0.7, X_plot[:, 0].max() + 0.7
        y_min, y_max = X_plot[:, 1].min() - 0.7, X_plot[:, 1].max() + 0.7
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
        grid_2d = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        if X_plot.shape[1] > 2:
            fixed_vals = np.zeros(X_plot.shape[1] - 2, dtype=np.float32)
            grid_scaled = np.hstack([grid_2d, np.tile(fixed_vals, (len(grid_2d), 1))])
        else:
            grid_scaled = grid_2d
        angles = (np.pi * np.clip(grid_scaled / 2.5, -1.0, 1.0)).astype(np.float32)
        for col, model_name in enumerate(["MLP", "QNN"]):
            ax = axes[row, col]
            model = models[(dataset_name, model_name)]
            inp = grid_scaled if model_name == "MLP" else angles
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(inp, dtype=torch.float32))).cpu().numpy().reshape(xx.shape)
            ax.contourf(xx, yy, p, levels=[0, 0.5, 1], alpha=0.25, cmap="coolwarm")
            ax.contour(xx, yy, p, levels=[0.5], colors=["#202020"], linewidths=1.8)
            ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_all, cmap="coolwarm", edgecolors="white", linewidths=0.5, s=24)
            model_label = "Classical MLP (Wide)" if model_name == "MLP" else f"{X_plot.shape[1]}-qubit QNN (Depth 2)"
            slice_note = " (2D Projection Slice)" if X_plot.shape[1] > 2 else ""
            clean_name = dataset_name.replace("_", " ").title()
            ax.set_title(f"{clean_name}{slice_note} — {model_label}", fontsize=12, fontweight="bold", pad=6)
            ax.set_xlabel("Normalized Feature 1", fontsize=11)
            ax.set_ylabel("Normalized Feature 2", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(alpha=0.20)
    fig.suptitle("Empirical Decision Boundaries & Level Sets Across Benchmarks", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "decision_boundaries.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(summary_df: pd.DataFrame, output_dir: Path) -> None:
    main = summary_df[summary_df["config"].isin(["MLP", "QNN_depth2_entangled"])].copy()
    datasets = list(main["dataset"].unique())
    display_names = {
        "iris": "Iris",
        "two_moons": "Two Moons",
        "concentric_circles": "Concentric Circles",
        "qdataset_hamiltonian": "QDataSet Dynamics"
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["accuracy", "f1"], ["Test Accuracy", "Test F1-Score"]):
        categories = []
        labels = []
        colors = []
        for d in datasets:
            d_name = display_names.get(d, d.replace("_", " ").title())
            categories.append((d, "MLP"))
            labels.append(f"{d_name}\nMLP")
            colors.append("#5176A8")
            categories.append((d, "QNN_depth2_entangled"))
            labels.append(f"{d_name}\nQNN")
            colors.append("#D98E32")
        vals, errs = [], []
        for d, c in categories:
            sub = main[(main.dataset == d) & (main.config == c)][metric]
            vals.append(sub.mean() if not sub.empty else 0.0)
            errs.append(sub.std() if len(sub) > 1 and np.isfinite(sub.std()) else 0.0)
        x = np.arange(len(categories))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor="#303030", linewidth=0.8)
        ax.set_xticks(x, labels, fontsize=10, fontweight="medium")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.30)
    fig.suptitle("Primary Benchmark: Wide MLP vs Variational QNN (10 Paired Splits)", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "primary_metrics.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Compare a compact MLP, a wider MLP, and the primary QNN."""
    configs = ["MLP_small", "MLP", "QNN_depth2_entangled"]
    colors = ["#6C9A8B", "#5176A8", "#D98E32"]
    datasets = list(results_df["dataset"].unique())
    display_names = {
        "iris": "Iris",
        "two_moons": "Two Moons",
        "concentric_circles": "Concentric Circles",
        "qdataset_hamiltonian": "QDataSet Dynamics"
    }
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 4.8), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        d = results_df[results_df.dataset == dataset]
        means = d.groupby("config").accuracy.mean()
        stds = d.groupby("config").accuracy.std().fillna(0)
        x = np.arange(len(configs))
        labels = []
        for c in configs:
            sub = d[d.config == c]
            p_cnt = int(sub.parameters.iloc[0]) if not sub.empty else 0
            if c == "MLP_small":
                labels.append(f"Compact MLP\n({p_cnt}p)")
            elif c == "MLP":
                labels.append(f"Wide MLP\n({p_cnt}p)")
            else:
                labels.append(f"QNN Depth 2\n({p_cnt}p)")
        ax.bar(
            x,
            [means.get(c, 0.0) for c in configs],
            yerr=[stds.get(c, 0.0) for c in configs],
            capsize=4,
            color=colors,
            edgecolor="#303030",
            linewidth=0.8,
        )
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Test Accuracy", fontsize=11, fontweight="bold")
        d_name = display_names.get(dataset, dataset.replace("_", " ").title())
        ax.set_title(d_name, fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.30)
    fig.suptitle("Capacity Scaling & Inductive Bias: Compact MLP vs Wide MLP vs Variational QNN", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "capacity_control.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(results_df: pd.DataFrame, output_dir: Path) -> None:
    q = results_df[results_df["model_type"] == "QNN"].copy()
    q = q[
        q["config"].isin([
            "QNN_depth1_entangled",
            "QNN_depth2_entangled",
            "QNN_depth3_entangled",
            "QNN_depth2_no_entanglement",
        ])
    ]
    datasets = list(q["dataset"].unique())
    display_names = {
        "iris": "Iris",
        "two_moons": "Two Moons",
        "concentric_circles": "Concentric Circles",
        "qdataset_hamiltonian": "QDataSet Dynamics"
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    order = [
        "QNN_depth1_entangled",
        "QNN_depth2_entangled",
        "QNN_depth3_entangled",
        "QNN_depth2_no_entanglement",
    ]
    labels = ["Depth 1\nEntangled", "Depth 2\nEntangled", "Depth 3\nEntangled", "Depth 2\nUnentangled"]
    palette = ["#247A67", "#B36B28", "#5B5EA6", "#9B2335"]
    for ax, metric, title in zip(axes, ["accuracy", "f1"], ["Test Accuracy", "Test F1-Score"]):
        offsets = np.linspace(-0.15, 0.15, len(datasets))
        for di, dataset in enumerate(datasets):
            d = q[q.dataset == dataset].groupby("config")[metric].agg(["mean", "std"]).reindex(order)
            x = np.arange(len(order)) + offsets[di]
            d_name = display_names.get(dataset, dataset.replace("_", " ").title())
            ax.errorbar(
                x,
                d["mean"],
                yerr=d["std"].fillna(0),
                marker="o",
                markersize=6,
                capsize=4,
                linewidth=1.8,
                label=d_name,
                color=palette[di % len(palette)],
            )
        ax.set_xticks(np.arange(len(order)), labels, fontsize=10, fontweight="medium")
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.30)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    fig.suptitle("Quantum Circuit Architecture Ablation: Expressivity Across Depth & Entanglement", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "qnn_ablation.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(histories: Dict[Tuple[str, str], List[List[float]]], output_dir: Path) -> None:
    datasets = sorted(list(set(k[0] for k in histories.keys())))
    display_names = {
        "iris": "Iris",
        "two_moons": "Two Moons",
        "concentric_circles": "Concentric Circles",
        "qdataset_hamiltonian": "QDataSet Dynamics"
    }
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.5), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        for config, color, label in [
            ("MLP", "#5176A8", "Wide Classical MLP"),
            ("QNN_depth2_entangled", "#D98E32", "Variational Quantum Classifier"),
        ]:
            runs = histories.get((dataset, config), [])
            if not runs:
                continue
            arr = np.asarray(runs)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0)
            epochs = np.arange(1, arr.shape[1] + 1)
            ax.plot(epochs, mean, color=color, label=label, linewidth=2.0)
            ax.fill_between(epochs, np.maximum(0, mean - sd), mean + sd, color=color, alpha=0.18)
        d_name = display_names.get(dataset, dataset.replace("_", " ").title())
        ax.set_title(d_name, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Training BCE Loss", fontsize=11, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    fig.suptitle("Empirical Convergence & Optimization Trajectories Across Data Geometries", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "loss_curves.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(confusions: Dict[Tuple[str, str], np.ndarray], output_dir: Path) -> None:
    datasets = ['concentric_circles', 'iris', 'two_moons', 'qdataset_hamiltonian']
    display_names = {
        'concentric_circles': 'Concentric Circles',
        'iris': 'Iris (Versicolor vs Virginica)',
        'two_moons': 'Two Moons',
        'qdataset_hamiltonian': 'QDataSet Hamiltonian Dynamics'
    }
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.0, 4.4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(["MLP", "QNN_depth2_entangled"]):
            ax = axes[i, j]
            cm = confusions.get((dataset, model), np.zeros((2, 2), dtype=int))
            row_sums = cm.sum(axis=1, keepdims=True)
            normalized = cm / np.maximum(row_sums, 1)
            im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    ax.text(
                        c,
                        r,
                        f"{cm[r, c]}\n({normalized[r, c]:.2%})",
                        ha="center",
                        va="center",
                        color="white" if normalized[r, c] > 0.58 else "#111111",
                        fontsize=14,
                        fontweight="bold",
                    )
            ax.set_xticks([0, 1], ["Pred Class 0", "Pred Class 1"], fontsize=12)
            ax.set_yticks([0, 1], ["True Class 0", "True Class 1"], fontsize=12)
            m_label = "Classical MLP (Wide)" if model == "MLP" else "Variational QNN (Depth 2)"
            d_name = display_names.get(dataset, dataset.replace("_", " ").title())
            ax.set_title(f"{d_name}\n{m_label}", fontsize=13, fontweight="bold", pad=8)
    fig.suptitle(
        "Pooled Test Confusion Matrices & Normalized Classification Distributions Across Benchmarks",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shot_sensitivity(shot_results: List[Dict[str, Any]], output_dir: Path) -> None:
    df = pd.DataFrame(shot_results)
    if df.empty:
        return
    datasets = list(df["dataset"].unique())
    display_names = {
        "iris": "Iris",
        "two_moons": "Two Moons",
        "concentric_circles": "Concentric Circles",
        "qdataset_hamiltonian": "QDataSet Dynamics"
    }
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 4.8), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        d = df[df.dataset == dataset]
        means = d.groupby("setting")["accuracy"].mean()
        stds = d.groupby("setting")["accuracy"].std().fillna(0)
        order = ["analytic", "256 shots", "1024 shots"]
        vals = [means.get(k, np.nan) for k in order]
        errs = [stds.get(k, 0) for k in order]
        ax.bar(
            np.arange(3),
            vals,
            yerr=errs,
            capsize=4,
            color=["#6B8E7C", "#D7A54B", "#B66D58"],
            edgecolor="#333333",
            linewidth=0.8,
        )
        ax.set_xticks(np.arange(3), ["Analytic", "256 Shots", "1024 Shots"], fontsize=10)
        ax.set_ylim(0, 1.08)
        d_name = display_names.get(dataset, dataset.replace("_", " ").title())
        ax.set_title(d_name, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Test Accuracy", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.30)
    fig.suptitle("Impact of Quantum Measurement Shot Noise on Classification Fidelity", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "shot_sensitivity.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


