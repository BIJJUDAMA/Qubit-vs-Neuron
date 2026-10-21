"""Exploratory Data Analysis (EDA) module for Quantum and Classical Datasets.

Evaluates and visualizes dataset properties before model training and benchmarking:
- Detailed column-level schemas (names, roles, dtypes, non-null counts, domain meanings).
- Explicit target column identification, data type, and class distribution.
- Sample sizes, feature dimensions (qubit count), and class balance.
- Summary statistics (mean, std, min, median, max, skewness, kurtosis).
- Class separability metrics (centroid distance, Fisher discriminant ratio, normalized separation).
- Inter-feature correlation matrices and principal component variance spectra.
- High-resolution visual figures (scatter projections, distributions, correlation heatmaps).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA


CATEGORY_MAP = {
    "pennylane_tfim": "1. Quantum State & Phase Classification",
    "qiskit_phase_of_matter": "1. Quantum State & Phase Classification",
    "havlicek_adhoc": "2. High-Dimensional / Non-Linear Quantum Kernel",
    "breast_cancer": "3. Classical Small-Data & Low-Sample",
    "mnist_digits": "3. Classical Small-Data & Low-Sample",
}

DISPLAY_NAMES = {
    "pennylane_tfim": "PennyLane TFIM",
    "qiskit_phase_of_matter": "Qiskit Phase of Matter",
    "havlicek_adhoc": "Havlicek Ad-Hoc",
    "breast_cancer": "Breast Cancer",
    "mnist_digits": "MNIST Digits (0 vs 1)",
}


def compute_dataset_eda(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    description: str,
    feature_names: Optional[List[str]] = None,
    target_name: Optional[str] = None,
    column_info: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive statistical, geometric, and column-level EDA metrics."""
    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    class_counts = {int(c): int(cnt) for c, cnt in zip(classes, counts)}
    balance_ratio = float(min(counts) / max(counts)) if len(counts) > 1 else 1.0

    feat_names = feature_names or [f"dim_{f+1}" for f in range(n_features)]
    tgt_name = target_name or "target"
    col_info = column_info or {}

    # Per-feature column statistics & schema
    columns_list = []
    feature_stats = []
    for f in range(n_features):
        feat_col = X[:, f]
        f_name = feat_names[f]
        f_meta = col_info.get(f_name, {})
        mean = float(np.mean(feat_col))
        std = float(np.std(feat_col, ddof=1)) if len(feat_col) > 1 else 0.0
        q25, median, q75 = np.percentile(feat_col, [25, 50, 75])
        skewness = float(scipy.stats.skew(feat_col))
        kurt = float(scipy.stats.kurtosis(feat_col))

        col_dict = {
            "column_name": f_name,
            "role": "Feature",
            "dtype": str(X.dtype),
            "non_null_count": int(np.count_nonzero(~np.isnan(feat_col))),
            "total_count": n_samples,
            "mean": mean,
            "std": std,
            "min": float(np.min(feat_col)),
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "max": float(np.max(feat_col)),
            "skewness": skewness,
            "kurtosis": kurt,
            "meaning": f_meta.get("meaning", f"Continuous feature {f+1}"),
            "purpose": f_meta.get("purpose", ""),
        }
        columns_list.append(col_dict)
        feature_stats.append({
            "feature_index": f,
            "feature_name": f_name,
            "mean": mean,
            "std": std,
            "min": float(np.min(feat_col)),
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "max": float(np.max(feat_col)),
            "skewness": skewness,
            "kurtosis": kurt,
        })

    # Target column metadata
    tgt_meta = col_info.get(tgt_name, {})
    target_column_info = {
        "column_name": tgt_name,
        "role": "Target",
        "dtype": str(y.dtype),
        "non_null_count": int(len(y)),
        "total_count": n_samples,
        "unique_values": [int(c) for c in classes],
        "class_distribution": class_counts,
        "class_mapping": {int(c): (labels[c] if c < len(labels) else f"Class {c}") for c in classes},
        "meaning": tgt_meta.get("meaning", f"Target label: {labels}"),
        "purpose": tgt_meta.get("purpose", "Supervised classification label"),
    }

    # Class separation metrics
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    mu0 = np.mean(X[idx0], axis=0) if len(idx0) > 0 else np.zeros(n_features)
    mu1 = np.mean(X[idx1], axis=0) if len(idx1) > 0 else np.zeros(n_features)
    std0 = np.std(X[idx0], axis=0) if len(idx0) > 0 else np.ones(n_features)
    std1 = np.std(X[idx1], axis=0) if len(idx1) > 0 else np.ones(n_features)

    centroid_dist = float(np.linalg.norm(mu0 - mu1))
    mean_intra_std = float(0.5 * (np.mean(std0) + np.mean(std1)))
    fisher_ratio = float((centroid_dist ** 2) / (np.mean(std0 ** 2) + np.mean(std1 ** 2) + 1e-8))
    norm_separation = float(centroid_dist / (mean_intra_std + 1e-8))

    # Correlation matrix
    if n_features > 1:
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        upper_tri = corr[np.triu_indices(n_features, k=1)]
        mean_abs_corr = float(np.mean(np.abs(upper_tri))) if len(upper_tri) > 0 else 0.0
        max_abs_corr = float(np.max(np.abs(upper_tri))) if len(upper_tri) > 0 else 0.0
    else:
        corr = np.array([[1.0]])
        mean_abs_corr = 0.0
        max_abs_corr = 0.0

    # PCA variance spectrum
    n_pca = min(n_features, n_samples)
    if n_pca >= 2:
        pca = PCA(n_components=n_pca)
        pca.fit(X)
        pca_var_ratio = [float(v) for v in pca.explained_variance_ratio_]
    else:
        pca_var_ratio = [1.0]

    sparsity = float(np.mean(np.isclose(X, 0.0, atol=1e-6)))

    return {
        "name": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "category": CATEGORY_MAP.get(name, "Other"),
        "description": description,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_qubits": n_features,
        "class_labels": labels,
        "class_counts": class_counts,
        "balance_ratio": balance_ratio,
        "feature_names": feat_names,
        "target_name": tgt_name,
        "target_column": target_column_info,
        "columns": columns_list,
        "centroid_distance": centroid_dist,
        "mean_intra_class_std": mean_intra_std,
        "fisher_discriminant_ratio": fisher_ratio,
        "normalized_separation": norm_separation,
        "mean_abs_correlation": mean_abs_corr,
        "max_abs_correlation": max_abs_corr,
        "pca_explained_variance_ratio": pca_var_ratio,
        "feature_statistics": feature_stats,
        "sparsity": sparsity,
        "correlation_matrix": corr.tolist(),
    }


def print_eda_report(all_eda: Dict[str, Dict[str, Any]]) -> None:
    """Print an informative, structured EDA summary with column schemas to the console."""
    print("\n" + "=" * 110)
    print("EXPLORATORY DATA ANALYSIS (EDA): DETAILED COLUMN SCHEMAS & TARGET DEFINITIONS")
    print("=" * 110)

    for name, eda in all_eda.items():
        disp = eda["display_name"]
        cat = eda["category"]
        n_samp = eda["n_samples"]
        n_q = eda["n_qubits"]
        tgt = eda.get("target_column", {})
        tgt_name = tgt.get("column_name", "target")
        tgt_dtype = tgt.get("dtype", "int64")
        class_map = tgt.get("class_mapping", {})
        class_dist = tgt.get("class_distribution", {})

        print(f"\n>> DATASET: {disp} [{cat}]")
        print(f"   Domain Scope : {eda['description']}")
        print(f"   Dimensions   : {n_samp} samples, {n_q} features ({n_q} qubits)")
        dist_str = ", ".join([f"{class_map.get(k, k)}: {cnt} ({cnt / n_samp * 100:.1f}%)" for k, cnt in class_dist.items()])
        print(f"   [TARGET COL] : '{tgt_name}' | Data Type: {tgt_dtype} | Class Distribution: {dist_str}")
        print("-" * 110)
        print(f"{'Column Name':<28} {'Role':<9} {'Data Type':<10} {'Non-Null':<10} {'Mean':<9} {'Std':<9} {'Min..Max':<16} {'Domain Meaning'}")
        print("-" * 110)

        for col in eda.get("columns", []):
            min_max = f"[{col['min']:.2f}, {col['max']:.2f}]"
            meaning = col.get("meaning", "")
            if len(meaning) > 28:
                meaning = meaning[:26] + ".."
            print(
                f"{col['column_name']:<28} "
                f"{col['role']:<9} "
                f"{col['dtype']:<10} "
                f"{col['non_null_count']}/{n_samp:<4} "
                f"{col['mean']:<9.3f} "
                f"{col['std']:<9.3f} "
                f"{min_max:<16} "
                f"{meaning}"
            )

        # Explicit target row
        tgt_meaning = tgt.get("meaning", "")
        if len(tgt_meaning) > 28:
            tgt_meaning = tgt_meaning[:26] + ".."
        u_vals = tgt.get("unique_values", [0, 1])
        val_str = f"[{min(u_vals)}, {max(u_vals)}]"
        print(
            f"{tgt_name:<28} "
            f"{'Target':<9} "
            f"{tgt_dtype:<10} "
            f"{n_samp}/{n_samp:<4} "
            f"{'-':<9} "
            f"{'-':<9} "
            f"{val_str:<16} "
            f"{tgt_meaning}"
        )
        print("-" * 110)

    # Consolidated cross-benchmark summary table
    print("\n" + "=" * 110)
    print("CONSOLIDATED CROSS-BENCHMARK GEOMETRIC & SEPARABILITY SUMMARY")
    print("=" * 110)
    print(f"{'Dataset':<24} {'Category':<32} {'Samples':<8} {'Qubits':<7} {'Balance':<8} {'Separation':<11} {'Target Column':<16}")
    print("-" * 110)
    for name, eda in all_eda.items():
        disp = eda["display_name"]
        cat = eda["category"]
        if len(cat) > 31:
            cat = cat[:28] + "..."
        n_samp = eda["n_samples"]
        n_q = eda["n_qubits"]
        bal = f"{eda['balance_ratio']:.2f}"
        sep = f"{eda['normalized_separation']:.3f}"
        tgt_col = eda.get("target_name", "target")
        print(f"{disp:<24} {cat:<32} {n_samp:<8} {n_q:<7} {bal:<8} {sep:<11} {tgt_col:<16}")
    print("=" * 110 + "\n", flush=True)


def plot_dataset_eda(all_eda: Dict[str, Dict[str, Any]], dataset_data: Dict[str, Any], figures_dir: Path) -> None:
    """Generate high-resolution EDA figures for all datasets."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_eda.keys())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    # Consolidated EDA figure: 4 panels per dataset in rows
    # Panel 1: 2D Feature Scatter (Projection)
    # Panel 2: Feature Boxplot by Class (Named features)
    # Panel 3: Correlation Matrix Heatmap
    # Panel 4: PCA Variance Spectrum & Class Balance
    fig, axes = plt.subplots(n_datasets, 4, figsize=(21.0, 4.2 * n_datasets), constrained_layout=True)
    if n_datasets == 1:
        axes = np.expand_dims(axes, 0)

    for row, name in enumerate(datasets):
        eda = all_eda[name]
        d_entry = dataset_data[name]
        X = d_entry["X"]
        y = d_entry["y"]
        labels = eda["class_labels"]
        disp_name = eda["display_name"]
        feat_names = eda.get("feature_names", [f"F{f+1}" for f in range(X.shape[1])])

        # Short clean names for plot labels
        short_feat_names = []
        for fn in feat_names:
            clean_fn = fn.replace("intensity", "int").replace("perimeter", "perim")
            if len(clean_fn) > 13:
                clean_fn = clean_fn[:11] + ".."
            short_feat_names.append(clean_fn)

        # Panel 1: 2D Scatter Projection
        ax1 = axes[row, 0]
        c_colors = ["#4A7BB0", "#D95F02"]
        for c_idx in (0, 1):
            mask = (y == c_idx)
            lbl = labels[c_idx] if c_idx < len(labels) else f"Class {c_idx}"
            ax1.scatter(
                X[mask, 0],
                X[mask, 1],
                c=c_colors[c_idx],
                label=f"{lbl} (n={np.sum(mask)})",
                alpha=0.82,
                edgecolors="w",
                linewidth=0.5,
                s=36,
            )
        mu0 = np.mean(X[y == 0], axis=0) if np.any(y == 0) else np.zeros(2)
        mu1 = np.mean(X[y == 1], axis=0) if np.any(y == 1) else np.zeros(2)
        ax1.scatter([mu0[0]], [mu0[1]], c="#1B385D", marker="X", s=130, edgecolors="white", linewidth=1.2, zorder=5)
        ax1.scatter([mu1[0]], [mu1[1]], c="#8A2C00", marker="X", s=130, edgecolors="white", linewidth=1.2, zorder=5)
        proj_note = f" ({short_feat_names[0]} vs {short_feat_names[1]})"
        ax1.set_title(f"{disp_name}{proj_note}", fontsize=11, fontweight="bold")
        ax1.set_xlabel(short_feat_names[0], fontsize=9)
        ax1.set_ylabel(short_feat_names[1], fontsize=9)
        ax1.legend(loc="best", fontsize=8, framealpha=0.8)
        ax1.grid(alpha=0.25)

        # Panel 2: Feature Boxplot per Class
        ax2 = axes[row, 1]
        n_feats = X.shape[1]
        positions = []
        data_to_plot = []
        box_colors = []
        for f in range(n_feats):
            data_to_plot.append(X[y == 0, f])
            data_to_plot.append(X[y == 1, f])
            positions.extend([f * 3 + 1, f * 3 + 2])
            box_colors.extend(["#4A7BB0", "#D95F02"])
        bp = ax2.boxplot(
            data_to_plot,
            positions=positions,
            widths=0.8,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xticks([f * 3 + 1.5 for f in range(n_feats)])
        ax2.set_xticklabels(short_feat_names, fontsize=8, rotation=15)
        ax2.set_title("Named Feature Distributions by Class", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Value Range", fontsize=9)
        ax2.grid(axis="y", alpha=0.25)

        # Panel 3: Correlation Matrix Heatmap
        ax3 = axes[row, 2]
        corr = np.array(eda["correlation_matrix"])
        im = ax3.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax3.set_xticks(range(n_feats))
        ax3.set_yticks(range(n_feats))
        ax3.set_xticklabels(short_feat_names, fontsize=7, rotation=20)
        ax3.set_yticklabels(short_feat_names, fontsize=7)
        for i in range(n_feats):
            for j in range(n_feats):
                ax3.text(
                    j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if abs(corr[i, j]) < 0.6 else "white",
                    fontsize=7,
                )
        ax3.set_title("Inter-Feature Correlation", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Panel 4: PCA Explained Variance Spectrum
        ax4 = axes[row, 3]
        pca_var = eda["pca_explained_variance_ratio"]
        x_pca = np.arange(1, len(pca_var) + 1)
        ax4.bar(x_pca, pca_var, color="#2CA02C", alpha=0.75, edgecolor="#202020", linewidth=0.8, label="Component Var")
        cum_var = np.cumsum(pca_var)
        ax4.plot(x_pca, cum_var, color="#D62728", marker="o", linewidth=1.5, label="Cumulative Var")
        ax4.set_xticks(x_pca)
        ax4.set_xticklabels([f"PC{k}" for k in x_pca], fontsize=8)
        ax4.set_ylim(0, 1.08)
        ax4.set_ylabel("Explained Variance Ratio", fontsize=9)
        ax4.set_title(f"PCA Variance Spectrum (Separation={eda['normalized_separation']:.2f})", fontsize=10, fontweight="bold")
        ax4.legend(loc="center right", fontsize=8, framealpha=0.8)
        ax4.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Exploratory Data Analysis (EDA): Geometric Manifolds, Distributions, & Correlative Structure",
        fontsize=16,
        fontweight="bold",
    )
    eda_fig_path = figures_dir / "eda_all_datasets.png"
    fig.savefig(eda_fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved EDA visualization: {eda_fig_path}")


def save_eda_summary(all_eda: Dict[str, Dict[str, Any]], outdir: Path) -> Path:
    """Save all EDA results to results/eda_summary.json."""
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_eda, f, indent=2)
    print(f"Saved EDA numerical summary: {summary_path}")
    return summary_path
