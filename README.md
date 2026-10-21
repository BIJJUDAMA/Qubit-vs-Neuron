# QNN vs Classical Neural Networks: Reproducible Case Study

This project presents a rigorous empirical and theoretical comparison between variational quantum classifiers (VQCs / QNNs) with data re-uploading and classical multilayer perceptrons (MLPs) across six benchmark datasets organized into three distinct inductive-bias categories:

1. **Quantum State & Phase Classification Datasets:**
   - **PennyLane TFIM:** Transverse-Field Ising Model 1D open chain ($L=4$) ground states across transverse field $h \in [0, 3.0]$. Classifies ordered ferromagnetic ($h < 1.0$) vs. disordered paramagnetic ($h > 1.0$) phases using 4 ground-state Pauli expectation observables ($\langle Z_0 Z_1 \rangle, \langle Z_1 Z_2 \rangle, \langle Z_2 Z_3 \rangle, \langle X_{\text{mean}} \rangle$).
   - **Qiskit Phase of Matter:** Bond-alternating XXX Heisenberg spin chain ground states (Qiskit ML). Classifies trivial vs. symmetry-protected topological (SPT) phases via 4 PCA ground-state features.

2. **High-Dimensional / Non-Linear Quantum Kernel Datasets:**
   - **Havlíček Ad-Hoc Benchmark (`ad_hoc_data`):** Havlíček et al. (*Nature* 2019) quantum feature map benchmark with separation gap 0.3. Explicitly constructed to be non-linearly separable by classical hyperplanes while admitting an exact quantum kernel separation via a second-order $ZZ$-feature map.
   - **Concentric Circles:** Nested circular geometry (factor 0.3, noise 0.05). Directly evaluates rotational trigonometric inductive bias against feedforward ReLU units.

3. **Classical Small-Data & Low-Sample Benchmarks:**
   - **Breast Cancer Wisconsin (Diagnostic):** Real-world clinical diagnostic dataset (UCI / Scikit-Learn). Reduced from 30 continuous clinical features to 4 principal components ($\approx 80\%$ variance explained), matching a 4-qubit NISQ variational classifier.
   - **MNIST Digits (0 vs. 1):** Binary handwritten digits downsampled and projected to 4 principal components ($\approx 75\%$ variance explained) in a low-sample regime.

---

## Exploratory Data Analysis (EDA)

Before model training, an automated EDA engine evaluates:
- Sample and class balance ($N_0 / N_1$)
- Feature distributional moments (mean, std, min, median, max, skewness, kurtosis)
- Normalized class separation $S = \|\mu_0 - \mu_1\|_2 / (\bar{\sigma}_0 + \bar{\sigma}_1)$ and Fisher discriminant ratio
- Inter-feature correlation matrices
- PCA spectral variance decomposition

EDA metrics are printed in structured tables, written to `results/eda_summary.json`, and plotted in `results/figures/eda_all_datasets.png`.

---

## Execution & Result Provenance

- **PennyLane execution:** Primary execution uses PennyLane `default.qubit` QNodes with the Torch interface and simulator backpropagation.
- **Independent PyTorch Statevector:** An independent, fully differentiable $N$-qubit tensor statevector implementation is maintained for offline parity verification and gradient flow testing (`pytest tests`).
- **Paired splits:** Stratified train/test splits are strictly shared across all competing models for each seed.

---

## One-Command Master Execution

To run the complete pipeline (test suite, EDA, PennyLane benchmarks, LaTeX macro updates, and PDF compilation), execute `run.py`:

```bash
python run.py
```

Optional arguments:
```bash
python run.py --seeds 5 --backend pennylane
python run.py --skip-pdf  # Skip LaTeX PDF compilation
```

---

## Repository Structure

- `src/qnn_vs_mlp/`:
  - `data.py`: Dataset loading for all 6 benchmarks across the 3 categories.
  - `eda.py`: Automated Exploratory Data Analysis computation, reports, and visualization.
  - `models/`: Classical MLPs, PennyLane QNode classifier, and PyTorch multi-qubit statevector reference.
  - `configs.py`: Dataclass configurations and seed utilities.
  - `training.py`: Model training loops and evaluation metrics.
  - `analysis.py`: Paired statistics and finite-shot measurement simulation.
  - `visualization.py`: Publication figure generation.
  - `reporting.py`: JSON summaries and LaTeX macros.
  - `pipeline.py`: Full experiment execution orchestration.
- `main.py`: Primary CLI entrypoint.
- `run.py`: Unified pipeline runner.
- `sections/`: Modular manuscript sections (`00_abstract.tex` through `07_conclusion.tex`).
- `case_study.tex`: Master LaTeX document.
- `case_study.pdf`: Compiled case study publication paper.
- `tests/`: Automated unit and backend parity tests (`pytest tests`).
- `results/`:
  - `eda_summary.json`: Quantitative EDA metrics across all 6 benchmarks.
  - `run_metrics.csv`: Metric rows across seeds, models, and datasets.
  - `shot_sensitivity.csv`: Finite-shot inference resample results.
  - `summary.json`: Aggregate statistics, paired tests, and execution metadata.
  - `results_macros.tex`: Direct LaTeX macro definitions.
  - `figures/`: All publication figures (EDA, decision boundaries, loss curves, confusion matrices, capacity controls, ablations, shot sensitivity).
