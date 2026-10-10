# QNN vs Classical Neural Networks: Reproducible Case Study

This project compares variational quantum classifiers (VQCs) with PyTorch multilayer perceptrons across four diverse binary classification benchmarks:
1. **Iris** (*Iris versicolor* vs. *Iris virginica*, overlapping petal features).
2. **Two Moons** (interleaving non-convex crescents with noise $\sigma = 0.25$).
3. **Concentric Circles** (inner vs. outer rings, rotational inductive bias benchmark).
4. **QDataSet 2-Qubit Hamiltonian Dynamics** (physical quantum dynamics classifying unentangled local drift vs. entangling $X \otimes X$ interaction).

The package includes a modular scientific architecture (`src/qnn_vs_mlp/`), an automated PennyLane `default.qubit` execution pipeline, paired statistical tests, publication figures, unit tests, and a publication-ready LaTeX manuscript (`case_study.tex` with `sections/`).

## Execution & Result Provenance

**All metrics, figures, and tables in `results/` and the compiled PDF manuscript (`case_study.pdf`) are generated from actual PennyLane executions (`backend=pennylane`).**
- Primary execution uses PennyLane `default.qubit` QNodes with the Torch interface and simulator backpropagation (`qml.device("default.qubit", wires=n_qubits, shots=None)`).
- An independent PyTorch statevector implementation is maintained for automated parity verification and gradient flow testing (`pytest tests`).
- Full reproducibility is maintained across 10 paired, stratified train/test splits (seeds 0 through 9). All competing models share identical splits at each seed.

## Benchmark Datasets

- **Iris Subset:** Petal length and width comparing *Iris versicolor* vs. *Iris virginica* (100 samples, overlapping boundary).
- **Two Moons:** 500 samples per seed, noise $\sigma = 0.25$.
- **Concentric Circles:** 500 samples per seed, factor 0.5, noise 0.08.
- **QDataSet 2-Qubit Dynamics:** 500 physical state realizations per seed generated from time-dependent Hamiltonian dynamics. Evaluates 4 Pauli expectations ($\langle XI \rangle, \langle IX \rangle, \langle XX \rangle, \langle ZZ \rangle$) to classify separable local evolution vs. bipartite interacting dynamics.

## One-Command Master Execution

To run the complete pipeline (dependencies check, test suite, PennyLane benchmarks, LaTeX macro updates, and PDF compilation), execute `run.py`:

```bash
python run.py
```

Optional arguments:
```bash
python run.py --seeds 10 --backend pennylane
python run.py --skip-pdf  # Skip LaTeX PDF compilation
```

## Repository Structure

- `src/qnn_vs_mlp/`: Clean, modular scientific package:
  - `models/`: Classical MLPs, PennyLane QNode classifier, and PyTorch statevector reference.
  - `qdataset.py`: QDataSet physical Hamiltonian dynamics generator.
  - `configs.py`: Hyperparameter configurations and seed utilities.
  - `data.py`: Dataset loading and stratified scaling.
  - `training.py`: Model training loops and evaluation metrics.
  - `analysis.py`: Paired statistics and finite-shot measurement simulation.
  - `visualization.py`: Publication figure generation.
  - `reporting.py`: JSON summaries and LaTeX macros.
  - `pipeline.py`: Full experiment execution orchestration.
- `main.py`: Primary clean CLI entrypoint.
- `run.py`: Unified cross-platform pipeline runner.
- `sections/`: Modular manuscript sections (`00_abstract.tex` through `07_conclusion.tex`).
- `case_study.tex`: Master LaTeX document.
- `case_study.pdf`: Compiled case study publication paper.
- `tests/`: Automated unit and backend parity tests (`pytest tests`).
- `results/`: Canonical PennyLane experiment artifacts:
  - `run_metrics.csv`: Run-level metric rows (4 datasets x 10 seeds x 6 models).
  - `shot_sensitivity.csv`: Finite-shot inference resample results.
  - `summary.json`: Aggregate statistics, paired tests, and execution metadata.
  - `results_macros.tex`: Direct LaTeX macro definitions.
  - `figures/`: All publication figures (decision boundaries, loss curves, confusion matrices, capacity controls, ablations, shot sensitivity).

## Citations & Attribution

The quantum Hamiltonian dynamics benchmark incorporates the simulation model and open-access protocols of **QDataSet**:
> **Perrier, E., Youssry, A. & Ferrie, C.**  
> *"QDataSet, quantum datasets for machine learning."*  
> *Scientific Data* **9**, 582 (2022).  
> DOI: [10.1038/s41597-022-01639-1](https://doi.org/10.1038/s41597-022-01639-1)  
> GitHub: [https://github.com/eperrier/QDataSet](https://github.com/eperrier/QDataSet)

## Claims and Limits

The study is an educational simulator-based case study on classical and quantum-simulated datasets. It does not run on physical quantum hardware and does not establish commercial quantum advantage. Runtime metrics represent software simulation on CPU. The finite-shot experiment samples measurements only at inference; device noise and decoherence are outside scope.
