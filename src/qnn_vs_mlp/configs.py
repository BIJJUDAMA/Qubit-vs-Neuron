"""Configuration dataclasses and utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class RunConfig:
    seeds: int = 10
    qnn_epochs: int = 100
    mlp_epochs: int = 100
    learning_rate_qnn: float = 0.025
    learning_rate_mlp: float = 0.01
    moons_samples: int = 500
    moons_noise: float = 0.25
    test_size: float = 0.25
    torch_dtype: str = "float32"
    datasets: tuple = ("iris", "two_moons", "concentric_circles", "qdataset_hamiltonian")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
