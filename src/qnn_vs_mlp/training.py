"""Model training and evaluation routines."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import nn


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> Tuple[List[float], List[float], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    losses, train_accuracies = [], []
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            train_accuracies.append(
                float(((torch.sigmoid(logits) >= 0.5) == (y_train >= 0.5)).float().mean().cpu())
            )
    elapsed = time.perf_counter() - start
    return losses, train_accuracies, elapsed


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    model_input_type: str = "classical",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    y_np = y.cpu().numpy().astype(int)
    try:
        auc = float(roc_auc_score(y_np, prob))
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_np, pred)),
        "precision": float(precision_score(y_np, pred, zero_division=0)),
        "recall": float(recall_score(y_np, pred, zero_division=0)),
        "f1": float(f1_score(y_np, pred, zero_division=0)),
        "roc_auc": auc,
        "log_loss": float(log_loss(y_np, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
    return metrics, prob, pred
