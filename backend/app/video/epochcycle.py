"""
Epoch & Training Cycle Tracker Module
=====================================
Provides lightweight metric calculation, epoch tracking, and baseline optimization step runner.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger("video.epochcycle")


def classification_metrics(
    targets: List[int] | np.ndarray,
    predictions: List[int] | np.ndarray,
    num_classes: int
) -> Dict[str, Any]:
    """Calculates top-1 accuracy, macro/micro precision, recall, and F1 score."""
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    total = len(targets)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_samples": 0
        }

    correct = int((targets == predictions).sum())
    accuracy = correct / total

    precisions = []
    recalls = []
    f1_scores = []

    for c in range(num_classes):
        tp = int(((predictions == c) & (targets == c)).sum())
        fp = int(((predictions == c) & (targets != c)).sum())
        fn = int(((predictions != c) & (targets == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": float(accuracy),
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "total_samples": total,
        "correct": correct
    }


class EpochCycleTracker:
    """Tracks training metrics per epoch, identifying best performing model checkpoints."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self.best_accuracy = -1.0
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.best_metrics: Dict[str, Any] = {}
        self.start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        metrics: Dict[str, Any]
    ) -> bool:
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        is_best = False

        if acc > self.best_accuracy or (acc == self.best_accuracy and f1 > self.best_f1):
            self.best_accuracy = acc
            self.best_f1 = f1
            self.best_epoch = epoch
            self.best_metrics = metrics
            is_best = True

        elapsed = time.time() - self.start_time
        star = " ★ NEW BEST MODEL SAVED" if is_best else ""
        pct_complete = (epoch / self.total_epochs) * 100
        logger.info(
            f"🎯 [Overall Progress: {pct_complete:5.1f}%] Epoch [{epoch:02d}/{self.total_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {acc * 100:.2f}% | "
            f"Val F1: {f1:.4f} | "
            f"Elapsed: {elapsed:.1f}s{star}"
        )
        return is_best


def run_baseline_epoch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    batch_size: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    mW: np.ndarray,
    vW: np.ndarray,
    mb: np.ndarray,
    vb: np.ndarray,
    step_count: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Runs 1 mini-batch SGD / Adam optimization epoch for HandCraftedBaseline."""
    n_samples, n_features = train_x.shape

    indices = rng.permutation(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        xb = train_x[batch_idx]
        yb = train_y[batch_idx]

        # Forward
        logits = xb @ weights + bias
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross entropy gradient
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(yb)), yb] = 1.0

        grad_logits = (probs - one_hot) / len(yb)
        grad_W = xb.T @ grad_logits
        grad_b = grad_logits.sum(axis=0, keepdims=True)

        step_count += 1

        # Adam updates
        mW = beta1 * mW + (1 - beta1) * grad_W
        vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
        mW_hat = mW / (1 - beta1 ** step_count)
        vW_hat = vW / (1 - beta2 ** step_count)
        weights -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        mb = beta1 * mb + (1 - beta1) * grad_b
        vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
        mb_hat = mb / (1 - beta1 ** step_count)
        vb_hat = vb / (1 - beta2 ** step_count)
        bias -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    return weights, bias, mW, vW, mb, vb, step_count
