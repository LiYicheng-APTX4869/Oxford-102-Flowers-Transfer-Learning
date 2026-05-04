from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def accuracy_from_logits(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def confusion_matrix_from_arrays(
    targets: np.ndarray, preds: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, preds):
        matrix[int(target), int(pred)] += 1
    return matrix


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    scores = []
    for idx in range(matrix.shape[0]):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def per_class_accuracy(matrix: np.ndarray) -> list[float]:
    values: list[float] = []
    for idx in range(matrix.shape[0]):
        total = matrix[idx, :].sum()
        values.append(float(matrix[idx, idx] / total) if total > 0 else 0.0)
    return values


def summarize_class_errors(
    matrix: np.ndarray, class_names: list[str] | None = None, top_k: int = 10
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j or matrix[i, j] == 0:
                continue
            errors.append(
                {
                    "true_idx": i,
                    "pred_idx": j,
                    "true_name": class_names[i] if class_names else str(i),
                    "pred_name": class_names[j] if class_names else str(j),
                    "count": int(matrix[i, j]),
                }
            )
    errors.sort(key=lambda item: item["count"], reverse=True)
    return errors[:top_k]


def save_confusion_heatmap(
    matrix: np.ndarray, output_path: str | Path, title: str = "Confusion Matrix"
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
