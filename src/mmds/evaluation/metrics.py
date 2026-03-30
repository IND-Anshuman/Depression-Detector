from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalMetrics:
    auroc: float | None
    f1: float | None
    precision: float | None
    recall: float | None
    accuracy: float | None

    mae: float | None
    rmse: float | None

    severity_accuracy: float | None


def compute_metrics(
    y_true_bin: np.ndarray,
    y_prob_bin: np.ndarray,
    y_true_ord: np.ndarray,
    y_pred_ord: np.ndarray,
    y_true_cont: np.ndarray,
    y_pred_cont: np.ndarray,
    m_bin: np.ndarray,
    m_ord: np.ndarray,
    m_cont: np.ndarray,
) -> EvalMetrics:
    auroc = None
    f1 = precision = recall = accuracy = None
    if m_bin.any():
        yt = y_true_bin[m_bin]
        yp = y_prob_bin[m_bin]
        yhat = (yp >= 0.5).astype(int)
        try:
            auroc = float(roc_auc_score(yt, yp))
        except Exception:
            auroc = None
        f1 = float(f1_score(yt, yhat))
        precision = float(precision_score(yt, yhat, zero_division=0))
        recall = float(recall_score(yt, yhat, zero_division=0))
        accuracy = float(accuracy_score(yt, yhat))

    mae = rmse = None
    if m_cont.any():
        yt = y_true_cont[m_cont]
        yp = y_pred_cont[m_cont]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(mean_squared_error(yt, yp, squared=False))

    sev_acc = None
    if m_ord.any():
        sev_acc = float(accuracy_score(y_true_ord[m_ord], y_pred_ord[m_ord]))

    return EvalMetrics(
        auroc=auroc,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        mae=mae,
        rmse=rmse,
        severity_accuracy=sev_acc,
    )


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, m: np.ndarray, num_classes: int) -> np.ndarray:
    if not m.any():
        return np.zeros((num_classes, num_classes), dtype=int)
    return confusion_matrix(y_true[m], y_pred[m], labels=list(range(num_classes)))
