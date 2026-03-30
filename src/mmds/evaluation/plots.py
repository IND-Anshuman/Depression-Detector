from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    try:
        from sklearn.metrics import RocCurveDisplay

        RocCurveDisplay.from_predictions(y_true, y_prob)
        plt.title("Binary risk ROC")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    except Exception:
        # best-effort
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.title("ROC unavailable")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def save_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, bins: int = 10) -> None:
    # Reliability diagram
    y_true = y_true.astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, bins + 1)
    binids = np.digitize(y_prob, edges) - 1

    acc = []
    conf = []
    for b in range(bins):
        m = binids == b
        if not m.any():
            continue
        acc.append(float(y_true[m].mean()))
        conf.append(float(y_prob[m].mean()))

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(conf, acc)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Calibration (reliability)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
