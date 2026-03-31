from __future__ import annotations

import numpy as np


def _series_energy_hist(x: np.ndarray | None, bins: int = 16) -> np.ndarray:
    if x is None:
        return np.zeros((bins,), dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((bins,), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    mag = np.linalg.norm(arr, axis=1)
    if np.allclose(mag.max(), mag.min()):
        hist = np.zeros((bins,), dtype=np.float64)
        hist[0] = 1.0
        return hist
    hist, _ = np.histogram(mag, bins=bins, range=(float(mag.min()), float(mag.max()) + 1e-6))
    return hist.astype(np.float64)


def _entropy(hist: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(hist.astype(np.float64), eps, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def expression_entropy(face_ts: np.ndarray | None, bins: int = 16) -> float:
    return _entropy(_series_energy_hist(face_ts, bins=bins)) / float(np.log(bins))


def action_entropy(body_ts: np.ndarray | None, bins: int = 16) -> float:
    return _entropy(_series_energy_hist(body_ts, bins=bins)) / float(np.log(bins))


def bdd_score(face_ts: np.ndarray | None, body_ts: np.ndarray | None, w_expr: float = 0.6, w_action: float = 0.4) -> float:
    expr = float(np.clip(expression_entropy(face_ts), 0.0, 1.0))
    action = float(np.clip(action_entropy(body_ts), 0.0, 1.0))
    return float(np.clip(w_expr * expr + w_action * action, 0.0, 1.0))
