from __future__ import annotations

import numpy as np


def _entropy_from_hist(p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p.astype(np.float64), eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _series_entropy(x: np.ndarray, bins: int = 16) -> float:
    """Compute entropy over time for a (T,D) feature series by binning magnitudes."""

    if x.size == 0:
        return 0.0
    x2 = np.asarray(x)
    if x2.ndim == 1:
        x2 = x2[:, None]
    mag = np.linalg.norm(x2, axis=1)
    hist, _ = np.histogram(mag, bins=bins, range=(float(mag.min()), float(mag.max()) + 1e-6))
    return _entropy_from_hist(hist)


def compute_behavioral_variability(
    face_au_ts: np.ndarray | None,
    body_ts: np.ndarray | None,
    w_face: float = 0.6,
    w_body: float = 0.4,
) -> float:
    """BDD-inspired *behavioral variability* score (auxiliary).

    This is intentionally separated from the depression screening heads.

    Inputs:
    - face_au_ts: (T,D) AU-like activity series (or proxy).
    - body_ts: (T,D) body-motion/pose series (or proxy).

    Returns:
    - variability score in [0, 1] (heuristic normalization).
    """

    e_face = _series_entropy(face_au_ts) if face_au_ts is not None else 0.0
    e_body = _series_entropy(body_ts) if body_ts is not None else 0.0

    # Normalize entropy to roughly [0,1] by dividing by log(bins).
    denom = np.log(16.0)
    s_face = float(np.clip(e_face / denom, 0.0, 1.0))
    s_body = float(np.clip(e_body / denom, 0.0, 1.0))

    score = float(np.clip(w_face * s_face + w_body * s_body, 0.0, 1.0))
    return score
