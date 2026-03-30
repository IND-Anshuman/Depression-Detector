from __future__ import annotations

import numpy as np


def summarize_quality(quality_ts: np.ndarray) -> dict[str, float]:
    if quality_ts.size == 0:
        return {"brightness": 0.0, "blur": 0.0, "motion": 0.0}
    q = np.asarray(quality_ts)
    if q.ndim != 2 or q.shape[1] < 3:
        return {"brightness": float(q.mean()), "blur": 0.0, "motion": 0.0}
    return {
        "brightness": float(q[:, 0].mean()),
        "blur": float(q[:, 1].mean()),
        "motion": float(q[:, 2].mean()),
    }
