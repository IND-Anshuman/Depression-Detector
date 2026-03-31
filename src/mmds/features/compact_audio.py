from __future__ import annotations

import numpy as np


def _resample_feature_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D audio feature array, received shape {array.shape}")
    if array.shape[1] == target_dim:
        return array.astype(np.float32, copy=False)
    src_x = np.linspace(0.0, 1.0, num=array.shape[1], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    out = np.empty((array.shape[0], target_dim), dtype=np.float32)
    for idx, row in enumerate(array):
        out[idx] = np.interp(dst_x, src_x, row.astype(np.float32, copy=False))
    return out


def build_compact_audio_features(audio_ts: np.ndarray, target_dim: int = 25) -> np.ndarray:
    """Normalize arbitrary per-frame audio features into a shared compact schema.

    This keeps D-Vlog, LMVD, and live microphone MFCCs aligned to a single
    modality width so one checkpoint can consume all three sources.
    """

    arr = np.asarray(audio_ts, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D audio feature array, received shape {arr.shape}")

    arr = _resample_feature_dim(arr, target_dim=target_dim)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-5, 1.0, std)
    return ((arr - mean) / std).astype(np.float32)
