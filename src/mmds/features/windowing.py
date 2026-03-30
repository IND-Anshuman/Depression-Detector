from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowingConfig:
    window_seconds: float
    fps: float
    audio_sr: int


def slice_windows_frames(frames: list[np.ndarray], cfg: WindowingConfig) -> list[list[np.ndarray]]:
    if not frames:
        return []
    w = int(round(cfg.window_seconds * cfg.fps))
    w = max(w, 1)
    out = []
    for i in range(0, len(frames), w):
        out.append(frames[i : i + w])
    return out


def slice_windows_audio(audio: np.ndarray, cfg: WindowingConfig) -> list[np.ndarray]:
    if audio.size == 0:
        return []
    w = int(round(cfg.window_seconds * cfg.audio_sr))
    w = max(w, 1)
    out = []
    for i in range(0, audio.size, w):
        out.append(audio[i : i + w])
    return out
