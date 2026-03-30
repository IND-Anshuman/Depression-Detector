from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class VizBundle:
    rolling_score_fig: np.ndarray
    au_trend_fig: np.ndarray
    importance_fig: np.ndarray


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()

    # Matplotlib's canvas API differs across versions.
    # Prefer `buffer_rgba()` when available; fall back to legacy `tostring_rgb()`.
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba())
        arr = np.ascontiguousarray(rgba[..., :3])
    else:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        arr = buf.reshape((h, w, 3))
    plt.close(fig)
    return arr


def plot_rolling_score(values: list[float], title: str) -> np.ndarray:
    fig = plt.figure(figsize=(5, 2.5), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(values, linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    return _fig_to_rgb(fig)


def plot_au_trend(face_au_ts: np.ndarray | None) -> np.ndarray:
    fig = plt.figure(figsize=(5, 2.5), dpi=150)
    ax = fig.add_subplot(111)
    if face_au_ts is None or face_au_ts.size == 0:
        ax.text(0.5, 0.5, "AU trend unavailable", ha="center", va="center")
    else:
        # Plot mean AU activity over time.
        x = np.asarray(face_au_ts)
        if x.ndim == 1:
            x = x[:, None]
        y = x.mean(axis=1)
        ax.plot(y, linewidth=2)
        ax.set_ylim(float(y.min()) - 0.1, float(y.max()) + 0.1)
        ax.set_title("Proxy AU activity (mean)")
        ax.set_xlabel("t")
        ax.set_ylabel("activity")
        ax.grid(True, alpha=0.3)
    return _fig_to_rgb(fig)


def plot_importance(time_importance: np.ndarray | None) -> np.ndarray:
    fig = plt.figure(figsize=(5, 2.5), dpi=150)
    ax = fig.add_subplot(111)
    if time_importance is None or time_importance.size == 0:
        ax.text(0.5, 0.5, "Importance unavailable", ha="center", va="center")
    else:
        ax.imshow(time_importance[None, :], aspect="auto", cmap="magma")
        ax.set_yticks([])
        ax.set_title("Temporal importance (token proxy)")
        ax.set_xlabel("t")
    return _fig_to_rgb(fig)
