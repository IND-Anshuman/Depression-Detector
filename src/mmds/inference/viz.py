from __future__ import annotations

import math

import cv2
import numpy as np


BG = (18, 24, 33)
FG = (236, 240, 245)
GRID = (58, 72, 89)
RED = (242, 100, 100)
TEAL = (78, 205, 196)
GOLD = (255, 196, 102)


def _canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), BG, dtype=np.uint8)


def _put(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55, color: tuple[int, int, int] = FG, thickness: int = 1) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_grid(img: np.ndarray, left: int, top: int, right: int, bottom: int, rows: int = 4) -> None:
    cv2.rectangle(img, (left, top), (right, bottom), GRID, 1, cv2.LINE_AA)
    for idx in range(1, rows):
        y = int(top + (bottom - top) * idx / rows)
        cv2.line(img, (left, y), (right, y), GRID, 1, cv2.LINE_AA)


def plot_rolling_score(values: list[float], title: str) -> np.ndarray:
    img = _canvas(750, 375)
    left, top, right, bottom = 72, 52, 708, 320
    _put(img, title, (left, 32), scale=0.7, thickness=2)
    _draw_grid(img, left, top, right, bottom)
    if not values:
        _put(img, "Waiting for predictions", (220, 205), scale=0.65)
        return img

    pts = []
    for idx, value in enumerate(values):
        x = int(left + (right - left) * idx / max(len(values) - 1, 1))
        y = int(bottom - np.clip(value, 0.0, 1.0) * (bottom - top))
        pts.append((x, y))
    poly = np.array(pts, dtype=np.int32)
    fill = np.vstack([poly, [right, bottom], [left, bottom]])
    overlay = img.copy()
    cv2.fillPoly(overlay, [fill], RED)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0.0)
    cv2.polylines(img, [poly], False, RED, 3, cv2.LINE_AA)
    for tick in [0.0, 0.5, 1.0]:
        y = int(bottom - tick * (bottom - top))
        _put(img, f"{tick:.1f}", (24, y + 5), scale=0.45, color=(177, 187, 198))
    return img


def plot_au_trend(face_au_ts: np.ndarray | None) -> np.ndarray:
    img = _canvas(750, 375)
    left, top, right, bottom = 72, 52, 708, 320
    _put(img, "Facial activity trend", (left, 32), scale=0.7, thickness=2)
    _draw_grid(img, left, top, right, bottom)
    if face_au_ts is None or np.asarray(face_au_ts).size == 0:
        _put(img, "Facial trend unavailable", (220, 205), scale=0.65)
        return img

    x = np.asarray(face_au_ts, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    y = x.mean(axis=1)
    y_min = float(y.min())
    y_max = float(y.max())
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pts = []
    for idx, value in enumerate(y):
        xx = int(left + (right - left) * idx / max(len(y) - 1, 1))
        yy = int(bottom - ((float(value) - y_min) / (y_max - y_min)) * (bottom - top))
        pts.append((xx, yy))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, TEAL, 3, cv2.LINE_AA)
    _put(img, f"min {y_min:.2f}", (left, bottom + 28), scale=0.45, color=(177, 187, 198))
    _put(img, f"max {y_max:.2f}", (right - 110, bottom + 28), scale=0.45, color=(177, 187, 198))
    return img


def plot_importance(time_importance: np.ndarray | None) -> np.ndarray:
    img = _canvas(750, 375)
    _put(img, "Temporal importance", (72, 32), scale=0.7, thickness=2)
    if time_importance is None or np.asarray(time_importance).size == 0:
        _put(img, "Importance unavailable", (250, 205), scale=0.65)
        return img

    arr = np.asarray(time_importance, dtype=np.float32).reshape(1, -1)
    arr = arr - arr.min()
    arr = arr / max(float(arr.max()), 1.0e-6)
    heat = cv2.applyColorMap((arr * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (636, 110), interpolation=cv2.INTER_LINEAR)
    img[130:240, 72:708] = heat
    cv2.rectangle(img, (72, 130), (708, 240), GRID, 1, cv2.LINE_AA)
    return img


def plot_risk_gauge(prob: float) -> np.ndarray:
    img = _canvas(675, 420)
    center = (337, 280)
    radius = 148
    cv2.ellipse(img, center, (radius, radius), 0, 180, 360, GRID, 18, cv2.LINE_AA)
    segments = [
        (180, 240, (102, 196, 123)),
        (240, 300, GOLD),
        (300, 360, RED),
    ]
    for start, end, color in segments:
        cv2.ellipse(img, center, (radius, radius), 0, start, end, color, 18, cv2.LINE_AA)
    angle = math.pi * (1.0 - float(np.clip(prob, 0.0, 1.0)))
    tip = (
        int(center[0] + math.cos(angle) * (radius - 12)),
        int(center[1] - math.sin(angle) * (radius - 12)),
    )
    cv2.line(img, center, tip, FG, 6, cv2.LINE_AA)
    cv2.circle(img, center, 12, FG, -1, cv2.LINE_AA)
    _put(img, "Live risk", (252, 66), scale=0.9, thickness=2)
    _put(img, f"{prob:.2f}", (285, 360), scale=1.4, thickness=3)
    return img


def plot_probability_bars(values: list[float], labels: list[str], title: str, color: str = "#f26464") -> np.ndarray:
    img = _canvas(750, 420)
    left, right = 210, 690
    bar_h = 34
    top = 82
    rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))
    _put(img, title, (48, 42), scale=0.78, thickness=2)
    if not values:
        _put(img, "No values", (300, 220), scale=0.65)
        return img

    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * 72
        cv2.rectangle(img, (left, y), (right, y + bar_h), GRID, 1, cv2.LINE_AA)
        fill_right = int(left + np.clip(value, 0.0, 1.0) * (right - left))
        cv2.rectangle(img, (left, y), (fill_right, y + bar_h), rgb, -1, cv2.LINE_AA)
        _put(img, label, (48, y + 24), scale=0.58)
        _put(img, f"{float(value):.2f}", (right + 12, y + 24), scale=0.56, color=(177, 187, 198))
    return img


def plot_attention_heatmap(modality_time: np.ndarray | None, modality_labels: list[str] | None) -> np.ndarray:
    img = _canvas(780, 420)
    _put(img, "Attention heatmap", (42, 38), scale=0.78, thickness=2)
    if modality_time is None or np.asarray(modality_time).size == 0 or not modality_labels:
        _put(img, "Attention heatmap unavailable", (220, 220), scale=0.65)
        return img

    arr = np.asarray(modality_time, dtype=np.float32)
    arr = arr - arr.min()
    arr = arr / max(float(arr.max()), 1.0e-6)
    heat = cv2.applyColorMap((arr * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (620, 260), interpolation=cv2.INTER_LINEAR)
    top, left = 92, 138
    img[top : top + 260, left : left + 620] = heat
    cv2.rectangle(img, (left, top), (left + 620, top + 260), GRID, 1, cv2.LINE_AA)
    for idx, label in enumerate(modality_labels):
        y = top + int((idx + 0.6) * 260 / max(len(modality_labels), 1))
        _put(img, label, (24, y), scale=0.5)
    return img
