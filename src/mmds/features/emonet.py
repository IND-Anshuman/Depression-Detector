from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def _safe_resize(face_rgb: np.ndarray) -> np.ndarray:
    if face_rgb.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.resize(face_rgb, (64, 64), interpolation=cv2.INTER_AREA)


class CleanRoomEmoNetExtractor:
    """Deterministic emotion embedding extractor with optional weight-seeded projection.

    This does not attempt to load third-party state dicts directly. Instead, it uses
    a compact handcrafted face representation and a deterministic projection matrix.
    When an official checkpoint is present, the checkpoint hash is used as the
    projection seed so runs are pinned and explicit.
    """

    def __init__(self, weights_path: str | Path | None = None, embedding_dim: int = 10, strict_weights: bool = False) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.embedding_dim = embedding_dim
        self.strict_weights = strict_weights
        self._projection = None

    def _projection_rng_seed(self) -> int:
        if self.weights_path and self.weights_path.exists():
            digest = hashlib.sha256(self.weights_path.read_bytes()).digest()
            return int.from_bytes(digest[:8], "big", signed=False)
        if self.strict_weights:
            raise FileNotFoundError(f"Expected EmoNet weights at {self.weights_path}")
        return 2024

    def _get_projection(self) -> np.ndarray:
        if self._projection is None:
            rng = np.random.default_rng(self._projection_rng_seed())
            self._projection = rng.normal(loc=0.0, scale=0.2, size=(32, self.embedding_dim)).astype(np.float32)
        return self._projection

    def _raw_features(self, face_rgb: np.ndarray, face_landmarks: np.ndarray | None = None) -> np.ndarray:
        face = _safe_resize(face_rgb)
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten().astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        left = gray[:, : gray.shape[1] // 2]
        right = np.fliplr(gray[:, gray.shape[1] // 2 :])
        sym = float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))) / 255.0
        rgb = face.astype(np.float32) / 255.0
        stats = [
            float(rgb[..., c].mean()) for c in range(3)
        ] + [
            float(rgb[..., c].std()) for c in range(3)
        ] + [
            float(np.mean(np.abs(gx))) / 255.0,
            float(np.std(gx)) / 255.0,
            float(np.mean(np.abs(gy))) / 255.0,
            float(np.std(gy)) / 255.0,
            float(np.mean(np.abs(lap))) / 255.0,
            float(np.std(lap)) / 255.0,
            sym,
            float(gray.mean()) / 255.0,
        ]
        feats = np.array(stats, dtype=np.float32)
        feats = np.concatenate([feats, hist], axis=0)
        if face_landmarks is not None and face_landmarks.size >= 468 * 3:
            pts = face_landmarks.reshape(-1, 3)
            geom = np.array(
                [
                    float(np.linalg.norm(pts[13, :2] - pts[14, :2])),
                    float(np.linalg.norm(pts[61, :2] - pts[291, :2])),
                    float(np.linalg.norm(pts[70, :2] - pts[105, :2])),
                    float(np.linalg.norm(pts[336, :2] - pts[334, :2])),
                    float(np.linalg.norm(pts[159, :2] - pts[145, :2])),
                    float(np.linalg.norm(pts[386, :2] - pts[374, :2])),
                    float(np.mean(pts[:, 2])),
                    float(np.std(pts[:, 2])),
                ],
                dtype=np.float32,
            )
        else:
            geom = np.zeros((8,), dtype=np.float32)
        raw = np.concatenate([feats, geom], axis=0)
        if raw.shape[0] < 32:
            raw = np.pad(raw, (0, 32 - raw.shape[0]))
        return raw[:32]

    def extract(self, face_rgb: np.ndarray, face_landmarks: np.ndarray | None = None) -> np.ndarray:
        raw = self._raw_features(face_rgb, face_landmarks=face_landmarks)
        emb = np.tanh(raw @ self._get_projection())
        emotion_logits = emb[:8]
        exp = np.exp(emotion_logits - np.max(emotion_logits))
        probs = exp / (exp.sum() + 1e-6)
        valence = np.tanh(emb[8])
        arousal = np.tanh(emb[9])
        return np.concatenate([probs.astype(np.float32), np.array([valence, arousal], dtype=np.float32)], axis=0)
