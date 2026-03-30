from __future__ import annotations

from typing import Any

import numpy as np

from .base import ExtractionResult, FeatureExtractor
from .simple_backend import SimpleExtractor


class MediaPipeExtractor(FeatureExtractor):
    """Optional MediaPipe backend.

    If MediaPipe isn't installed, this backend falls back to `SimpleExtractor` while
    emitting a warning.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._fallback = SimpleExtractor(cfg)
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._available = True
        except Exception:
            self._mp = None
            self._available = False

    @property
    def modality_dims(self) -> dict[str, int]:
        # Reuse simple dims for the first implementation.
        return self._fallback.modality_dims

    def extract_window(
        self,
        frames_bgr: list[np.ndarray],
        fps: float,
        audio_f32: np.ndarray | None,
        audio_sr: int | None,
    ) -> ExtractionResult:
        if not self._available:
            res = self._fallback.extract_window(frames_bgr, fps, audio_f32, audio_sr)
            return ExtractionResult(
                res.modality_payloads,
                res.modality_masks,
                ["mediapipe_not_installed"] + res.quality_warnings,
                res.debug,
            )

        # For now, we still rely on the simple backend features. The swap point is here:
        # implement FaceMesh + Pose landmarks and derive head/gaze/blink/body pose features.
        res = self._fallback.extract_window(frames_bgr, fps, audio_f32, audio_sr)
        return ExtractionResult(
            res.modality_payloads,
            res.modality_masks,
            ["mediapipe_backend_stub"] + res.quality_warnings,
            res.debug,
        )
