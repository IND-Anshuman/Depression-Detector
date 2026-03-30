from __future__ import annotations

from typing import Any

import numpy as np

from .base import ExtractionResult, FeatureExtractor


class OpenFaceExtractor(FeatureExtractor):
    """OpenFace backend placeholder for offline research extraction.

    OpenFace is not a pip-installable pure-Python dependency; deployments typically use:
    - a Docker image with OpenFace binaries, or
    - a local OpenFace installation.

    This class makes the swap-point explicit while keeping the demo path lightweight.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @property
    def modality_dims(self) -> dict[str, int]:
        # Typical AU output dims vary; keep as placeholders.
        return {
            "face_au": 18,
            "face_landmarks": 68 * 2,
            "head_pose": 6,
            "gaze": 4,
            "blink": 2,
            "body_pose": 33 * 3,
            "quality": 8,
            "audio": 64,
        }

    def extract_window(
        self,
        frames_bgr: list[np.ndarray],
        fps: float,
        audio_f32: np.ndarray | None,
        audio_sr: int | None,
    ) -> ExtractionResult:
        raise RuntimeError(
            "OpenFace backend is not implemented in pure Python. "
            "Use the offline extraction script with an OpenFace-enabled environment, "
            "or switch features.backend to 'simple' or 'mediapipe'."
        )
