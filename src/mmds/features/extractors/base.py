from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExtractionResult:
    modality_payloads: dict[str, np.ndarray]
    modality_masks: dict[str, bool]
    quality_warnings: list[str]
    debug: dict[str, Any]


class FeatureExtractor(ABC):
    """Feature extractor interface.

    The extractor consumes buffered frames/audio and returns time-series features per modality.
    """

    @abstractmethod
    def extract_window(
        self,
        frames_bgr: list[np.ndarray],
        fps: float,
        audio_f32: np.ndarray | None,
        audio_sr: int | None,
    ) -> ExtractionResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def modality_dims(self) -> dict[str, int]:
        """Feature dims for each modality emitted by this backend."""

        raise NotImplementedError
