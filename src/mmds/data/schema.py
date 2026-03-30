from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

ModalityName = Literal[
    "audio",
    "face_au",
    "face_landmarks",
    "head_pose",
    "gaze",
    "blink",
    "body_pose",
    "hand_pose",
    "emotion",
    "quality",
]


@dataclass(frozen=True)
class Sample:
    """Unified sample schema.

    Notes:
    - Labels are optional to support partially labeled datasets.
    - modality_payloads holds per-window arrays (T, D) or per-window scalars.
    - modality_masks indicates which modalities are present.
    """

    sample_id: str
    subject_id: str
    dataset_name: str

    binary_label: int | None = None
    ordinal_label: int | None = None
    continuous_score: float | None = None

    window_index: int | None = None
    timestamps_s: np.ndarray | None = None

    modality_payloads: dict[str, np.ndarray] = None  # type: ignore[assignment]
    modality_masks: dict[str, bool] = None  # type: ignore[assignment]
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality_payloads", self.modality_payloads or {})
        object.__setattr__(self, "modality_masks", self.modality_masks or {})
        object.__setattr__(self, "metadata", self.metadata or {})


@dataclass(frozen=True)
class ModalityBatch:
    """Collated batch returned by DataLoader."""

    sample_ids: list[str]
    subject_ids: list[str]
    dataset_names: list[str]

    x: dict[str, torch.Tensor]
    x_mask: dict[str, torch.Tensor]

    y_binary: torch.Tensor
    y_ordinal: torch.Tensor
    y_continuous: torch.Tensor

    y_mask_binary: torch.Tensor
    y_mask_ordinal: torch.Tensor
    y_mask_continuous: torch.Tensor
