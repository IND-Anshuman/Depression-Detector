from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    hidden_size: int
    dropout: float


class ModalityEncoder(nn.Module):
    """Project modality-specific features to shared hidden size."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.LazyLinear(cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        b, t, d = x.shape
        y = self.proj(x.reshape(b * t, d)).reshape(b, t, self.cfg.hidden_size)
        return y
