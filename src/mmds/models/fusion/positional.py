from __future__ import annotations

import math

import torch


def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """(length, dim) classic sinusoidal positional encoding."""

    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    position = torch.arange(0, length, device=device, dtype=torch.float32)[:, None]
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
