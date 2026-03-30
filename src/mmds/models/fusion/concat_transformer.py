from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConcatTransformerConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float


class ConcatTransformerFusion(nn.Module):
    """Baseline fusion: transformer encoder over concatenated tokens."""

    def __init__(self, cfg: ConcatTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_size * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TransformerEncoder uses src_key_padding_mask with True as padding.
        x = self.encoder(tokens, src_key_padding_mask=~token_mask)
        # Masked mean pool.
        m = token_mask.to(dtype=x.dtype)
        pooled = (x * m[:, :, None]).sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-6)
        # Importance proxy: token norm weighted by mask.
        importance = torch.linalg.vector_norm(x, dim=-1) * m
        return pooled, importance
