from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridFusionConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    lstm_hidden_size: int
    attention_dim: int


class HybridFusion(nn.Module):
    """Transformer encoder followed by Bi-LSTM and self-attention pooling."""

    def __init__(self, cfg: HybridFusionConfig) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_size * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(cfg.lstm_hidden_size * 2, cfg.hidden_size)
        self.attn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.attention_dim),
            nn.Tanh(),
            nn.Linear(cfg.attention_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.transformer(tokens, src_key_padding_mask=~token_mask)
        lstm_out, _ = self.lstm(x)
        h = self.proj(lstm_out)
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(~token_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        importance = weights * token_mask.to(dtype=weights.dtype)
        return pooled, importance
