from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PerceiverConfig:
    hidden_size: int
    num_latents: int
    num_layers: int
    num_heads: int
    dropout: float


class PerceiverFusion(nn.Module):
    """Perceiver-style latent fusion (cross-attention over multimodal tokens)."""

    def __init__(self, cfg: PerceiverConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(torch.randn(cfg.num_latents, cfg.hidden_size) * 0.02)

        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
                for _ in range(cfg.num_layers)
            ]
        )
        self.self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
                for _ in range(cfg.num_layers)
            ]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.hidden_size),
                    nn.Linear(cfg.hidden_size, cfg.hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_size * 4, cfg.hidden_size),
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.ln_lat = nn.LayerNorm(cfg.hidden_size)
        self.ln_tok = nn.LayerNorm(cfg.hidden_size)

    def forward(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse tokens.

        Args:
          tokens: (B,N,H)
          token_mask: (B,N) bool, True for valid

        Returns:
          pooled: (B,H)
          importance: (B,N) token importance (cross-attn aggregated)
        """

        b, n, h = tokens.shape
        lat = self.latents[None, :, :].expand(b, -1, -1)
        tok = self.ln_tok(tokens)
        kpm = ~token_mask

        importance_accum = torch.zeros((b, n), device=tokens.device, dtype=torch.float32)

        for i in range(self.cfg.num_layers):
            q = self.ln_lat(lat)
            lat2, attn = self.cross_attn[i](
                query=q,
                key=tok,
                value=tok,
                key_padding_mask=kpm,
                need_weights=True,
                average_attn_weights=False,
            )
            lat = lat + lat2

            # attn: (B, num_heads, L, N)
            if attn is not None:
                importance_accum = importance_accum + attn.mean(dim=1).mean(dim=1)

            lat3, _ = self.self_attn[i](query=lat, key=lat, value=lat, need_weights=False)
            lat = lat + lat3
            lat = lat + self.ff[i](lat)

        pooled = lat.mean(dim=1)
        importance = importance_accum / max(self.cfg.num_layers, 1)
        return pooled, importance
