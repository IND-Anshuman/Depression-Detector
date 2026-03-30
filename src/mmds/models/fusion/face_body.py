from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FaceBodyFusionConfig:
    hidden_size: int
    dropout: float
    num_heads: int = 4


class FaceBodyFusionBlock(nn.Module):
    """TSFFM-inspired face+body fusion module.

    This is a *module* inside the larger architecture, not the whole system.
    It models local temporal patterns via lightweight temporal conv and cross-stream attention.
    """

    def __init__(self, cfg: FaceBodyFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Depthwise temporal conv for each stream.
        self.face_conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1, groups=cfg.hidden_size)
        self.body_conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1, groups=cfg.hidden_size)

        self.attn_f_to_b = nn.MultiheadAttention(
            cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.attn_b_to_f = nn.MultiheadAttention(
            cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * 4, cfg.hidden_size),
        )
        self.ln_face = nn.LayerNorm(cfg.hidden_size)
        self.ln_body = nn.LayerNorm(cfg.hidden_size)

    def forward(
        self,
        face: torch.Tensor,
        face_mask: torch.Tensor,
        body: torch.Tensor,
        body_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # face/body: (B,T,H); masks: (B,T) bool
        f = self.ln_face(face)
        b = self.ln_body(body)

        # temporal conv expects (B,H,T)
        f2 = self.face_conv(f.transpose(1, 2)).transpose(1, 2)
        b2 = self.body_conv(b.transpose(1, 2)).transpose(1, 2)

        # Cross attention with padding masks.
        face_kpm = ~face_mask
        body_kpm = ~body_mask

        f_attn, _ = self.attn_f_to_b(query=f2, key=b2, value=b2, key_padding_mask=body_kpm)
        b_attn, _ = self.attn_b_to_f(query=b2, key=f2, value=f2, key_padding_mask=face_kpm)

        f_out = face + f_attn
        b_out = body + b_attn

        f_out = f_out + self.ff(f_out)
        b_out = b_out + self.ff(b_out)

        return f_out, b_out
