from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class FusionExtractConfig:
    hidden_size: int
    dropout: float
    num_heads: int


class FusionExtract(nn.Module):
    """TSFFM-like local interaction block for face/body streams."""

    def __init__(self, cfg: FusionExtractConfig) -> None:
        super().__init__()
        self.face_conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1, groups=cfg.hidden_size)
        self.body_conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1, groups=cfg.hidden_size)
        self.face_to_body = nn.MultiheadAttention(cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.body_to_face = nn.MultiheadAttention(cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * 4, cfg.hidden_size),
        )

    def forward(self, face, face_mask, body, body_mask):
        face_local = self.face_conv(face.transpose(1, 2)).transpose(1, 2)
        body_local = self.body_conv(body.transpose(1, 2)).transpose(1, 2)
        face_cross, _ = self.face_to_body(face_local, body_local, body_local, key_padding_mask=~body_mask)
        body_cross, _ = self.body_to_face(body_local, face_local, face_local, key_padding_mask=~face_mask)
        face_out = face + face_cross
        body_out = body + body_cross
        return face_out + self.ffn(face_out), body_out + self.ffn(body_out)
