from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .encoders.modality_encoder import EncoderConfig, ModalityEncoder
from .fusion.concat_transformer import ConcatTransformerConfig, ConcatTransformerFusion
from .fusion.face_body import FaceBodyFusionBlock, FaceBodyFusionConfig
from .fusion.perceiver import PerceiverConfig, PerceiverFusion
from .fusion.positional import sinusoidal_positional_encoding
from .heads.bdd import BDDHead
from .heads.binary import BinaryHead
from .heads.ordinal import OrdinalCoralHead
from .heads.regression import RegressionHead
from .outputs import ModelOutputs


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    dropout: float
    num_latents: int
    num_layers: int
    num_heads: int
    fusion: str
    use_bdd_head: bool = True


class MMDSModel(nn.Module):
    """Main multimodal backbone + multitask heads.

    Design goals:
    - modular encoders
    - missing modality support
    - interpretable token importance export
    - buffered-inference friendly
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoders = nn.ModuleDict()
        self.modality_embed = nn.Embedding(32, cfg.hidden_size)
        self.modality_to_id: dict[str, int] = {}

        self.face_body = FaceBodyFusionBlock(
            FaceBodyFusionConfig(hidden_size=cfg.hidden_size, dropout=cfg.dropout, num_heads=cfg.num_heads)
        )

        if cfg.fusion == "perceiver":
            self.fusion = PerceiverFusion(
                PerceiverConfig(
                    hidden_size=cfg.hidden_size,
                    num_latents=cfg.num_latents,
                    num_layers=cfg.num_layers,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                )
            )
        elif cfg.fusion == "concat_transformer":
            self.fusion = ConcatTransformerFusion(
                ConcatTransformerConfig(
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                )
            )
        else:
            raise ValueError(f"Unknown fusion: {cfg.fusion}")

        self.binary_head = BinaryHead(cfg.hidden_size)
        self.ordinal_head = OrdinalCoralHead(cfg.hidden_size, num_classes=3)
        self.regression_head = RegressionHead(cfg.hidden_size)
        self.bdd_head = BDDHead(cfg.hidden_size) if cfg.use_bdd_head else None

        self.dropout = nn.Dropout(cfg.dropout)

    def _get_encoder(self, modality: str) -> ModalityEncoder:
        if modality not in self.encoders:
            self.encoders[modality] = ModalityEncoder(
                EncoderConfig(hidden_size=self.cfg.hidden_size, dropout=self.cfg.dropout)
            )
        return self.encoders[modality]

    def _modality_id(self, modality: str) -> int:
        if modality not in self.modality_to_id:
            self.modality_to_id[modality] = len(self.modality_to_id)
        return self.modality_to_id[modality]

    def forward(self, x: dict[str, torch.Tensor], x_mask: dict[str, torch.Tensor]) -> ModelOutputs:
        device = next(self.parameters()).device

        tokens_l: list[torch.Tensor] = []
        masks_l: list[torch.Tensor] = []
        token_modality: list[str] = []
        token_time_idx: list[torch.Tensor] = []

        # Optional TSFFM-inspired face-body fusion if both present.
        face = x.get("face_au")
        body = x.get("body_pose")
        if face is not None and body is not None:
            fm = x_mask.get("face_au")
            bm = x_mask.get("body_pose")
            if fm is not None and bm is not None:
                f_enc = self._get_encoder("face_au")(face)
                b_enc = self._get_encoder("body_pose")(body)
                f_fused, b_fused = self.face_body(f_enc, fm, b_enc, bm)
                x = dict(x)
                x_mask = dict(x_mask)
                x["face_au"] = f_fused
                x["body_pose"] = b_fused

        for modality in sorted(x.keys()):
            x_mod = x[modality].to(device)
            m_mod = x_mask[modality].to(device)

            if modality in ["face_au", "body_pose"] and x_mod.shape[-1] == self.cfg.hidden_size:
                enc = nn.Identity()
                emb = x_mod
            else:
                enc = self._get_encoder(modality)
                emb = enc(x_mod)

            emb = self.dropout(emb)
            b, t, h = emb.shape
            pos = sinusoidal_positional_encoding(t, h, device=device)[None, :, :]
            mid = self._modality_id(modality)
            mod_e = self.modality_embed(torch.tensor([mid], device=device)).view(1, 1, h)
            emb = emb + pos + mod_e

            tokens_l.append(emb)
            masks_l.append(m_mod)
            token_modality.extend([modality] * t)
            token_time_idx.append(torch.arange(t, device=device, dtype=torch.long))

        tokens = torch.cat(tokens_l, dim=1)  # (B,N,H)
        token_mask = torch.cat(masks_l, dim=1)  # (B,N)
        time_idx = torch.cat(token_time_idx, dim=0)  # (N,)

        pooled, importance = self.fusion(tokens, token_mask)

        b_logit, b_prob = self.binary_head(pooled)
        o_logits, o_probs = self.ordinal_head(pooled)
        c_mean, c_log_var = self.regression_head(pooled)
        bdd_pred = self.bdd_head(pooled) if self.bdd_head is not None else None

        return ModelOutputs(
            binary_logit=b_logit,
            binary_prob=b_prob,
            ordinal_logits=o_logits,
            severity_probs=o_probs,
            continuous_mean=c_mean,
            continuous_log_var=c_log_var,
            bdd_pred=bdd_pred,
            token_importance=importance,
            token_modality=token_modality,
            token_time_index=time_idx,
        )


def model_from_cfg(cfg: Any) -> MMDSModel:
    mc = ModelConfig(
        hidden_size=int(cfg.model.hidden_size),
        dropout=float(cfg.model.dropout),
        num_latents=int(cfg.model.num_latents),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        fusion=str(cfg.model.fusion),
        use_bdd_head=True,
    )
    return MMDSModel(mc)
