from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mmds.models.outputs import ModelOutputs


@dataclass(frozen=True)
class LossWeights:
    binary: float
    ordinal: float
    continuous: float
    bdd: float


def coral_targets(y: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Convert ordinal class labels (0..K-1) to CORAL binary targets (K-1)."""

    if num_classes != 3:
        raise ValueError("This implementation is fixed to 3 classes")
    # thresholds: y > 0 and y > 1
    t0 = (y > 0).to(dtype=torch.float32)
    t1 = (y > 1).to(dtype=torch.float32)
    return torch.stack([t0, t1], dim=-1)


class MultitaskLoss(nn.Module):
    def __init__(self, weights: LossWeights, binary_pos_weight: float | None = None) -> None:
        super().__init__()
        self.w = weights
        self.binary_pos_weight = binary_pos_weight

    def forward(
        self,
        out: ModelOutputs,
        y_binary: torch.Tensor,
        y_ordinal: torch.Tensor,
        y_cont: torch.Tensor,
        m_binary: torch.Tensor,
        m_ordinal: torch.Tensor,
        m_cont: torch.Tensor,
        y_bdd: torch.Tensor | None = None,
        m_bdd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses: dict[str, torch.Tensor] = {}

        # Binary
        if m_binary.any():
            pos_weight = None
            if self.binary_pos_weight is not None:
                pos_weight = torch.tensor(float(self.binary_pos_weight), dtype=torch.float32, device=out.binary_logit.device)
            bce = F.binary_cross_entropy_with_logits(
                out.binary_logit[m_binary],
                y_binary[m_binary].to(dtype=torch.float32),
                pos_weight=pos_weight,
            )
            losses["binary"] = bce
        else:
            losses["binary"] = out.binary_logit.mean() * 0.0

        # Severity classification: support either legacy CORAL thresholds or direct 3-way logits.
        if m_ordinal.any():
            logits = out.ordinal_logits[m_ordinal]
            if logits.shape[-1] == 2:
                t = coral_targets(y_ordinal[m_ordinal], num_classes=3)
                losses["ordinal"] = F.binary_cross_entropy_with_logits(logits, t)
            else:
                losses["ordinal"] = F.cross_entropy(logits, y_ordinal[m_ordinal].long())
        else:
            losses["ordinal"] = out.ordinal_logits.mean() * 0.0

        # Continuous regression with aleatoric log_var (legacy) or MAE-driven hybrid objective.
        if m_cont.any():
            mu = out.continuous_mean[m_cont]
            if out.continuous_log_var is not None:
                lv = out.continuous_log_var[m_cont]
                var = torch.exp(lv)
                mae = torch.abs(y_cont[m_cont] - mu)
                nll = 0.5 * (torch.log(var + 1e-6) + (y_cont[m_cont] - mu) ** 2 / (var + 1e-6))
                losses["continuous"] = 0.5 * nll.mean() + 0.5 * mae.mean()
            else:
                losses["continuous"] = F.l1_loss(mu, y_cont[m_cont])
        else:
            losses["continuous"] = out.continuous_mean.mean() * 0.0

        # Optional BDD auxiliary
        if out.bdd_pred is not None and y_bdd is not None and m_bdd is not None and m_bdd.any():
            losses["bdd"] = F.mse_loss(out.bdd_pred[m_bdd], y_bdd[m_bdd])
        else:
            losses["bdd"] = out.binary_logit.mean() * 0.0

        total = (
            self.w.binary * losses["binary"]
            + self.w.ordinal * losses["ordinal"]
            + self.w.continuous * losses["continuous"]
            + self.w.bdd * losses["bdd"]
        )

        scalars = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
        scalars["total"] = float(total.detach().cpu().item())
        return total, scalars
