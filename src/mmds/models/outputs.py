from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOutputs:
    binary_logit: torch.Tensor  # (B,)
    binary_prob: torch.Tensor  # (B,)

    ordinal_logits: torch.Tensor  # (B,2) for legacy CORAL thresholds or (B,3) direct severity logits
    severity_probs: torch.Tensor  # (B,3)

    continuous_mean: torch.Tensor  # (B,)
    continuous_log_var: torch.Tensor | None  # (B,)

    bdd_pred: torch.Tensor | None  # (B,)

    # Interpretability / debugging
    token_importance: torch.Tensor | None  # (B,N)
    token_modality: list[str] | None
    token_time_index: torch.Tensor | None  # (N,) global token->time mapping
