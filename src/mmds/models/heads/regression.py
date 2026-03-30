from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """Predict a continuous severity score and an aleatoric log-variance."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden_size, 1)
        self.log_var = nn.Linear(hidden_size, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.sigmoid(self.mean(z).squeeze(-1))
        log_var = self.log_var(z).squeeze(-1).clamp(-6.0, 3.0)
        return mean, log_var
