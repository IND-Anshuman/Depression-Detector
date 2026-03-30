from __future__ import annotations

import torch
from torch import nn


class BDDHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(z).squeeze(-1))
