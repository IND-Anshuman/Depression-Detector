from __future__ import annotations

import torch
from torch import nn


class OrdinalCoralHead(nn.Module):
    """CORAL-style ordinal head for 3-class severity.

    Produces K-1 logits (here 2) corresponding to P(y > k).
    """

    def __init__(self, hidden_size: int, num_classes: int = 3) -> None:
        super().__init__()
        if num_classes != 3:
            raise ValueError("This implementation is fixed to 3 classes for simplicity")
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z)  # (B,2)
        p_gt = torch.sigmoid(logits)  # P(y > k)
        # Convert to class probs:
        # P(y=0)=1-p(y>0)
        # P(y=1)=p(y>0)-p(y>1)
        # P(y=2)=p(y>1)
        p0 = 1.0 - p_gt[:, 0]
        p1 = p_gt[:, 0] - p_gt[:, 1]
        p2 = p_gt[:, 1]
        probs = torch.stack([p0, p1, p2], dim=-1).clamp_min(0.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
        return logits, probs
