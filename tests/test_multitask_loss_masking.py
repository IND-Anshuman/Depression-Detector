from __future__ import annotations

import torch

from mmds.models.outputs import ModelOutputs
from mmds.training.losses import LossWeights, MultitaskLoss


def test_multitask_loss_respects_label_masks() -> None:
    out = ModelOutputs(
        binary_logit=torch.tensor([0.0, 0.0]),
        binary_prob=torch.tensor([0.5, 0.5]),
        ordinal_logits=torch.zeros((2, 2)),
        severity_probs=torch.ones((2, 3)) / 3.0,
        continuous_mean=torch.tensor([0.2, 0.8]),
        continuous_log_var=torch.zeros((2,)),
        bdd_pred=None,
        token_importance=None,
        token_modality=None,
        token_time_index=None,
    )

    loss_fn = MultitaskLoss(LossWeights(binary=1.0, ordinal=1.0, continuous=1.0, bdd=0.0))

    yb = torch.tensor([1, 0])
    yo = torch.tensor([2, 1])
    yc = torch.tensor([0.3, 0.7])

    mb = torch.tensor([True, False])
    mo = torch.tensor([False, False])
    mc = torch.tensor([True, True])

    total, scalars = loss_fn(out, yb, yo, yc, mb, mo, mc)
    assert total.isfinite().item()
    assert scalars["ordinal"] == 0.0  # fully masked
