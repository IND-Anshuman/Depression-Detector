from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mmds.models.outputs import ModelOutputs


@dataclass(frozen=True)
class UncertaintySummary:
    risk_prob_mean: float
    risk_prob_std: float
    severity_probs_mean: list[float]
    severity_probs_std: list[float]
    continuous_mean_mean: float
    continuous_mean_std: float
    confidence: float
    uncertainty: float


def _stack(outputs: list[ModelOutputs]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    risk = np.stack([o.binary_prob.detach().cpu().numpy() for o in outputs], axis=0)  # (S,B)
    sev = np.stack([o.severity_probs.detach().cpu().numpy() for o in outputs], axis=0)  # (S,B,3)
    cont = np.stack([o.continuous_mean.detach().cpu().numpy() for o in outputs], axis=0)  # (S,B)
    return risk, sev, cont


def summarize_mc(outputs: list[ModelOutputs], batch_index: int = 0) -> UncertaintySummary:
    risk, sev, cont = _stack(outputs)

    r_mean = float(risk[:, batch_index].mean())
    r_std = float(risk[:, batch_index].std())

    s_mean = sev[:, batch_index, :].mean(axis=0)
    s_std = sev[:, batch_index, :].std(axis=0)

    c_mean = float(cont[:, batch_index].mean())
    c_std = float(cont[:, batch_index].std())

    # A simple combined uncertainty score.
    unc = float(np.clip(0.7 * r_std + 0.3 * c_std, 0.0, 1.0))
    conf = float(np.clip(1.0 - unc, 0.0, 1.0))

    return UncertaintySummary(
        risk_prob_mean=r_mean,
        risk_prob_std=r_std,
        severity_probs_mean=[float(x) for x in s_mean.tolist()],
        severity_probs_std=[float(x) for x in s_std.tolist()],
        continuous_mean_mean=c_mean,
        continuous_mean_std=c_std,
        confidence=conf,
        uncertainty=unc,
    )


def mc_forward(
    model: torch.nn.Module,
    x: dict[str, torch.Tensor],
    x_mask: dict[str, torch.Tensor],
    passes: int,
) -> list[ModelOutputs]:
    model.train()  # enable dropout
    outs: list[ModelOutputs] = []
    with torch.no_grad():
        for _ in range(passes):
            outs.append(model(x, x_mask))
    model.eval()
    return outs
