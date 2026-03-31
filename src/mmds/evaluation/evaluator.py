from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mmds.data.collate import collate_samples
from mmds.data.schema import Sample
from mmds.evaluation.metrics import EvalMetrics, compute_confusion, compute_metrics
from mmds.evaluation.plots import save_calibration_plot, save_roc_curve
from mmds.models.mmds_model import model_from_cfg
from mmds.training.trainer import InMemorySampleDataset, _batch_to_device
from mmds.utils.device import get_default_device


@dataclass(frozen=True)
class EvalResult:
    metrics: EvalMetrics
    out_dir: Path


def evaluate(cfg: DictConfig, ckpt_path: Path, samples: list[Sample], out_dir: Path) -> EvalResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    model = model_from_cfg(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dl = DataLoader(
        InMemorySampleDataset(samples),
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    rows = []
    with torch.no_grad():
        for batch in dl:
            batch = _batch_to_device(batch, device)
            out = model(batch.x, batch.x_mask)

            bprob = out.binary_prob.detach().cpu().numpy()
            sev = out.severity_probs.detach().cpu().numpy()
            sev_pred = sev.argmax(axis=1)
            cont = out.continuous_mean.detach().cpu().numpy()

            for i in range(len(batch.sample_ids)):
                rows.append(
                    {
                        "sample_id": batch.sample_ids[i],
                        "subject_id": batch.subject_ids[i],
                        "dataset": batch.dataset_names[i],
                        "y_binary": int(batch.y_binary[i].detach().cpu().item()),
                        "m_binary": bool(batch.y_mask_binary[i].detach().cpu().item()),
                        "y_ordinal": int(batch.y_ordinal[i].detach().cpu().item()),
                        "m_ordinal": bool(batch.y_mask_ordinal[i].detach().cpu().item()),
                        "y_cont": float(batch.y_continuous[i].detach().cpu().item()),
                        "m_cont": bool(batch.y_mask_continuous[i].detach().cpu().item()),
                        "y_bdd": float(batch.y_bdd[i].detach().cpu().item()),
                        "m_bdd": bool(batch.y_mask_bdd[i].detach().cpu().item()),
                        "p_risk": float(bprob[i]),
                        "sev0": float(sev[i, 0]),
                        "sev1": float(sev[i, 1]),
                        "sev2": float(sev[i, 2]),
                        "sev_pred": int(sev_pred[i]),
                        "cont_pred": float(cont[i]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)

    y_true_bin = df["y_binary"].to_numpy()
    y_prob_bin = df["p_risk"].to_numpy()
    y_true_ord = df["y_ordinal"].to_numpy()
    y_pred_ord = df["sev_pred"].to_numpy()
    y_true_cont = df["y_cont"].to_numpy(dtype=float)
    y_pred_cont = df["cont_pred"].to_numpy(dtype=float)

    m_bin = df["m_binary"].to_numpy(dtype=bool)
    m_ord = df["m_ordinal"].to_numpy(dtype=bool)
    m_cont = df["m_cont"].to_numpy(dtype=bool)

    metrics = compute_metrics(
        y_true_bin,
        y_prob_bin,
        y_true_ord,
        y_pred_ord,
        y_true_cont,
        y_pred_cont,
        m_bin,
        m_ord,
        m_cont,
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    if m_bin.any():
        save_roc_curve(y_true_bin[m_bin], y_prob_bin[m_bin], out_dir / "roc.png")
        save_calibration_plot(y_true_bin[m_bin], y_prob_bin[m_bin], out_dir / "calibration.png")

    cm = compute_confusion(y_true_ord, y_pred_ord, m_ord, num_classes=3)
    np.save(out_dir / "severity_confusion.npy", cm)

    return EvalResult(metrics=metrics, out_dir=out_dir)
