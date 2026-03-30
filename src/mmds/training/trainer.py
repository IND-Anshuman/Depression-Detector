from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mmds.data.collate import collate_samples
from mmds.data.schema import Sample
from mmds.models.mmds_model import model_from_cfg
from mmds.training.losses import LossWeights, MultitaskLoss
from mmds.utils.device import get_default_device
from mmds.utils.seed import seed_everything


class InMemorySampleDataset(Dataset[Sample]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


@dataclass(frozen=True)
class TrainResult:
    ckpt_path: Path


def _batch_to_device(batch: Any, device: torch.device) -> Any:
    batch.x = {k: v.to(device) for k, v in batch.x.items()}
    batch.x_mask = {k: v.to(device) for k, v in batch.x_mask.items()}
    batch.y_binary = batch.y_binary.to(device)
    batch.y_ordinal = batch.y_ordinal.to(device)
    batch.y_continuous = batch.y_continuous.to(device)
    batch.y_mask_binary = batch.y_mask_binary.to(device)
    batch.y_mask_ordinal = batch.y_mask_ordinal.to(device)
    batch.y_mask_continuous = batch.y_mask_continuous.to(device)
    return batch


def train(cfg: DictConfig, train_samples: list[Sample], val_samples: list[Sample]) -> TrainResult:
    seed_everything(int(cfg.training.seed))

    device = get_default_device()
    model = model_from_cfg(cfg).to(device)

    lw = cfg.loss.weights
    loss_fn = MultitaskLoss(
        LossWeights(
            binary=float(lw.binary), ordinal=float(lw.ordinal), continuous=float(lw.continuous), bdd=float(lw.bdd)
        )
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay)
    )

    amp = bool(cfg.training.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    train_dl = DataLoader(
        InMemorySampleDataset(train_samples),
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_samples,
    )
    val_dl = DataLoader(
        InMemorySampleDataset(val_samples),
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    best_val = float("inf")
    patience = int(cfg.training.early_stop_patience)
    bad = 0

    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "checkpoint.pt"

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        pbar = tqdm(train_dl, desc=f"train e{epoch}")
        for batch in pbar:
            batch = _batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(batch.x, batch.x_mask)
                total, scalars = loss_fn(
                    out,
                    batch.y_binary,
                    batch.y_ordinal,
                    batch.y_continuous,
                    batch.y_mask_binary,
                    batch.y_mask_ordinal,
                    batch.y_mask_continuous,
                )
            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": f"{scalars['total']:.4f}"})

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                batch = _batch_to_device(batch, device)
                out = model(batch.x, batch.x_mask)
                total, _ = loss_fn(
                    out,
                    batch.y_binary,
                    batch.y_ordinal,
                    batch.y_continuous,
                    batch.y_mask_binary,
                    batch.y_mask_ordinal,
                    batch.y_mask_continuous,
                )
                val_losses.append(float(total.detach().cpu().item()))

        v = float(sum(val_losses) / max(len(val_losses), 1))
        if v < best_val - 1e-4:
            best_val = v
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    return TrainResult(ckpt_path=ckpt_path)
