from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
    return replace(
        batch,
        x={k: v.to(device) for k, v in batch.x.items()},
        x_mask={k: v.to(device) for k, v in batch.x_mask.items()},
        y_binary=batch.y_binary.to(device),
        y_ordinal=batch.y_ordinal.to(device),
        y_continuous=batch.y_continuous.to(device),
        y_bdd=batch.y_bdd.to(device),
        y_mask_binary=batch.y_mask_binary.to(device),
        y_mask_ordinal=batch.y_mask_ordinal.to(device),
        y_mask_continuous=batch.y_mask_continuous.to(device),
        y_mask_bdd=batch.y_mask_bdd.to(device),
    )


def _balanced_sampler(samples: list[Sample]) -> WeightedRandomSampler | None:
    if not samples:
        return None

    dataset_counts: dict[str, int] = {}
    label_counts: dict[int, int] = {}
    for sample in samples:
        dataset_counts[sample.dataset_name] = dataset_counts.get(sample.dataset_name, 0) + 1
        label = int(sample.binary_label) if sample.binary_label is not None else 0
        label_counts[label] = label_counts.get(label, 0) + 1

    weights = []
    for sample in samples:
        dataset_weight = 1.0 / max(dataset_counts.get(sample.dataset_name, 1), 1)
        label = int(sample.binary_label) if sample.binary_label is not None else 0
        label_weight = 1.0 / max(label_counts.get(label, 1), 1)
        weights.append(float((dataset_weight * label_weight) ** 0.5))

    return WeightedRandomSampler(weights=weights, num_samples=len(samples), replacement=True)


def _binary_pos_weight(samples: list[Sample]) -> float | None:
    positives = sum(1 for sample in samples if sample.binary_label == 1)
    negatives = sum(1 for sample in samples if sample.binary_label == 0)
    if positives == 0 or negatives == 0:
        return None
    return float(negatives / max(positives, 1))


def train(cfg: DictConfig, train_samples: list[Sample], val_samples: list[Sample]) -> TrainResult:
    seed_everything(int(cfg.training.seed))

    device = get_default_device()
    model = model_from_cfg(cfg).to(device)

    lw = cfg.loss.weights
    loss_fn = MultitaskLoss(
        LossWeights(
            binary=float(lw.binary), ordinal=float(lw.ordinal), continuous=float(lw.continuous), bdd=float(lw.bdd)
        ),
        binary_pos_weight=_binary_pos_weight(train_samples) if bool(getattr(cfg.training, "use_class_pos_weight", True)) else None,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay)
    )

    amp = bool(cfg.training.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    sampler = _balanced_sampler(train_samples) if bool(getattr(cfg.training, "balanced_sampling", True)) else None
    train_dl = DataLoader(
        InMemorySampleDataset(train_samples),
        batch_size=int(cfg.training.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
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
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(batch.x, batch.x_mask)
                total, scalars = loss_fn(
                    out,
                    batch.y_binary,
                    batch.y_ordinal,
                    batch.y_continuous,
                    batch.y_mask_binary,
                    batch.y_mask_ordinal,
                    batch.y_mask_continuous,
                    batch.y_bdd,
                    batch.y_mask_bdd,
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
                    batch.y_bdd,
                    batch.y_mask_bdd,
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
