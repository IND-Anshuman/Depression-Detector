from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .schema import ModalityBatch, Sample


def _pad_time_series(arrays: list[np.ndarray], pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad (T,D) arrays to max T in batch.

    Returns:
      x: (B,T,D)
      mask: (B,T) with 1 for valid timesteps
    """

    if not arrays:
        raise ValueError("No arrays to pad")

    lengths = [a.shape[0] for a in arrays]
    max_t = max(lengths)
    feat_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else 1

    x = torch.full((len(arrays), max_t, feat_dim), float(pad_value), dtype=torch.float32)
    m = torch.zeros((len(arrays), max_t), dtype=torch.bool)

    for i, a in enumerate(arrays):
        if a.ndim == 1:
            a2 = a[:, None]
        else:
            a2 = a
        t = a2.shape[0]
        x[i, :t] = torch.from_numpy(a2).to(dtype=torch.float32)
        m[i, :t] = True

    return x, m


def collate_samples(samples: Iterable[Sample]) -> ModalityBatch:
    samples_l = list(samples)
    if not samples_l:
        raise ValueError("Empty batch")

    sample_ids = [s.sample_id for s in samples_l]
    subject_ids = [s.subject_id for s in samples_l]
    dataset_names = [s.dataset_name for s in samples_l]

    all_modalities: set[str] = set()
    for s in samples_l:
        all_modalities.update(s.modality_payloads.keys())

    x: dict[str, torch.Tensor] = {}
    x_mask: dict[str, torch.Tensor] = {}

    for mod in sorted(all_modalities):
        arrays: list[np.ndarray] = []
        present: list[bool] = []
        for s in samples_l:
            a = s.modality_payloads.get(mod)
            if a is None:
                arrays.append(np.zeros((0, 1), dtype=np.float32))
                present.append(False)
            else:
                arrays.append(a.astype(np.float32))
                present.append(True)
        x_mod, tmask = _pad_time_series(arrays)
        # Present mask gates whole modality as well.
        present_mask = torch.tensor(present, dtype=torch.bool)[:, None]
        x[mod] = x_mod
        x_mask[mod] = tmask & present_mask

    def _labels_int(values: list[int | None]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.tensor([v is not None for v in values], dtype=torch.bool)
        y = torch.tensor([int(v) if v is not None else 0 for v in values], dtype=torch.long)
        return y, mask

    def _labels_float(values: list[float | None]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.tensor([v is not None for v in values], dtype=torch.bool)
        y = torch.tensor([float(v) if v is not None else 0.0 for v in values], dtype=torch.float32)
        return y, mask

    yb, mb = _labels_int([s.binary_label for s in samples_l])
    yo, mo = _labels_int([s.ordinal_label for s in samples_l])
    yc, mc = _labels_float([s.continuous_score for s in samples_l])

    return ModalityBatch(
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        dataset_names=dataset_names,
        x=x,
        x_mask=x_mask,
        y_binary=yb,
        y_ordinal=yo,
        y_continuous=yc,
        y_mask_binary=mb,
        y_mask_ordinal=mo,
        y_mask_continuous=mc,
    )
