from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Sample


@dataclass(frozen=True)
class FeatureManifestConfig:
    manifest_csv: Path


def load_feature_manifest(cfg: FeatureManifestConfig) -> list[Sample]:
    if not cfg.manifest_csv.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {cfg.manifest_csv}\n"
            "Run `scripts/extract_features.py` first, then pass the generated features manifest to training/evaluation."
        )
    df = pd.read_csv(cfg.manifest_csv)
    out: list[Sample] = []

    for _, r in df.iterrows():
        fpath = Path(str(r["features_path"]))
        npz = np.load(fpath, allow_pickle=False)

        payloads: dict[str, np.ndarray] = {}
        for k in npz.files:
            if k.startswith("x__"):
                payloads[k.replace("x__", "", 1)] = np.array(npz[k], dtype=np.float32)

        masks: dict[str, bool] = {}
        if "mask_keys" in npz.files and "mask" in npz.files:
            keys = [str(x) for x in npz["mask_keys"].tolist()]
            vals = npz["mask"].astype(int).tolist()
            masks = {k: bool(v) for k, v in zip(keys, vals)}
        else:
            masks = {k: True for k in payloads.keys()}

        def _opt_int(name: str) -> int | None:
            if name not in r or pd.isna(r[name]):
                return None
            return int(r[name])

        def _opt_float(name: str) -> float | None:
            if name not in r or pd.isna(r[name]):
                return None
            return float(r[name])

        metadata = {
            k: (None if pd.isna(v) else v)
            for k, v in r.to_dict().items()
            if k
            not in {
                "sample_id",
                "subject_id",
                "dataset_name",
                "binary_label",
                "ordinal_label",
                "ordinal_label_3class",
                "continuous_score",
                "bdd_score",
                "window_index",
                "features_path",
                "split",
                "severity_label_4class",
            }
        }

        out.append(
            Sample(
                sample_id=str(r["sample_id"]),
                subject_id=str(r["subject_id"]),
                dataset_name=str(r.get("dataset_name", "features")),
                binary_label=_opt_int("binary_label"),
                ordinal_label=_opt_int("ordinal_label_3class") if "ordinal_label_3class" in r else _opt_int("ordinal_label"),
                continuous_score=_opt_float("continuous_score"),
                bdd_score=_opt_float("bdd_score"),
                split=str(r["split"]) if "split" in r and not pd.isna(r["split"]) else None,
                severity_label_4class=_opt_int("severity_label_4class"),
                window_index=int(r["window_index"]) if "window_index" in r and not pd.isna(r["window_index"]) else None,
                modality_payloads=payloads,
                modality_masks=masks,
                metadata={"features_path": str(fpath), **metadata},
            )
        )

    return out
