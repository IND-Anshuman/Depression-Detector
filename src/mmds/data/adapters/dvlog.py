from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..schema import Sample
from .base import AdapterConfig, DatasetAdapter


@dataclass(frozen=True)
class DVlogConfig:
    """Prepared D-Vlog manifest loader."""

    manifest_csv: Path


class DVlogAdapter(DatasetAdapter):
    def __init__(self, cfg: AdapterConfig, dcfg: DVlogConfig) -> None:
        super().__init__(cfg)
        self.dcfg = dcfg

    def iter_samples(self) -> list[Sample]:
        df = pd.read_csv(self.dcfg.manifest_csv)
        samples: list[Sample] = []
        for _, r in df.iterrows():
            samples.append(
                Sample(
                    sample_id=str(r.get("sample_id", Path(str(r.get("video_path", "dvlog"))).stem)),
                    subject_id=str(r.get("subject_id", r.get("sample_id", "unknown"))),
                    dataset_name="dvlog",
                    binary_label=int(r["binary_label"]) if "binary_label" in r and pd.notna(r["binary_label"]) else None,
                    ordinal_label=int(r["ordinal_label_3class"]) if "ordinal_label_3class" in r and pd.notna(r["ordinal_label_3class"]) else None,
                    continuous_score=float(r["continuous_score"]) if "continuous_score" in r and pd.notna(r["continuous_score"]) else None,
                    bdd_score=float(r["bdd_score"]) if "bdd_score" in r and pd.notna(r["bdd_score"]) else None,
                    split=str(r["split"]) if "split" in r and pd.notna(r["split"]) else None,
                    severity_label_4class=int(r["severity_label_4class"]) if "severity_label_4class" in r and pd.notna(r["severity_label_4class"]) else None,
                    modality_payloads={},
                    modality_masks={},
                    metadata={
                        "audio_path": str(r.get("audio_path")) if "audio_path" in r and pd.notna(r.get("audio_path")) else None,
                        "video_path": str(r.get("video_path")) if "video_path" in r and pd.notna(r.get("video_path")) else None,
                    },
                )
            )
        return samples
