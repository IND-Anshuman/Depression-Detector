from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..schema import Sample
from .base import AdapterConfig, DatasetAdapter


@dataclass(frozen=True)
class DaicWozPaths:
    labels_csv: Path
    manifest_csv: Path | None = None


class DaicWozAdapter(DatasetAdapter):
    """DAIC-WOZ-style adapter scaffold.

    This implementation is intentionally conservative: it expects a *prepared* manifest
    rather than assuming any private dataset structure.

    Required files:
    - labels_csv: columns [subject_id, binary_label(optional), ordinal_label(optional), continuous_score(optional)]
    - manifest_csv (optional): columns [subject_id, sample_id, window_index, audio_path, video_path]
    """

    def __init__(self, cfg: AdapterConfig, paths: DaicWozPaths) -> None:
        super().__init__(cfg)
        self.paths = paths

    def iter_samples(self) -> list[Sample]:
        labels = pd.read_csv(self.paths.labels_csv)
        labels = labels.set_index("subject_id")

        if self.paths.manifest_csv is None:
            # Minimal adapter: yields a subject-level sample with only labels.
            samples: list[Sample] = []
            for subject_id, row in labels.iterrows():
                samples.append(
                    Sample(
                        sample_id=f"{subject_id}-subject",
                        subject_id=str(subject_id),
                        dataset_name="daic_woz",
                        binary_label=int(row["binary_label"]) if "binary_label" in row and pd.notna(row["binary_label"]) else None,
                        ordinal_label=int(row["ordinal_label"]) if "ordinal_label" in row and pd.notna(row["ordinal_label"]) else None,
                        continuous_score=float(row["continuous_score"]) if "continuous_score" in row and pd.notna(row["continuous_score"]) else None,
                        modality_payloads={},
                        modality_masks={},
                        metadata={},
                    )
                )
            return samples

        manifest = pd.read_csv(self.paths.manifest_csv)
        samples = []
        for _, r in manifest.iterrows():
            subject_id = str(r["subject_id"])
            lr = labels.loc[subject_id] if subject_id in labels.index else None
            samples.append(
                Sample(
                    sample_id=str(r.get("sample_id", f"{subject_id}-{int(r.get('window_index', 0))}")),
                    subject_id=subject_id,
                    dataset_name="daic_woz",
                    binary_label=int(lr["binary_label"]) if lr is not None and "binary_label" in lr and pd.notna(lr["binary_label"]) else None,
                    ordinal_label=int(lr["ordinal_label"]) if lr is not None and "ordinal_label" in lr and pd.notna(lr["ordinal_label"]) else None,
                    continuous_score=float(lr["continuous_score"]) if lr is not None and "continuous_score" in lr and pd.notna(lr["continuous_score"]) else None,
                    window_index=int(r.get("window_index", 0)) if pd.notna(r.get("window_index", 0)) else None,
                    modality_payloads={},
                    modality_masks={},
                    metadata={
                        "audio_path": str(r.get("audio_path")) if "audio_path" in r and pd.notna(r.get("audio_path")) else None,
                        "video_path": str(r.get("video_path")) if "video_path" in r and pd.notna(r.get("video_path")) else None,
                    },
                )
            )
        return samples
