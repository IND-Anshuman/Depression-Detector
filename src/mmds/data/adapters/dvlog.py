from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema import Sample
from .base import AdapterConfig, DatasetAdapter


@dataclass(frozen=True)
class DVlogConfig:
    """Optional D-Vlog-style adapter scaffold.

    This is a placeholder; D-Vlog datasets vary in structure.
    Provide a prepared manifest CSV and use the feature-manifest training path.
    """

    manifest_csv: Path


class DVlogAdapter(DatasetAdapter):
    def __init__(self, cfg: AdapterConfig, dcfg: DVlogConfig) -> None:
        super().__init__(cfg)
        self.dcfg = dcfg

    def iter_samples(self) -> list[Sample]:
        raise NotImplementedError(
            "DVlogAdapter is a scaffold. Use a prepared manifest + offline extraction "
            "(scripts/extract_features.py) to build a features manifest for training."
        )
