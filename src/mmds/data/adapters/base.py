from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..schema import Sample


@dataclass(frozen=True)
class AdapterConfig:
    root: Path


class DatasetAdapter(ABC):
    """Adapter interface to map raw datasets into unified `Sample`s."""

    def __init__(self, cfg: AdapterConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def iter_samples(self) -> list[Sample]:
        raise NotImplementedError
