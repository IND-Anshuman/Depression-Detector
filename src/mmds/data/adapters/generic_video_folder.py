from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema import Sample
from .base import AdapterConfig, DatasetAdapter


@dataclass(frozen=True)
class GenericVideoFolderConfig:
    """Expect a directory layout like:

    root/
      subject_001/
        clip_001.mp4
        clip_002.mp4
      subject_002/
        clip_001.mp4

    Labels are optional and can be provided via a CSV prepared by the user.
    """

    video_glob: str = "*.mp4"


class GenericVideoFolderAdapter(DatasetAdapter):
    def __init__(self, cfg: AdapterConfig, gcfg: GenericVideoFolderConfig | None = None) -> None:
        super().__init__(cfg)
        self.gcfg = gcfg or GenericVideoFolderConfig()

    def iter_samples(self) -> list[Sample]:
        out: list[Sample] = []
        for subject_dir in sorted([p for p in self.cfg.root.iterdir() if p.is_dir()]):
            subject_id = subject_dir.name
            for i, video_path in enumerate(sorted(subject_dir.glob(self.gcfg.video_glob))):
                out.append(
                    Sample(
                        sample_id=f"{subject_id}-{i}",
                        subject_id=subject_id,
                        dataset_name="generic_video_folder",
                        window_index=i,
                        modality_payloads={},
                        modality_masks={},
                        metadata={"video_path": str(video_path)},
                    )
                )
        return out
