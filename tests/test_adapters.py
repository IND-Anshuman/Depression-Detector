from __future__ import annotations

from pathlib import Path

from mmds.data.adapters.base import AdapterConfig
from mmds.data.adapters.generic_video_folder import GenericVideoFolderAdapter


def test_generic_video_folder_adapter_discovers_videos(tmp_path: Path) -> None:
    subj = tmp_path / "subject_001"
    subj.mkdir()
    (subj / "clip_001.mp4").write_bytes(b"")

    adapter = GenericVideoFolderAdapter(AdapterConfig(root=tmp_path))
    samples = adapter.iter_samples()
    assert len(samples) == 1
    assert samples[0].subject_id == "subject_001"
    assert "video_path" in samples[0].metadata
