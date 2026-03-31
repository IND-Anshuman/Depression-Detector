from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_depvidmood_manifest_from_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "subject01_mild_clip.mp4").write_bytes(b"")
    (raw_dir / "subject02_severe_clip.mp4").write_bytes(b"")

    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {"video_path": "subject01_mild_clip.mp4", "severity": "mild", "subject_id": "subject01", "sample_id": "s1"},
            {"video_path": "subject02_severe_clip.mp4", "severity": "severe", "subject_id": "subject02", "sample_id": "s2"},
        ]
    ).to_csv(metadata, index=False)

    output = tmp_path / "manifest.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_depvidmood_manifest.py",
            "--raw-dir",
            str(raw_dir),
            "--metadata-csv",
            str(metadata),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    manifest = pd.read_csv(output)
    assert set(["sample_id", "subject_id", "video_path", "severity_label_4class", "ordinal_label_3class", "binary_label", "continuous_score", "split"]).issubset(manifest.columns)
    assert sorted(manifest["severity_label_4class"].tolist()) == [1, 3]
