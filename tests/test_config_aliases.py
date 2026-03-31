from __future__ import annotations

from pathlib import Path

from mmds.config.load import load_config


def test_top_level_feature_aliases_are_normalized(tmp_path: Path) -> None:
    base_cfg = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
defaults:
  - {base_cfg.as_posix()}
feature_backend: mediapipe_full
include_emonet: true
include_entropy: true
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path).cfg
    assert cfg.features.backend == "mediapipe_full"
    assert bool(cfg.features.include_emonet) is True
    assert bool(cfg.features.include_entropy) is True
