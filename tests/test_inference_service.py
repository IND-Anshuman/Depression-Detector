from __future__ import annotations

import numpy as np
from pathlib import Path

from mmds.config import load_config
from mmds.inference.service import BufferedInferenceService


def test_buffered_inference_runs_on_fake_frames() -> None:
    cfg = load_config("configs/demo.yaml").cfg

    service = BufferedInferenceService(cfg, ckpt_path=None)

    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    audio = (np.random.randn(16000).astype(np.float32) * 0.01)

    for _ in range(5):
        service.push(frame, audio)

    res = service.run_once()
    assert 0.0 <= res.risk_prob <= 1.0
    assert len(res.severity_probs) == 3
    assert 0.0 <= res.continuous_score <= 1.0
    assert res.au_trend_img.ndim == 3
    assert res.risk_gauge_img.ndim == 3
    assert res.attention_heatmap_img.ndim == 3


def test_live_dvlog_service_status_without_checkpoint() -> None:
    cfg = load_config("configs/live_dvlog.yaml").cfg
    service = BufferedInferenceService(cfg, ckpt_path=Path("artifacts/does-not-exist.pt"))
    status = service.status_markdown()
    assert "Checkpoint" in status
    assert "mediapipe_dvlog" in status
