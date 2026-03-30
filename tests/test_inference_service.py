from __future__ import annotations

import numpy as np

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
