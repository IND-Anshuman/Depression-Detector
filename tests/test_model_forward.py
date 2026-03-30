from __future__ import annotations

import torch

from mmds.models.mmds_model import MMDSModel, ModelConfig


def test_model_forward_with_missing_modality() -> None:
    cfg = ModelConfig(
        hidden_size=64,
        dropout=0.1,
        num_latents=8,
        num_layers=1,
        num_heads=2,
        fusion="perceiver",
        use_bdd_head=True,
    )
    model = MMDSModel(cfg)

    x = {
        "face_au": torch.randn(2, 10, 16),
        # body_pose is missing
        "audio": torch.randn(2, 8, 32),
        "quality": torch.randn(2, 10, 6),
    }
    x_mask = {
        "face_au": torch.ones(2, 10, dtype=torch.bool),
        "audio": torch.ones(2, 8, dtype=torch.bool),
        "quality": torch.ones(2, 10, dtype=torch.bool),
    }

    out = model(x, x_mask)
    assert out.binary_prob.shape == (2,)
    assert out.severity_probs.shape == (2, 3)
    assert out.continuous_mean.shape == (2,)
    assert out.token_importance is not None
