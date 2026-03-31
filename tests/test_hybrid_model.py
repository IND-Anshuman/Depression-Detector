from __future__ import annotations

import torch

from mmds.models.mmds_model import HybridMMDSModel, ModelConfig


def test_hybrid_model_forward_with_face_body_streams() -> None:
    cfg = ModelConfig(
        hidden_size=64,
        dropout=0.1,
        num_latents=8,
        num_layers=2,
        num_heads=2,
        fusion="hybrid",
        lstm_hidden_size=32,
        attention_dim=32,
        use_bdd_head=True,
    )
    model = HybridMMDSModel(cfg)
    x = {
        "face_landmarks": torch.randn(2, 10, 1404),
        "body_pose": torch.randn(2, 10, 132),
        "emotion": torch.randn(2, 10, 10),
    }
    x_mask = {key: torch.ones(2, 10, dtype=torch.bool) for key in x}

    out = model(x, x_mask)
    assert out.binary_prob.shape == (2,)
    assert out.ordinal_logits.shape == (2, 3)
    assert out.severity_probs.shape == (2, 3)
    assert out.bdd_pred is not None
