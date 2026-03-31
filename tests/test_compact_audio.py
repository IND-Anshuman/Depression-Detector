from __future__ import annotations

import numpy as np

from mmds.features.compact_audio import build_compact_audio_features


def test_build_compact_audio_features_resamples_to_shared_width() -> None:
    audio = np.random.default_rng(42).normal(size=(32, 128)).astype(np.float32)
    compact = build_compact_audio_features(audio, target_dim=25)
    assert compact.shape == (32, 25)
    assert np.isfinite(compact).all()


def test_build_compact_audio_features_preserves_sequence_length() -> None:
    audio = np.random.default_rng(7).normal(size=(19, 25)).astype(np.float32)
    compact = build_compact_audio_features(audio, target_dim=25)
    assert compact.shape == (19, 25)
