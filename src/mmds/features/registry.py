from __future__ import annotations

from omegaconf import DictConfig

from .extractors.base import FeatureExtractor
from .extractors.mediapipe_backend import MediaPipeExtractor
from .extractors.openface_backend import OpenFaceExtractor
from .extractors.simple_backend import SimpleExtractor


def build_extractor(cfg: DictConfig) -> FeatureExtractor:
    backend = str(cfg.features.backend)
    if backend == "simple":
        return SimpleExtractor(cfg)
    if backend == "mediapipe":
        return MediaPipeExtractor(cfg)
    if backend == "openface":
        return OpenFaceExtractor(cfg)
    raise ValueError(f"Unknown features.backend={backend!r}")
