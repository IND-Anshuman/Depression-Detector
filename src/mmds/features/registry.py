from __future__ import annotations

from omegaconf import DictConfig

from .extractors.base import FeatureExtractor
from .extractors.mediapipe_backend import MediaPipeExtractor
from .extractors.mediapipe_dvlog_backend import MediaPipeDVlogExtractor
from .extractors.mediapipe_full_backend import MediaPipeFullExtractor
from .extractors.openface_backend import OpenFaceExtractor
from .extractors.simple_backend import SimpleExtractor


def build_extractor(cfg: DictConfig) -> FeatureExtractor:
    backend = str(cfg.features.backend)
    if backend == "simple":
        return SimpleExtractor(cfg)
    if backend == "mediapipe":
        return MediaPipeExtractor(cfg)
    if backend == "mediapipe_dvlog":
        return MediaPipeDVlogExtractor(cfg)
    if backend == "mediapipe_full":
        return MediaPipeFullExtractor(cfg)
    if backend == "openface":
        return OpenFaceExtractor(cfg)
    raise ValueError(f"Unknown features.backend={backend!r}")
