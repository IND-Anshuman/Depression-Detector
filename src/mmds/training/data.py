from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from mmds.data.adapters.base import AdapterConfig
from mmds.data.adapters.daic_woz import DaicWozAdapter, DaicWozPaths
from mmds.data.adapters.edaic import EDaicAdapter, EDaicPaths
from mmds.data.adapters.generic_video_folder import GenericVideoFolderAdapter
from mmds.data.feature_manifest import FeatureManifestConfig, load_feature_manifest
from mmds.data.schema import Sample
from mmds.data.synthetic import SyntheticDatasetConfig, make_synthetic_samples


@dataclass(frozen=True)
class DatasetBundle:
    samples: list[Sample]


def build_samples_from_cfg(cfg: DictConfig) -> DatasetBundle:
    name = str(cfg.dataset.name)

    if name == "synthetic":
        scfg = cfg.dataset.synthetic
        samples = make_synthetic_samples(
            SyntheticDatasetConfig(
                num_subjects=int(scfg.num_subjects),
                windows_per_subject=int(scfg.windows_per_subject),
                seq_len=int(scfg.seq_len),
                seed=int(cfg.training.seed),
            )
        )
        return DatasetBundle(samples=samples)

    if name == "feature_manifest":
        manifest_csv = Path(cfg.dataset.manifest_csv)
        samples = load_feature_manifest(FeatureManifestConfig(manifest_csv=manifest_csv))
        return DatasetBundle(samples=samples)

    root = Path(getattr(cfg.dataset, "root", "."))
    if name == "generic_video_folder":
        adapter = GenericVideoFolderAdapter(AdapterConfig(root=root))
        return DatasetBundle(samples=adapter.iter_samples())

    if name == "daic_woz":
        paths = DaicWozPaths(labels_csv=Path(cfg.dataset.labels_csv), manifest_csv=getattr(cfg.dataset, "manifest_csv", None))
        adapter = DaicWozAdapter(AdapterConfig(root=root), paths=paths)
        return DatasetBundle(samples=adapter.iter_samples())

    if name == "e_daic":
        paths = EDaicPaths(labels_csv=Path(cfg.dataset.labels_csv), manifest_csv=getattr(cfg.dataset, "manifest_csv", None))
        adapter = EDaicAdapter(AdapterConfig(root=root), paths=paths)
        return DatasetBundle(samples=adapter.iter_samples())

    raise ValueError(f"Unsupported dataset.name={name!r}")
