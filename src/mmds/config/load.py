from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass(frozen=True)
class LoadedConfig:
    cfg: DictConfig
    sources: list[Path]


def _normalize_default_entry(entry: object, base_dir: Path) -> Path:
    # Support both string entries ("base" or "base.yaml") and single-key dicts.
    if isinstance(entry, str):
        name = entry
    elif isinstance(entry, Mapping) and len(entry) == 1:
        name = str(next(iter(entry.keys())))
    else:
        raise ValueError(f"Unsupported defaults entry: {entry!r}")

    p = Path(name)
    if p.suffix.lower() != ".yaml":
        p = p.with_suffix(".yaml")
    if not p.is_absolute():
        p = base_dir / p
    return p


def _load_yaml_with_defaults(path: Path, seen: set[Path]) -> tuple[DictConfig, list[Path]]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config defaults cycle detected at: {path}")
    seen.add(path)

    raw: DictConfig = OmegaConf.load(path)
    sources: list[Path] = []

    defaults = raw.get("defaults", None)
    if defaults is not None:
        if not isinstance(defaults, (list, ListConfig)):
            raise ValueError("Config 'defaults' must be a list")
        merged = OmegaConf.create({})
        for entry in defaults:
            dp = _normalize_default_entry(entry, base_dir=path.parent)
            if not dp.exists():
                raise FileNotFoundError(f"Default config not found: {dp}")
            child_cfg, child_sources = _load_yaml_with_defaults(dp, seen=seen)
            merged = OmegaConf.merge(merged, child_cfg)
            sources.extend(child_sources)

        # Apply current file as override, but drop 'defaults' key.
        override = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
        override.pop("defaults", None)
        merged = OmegaConf.merge(merged, override)
        sources.append(path)
        return merged, sources

    sources.append(path)
    return raw, sources


def load_config(*yaml_paths: str | Path, overrides: Mapping[str, Any] | None = None) -> LoadedConfig:
    """Load one or more YAML configs and merge them (later files override earlier ones).

    This intentionally stays lightweight (OmegaConf) rather than full Hydra.
    """

    if not yaml_paths:
        raise ValueError("At least one YAML config path is required")

    sources_in: list[Path] = [Path(p) for p in yaml_paths]
    missing = [str(p) for p in sources_in if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing config files: {missing}")

    cfgs: list[DictConfig] = []
    sources: list[Path] = []
    for p in sources_in:
        c, s = _load_yaml_with_defaults(Path(p), seen=set())
        cfgs.append(c)
        sources.extend(s)

    merged: DictConfig = OmegaConf.merge(*cfgs)
    if overrides:
        merged = OmegaConf.merge(merged, dict(overrides))
    merged = OmegaConf.create(OmegaConf.to_container(merged, resolve=True))
    return LoadedConfig(cfg=merged, sources=sources)
