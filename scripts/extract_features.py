from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmds.config import load_config
from mmds.features.offline import OfflineExtractionPaths, extract_features_from_manifest


def _validate_manifest(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Input manifest not found: {path}\n"
            "Run `scripts/build_depvidmood_manifest.py` first, or point `--in-manifest` at an existing CSV."
        )
    df = pd.read_csv(path, nrows=5)
    required = {"sample_id", "subject_id", "video_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input manifest is missing required columns: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--in-manifest", required=True, help="CSV with video_path/audio_path and IDs")
    ap.add_argument("--out-dir", default="artifacts/features")
    ap.add_argument("--out-manifest", default=None)
    ap.add_argument("--dataset", default="custom")
    args = ap.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.cfg
    in_manifest = Path(args.in_manifest)
    _validate_manifest(in_manifest)

    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest) if args.out_manifest else out_dir / f"{args.dataset}_features_manifest.csv"

    paths = OfflineExtractionPaths(out_dir=out_dir, manifest_csv=out_manifest)
    extract_features_from_manifest(cfg, in_manifest, paths)
    print(f"Wrote features to: {paths.out_dir}")
    print(f"Wrote manifest to: {paths.manifest_csv}")


if __name__ == "__main__":
    main()
