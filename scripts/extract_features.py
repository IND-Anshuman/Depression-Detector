from __future__ import annotations

import argparse
from pathlib import Path

from mmds.config import load_config
from mmds.features.offline import OfflineExtractionPaths, extract_features_from_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--in-manifest", required=True, help="CSV with video_path/audio_path and IDs")
    ap.add_argument("--out-dir", default="artifacts/features")
    ap.add_argument("--out-manifest", default="artifacts/features_manifest.csv")
    args = ap.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.cfg

    paths = OfflineExtractionPaths(out_dir=Path(args.out_dir), manifest_csv=Path(args.out_manifest))
    extract_features_from_manifest(cfg, Path(args.in_manifest), paths)
    print(f"Wrote features to: {paths.out_dir}")
    print(f"Wrote manifest to: {paths.manifest_csv}")


if __name__ == "__main__":
    main()
