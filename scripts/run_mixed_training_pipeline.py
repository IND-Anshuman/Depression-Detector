from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and train the strongest locally available mixed dataset pipeline.")
    ap.add_argument("--dvlog-zip", required=True)
    ap.add_argument("--dvlog-links", required=True)
    ap.add_argument("--lmvd-zip", required=True)
    ap.add_argument("--daic-root", default=None)
    ap.add_argument("--daic-splits-dir", default=None)
    ap.add_argument("--extract-config", default="configs/compact_av_extract.yaml")
    ap.add_argument("--train-config", default="configs/mixed_visual_train.yaml")
    ap.add_argument("--artifacts-dir", default="artifacts/mixed_visual_run")
    ap.add_argument("--max-frames", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    py = sys.executable
    artifacts_dir = Path(args.artifacts_dir)
    features_dir = artifacts_dir.parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    dvlog_manifest = features_dir / "dvlog_compact_manifest.csv"
    lmvd_manifest = features_dir / "lmvd_manifest.csv"
    merged_inputs = [str(dvlog_manifest), str(lmvd_manifest)]

    _run(
        [
            py,
            "scripts/build_dvlog_feature_manifest.py",
            "--zip-path",
            args.dvlog_zip,
            "--links-csv",
            args.dvlog_links,
            "--out-dir",
            str(features_dir / "dvlog_compact"),
            "--out-manifest",
            str(dvlog_manifest),
            "--max-frames",
            str(args.max_frames),
            "--schema",
            "compact_visual",
        ]
    )
    _run(
        [
            py,
            "scripts/build_lmvd_feature_manifest.py",
            "--zip-path",
            args.lmvd_zip,
            "--out-dir",
            str(features_dir / "lmvd"),
            "--out-manifest",
            str(lmvd_manifest),
            "--max-frames",
            str(args.max_frames),
        ]
    )

    if args.daic_root:
        daic_raw_manifest = features_dir / "daic_raw_manifest.csv"
        daic_features_manifest = features_dir / "daic_features_manifest.csv"
        _run(
            [
                py,
                "scripts/build_daic_feature_manifest.py",
                "--root-dir",
                args.daic_root,
                "--out-manifest",
                str(daic_raw_manifest),
                *(
                    ["--splits-dir", args.daic_splits_dir]
                    if args.daic_splits_dir
                    else []
                ),
            ]
        )
        _run(
            [
                py,
                "scripts/extract_features.py",
                "--config",
                args.extract_config,
                "--in-manifest",
                str(daic_raw_manifest),
                "--out-dir",
                str(features_dir / "daic"),
                "--out-manifest",
                str(daic_features_manifest),
                "--dataset",
                "daic",
            ]
        )
        merged_inputs.append(str(daic_features_manifest))

    merged_manifest = features_dir / "mixed_visual_manifest.csv"
    _run(
        [
            py,
            "scripts/merge_feature_manifests.py",
            "--inputs",
            *merged_inputs,
            "--output",
            str(merged_manifest),
        ]
    )
    _run(
        [
            py,
            "scripts/train.py",
            "--config",
            args.train_config,
            "--out",
            str(artifacts_dir),
            "--use_real_data",
            "--real_dataset",
            "feature_manifest",
            "--manifest_csv",
            str(merged_manifest),
            "--epochs",
            str(args.epochs),
        ]
    )
    _run(
        [
            py,
            "scripts/evaluate.py",
            "--config",
            args.train_config,
            "--ckpt",
            str(artifacts_dir / "checkpoint.pt"),
            "--out",
            str(artifacts_dir / "eval"),
            "--use_real_data",
            "--real_dataset",
            "feature_manifest",
            "--manifest_csv",
            str(merged_manifest),
        ]
    )


if __name__ == "__main__":
    main()
