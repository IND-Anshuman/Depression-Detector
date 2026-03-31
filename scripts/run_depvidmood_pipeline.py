from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the DepVidMood manifest -> extraction -> train -> eval pipeline.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--config", default="configs/research.yaml")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", default="data/manifests/depvidmood_feature_manifest.csv")
    ap.add_argument("--feature-dir", default="artifacts/features/depvidmood")
    ap.add_argument("--feature-manifest", default="artifacts/features/depvidmood_features_manifest.csv")
    ap.add_argument("--artifacts-dir", default="artifacts/depvidmood_run")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--wandb-project", default="MMDS-Final")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    feature_manifest = Path(args.feature_manifest)
    artifacts_dir = Path(args.artifacts_dir)
    eval_dir = artifacts_dir / "eval"

    _run(
        [
            args.python,
            "scripts/build_depvidmood_manifest.py",
            "--raw-dir",
            args.raw_dir,
            "--output",
            str(manifest),
        ]
    )
    _run(
        [
            args.python,
            "scripts/extract_features.py",
            "--config",
            args.config,
            "--in-manifest",
            str(manifest),
            "--out-dir",
            args.feature_dir,
            "--out-manifest",
            str(feature_manifest),
            "--dataset",
            "depvidmood",
        ]
    )
    _run(
        [
            args.python,
            "scripts/train.py",
            "--config",
            args.config,
            "--out",
            str(artifacts_dir),
            "--use_real_data",
            "--real_dataset",
            "depvidmood",
            "--manifest_csv",
            str(feature_manifest),
            "--epochs",
            str(args.epochs),
            "--wandb_project",
            args.wandb_project,
        ]
    )
    _run(
        [
            args.python,
            "scripts/evaluate.py",
            "--config",
            args.config,
            "--ckpt",
            str(artifacts_dir / "checkpoint.pt"),
            "--out",
            str(eval_dir),
            "--use_real_data",
            "--real_dataset",
            "depvidmood",
            "--manifest_csv",
            str(feature_manifest),
        ]
    )


if __name__ == "__main__":
    main()
