from __future__ import annotations

import argparse
from pathlib import Path

from mmds.config import load_config
from mmds.training.data import build_samples_from_cfg
from mmds.training.split import subject_stratified_split
from mmds.training.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/research.yaml")
    ap.add_argument("--out", default=None, help="Override artifacts directory")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--use_real_data", action="store_true")
    ap.add_argument("--real_dataset", default="depvidmood")
    ap.add_argument("--manifest_csv", default=None)
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.cfg
    if args.out:
        cfg.paths.artifacts_dir = str(Path(args.out))
    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)
    if args.use_real_data:
        cfg.dataset.name = "depvidmood_feature_manifest" if args.real_dataset == "depvidmood" else str(args.real_dataset)
    if args.manifest_csv:
        cfg.dataset.manifest_csv = str(Path(args.manifest_csv))
    if args.wandb_project:
        cfg.training.wandb_project = str(args.wandb_project)

    bundle = build_samples_from_cfg(cfg)
    split = subject_stratified_split(bundle.samples, seed=int(cfg.training.seed))
    print(
        f"Training dataset={cfg.dataset.name} "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)} "
        f"manifest={getattr(cfg.dataset, 'manifest_csv', 'n/a')}"
    )

    result = train(cfg, split.train, split.val)
    print(f"Saved checkpoint: {result.ckpt_path}")


if __name__ == "__main__":
    main()
