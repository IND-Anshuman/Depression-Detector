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
    args = ap.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.cfg
    if args.out:
        cfg.paths.artifacts_dir = str(Path(args.out))

    bundle = build_samples_from_cfg(cfg)
    split = subject_stratified_split(bundle.samples, seed=int(cfg.training.seed))

    result = train(cfg, split.train, split.val)
    print(f"Saved checkpoint: {result.ckpt_path}")


if __name__ == "__main__":
    main()
