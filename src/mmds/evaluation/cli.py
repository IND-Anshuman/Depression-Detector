from __future__ import annotations

import argparse
from pathlib import Path

from mmds.config import load_config
from mmds.training.data import build_samples_from_cfg
from mmds.training.split import subject_stratified_split

from .evaluator import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/research.yaml")
    ap.add_argument("--ckpt", default="artifacts/checkpoint.pt")
    ap.add_argument("--out", default="artifacts/eval")
    args = ap.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.cfg

    bundle = build_samples_from_cfg(cfg)
    split = subject_stratified_split(bundle.samples, seed=int(cfg.training.seed))

    res = evaluate(cfg, Path(args.ckpt), split.test, Path(args.out))
    print(f"Wrote evaluation to: {res.out_dir}")
    print(res.metrics)


if __name__ == "__main__":
    main()
