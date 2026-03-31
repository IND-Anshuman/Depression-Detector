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
    ap.add_argument("--use_real_data", action="store_true")
    ap.add_argument("--real_dataset", default="depvidmood")
    ap.add_argument("--manifest_csv", default=None)
    args = ap.parse_args()

    loaded = load_config(args.config)
    cfg = loaded.cfg
    if args.use_real_data:
        cfg.dataset.name = "depvidmood_feature_manifest" if args.real_dataset == "depvidmood" else str(args.real_dataset)
    if args.manifest_csv:
        cfg.dataset.manifest_csv = str(Path(args.manifest_csv))

    bundle = build_samples_from_cfg(cfg)
    split = subject_stratified_split(bundle.samples, seed=int(cfg.training.seed))
    print(
        f"Evaluating dataset={cfg.dataset.name} "
        f"test={len(split.test)} manifest={getattr(cfg.dataset, 'manifest_csv', 'n/a')}"
    )

    res = evaluate(cfg, Path(args.ckpt), split.test, Path(args.out))
    print(f"Wrote evaluation to: {res.out_dir}")
    print(res.metrics)


if __name__ == "__main__":
    main()
