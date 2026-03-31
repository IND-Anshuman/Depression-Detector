from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple MMDS feature manifests into one CSV.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    frames = []
    for path_s in args.inputs:
        path = Path(path_s)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        df = pd.read_csv(path)
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote merged manifest with {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
