from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/space_assets")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Minimal helper: collect the Space runtime config.
    (out / "space.yaml").write_text(Path("configs/space.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "Dockerfile").write_text(Path("Dockerfile").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Exported Space assets to: {out}")


if __name__ == "__main__":
    main()
