from __future__ import annotations

import argparse

from mmds.config import load_config
from mmds.ui.gradio_app import launch_gradio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    args = ap.parse_args()

    loaded = load_config(args.config)
    launch_gradio(loaded.cfg)


if __name__ == "__main__":
    main()
