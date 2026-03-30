from __future__ import annotations

from mmds.config import load_config
from mmds.ui.gradio_app import launch_gradio


def main() -> None:
    loaded = load_config("configs/space.yaml")
    launch_gradio(loaded.cfg)


if __name__ == "__main__":
    main()
