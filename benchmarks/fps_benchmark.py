from __future__ import annotations

import argparse
import time

import numpy as np

from mmds.config import load_config
from mmds.inference.service import BufferedInferenceService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/space.yaml")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--sleep-ms", type=float, default=200.0, help="Delay between pushed frames to mimic webcam cadence.")
    args = ap.parse_args()

    cfg = load_config(args.config).cfg
    service = BufferedInferenceService(cfg, ckpt_path=None)
    frame = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    audio = (np.random.randn(int(cfg.features.audio_sr * 0.2)).astype(np.float32) * 0.01)

    start = time.perf_counter()
    last = None
    completed = 0
    for _ in range(args.frames):
        service.push(frame, audio)
        last = service.run_live()
        if last.latency_ms > 0.0:
            completed += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)
    elapsed = time.perf_counter() - start
    fps = args.frames / max(elapsed, 1e-6)
    print(f"frames={args.frames} total_seconds={elapsed:.3f} fps={fps:.2f}")
    if last is not None:
        print(f"last_latency_ms={last.latency_ms:.2f} reported_fps={last.fps:.2f}")
    print(f"completed_inference_updates={completed}")


if __name__ == "__main__":
    main()
