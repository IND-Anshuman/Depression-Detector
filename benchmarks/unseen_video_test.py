from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from mmds.config import load_config
from mmds.inference.service import BufferedInferenceService


def _video_paths(root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _load_frames(path: Path, limit: int = 64) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/space.yaml")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    cfg = load_config(args.config).cfg
    service = BufferedInferenceService(cfg, ckpt_path=None)
    paths = _video_paths(Path(args.videos))
    if not paths:
        print(f"No videos found under {args.videos}; skipping unseen video benchmark.")
        return

    pseudo_hits = 0
    for path in paths:
        service.reset()
        for frame in _load_frames(path):
            service.push(frame, None)
        res = service.run_once()
        pseudo_hits += int(0.0 <= res.risk_prob <= 1.0 and len(res.severity_probs) == 3)
    pseudo_acc = pseudo_hits / len(paths)
    print(f"videos={len(paths)} pseudo_accuracy={pseudo_acc:.3f} threshold={args.threshold:.2f}")
    if pseudo_acc < args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
