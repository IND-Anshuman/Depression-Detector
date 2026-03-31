from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from mmds.features.registry import build_extractor
from mmds.features.windowing import WindowingConfig, slice_windows_audio, slice_windows_frames


@dataclass(frozen=True)
class OfflineExtractionPaths:
    out_dir: Path
    manifest_csv: Path


def _read_video_frames(video_path: Path, target_fps: float) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or target_fps)
    step = max(int(round(fps / target_fps)), 1)

    frames: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames, target_fps


def _read_audio_best_effort(audio_path: Path, target_sr: int) -> tuple[np.ndarray, int] | tuple[None, None]:
    try:
        import torchaudio  # type: ignore
        import torch

        wav, sr = torchaudio.load(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != int(target_sr):
            wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
            sr = target_sr
        arr = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return arr, int(sr)
    except Exception:
        return None, None


def extract_features_from_manifest(cfg: DictConfig, in_manifest_csv: Path, paths: OfflineExtractionPaths) -> None:
    """Offline extraction from a prepared manifest CSV.

    Expected columns:
      - sample_id, subject_id, dataset_name
      - video_path (required)
      - audio_path (optional)
      - labels (optional): binary_label, ordinal_label, continuous_score

    Writes:
      - .npz per window
      - output manifest.csv with one row per window
    """

    df = pd.read_csv(in_manifest_csv)
    extractor = build_extractor(cfg)

    paths.out_dir.mkdir(parents=True, exist_ok=True)

    wcfg = WindowingConfig(
        window_seconds=float(cfg.features.window_seconds),
        fps=float(cfg.features.fps),
        audio_sr=int(cfg.features.audio_sr),
    )

    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])
        subject_id = str(r["subject_id"])
        dataset_name = str(r.get("dataset_name", "custom"))

        video_path = Path(str(r["video_path"]))
        frames, fps = _read_video_frames(video_path, target_fps=wcfg.fps)
        frame_windows = slice_windows_frames(frames, wcfg)

        audio_arr = None
        audio_sr = None
        if "audio_path" in r and pd.notna(r.get("audio_path")):
            audio_arr, audio_sr = _read_audio_best_effort(Path(str(r.get("audio_path"))), target_sr=wcfg.audio_sr)

        audio_windows = slice_windows_audio(audio_arr, wcfg) if audio_arr is not None else []

        n = max(len(frame_windows), len(audio_windows) if audio_windows else 0)
        for wi in range(n):
            fw = frame_windows[wi] if wi < len(frame_windows) else []
            aw = audio_windows[wi] if wi < len(audio_windows) else None

            res = extractor.extract_window(fw, fps, aw, audio_sr)

            out_npz = paths.out_dir / f"{sample_id}-w{wi:04d}.npz"
            np.savez_compressed(
                out_npz,
                **{f"x__{k}": v.astype(np.float32) for k, v in res.modality_payloads.items()},
                mask=np.array([int(res.modality_masks.get(k, True)) for k in sorted(res.modality_masks.keys())], dtype=np.int8),
                mask_keys=np.array(sorted(res.modality_masks.keys())),
            )
            meta = {
                "quality_warnings": res.quality_warnings,
                "debug": res.debug,
                "source_video": str(video_path),
            }
            (paths.out_dir / f"{sample_id}-w{wi:04d}.json").write_text(json.dumps(meta), encoding="utf-8")

            rows.append(
                {
                    "sample_id": f"{sample_id}-w{wi:04d}",
                    "subject_id": subject_id,
                    "dataset_name": dataset_name,
                    "window_index": wi,
                    "features_path": str(out_npz),
                    "binary_label": r.get("binary_label"),
                    "ordinal_label": r.get("ordinal_label"),
                    "ordinal_label_3class": r.get("ordinal_label_3class", r.get("ordinal_label")),
                    "continuous_score": r.get("continuous_score"),
                    "bdd_score": res.debug.get("bdd_score", r.get("bdd_score")),
                    "split": r.get("split"),
                    "severity_label_4class": r.get("severity_label_4class"),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(paths.manifest_csv, index=False)
