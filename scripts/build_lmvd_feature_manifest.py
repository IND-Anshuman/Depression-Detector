from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mmds.features.compact_audio import build_compact_audio_features
from mmds.features.compact_visual import build_compact_visual_modalities, derive_behavioral_stats


LMVD_AU_R_COLS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
]
LMVD_HEAD_POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
LMVD_GAZE_COLS = ["gaze_0_x", "gaze_0_y", "gaze_1_x", "gaze_1_y"]
LMVD_BLINK_COLS = ["AU45_r", "AU45_c"]


def _binary_from_video_id(video_id: int) -> int:
    if 1 <= video_id <= 601 or 1117 <= video_id <= 1423:
        return 1
    if 602 <= video_id <= 1116 or 1425 <= video_id <= 1824:
        return 0
    raise ValueError(f"Unsupported LMVD video id for label mapping: {video_id}")


def _subject_split(df: pd.DataFrame, seed: int, subject_col: str = "subject_id") -> dict[str, str]:
    rng = np.random.default_rng(seed)
    subjects = df[[subject_col, "binary_label"]].drop_duplicates().sort_values(subject_col)
    split_map: dict[str, str] = {}
    for label_value in [0, 1]:
        ids = subjects.loc[subjects["binary_label"] == label_value, subject_col].tolist()
        rng.shuffle(ids)
        n = len(ids)
        test_n = max(1, int(round(n * 0.15))) if n >= 3 else max(0, int(round(n * 0.15)))
        val_n = max(1, int(round(n * 0.15))) if n >= 3 else max(0, int(round(n * 0.15)))
        if test_n + val_n >= n and n >= 3:
            val_n = max(1, val_n - 1)
        for sid in ids[:test_n]:
            split_map[sid] = "test"
        for sid in ids[test_n : test_n + val_n]:
            split_map[sid] = "val"
        for sid in ids[test_n + val_n :]:
            split_map[sid] = "train"
    return split_map


def _downsample_indices(length: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or length <= max_frames:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_frames, dtype=np.int64)


def _temporal_downsample(array: np.ndarray, max_frames: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, received shape {array.shape}")
    return array[_downsample_indices(array.shape[0], max_frames)]


def _standardize(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-5, 1.0, std)
    return np.clip((arr - mean) / std, -3.0, 3.0).astype(np.float32)


def _load_visual_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "frame" in df.columns:
        sort_cols = ["frame"]
    else:
        sort_cols = []
    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False)
    if sort_cols:
        df = df.drop_duplicates(subset=sort_cols).sort_values(sort_cols)

    x_cols = [f"x_{i}" for i in range(68)]
    y_cols = [f"y_{i}" for i in range(68)]
    missing = [c for c in x_cols + y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"LMVD video CSV missing expected landmark columns: {missing[:8]}")
    return df


def _build_lmvd_visual_payloads(df: pd.DataFrame, max_frames: int) -> dict[str, np.ndarray]:
    x_cols = [f"x_{i}" for i in range(68)]
    y_cols = [f"y_{i}" for i in range(68)]
    landmarks = np.concatenate(
        [df[x_cols].to_numpy(dtype=np.float32), df[y_cols].to_numpy(dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    confidence = df["confidence"].to_numpy(dtype=np.float32) if "confidence" in df.columns else np.ones((len(df),), dtype=np.float32)
    success = df["success"].to_numpy(dtype=np.float32) if "success" in df.columns else np.ones((len(df),), dtype=np.float32)

    idx = _downsample_indices(len(df), max_frames)
    landmarks = landmarks[idx]
    confidence = confidence[idx]
    success = success[idx]
    payloads = build_compact_visual_modalities(landmarks, confidence=confidence, success=success)

    if all(col in df.columns for col in LMVD_AU_R_COLS):
        payloads["face_au"] = (df.loc[:, LMVD_AU_R_COLS].to_numpy(dtype=np.float32)[idx] / 5.0).astype(np.float32)
    if all(col in df.columns for col in LMVD_HEAD_POSE_COLS):
        payloads["head_pose"] = _standardize(df.loc[:, LMVD_HEAD_POSE_COLS].to_numpy(dtype=np.float32)[idx])
    if all(col in df.columns for col in LMVD_GAZE_COLS):
        payloads["gaze"] = _standardize(df.loc[:, LMVD_GAZE_COLS].to_numpy(dtype=np.float32)[idx])
    if all(col in df.columns for col in LMVD_BLINK_COLS):
        blink = df.loc[:, LMVD_BLINK_COLS].to_numpy(dtype=np.float32)[idx]
        blink[:, 0] = blink[:, 0] / 5.0
        payloads["blink"] = blink.astype(np.float32)

    payloads["behavioral_stats"] = derive_behavioral_stats(payloads["face_au"], payloads["head_pose"])
    quality = np.asarray(payloads["quality"], dtype=np.float32)
    quality[:, 0] = confidence.astype(np.float32)
    quality[:, 1] = success.astype(np.float32)
    payloads["quality"] = quality.astype(np.float32)
    return payloads


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert LMVD feature zip into an MMDS compact visual manifest.")
    ap.add_argument("--zip-path", required=True)
    ap.add_argument("--out-dir", default="artifacts/features/lmvd")
    ap.add_argument("--out-manifest", default="artifacts/features/lmvd_features_manifest.csv")
    ap.add_argument("--max-frames", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for smoke testing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"LMVD zip not found: {zip_path}")

    rows = []
    missing_entries: list[str] = []
    id_records = []

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        video_ids = sorted(
            {
                Path(name).stem
                for name in names
                if name.startswith("Video_feature/") and name.lower().endswith(".csv")
            }
        )
        if args.limit > 0:
            video_ids = video_ids[: args.limit]

        for video_id in video_ids:
            vid_int = int(video_id)
            csv_name = f"Video_feature/{video_id}.csv"
            if csv_name not in names:
                missing_entries.append(video_id)
                continue
            try:
                binary = _binary_from_video_id(vid_int)
            except ValueError:
                continue
            id_records.append({"subject_id": video_id, "binary_label": binary})

        split_df = pd.DataFrame(id_records)
        split_map = _subject_split(split_df, seed=args.seed) if not split_df.empty else {}

        for rec in id_records:
            video_id = rec["subject_id"]
            vid_int = int(video_id)
            csv_name = f"Video_feature/{video_id}.csv"
            audio_name = f"Audio_feature/{video_id}.npy"
            visual_df = _load_visual_csv(zf.read(csv_name))
            payloads = _build_lmvd_visual_payloads(visual_df, args.max_frames)
            if audio_name in names:
                audio = np.load(io.BytesIO(zf.read(audio_name))).astype(np.float32)
                audio = _temporal_downsample(audio, args.max_frames)
                payloads["audio"] = build_compact_audio_features(audio)

            out_npz = out_dir / f"lmvd-{video_id}.npz"
            np.savez_compressed(
                out_npz,
                **{f"x__{k}": v for k, v in payloads.items()},
                mask=np.ones((len(payloads),), dtype=np.int8),
                mask_keys=np.array(list(payloads.keys())),
            )
            rows.append(
                {
                    "sample_id": f"lmvd-{video_id}",
                    "subject_id": video_id,
                    "dataset_name": "lmvd",
                    "window_index": 0,
                    "features_path": str(out_npz),
                    "binary_label": int(rec["binary_label"]),
                    "ordinal_label_3class": int(2 if rec["binary_label"] == 1 else 0),
                    "severity_label_4class": int(3 if rec["binary_label"] == 1 else 0),
                    "continuous_score": float(rec["binary_label"]),
                    "split": split_map.get(video_id, "train"),
                    "video_id": video_id,
                    "source_zip": str(zip_path),
                }
            )

    if not rows:
        raise RuntimeError("No LMVD feature rows were created from the archive.")
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Wrote {len(rows)} LMVD feature rows to {out_manifest}")
    if missing_entries:
        print(f"Skipped {len(missing_entries)} LMVD entries missing expected files.")


if __name__ == "__main__":
    main()
