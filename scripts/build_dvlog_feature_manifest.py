from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mmds.features.compact_audio import build_compact_audio_features
from mmds.features.compact_visual import build_compact_visual_modalities


def _binary_from_label(label: str) -> int:
    lowered = label.strip().lower()
    if lowered == "depression":
        return 1
    if lowered == "normal":
        return 0
    raise ValueError(f"Unsupported DVlog label: {label!r}")


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


def _temporal_downsample(array: np.ndarray, max_frames: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, received shape {array.shape}")
    if max_frames <= 0 or array.shape[0] <= max_frames:
        return array
    indices = np.linspace(0, array.shape[0] - 1, num=max_frames, dtype=np.int64)
    return array[indices]


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a DVlog zip archive + labels CSV into an MMDS feature manifest.")
    ap.add_argument("--zip-path", required=True)
    ap.add_argument("--links-csv", required=True)
    ap.add_argument("--out-dir", default="artifacts/features/dvlog")
    ap.add_argument("--out-manifest", default="artifacts/features/dvlog_features_manifest.csv")
    ap.add_argument("--max-frames", type=int, default=192)
    ap.add_argument("--schema", choices=["legacy", "compact_visual"], default="legacy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    links_csv = Path(args.links_csv)
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"DVlog zip not found: {zip_path}")
    if not links_csv.exists():
        raise FileNotFoundError(f"DVlog links CSV not found: {links_csv}")

    labels = pd.read_csv(links_csv)
    required = {"video_id", "label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"DVlog CSV is missing required columns: {missing}")

    labels = labels.copy()
    labels["video_id"] = labels["video_id"].astype(str)
    labels["binary_label"] = labels["label"].map(_binary_from_label)
    labels["ordinal_label_3class"] = labels["binary_label"].map(lambda x: 2 if x == 1 else 0)
    labels["severity_label_4class"] = labels["binary_label"].map(lambda x: 3 if x == 1 else 0)
    labels["continuous_score"] = labels["binary_label"].astype(float)
    labels["subject_id"] = labels["channelId"].astype(str) if "channelId" in labels.columns else labels["video_id"]
    split_map = _subject_split(labels[["subject_id", "binary_label"]].drop_duplicates(), seed=args.seed)
    labels["split"] = labels["subject_id"].map(split_map).fillna("train")

    rows = []
    missing_entries: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for _, row in labels.iterrows():
            video_id = str(row["video_id"])
            acoustic_name = f"dvlog-dataset/{video_id}/{video_id}_acoustic.npy"
            visual_name = f"dvlog-dataset/{video_id}/{video_id}_visual.npy"
            if acoustic_name not in names or visual_name not in names:
                missing_entries.append(video_id)
                continue

            acoustic = np.load(io.BytesIO(zf.read(acoustic_name))).astype(np.float32)
            visual = np.load(io.BytesIO(zf.read(visual_name))).astype(np.float32)
            seq_len = min(acoustic.shape[0], visual.shape[0])
            acoustic = acoustic[:seq_len]
            visual = visual[:seq_len]
            acoustic = _temporal_downsample(acoustic, args.max_frames)
            visual = _temporal_downsample(visual, args.max_frames)
            out_npz = out_dir / f"dvlog-{video_id}.npz"
            if args.schema == "compact_visual":
                payloads = build_compact_visual_modalities(visual)
                payloads["audio"] = build_compact_audio_features(acoustic)
                np.savez_compressed(
                    out_npz,
                    **{f"x__{k}": v for k, v in payloads.items()},
                    mask=np.ones((len(payloads),), dtype=np.int8),
                    mask_keys=np.array(list(payloads.keys())),
                )
            else:
                np.savez_compressed(
                    out_npz,
                    x__audio=acoustic,
                    x__face_landmarks=visual,
                    mask=np.array([1, 1], dtype=np.int8),
                    mask_keys=np.array(["audio", "face_landmarks"]),
                )
            rows.append(
                {
                    "sample_id": f"dvlog-{video_id}",
                    "subject_id": str(row["subject_id"]),
                    "dataset_name": "dvlog_compact" if args.schema == "compact_visual" else "dvlog",
                    "window_index": 0,
                    "features_path": str(out_npz),
                    "binary_label": int(row["binary_label"]),
                    "ordinal_label_3class": int(row["ordinal_label_3class"]),
                    "severity_label_4class": int(row["severity_label_4class"]),
                    "continuous_score": float(row["continuous_score"]),
                    "split": str(row["split"]),
                    "video_id": video_id,
                    "youtube_key": row.get("key"),
                    "gender": row.get("gender"),
                    "duration": row.get("duration"),
                    "channelId": row.get("channelId"),
                }
            )

    if not rows:
        raise RuntimeError("No DVlog feature pairs were found in the zip archive.")
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Wrote {len(rows)} DVlog feature rows to {out_manifest} with max_frames={args.max_frames}")
    if missing_entries:
        print(f"Skipped {len(missing_entries)} video IDs missing feature files inside the zip.")


if __name__ == "__main__":
    main()
