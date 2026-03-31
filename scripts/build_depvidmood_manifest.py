from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEVERITY_MAP = {
    "not": 0,
    "none": 0,
    "normal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


def _normalize_severity(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing severity value")
    if isinstance(value, (int, np.integer)):
        iv = int(value)
        if iv < 0 or iv > 3:
            raise ValueError(f"Unsupported severity integer: {iv}")
        return iv
    if isinstance(value, float):
        iv = int(value)
        if iv < 0 or iv > 3:
            raise ValueError(f"Unsupported severity float: {value}")
        return iv
    lowered = str(value).strip().lower()
    if lowered.isdigit():
        return _normalize_severity(int(lowered))
    if lowered in SEVERITY_MAP:
        return SEVERITY_MAP[lowered]
    raise ValueError(f"Unsupported severity label: {value!r}")


def _severity_from_path(path: Path) -> int:
    for part in [path.parent.name, path.parent.parent.name, path.stem]:
        lowered = part.lower()
        for key, value in SEVERITY_MAP.items():
            if key in lowered:
                return value
    raise ValueError(f"Could not infer severity from path: {path}")


def _discover_videos(raw_dir: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return sorted([p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(metadata_path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("data", raw.get("items", []))
            return pd.DataFrame(raw)
        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported metadata file type: {metadata_path.suffix}")


def _assign_subject_splits(subjects: list[str], seed: int, train_frac: float, val_frac: float, test_frac: float) -> dict[str, str]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    rng = np.random.default_rng(seed)
    ordered = list(subjects)
    rng.shuffle(ordered)
    n = len(ordered)
    test_n = max(1, int(round(n * test_frac))) if n > 2 else max(0, int(round(n * test_frac)))
    val_n = max(1, int(round(n * val_frac))) if n > 2 else max(0, int(round(n * val_frac)))
    if test_n + val_n >= n and n >= 3:
        val_n = max(1, val_n - 1)
    split_map: dict[str, str] = {}
    for sid in ordered[:test_n]:
        split_map[sid] = "test"
    for sid in ordered[test_n : test_n + val_n]:
        split_map[sid] = "val"
    for sid in ordered[test_n + val_n :]:
        split_map[sid] = "train"
    return split_map


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a DepVidMood manifest from a raw extracted dataset directory.")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metadata-csv", default=None, help="Optional CSV/JSON metadata with explicit labels and paths.")
    ap.add_argument("--video-column", default="video_path")
    ap.add_argument("--severity-column", default="severity")
    ap.add_argument("--subject-column", default="subject_id")
    ap.add_argument("--sample-column", default="sample_id")
    ap.add_argument("--audio-dir", default=None, help="Optional directory used to attach audio paths by stem.")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--strict", action="store_true", help="Fail if any metadata row cannot be resolved.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_map: dict[str, Path] = {}
    if args.audio_dir:
        audio_root = Path(args.audio_dir)
        audio_exts = {".wav", ".mp3", ".flac", ".m4a"}
        audio_map = {p.stem: p for p in audio_root.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts}

    rows = []
    if args.metadata_csv:
        meta = _load_metadata(Path(args.metadata_csv))
        required = {args.video_column, args.severity_column}
        missing = [c for c in required if c not in meta.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        unresolved = []
        for idx, record in meta.iterrows():
            raw_video = str(record[args.video_column])
            video_path = Path(raw_video)
            if not video_path.is_absolute():
                candidate = raw_dir / raw_video
                if candidate.exists():
                    video_path = candidate
                else:
                    matches = list(raw_dir.rglob(video_path.name))
                    if matches:
                        video_path = matches[0]
            if not video_path.exists():
                unresolved.append((idx, raw_video))
                continue

            sample_id = (
                str(record[args.sample_column])
                if args.sample_column in meta.columns and pd.notna(record.get(args.sample_column))
                else video_path.stem
            )
            subject_id = (
                str(record[args.subject_column])
                if args.subject_column in meta.columns and pd.notna(record.get(args.subject_column))
                else sample_id.split("_")[0]
            )
            severity4 = _normalize_severity(record[args.severity_column])
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "dataset_name": "depvidmood",
                    "video_path": str(video_path),
                    "audio_path": str(audio_map.get(video_path.stem)) if video_path.stem in audio_map else None,
                    "severity_label_4class": severity4,
                    "ordinal_label_3class": min(severity4, 2),
                    "binary_label": int(severity4 > 0),
                    "continuous_score": float(severity4 / 3.0),
                }
            )
        if unresolved and args.strict:
            preview = ", ".join(f"row {i}: {p}" for i, p in unresolved[:5])
            raise FileNotFoundError(f"Could not resolve {len(unresolved)} metadata video paths. Examples: {preview}")
    else:
        videos = _discover_videos(raw_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found under {raw_dir}")
        for path in videos:
            severity4 = _severity_from_path(path)
            sample_id = path.stem
            subject_id = sample_id.split("_")[0]
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "dataset_name": "depvidmood",
                    "video_path": str(path),
                    "audio_path": str(audio_map.get(path.stem)) if path.stem in audio_map else None,
                    "severity_label_4class": severity4,
                    "ordinal_label_3class": min(severity4, 2),
                    "binary_label": int(severity4 > 0),
                    "continuous_score": float(severity4 / 3.0),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid DepVidMood rows were generated.")
    split_map = _assign_subject_splits(
        sorted(df["subject_id"].drop_duplicates().tolist()),
        seed=args.seed,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
    )
    df["split"] = df["subject_id"].map(split_map).fillna("train")
    df.to_csv(out_path, index=False)
    severity_counts = df["severity_label_4class"].value_counts().sort_index().to_dict()
    split_counts = df["split"].value_counts().sort_index().to_dict()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Subjects: {df['subject_id'].nunique()} | Severity counts: {severity_counts} | Split counts: {split_counts}")


if __name__ == "__main__":
    main()
