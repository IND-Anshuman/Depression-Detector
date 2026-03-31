from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _severity_bucket(phq8: float | None) -> tuple[int | None, int | None, float | None]:
    if phq8 is None:
        return None, None, None
    value = float(phq8)
    if value < 5:
        sev4 = 0
        sev3 = 0
    elif value < 10:
        sev4 = 1
        sev3 = 1
    elif value < 15:
        sev4 = 2
        sev3 = 2
    else:
        sev4 = 3
        sev3 = 2
    return sev4, sev3, min(max(value / 24.0, 0.0), 1.0)


def _read_split_csv(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    participant_col = cols.get("participant_id") or cols.get("participant id")
    if participant_col is None:
        raise ValueError(f"{path} is missing a participant_id column")
    phq_col = cols.get("phq8_score") or cols.get("phq8 score") or cols.get("phq_score")
    records = []
    for _, row in df.iterrows():
        subject_id = int(row[participant_col])
        phq_value = None
        if phq_col is not None and pd.notna(row[phq_col]):
            phq_value = float(row[phq_col])
        severity4, ordinal3, continuous = _severity_bucket(phq_value)
        binary = int(phq_value >= 10.0) if phq_value is not None else None
        records.append(
            {
                "subject_id": f"{subject_id}",
                "binary_label": binary,
                "ordinal_label_3class": ordinal3,
                "severity_label_4class": severity4,
                "continuous_score": continuous,
                "split": split_name,
            }
        )
    return pd.DataFrame.from_records(records)


def _locate_session_dir(root: Path, subject_id: str) -> Path | None:
    candidates = [
        root / f"{subject_id}_P",
        root / subject_id,
        root / "DAIC-WOZ" / f"{subject_id}_P",
        root / "DAIC-WOZ" / subject_id,
        root / "AVEC2017" / f"{subject_id}_P",
        root / "AVEC2017" / subject_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in root.rglob(f"{subject_id}_P"):
        if candidate.is_dir():
            return candidate
    for candidate in root.rglob(subject_id):
        if candidate.is_dir():
            return candidate
    return None


def _find_media_file(session_dir: Path, subject_id: str, suffixes: list[str]) -> str | None:
    stems = [
        f"{subject_id}_P",
        f"{subject_id}_P_SYNC",
        f"{subject_id}",
    ]
    for stem in stems:
        for suffix in suffixes:
            candidate = session_dir / f"{stem}{suffix}"
            if candidate.exists():
                return str(candidate)
    for suffix in suffixes:
        matches = list(session_dir.glob(f"*{suffix}"))
        if matches:
            return str(matches[0])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a DAIC-WOZ/AVEC-style raw manifest for offline feature extraction.")
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--splits-dir", default=None, help="Directory containing AVEC/DAIC split CSVs. Defaults to root-dir.")
    ap.add_argument("--out-manifest", default="data/manifests/daic_raw_manifest.csv")
    ap.add_argument("--require-video", action="store_true")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    splits_dir = Path(args.splits_dir) if args.splits_dir else root_dir
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", splits_dir / "full_train_split_Depression_AVEC2017.csv"),
        ("val", splits_dir / "full_dev_split_Depression_AVEC2017.csv"),
        ("test", splits_dir / "full_test_split.csv"),
    ]

    frames = []
    for split_name, csv_path in split_specs:
        if csv_path.exists():
            frames.append(_read_split_csv(csv_path, split_name))

    if not frames:
        raise FileNotFoundError(
            f"No DAIC/AVEC split CSVs found under {splits_dir}. "
            "Expected files like full_train_split_Depression_AVEC2017.csv and full_dev_split_Depression_AVEC2017.csv."
        )

    labels = pd.concat(frames, ignore_index=True)
    rows = []
    missing_sessions = []
    for _, row in labels.iterrows():
        subject_id = str(row["subject_id"])
        session_dir = _locate_session_dir(root_dir, subject_id)
        if session_dir is None:
            missing_sessions.append(subject_id)
            continue
        video_path = _find_media_file(session_dir, subject_id, [".mp4", ".avi", ".mov"])
        audio_path = _find_media_file(session_dir, subject_id, ["_AUDIO.wav", ".wav"])
        if args.require_video and video_path is None:
            missing_sessions.append(subject_id)
            continue

        rows.append(
            {
                "sample_id": f"daic-{subject_id}",
                "subject_id": subject_id,
                "dataset_name": "daic_woz",
                "video_path": video_path,
                "audio_path": audio_path,
                "binary_label": row["binary_label"],
                "ordinal_label_3class": row["ordinal_label_3class"],
                "severity_label_4class": row["severity_label_4class"],
                "continuous_score": row["continuous_score"],
                "split": row["split"],
            }
        )

    if not rows:
        raise RuntimeError("No DAIC sessions were resolved into a raw manifest.")

    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Wrote {len(rows)} DAIC raw rows to {out_manifest}")
    if missing_sessions:
        print(f"Skipped {len(missing_sessions)} DAIC subjects with unresolved media/session paths.")


if __name__ == "__main__":
    main()
