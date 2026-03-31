from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LIKELY_DEPRESSION_TERMS = {"depression", "depvidmood", "phq", "bdd", "severity"}
LIKELY_EMOTION_TERMS = {"angry", "happy", "fear", "disgust", "neutral", "surprize", "surprise", "actor", "ravdess"}


def _discover_files(root: Path) -> tuple[list[Path], list[Path]]:
    videos: list[Path] = []
    metas: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTS:
            videos.append(path)
        if suffix in {".csv", ".json", ".jsonl"}:
            metas.append(path)
    return videos, metas


def _contains_terms(paths: list[Path], terms: set[str]) -> bool:
    text = " ".join(str(p).lower() for p in paths[:200])
    return any(term in text for term in terms)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check whether local real-data prerequisites are ready.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--manifest", default="data/manifests/depvidmood_feature_manifest.csv")
    ap.add_argument("--feature-manifest", default="artifacts/features/depvidmood_features_manifest.csv")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    manifest = Path(args.manifest)
    feature_manifest = Path(args.feature_manifest)

    print(f"raw_dir_exists={raw_dir.exists()} path={raw_dir}")
    if raw_dir.exists():
        videos, metas = _discover_files(raw_dir)
        print(f"video_count={len(videos)} metadata_files={len(metas)}")
        if videos:
            print(f"first_video={videos[0]}")
        if metas:
            print(f"first_metadata={metas[0]}")
        if _contains_terms(videos + metas, LIKELY_EMOTION_TERMS) and not _contains_terms(videos + metas, LIKELY_DEPRESSION_TERMS):
            print("warning=raw data appears emotion/actor-style rather than depression-labeled")
        elif not metas:
            print("warning=no metadata CSV/JSON files found; manifest building will rely on path heuristics only")

    print(f"manifest_exists={manifest.exists()} path={manifest}")
    if manifest.exists():
        df = pd.read_csv(manifest, nrows=5)
        print(f"manifest_columns={list(df.columns)}")

    print(f"feature_manifest_exists={feature_manifest.exists()} path={feature_manifest}")
    if feature_manifest.exists():
        df = pd.read_csv(feature_manifest, nrows=5)
        print(f"feature_manifest_columns={list(df.columns)}")


if __name__ == "__main__":
    main()
