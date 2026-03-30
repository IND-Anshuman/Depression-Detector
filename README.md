# MMDS — Multimodal Depression Screening (Research + Demo)

**MMDS** is a research-grade, demo-ready multimodal *screening* pipeline that produces **depression risk scores** from **non-verbal video/audio behavioral indicators** (e.g., proxy facial activity, motion variability, signal quality, and uncertainty).

## Disclaimer (read first)

- This is a **screening/research** tool. It is **not** a diagnostic medical system and must not be used to diagnose, treat, or prevent any disease.
- Outputs are **risk scores with uncertainty** derived from behavioral indicators and signal quality.
- Model predictions can be wrong, biased, or poorly calibrated outside the data distribution.

## What it does

Multi-task outputs (per buffered window):
- **Binary depression risk** probability
- **3-class severity** probability (low / moderate / high)
- **Continuous severity score** in $[0,1]$
- **Uncertainty / confidence** + **abstain recommendation** (MC-dropout)
- **BDD-style behavioral variability** score (auxiliary; kept separate from depression outputs)

Interpretable visualizations (demo path ships lightweight proxies):
- Proxy **AU activity trend**
- **Temporal importance** proxy (cross-attention aggregation or token norms)
- **Signal quality** indicators + warnings
- Rolling risk score

## Architecture

```mermaid
flowchart LR
  A[Webcam frames + optional mic] --> B[Buffered windowing]
  B --> C[Feature extractor backend]
  C -->|modality time-series + masks| D[Modality encoders]
  D --> E[TSFFM-inspired FaceBodyFusionBlock (module)]
  E --> F[Global fusion backbone (Perceiver-style / Transformer)]
  F --> G[Multi-task heads]
  G --> H[Risk + severity + continuous + (optional) BDD]
  F --> I[Token importance + summaries]
  C --> J[Signal quality + warnings]
  H --> K[MC-dropout uncertainty + abstain]
  H --> L[Gradio dashboard]
  I --> L
  J --> L
  K --> L
```

### Feature backends (pluggable)
- **`simple` (default demo)**: pure Python/OpenCV *proxy* indicators for AU-like activity, motion, brightness/blur.
- **`mediapipe` (optional)**: currently falls back to `simple` with a clear swap-point for FaceMesh/Pose.
- **`openface` (research)**: stub that intentionally fails unless you provide an OpenFace-enabled environment; use offline extraction.

## Quickstart

### 1) Install

Python 3.11 is required.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
```

### 2) Run the demo (buffered webcam)

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

The dashboard provides **risk scores** with **uncertainty** and a **privacy note**. It runs on CPU by default.

### 3) Train (out-of-the-box synthetic data)

This repo ships a synthetic dataset path so the entire pipeline runs without private data.

```bash
python scripts/train.py --config configs/demo.yaml
# checkpoint saved to artifacts/checkpoint.pt
```

### 4) Evaluate

```bash
python scripts/evaluate.py --config configs/demo.yaml --ckpt artifacts/checkpoint.pt --out artifacts/eval
```

Artifacts:
- `artifacts/eval/predictions.csv`
- `artifacts/eval/metrics.json`
- `artifacts/eval/roc.png`, `artifacts/eval/calibration.png`

## Offline feature extraction (research-friendly)

Training should not depend on video decoding at runtime. The intended workflow is:

1) Prepare an input manifest CSV, e.g. `my_manifest.csv` with columns:
   - `sample_id`, `subject_id`, `dataset_name`, `video_path`
   - optional: `audio_path`, `binary_label`, `ordinal_label`, `continuous_score`
2) Extract windowed features into `.npz` + write a features manifest:

```bash
python scripts/extract_features.py \
  --config configs/demo.yaml \
  --in-manifest my_manifest.csv \
  --out-dir artifacts/features \
  --out-manifest artifacts/features_manifest.csv
```

3) Train using the features manifest:
- set `dataset.name: feature_manifest`
- set `dataset.manifest_csv: artifacts/features_manifest.csv`

## Hugging Face Spaces (Docker Space)

- Entry point: `app/gradio_app.py`
- Default config: `configs/space.yaml`

Build/run locally:
```bash
docker build -t mmds .
docker run -p 7860:7860 mmds
```

## Output interpretation

- **Risk probability**: screening-oriented estimate, not a diagnosis.
- **Severity (3-class)**: ordinal severity bucket; interpretation depends on dataset mapping.
- **Continuous score**: normalized severity indicator in $[0,1]$.
- **Uncertainty/confidence**: derived from MC-dropout variability (epistemic) and regression variance.
- **Abstain recommendation**: triggers when uncertainty exceeds `inference.abstain_uncertainty_threshold`.

### BDD-style behavioral variability (auxiliary)
- Computed post-hoc from the entropy of AU-like activity and body-motion signals.
- Intended as an **auxiliary behavioral descriptor**, not a depression label.
- Kept **separate** from the depression heads in code and UI.

## Roadmap / extensions

- Real MediaPipe landmarks → head pose / blink / gaze features
- Subject-level temporal aggregation (per-video) and calibration metrics
- Proper ablation + missing-modality robustness reports
- Optional W&B logging behind `training.use_wandb`
- OpenFace extraction container + AU trend UI that uses true AU semantics

## Limitations

- The default demo backend uses **proxy** features (not clinically validated AUs/pose).
- Uncertainty is **heuristic** and may be miscalibrated.
- Dataset adapters are scaffolds; you should provide prepared manifests for reproducibility.
