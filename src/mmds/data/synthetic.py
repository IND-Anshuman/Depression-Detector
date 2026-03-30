from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Sample


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    num_subjects: int = 8
    windows_per_subject: int = 6
    seq_len: int = 32
    seed: int = 123


def make_synthetic_samples(cfg: SyntheticDatasetConfig) -> list[Sample]:
    """Synthetic multimodal dataset used for tests and a lightweight demo.

    The goal is *runnable* plumbing, not realism.
    """

    rng = np.random.default_rng(cfg.seed)
    samples: list[Sample] = []

    for subj in range(cfg.num_subjects):
        subject_id = f"S{subj:03d}"
        # Make a correlated continuous score, then derive labels.
        base = float(rng.normal(loc=0.0, scale=1.0))
        continuous = float(np.clip(0.5 + 0.2 * base + rng.normal(0, 0.05), 0.0, 1.0))
        binary = int(continuous > 0.55)
        ordinal = int(0 if continuous < 0.4 else (1 if continuous < 0.7 else 2))

        for w in range(cfg.windows_per_subject):
            t = cfg.seq_len

            def ts(dim: int, shift: float = 0.0) -> np.ndarray:
                return (rng.normal(size=(t, dim)).astype(np.float32) + shift).astype(np.float32)

            # Make face/body slightly more predictive.
            face_au = ts(16, shift=0.5 * base)
            body_pose = ts(24, shift=0.3 * base)
            audio = ts(32, shift=0.2 * base)

            # Randomly drop a modality to exercise masking.
            drop_audio = rng.random() < 0.15
            drop_body = rng.random() < 0.10

            payloads = {
                "face_au": face_au,
                "body_pose": body_pose if not drop_body else None,
                "audio": audio if not drop_audio else None,
                "quality": ts(6),
            }
            payloads = {k: v for k, v in payloads.items() if v is not None}
            masks = {k: (k in payloads) for k in ["face_au", "body_pose", "audio", "quality"]}

            samples.append(
                Sample(
                    sample_id=f"{subject_id}-w{w:02d}",
                    subject_id=subject_id,
                    dataset_name="synthetic",
                    binary_label=binary,
                    ordinal_label=ordinal,
                    continuous_score=continuous,
                    window_index=w,
                    modality_payloads=payloads,
                    modality_masks=masks,
                    metadata={},
                )
            )

    return samples
