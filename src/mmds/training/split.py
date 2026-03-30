from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmds.data.schema import Sample


@dataclass(frozen=True)
class Split:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def subject_stratified_split(
    samples: list[Sample],
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Split:
    """Split by subject id to avoid leakage.

    For partially labeled datasets this uses best-effort stratification on binary_label.
    """

    rng = np.random.default_rng(seed)
    subjects = sorted({s.subject_id for s in samples})

    # Build subject-level label for stratification if available.
    subj_to_label: dict[str, int] = {}
    for sid in subjects:
        ys = [s.binary_label for s in samples if s.subject_id == sid and s.binary_label is not None]
        subj_to_label[sid] = int(round(float(np.mean(ys)))) if ys else 0

    # Shuffle within strata.
    strata: dict[int, list[str]] = {0: [], 1: []}
    for sid in subjects:
        strata[subj_to_label.get(sid, 0)].append(sid)
    for k in strata:
        rng.shuffle(strata[k])

    ordered = strata[0] + strata[1]

    n = len(ordered)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_s = set(ordered[:n_test])
    val_s = set(ordered[n_test : n_test + n_val])
    train_s = set(ordered[n_test + n_val :])

    train = [s for s in samples if s.subject_id in train_s]
    val = [s for s in samples if s.subject_id in val_s]
    test = [s for s in samples if s.subject_id in test_s]

    return Split(train=train, val=val, test=test)
