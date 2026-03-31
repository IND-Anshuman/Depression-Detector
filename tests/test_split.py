from __future__ import annotations

from mmds.data.schema import Sample
from mmds.training.split import subject_stratified_split


def test_subject_split_includes_class_two_subjects() -> None:
    samples = [
        Sample(sample_id="s0", subject_id="a", dataset_name="x", ordinal_label=0, modality_payloads={}, modality_masks={}, metadata={}),
        Sample(sample_id="s1", subject_id="b", dataset_name="x", ordinal_label=1, modality_payloads={}, modality_masks={}, metadata={}),
        Sample(sample_id="s2", subject_id="c", dataset_name="x", ordinal_label=2, modality_payloads={}, modality_masks={}, metadata={}),
        Sample(sample_id="s3", subject_id="d", dataset_name="x", ordinal_label=2, modality_payloads={}, modality_masks={}, metadata={}),
    ]
    split = subject_stratified_split(samples, seed=42, val_frac=0.25, test_frac=0.25)
    all_subjects = {sample.subject_id for sample in split.train + split.val + split.test}
    assert all_subjects == {"a", "b", "c", "d"}
