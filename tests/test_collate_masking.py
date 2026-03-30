from __future__ import annotations

import numpy as np

from mmds.data.collate import collate_samples
from mmds.data.schema import Sample


def test_collate_pads_and_masks_modalities() -> None:
    s1 = Sample(
        sample_id="a",
        subject_id="s1",
        dataset_name="x",
        binary_label=1,
        ordinal_label=2,
        continuous_score=0.7,
        modality_payloads={"audio": np.ones((5, 3), dtype=np.float32)},
        modality_masks={"audio": True},
        metadata={},
    )
    s2 = Sample(
        sample_id="b",
        subject_id="s2",
        dataset_name="x",
        binary_label=None,
        ordinal_label=1,
        continuous_score=None,
        modality_payloads={"audio": np.ones((2, 3), dtype=np.float32) * 2.0},
        modality_masks={"audio": True},
        metadata={},
    )

    batch = collate_samples([s1, s2])
    assert batch.x["audio"].shape[0] == 2
    assert batch.x["audio"].shape[1] == 5
    assert batch.x_mask["audio"].shape == (2, 5)

    # label masking
    assert batch.y_mask_binary.tolist() == [True, False]
    assert batch.y_mask_continuous.tolist() == [True, False]
