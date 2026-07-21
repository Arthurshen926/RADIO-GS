import numpy as np
import torch

from radio_gs.scripts.audit_scannet_sam3_membership_coverage import _mask_statistics


def test_mask_statistics_reports_purity_completeness_and_union_support() -> None:
    rows, support = _mask_statistics(
        torch.tensor([[1.0, 0.9, 0.1, 0.0], [0.0, 0.8, 0.9, 0.0]]),
        torch.tensor([0.2, 0.5]), np.asarray([1, 1, 2, 0], dtype=np.int64),
        inside_threshold=0.8, frame="12.pt", source_mask_indices=torch.tensor([3, 5]),
    )
    assert support.tolist() == [True, True, True, False]
    assert rows[0]["dominant_instance_id"] == 1
    assert rows[0]["dominant_instance_purity"] == 1.0
    # The first mask covers both support primitives of instance 1.
    assert rows[0]["dominant_instance_completeness"] == 1.0
    assert rows[1]["dominant_instance_id"] == 1
    assert rows[1]["dominant_instance_purity"] == 0.5
