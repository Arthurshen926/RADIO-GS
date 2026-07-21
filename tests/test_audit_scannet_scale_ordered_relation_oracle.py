import torch

from radio_gs.scripts.audit_scannet_scale_ordered_relation_oracle import conservative_join_mask


def test_conservative_join_uses_only_consistent_same_upper_bounds() -> None:
    upper = torch.tensor([0.0, float("nan"), 0.5, 0.0])
    consistent = torch.tensor([True, True, True, False])
    early = conservative_join_mask(upper, consistent, log_radius=0.1)
    late = conservative_join_mask(upper, consistent, log_radius=0.6)
    assert early.tolist() == [True, False, False, False]
    assert late.tolist() == [True, False, True, False]
