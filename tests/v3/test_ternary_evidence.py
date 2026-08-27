import torch

from radio_gs.v3.lifting.ternary_evidence import aggregate_mask_evidence


def test_unknown_is_preserved_and_never_folded_into_negative():
    evidence = aggregate_mask_evidence(
        gaussian_ids=torch.tensor([0, 0, 0, 1]),
        weights=torch.tensor([0.5, 0.25, 0.25, 1.0]),
        labels=torch.tensor([1, 0, -1, -1]),
        boundary=torch.tensor([0, 1, 0, 1]),
        scales=torch.tensor([0.2, 0.2, 0.2, 0.8]),
        qualities=torch.ones(4),
        view_ids=torch.tensor([2, 2, 3, 2]),
        num_gaussians=2,
    )
    torch.testing.assert_close(evidence.positive_mass, torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(evidence.negative_mass, torch.tensor([0.25, 0.0]))
    torch.testing.assert_close(evidence.unknown_mass, torch.tensor([0.25, 1.0]))
    assert evidence.view_count.tolist() == [2.0, 1.0]
