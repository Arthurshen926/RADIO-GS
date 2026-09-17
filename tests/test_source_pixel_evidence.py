import torch
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights


def test_partition_conserves_covered_pixel_evidence_and_excludes_uncovered():
    masks = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.bool)
    pid = torch.tensor([0, 0, 1, 2])
    gid = torch.tensor([0, 1, 1, 2])
    weights, mass = source_pixel_evidence_weights(
        masks, pid, gid, torch.tensor([1., 3., 2., 1.]), 3, partition_overlap=True)
    torch.testing.assert_close(weights.sum(0), torch.tensor([.25, 1.75, 0.]))
    torch.testing.assert_close(mass, torch.tensor([.25, 1.75, 1.]))
    raw, _ = source_pixel_evidence_weights(
        masks, pid, gid, torch.tensor([1., 3., 2., 1.]), 3, partition_overlap=False)
    assert raw.sum() > weights.sum()


def test_duplicate_identical_evidence_does_not_gain_total_partition_mass():
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    args = (torch.tensor([0, 1]), torch.tensor([0, 0]), torch.ones(2), 1)
    single, _ = source_pixel_evidence_weights(mask, *args, partition_overlap=True)
    duplicate, _ = source_pixel_evidence_weights(mask.repeat(3, 1), *args, partition_overlap=True)
    torch.testing.assert_close(single.sum(0), duplicate.sum(0))


def test_tiny_element_support_is_not_promoted_to_full_evidence():
    weights, mass = source_pixel_evidence_weights(
        torch.ones((1, 1), dtype=torch.bool), torch.tensor([0, 0]),
        torch.tensor([0, 1]), torch.tensor([1., 1e-8]), 2, partition_overlap=True)
    assert mass[1] > 0
    assert weights[0, 1] == 0
