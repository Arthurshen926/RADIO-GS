import pytest
import torch
from radio_gs.training.proposal_invariant_fusion import proposal_invariant_pixel_weights, inverse_evidence_multiplicity
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights


def test_duplicate_whole_does_not_overrule_overlapping_part():
    masks = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    descriptors = torch.eye(2)
    ids = torch.arange(2)
    w, mass, _ = proposal_invariant_pixel_weights(masks, descriptors, ids, ids, torch.ones(2), 2)
    duplicated_masks = masks[[0, 0, 0, 1]]
    duplicated_desc = descriptors[[0, 0, 0, 1]]
    new, new_mass, _ = proposal_invariant_pixel_weights(duplicated_masks, duplicated_desc, ids, ids, torch.ones(2), 2)
    torch.testing.assert_close(w.T@descriptors, new.T@duplicated_desc)
    torch.testing.assert_close(mass, new_mass)
    legacy, _ = source_pixel_evidence_weights(duplicated_masks, ids, ids, torch.ones(2), 2, partition_overlap=True)
    assert not torch.allclose(w.T@descriptors, legacy.T@duplicated_desc)


def test_distinct_descriptor_is_distinct_evidence_and_unique_case_exact():
    masks = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    descriptors = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    ids = torch.arange(2)
    new, mass, mult = proposal_invariant_pixel_weights(masks, descriptors, ids, ids, torch.ones(2), 2)
    old, old_mass = source_pixel_evidence_weights(masks, ids, ids, torch.ones(2), 2, partition_overlap=True)
    assert torch.equal(mult, torch.ones(3)) and torch.equal(new, old) and torch.equal(mass, old_mass)


def test_row_order_and_partial_visibility_invariant():
    masks = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)
    desc = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    pid, gid = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])
    values = torch.tensor([1., 1e-8, 1.])
    w, m, _ = proposal_invariant_pixel_weights(masks, desc, pid, gid, values, 2)
    order = torch.tensor([1, 2, 0])
    other, om, _ = proposal_invariant_pixel_weights(masks[order], desc[order], pid, gid, values, 2)
    torch.testing.assert_close(w.T@desc, other.T@desc[order])
    torch.testing.assert_close(m, om)


def test_invalid_evidence_fails():
    with pytest.raises(ValueError):
        inverse_evidence_multiplicity(torch.tensor([[.5]]), torch.ones(1, 2))
    with pytest.raises(ValueError):
        inverse_evidence_multiplicity(torch.ones(1, 2), torch.zeros(1, 2))
