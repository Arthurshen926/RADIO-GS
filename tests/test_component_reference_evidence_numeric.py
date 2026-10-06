import pytest
import torch
from radio_gs.querying.component_reference_evidence_numeric import retain_numerically_checked_components


def test_roundoff_at_large_pixel_mass_is_clamped_without_changing_decision():
    keep, report = retain_numerically_checked_components(
        torch.tensor([0]), torch.tensor([True]), torch.tensor([False]),
        torch.tensor([600.698974609375]), torch.tensor([600.69873046875]))
    assert keep.item() and report['numeric_subset_clamped_rows'] == 1


def test_real_inconsistent_partition_still_fails_closed():
    with pytest.raises(ValueError, match='partition mismatch'):
        retain_numerically_checked_components(
            torch.tensor([0]), torch.tensor([True]), torch.tensor([False]),
            torch.tensor([2.]), torch.tensor([1.]))
