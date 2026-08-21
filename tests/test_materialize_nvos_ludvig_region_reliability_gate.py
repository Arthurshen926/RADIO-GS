import pytest

from radio_gs.scripts.materialize_nvos_ludvig_region_reliability_gate import (
    authorize_region_agreement,
)


@pytest.mark.parametrize(
    ("quality", "overlap", "expected"),
    [
        (0.5, 1.0, True),
        (0.0, 0.49, True),
        (0.49, 0.5, False),
        (0.0, 1.0, False),
    ],
)
def test_region_gate_uses_neutral_probability_and_majority_boundaries(
    quality, overlap, expected
):
    assert authorize_region_agreement(quality, overlap) is expected


@pytest.mark.parametrize("quality,overlap", [(float("nan"), 0.5), (0.5, 1.1)])
def test_region_gate_rejects_invalid_diagnostics(quality, overlap):
    with pytest.raises(ValueError):
        authorize_region_agreement(quality, overlap)

