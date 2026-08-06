import pytest
import torch

from radio_gs.scripts.audit_lerf_source_probability_parity import (
    audit_source_probability_parity,
)


def _payloads():
    xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    valid = torch.tensor([True, True, False, True])
    positive_scores = torch.tensor(
        [[[0.2], [0.3]], [[0.4], [0.1]], [[0.0], [0.0]], [[0.3], [0.5]]]
    )
    negative_scores = torch.tensor(
        [
            [[0.1, 0.05], [0.2, 0.0]],
            [[0.3, 0.2], [0.0, -0.1]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.1, 0.2], [0.4, 0.2]],
        ]
    )
    expected = torch.sigmoid(
        10.0 * (positive_scores - negative_scores.amax(-1, keepdim=True))
    )
    expected[~valid] = 0
    positive = {
        "query_scores": positive_scores,
        "query_ids": ["object"],
        "scale_ids": ["near", "far"],
        "scale_radii_m": [0.25, 0.45],
        "xyz": xyz,
        "valid": valid,
    }
    negative = {
        "query_scores": negative_scores,
        "query_ids": ["thing", "stuff"],
        "scale_ids": ["near", "far"],
        "scale_radii_m": [0.25, 0.45],
        "xyz": xyz,
        "valid": valid,
    }
    control = {
        "features": expected.half(),
        "xyz": xyz,
        "valid": valid,
        "metadata": {
            "query_names": ["object"],
            "scale_radii_m": [0.25, 0.45],
        },
    }
    return positive, negative, control


def test_source_probability_parity_passes_fp16_streaming_roundoff() -> None:
    positive, negative, control = _payloads()
    result = audit_source_probability_parity(
        positive=positive,
        negative=negative,
        control=control,
    )
    assert result["passed"] is True
    assert result["metrics"]["pearson"] > 0.9999


def test_source_probability_parity_fails_changed_probabilities() -> None:
    positive, negative, control = _payloads()
    control["features"] = control["features"].clone()
    control["features"][0, 0, 0] = 0
    result = audit_source_probability_parity(
        positive=positive,
        negative=negative,
        control=control,
    )
    assert result["passed"] is False


def test_source_probability_parity_fails_axis_mismatch() -> None:
    positive, negative, control = _payloads()
    control["metadata"]["query_names"] = ["different"]
    with pytest.raises(ValueError, match="query axis"):
        audit_source_probability_parity(
            positive=positive,
            negative=negative,
            control=control,
        )
