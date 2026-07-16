from __future__ import annotations

import torch

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.scripts.build_free_canonical_radio_field import build_free_field_payload


_SIGNATURE = {
    "radio_version": "c-radio-test",
    "radio_checkpoint_sha256": "abc123",
    "raw_feature_dim": 3,
    "adaptor_name": "backbone",
    "token_type": "primitive",
    "normalization": "l2",
}


def _mpr_payload() -> dict:
    return {
        "features": torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        "valid": torch.tensor([True, False]),
        "reliability": torch.tensor([[1.0], [0.0]]),
        "geometry_fingerprint": {"num_gaussians": 2, "xyz_sha256": "abc"},
        "metadata": {
            "normalize_each_view": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }


def test_free_field_is_exact_identity_mpr_initialization(tmp_path) -> None:
    payload = build_free_field_payload(
        _mpr_payload(), source_path="mpr.pt", feature_signature=_SIGNATURE
    )
    checkpoint = tmp_path / "field.pth"
    torch.save(payload, checkpoint)
    field, restored = load_canonical_field_checkpoint(checkpoint)

    torch.testing.assert_close(field.radio_features(), _mpr_payload()["features"])
    assert restored["architecture"]["coefficient_dim"] == 3
    assert restored["render_ceiling"]["query_free"] is True


def test_free_field_rejects_query_contaminated_mpr() -> None:
    payload = _mpr_payload()
    payload["metadata"]["text_queries_opened"] = True
    try:
        build_free_field_payload(
            payload, source_path="mpr.pt", feature_signature=_SIGNATURE
        )
    except ValueError as error:
        assert "text-query" in str(error)
    else:
        raise AssertionError("query-contaminated MPR must be rejected")
