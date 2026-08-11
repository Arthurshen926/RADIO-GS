import json

import pytest

from radio_gs.scripts.seal_lerf_rgb_free_query_score_batch import (
    _validate_query_independent,
)


FIELD_SHA = "d38256ed91c4f373759355395bd8c8f6ddcd4b4f59018f9d27e53525a35f31b9"
EXACT_CONSTRUCTION = (
    "canonical_radio_surface_region_readout_then_official_summary_head"
)


def _write_descriptor(tmp_path, **overrides):
    descriptor = tmp_path / "descriptor_v3_v2_isomorphic.pt"
    descriptor.write_bytes(b"sealed-test-descriptor")
    metadata = {
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "query_set_invariant": True,
        "construction": EXACT_CONSTRUCTION,
        "field_checkpoint_sha256": FIELD_SHA,
    }
    metadata.update(overrides)
    descriptor.with_suffix(".pt.json").write_text(
        json.dumps({"metadata": metadata}), encoding="utf-8"
    )
    return descriptor


def test_exact_descriptor_query_invariant_alias_is_accepted(tmp_path):
    descriptor = _write_descriptor(tmp_path)
    record = _validate_query_independent(descriptor, FIELD_SHA)
    assert record["sha256"]
    assert record["metadata"]["sha256"]


@pytest.mark.parametrize(
    "override",
    [
        {"query_set_invariant": False},
        {"text_queries_opened": True},
        {"construction": "query_conditioned_descriptor"},
    ],
)
def test_exact_descriptor_alias_fails_if_any_joint_field_differs(
    tmp_path, override
):
    descriptor = _write_descriptor(tmp_path, **override)
    with pytest.raises(ValueError):
        _validate_query_independent(descriptor, FIELD_SHA)
