import pytest

from radio_gs.scripts.build_canonical_capability_views import (
    _validate_compatible_legacy_observation,
)


def test_compatible_legacy_capability_observation_is_narrow_and_query_free() -> None:
    metadata = {
        "construction": "dominant_primary_with_query_free_support_completion",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    _validate_compatible_legacy_observation(metadata)
    with pytest.raises(ValueError, match="construction"):
        _validate_compatible_legacy_observation({**metadata, "construction": "other"})
    with pytest.raises(ValueError, match="query-independent"):
        _validate_compatible_legacy_observation(
            {**metadata, "text_queries_opened": True}
        )
