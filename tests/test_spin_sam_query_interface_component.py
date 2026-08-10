import numpy as np
import pytest
import torch
from types import SimpleNamespace

from radio_gs.scripts.eval_spin_sam_query_interface_component import (
    _exact_subset_mapping,
    _normalize_candidates,
    _resolve_render_resolution,
)


def _void_keys(values: list[int]) -> np.ndarray:
    array = np.asarray(values, dtype="<i4")
    return array.view(np.dtype((np.void, 4))).reshape(-1)


def test_exact_subset_mapping_preserves_subset_order() -> None:
    mapping = _exact_subset_mapping(
        _void_keys([30, 10, 20]),
        _void_keys([20, 30]),
    )
    assert mapping.tolist() == [2, 0]


def test_exact_subset_mapping_rejects_missing_row() -> None:
    with pytest.raises(ValueError, match="not an exact carrier subset"):
        _exact_subset_mapping(_void_keys([10, 20]), _void_keys([30]))


def test_candidate_normalization_is_independent_per_channel() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0]],
            [[-4.0, -2.0], [0.0, 4.0]],
        ]
    )
    normalized = _normalize_candidates(values)
    assert torch.allclose(normalized.amin(dim=(1, 2)), torch.zeros(2))
    assert torch.allclose(normalized.amax(dim=(1, 2)), torch.ones(2))


def test_query_scalar_render_can_use_native_resolution() -> None:
    config = SimpleNamespace(
        feature_height=48,
        feature_width=63,
        image_height=764,
        image_width=1015,
    )

    assert _resolve_render_resolution(config, "feature") == (48, 63)
    assert _resolve_render_resolution(config, "native") == (764, 1015)
    assert _resolve_render_resolution(
        config,
        "registered",
        registered_resolution=(600, 800),
    ) == (600, 800)


def test_registered_query_scalar_render_requires_declared_resolution() -> None:
    config = SimpleNamespace(
        feature_height=48,
        feature_width=63,
        image_height=764,
        image_width=1015,
    )

    with pytest.raises(ValueError, match="was not provided"):
        _resolve_render_resolution(config, "registered")


def test_query_scalar_render_rejects_unknown_resolution_mode() -> None:
    config = SimpleNamespace(
        feature_height=48,
        feature_width=63,
        image_height=764,
        image_width=1015,
    )

    with pytest.raises(ValueError, match="render-resolution"):
        _resolve_render_resolution(config, "quarter")
