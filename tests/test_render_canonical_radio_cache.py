import pytest

from radio_gs.scripts.render_canonical_radio_cache import (
    _benchmark_frames_from_payload,
    _parse_region_kernel_sizes,
)


def test_benchmark_frames_use_render_optimization_manifest() -> None:
    payload = {
        "render_optimization": {
            "excluded_benchmark_frames": [195, 41, 105, 152]
        }
    }

    assert _benchmark_frames_from_payload(payload) == [41, 105, 152, 195]


def test_benchmark_frames_fall_back_to_factorized_mpr_authority() -> None:
    payload = {
        "mpr_cache_metadata": {
            "registration_responsibility_contract": {
                "excluded_frame_ids": [41, 105, 152, 195]
            }
        }
    }

    assert _benchmark_frames_from_payload(payload) == [41, 105, 152, 195]


def test_region_summary_kernel_contract_is_fixed_and_validated() -> None:
    assert _parse_region_kernel_sizes("3,7,15") == (3, 7, 15)
    with pytest.raises(ValueError, match="positive odd"):
        _parse_region_kernel_sizes("3,8")
