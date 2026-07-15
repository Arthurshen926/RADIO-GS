from argparse import Namespace

import pytest

from radio_gs.field.observation_lifting_contract import (
    apply_canonical_observation_contract,
    canonical_observation_contract,
    observation_contract_sha256,
    validate_observation_contract_metadata,
)


def _metadata(*, declared: bool = True):
    contract = canonical_observation_contract()
    metadata = {
        "aggregation_mode": contract["aggregation_mode"],
        "registration_weight_mode": contract["registration_weight_mode"],
        "raster_view_fusion": contract["raster_view_fusion"],
        "normalize_each_view": contract["normalize_each_view"],
        "per_view_normalization_applied": True,
        "depth_tolerance": contract["depth_tolerance"],
        "relative_depth_tolerance": contract["relative_depth_tolerance"],
        "alpha_threshold": contract["alpha_threshold"],
        "num_declared_views": 73,
        "robust_mpr": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    if declared:
        metadata["observation_lifting_contract"] = contract
        metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(
            contract
        )
    return metadata


def test_contract_overrides_all_method_policy_knobs():
    args = Namespace(
        max_views=7,
        aggregation_mode="center",
        registration_weight_mode="uniform",
        raster_view_fusion="view_mean",
        normalize_each_view=False,
        depth_tolerance=1.0,
        relative_depth_tolerance=1.0,
        alpha_threshold=1.0,
        robust_mpr=True,
    )

    contract = apply_canonical_observation_contract(args)

    assert args.max_views == 120
    assert args.aggregation_mode == "raster_gaussian_top1"
    assert args.registration_weight_mode == "alpha_depth"
    assert args.raster_view_fusion == "contribution_mean"
    assert args.normalize_each_view is True
    assert args.robust_mpr is False
    assert contract == canonical_observation_contract()


def test_declared_contract_round_trip_validates():
    assert validate_observation_contract_metadata(_metadata()) == (
        canonical_observation_contract()
    )


def test_compatible_legacy_requires_explicit_certification_mode():
    with pytest.raises(ValueError, match="does not declare"):
        validate_observation_contract_metadata(_metadata(declared=False))

    validate_observation_contract_metadata(
        _metadata(declared=False), require_declaration=False
    )


def test_contract_rejects_dataset_specific_policy_drift():
    metadata = _metadata()
    metadata["raster_view_fusion"] = "view_mean"

    with pytest.raises(ValueError, match="violates canonical"):
        validate_observation_contract_metadata(metadata)
