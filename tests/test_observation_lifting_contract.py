from argparse import Namespace

import pytest

from radio_gs.field.observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    apply_canonical_observation_contract,
    canonical_observation_contract,
    select_full_observation_coverage_ranked_dataset_indices,
    observation_contract_sha256,
    validate_observation_contract_metadata,
)
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _select_full_observation_coverage_views,
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


def test_full_observation_contract_selects_the_label_free_coverage_prefix():
    args = Namespace(
        observation_contract=CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
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
    selected = select_full_observation_coverage_ranked_dataset_indices(
        dataset_frame_ids=[0, 20, 40, 60],
        candidate_dataset_indices=[0, 2, 3],
        ranked_frame_ids=[60, 20, 40, 0],
        maximum_views=2,
    )

    assert contract["name"] == CANONICAL_FULL_OBSERVATION_CONTRACT_NAME
    assert contract["view_selection"] == "field_source_coverage_ranked_deterministic"
    assert args.max_views == 240
    # frame 20 is held out; ranking is preserved for the remaining candidates.
    assert selected == [3, 2]


def test_full_observation_v2_preserves_a_480_view_source_prefix():
    args = Namespace(
        observation_contract=CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
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
    metadata = _metadata()
    metadata["observation_lifting_contract"] = contract
    metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(contract)
    metadata["num_declared_views"] = 476
    metadata["full_observation_source_view_count"] = 480

    assert contract["name"] == CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME
    assert args.max_views == 480
    assert validate_observation_contract_metadata(metadata) == contract

    metadata["full_observation_source_view_count"] = 240
    with pytest.raises(ValueError, match="480-view source prefix"):
        validate_observation_contract_metadata(metadata)


def test_full_observation_v3_requires_an_independent_960_view_source_prefix():
    args = Namespace(
        observation_contract=CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
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
    metadata = _metadata()
    metadata["observation_lifting_contract"] = contract
    metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(
        contract
    )
    metadata["num_declared_views"] = 956
    metadata["full_observation_source_view_count"] = 960

    assert contract["name"] == CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME
    assert args.max_views == 960
    assert validate_observation_contract_metadata(metadata) == contract

    metadata["full_observation_source_view_count"] = 480
    with pytest.raises(ValueError, match="960-view source prefix"):
        validate_observation_contract_metadata(metadata)


def test_full_observation_mpr_fails_closed_on_a_labelled_source_manifest(tmp_path):
    contract = tmp_path / "field_source_contract.json"
    contract.write_text(
        """{
          "field_contract_version": "scannet_full_observation_v1",
          "selected_frame_indices": [0, 20, 40],
          "selection_order_frame_indices": [40, 0, 20],
          "uses_private_anchor": false,
          "uses_private_depth_pixel": false,
          "uses_instances_or_semantic_labels": false,
          "contains_instance_or_label_directories": false
        }""",
        encoding="utf-8",
    )

    selected, audit = _select_full_observation_coverage_views(
        scene_root=tmp_path,
        dataset_frame_ids=[0, 20, 40],
        candidates=[0, 2],
        maximum_views=2,
    )
    assert selected == [2, 0]
    assert audit["full_observation_coverage_order_applied"] is True

    with pytest.raises(ValueError, match="materialized 4-view source prefix"):
        _select_full_observation_coverage_views(
            scene_root=tmp_path,
            dataset_frame_ids=[0, 20, 40],
            candidates=[0, 2],
            maximum_views=4,
            minimum_source_views=4,
        )

    contract.write_text(
        contract.read_text(encoding="utf-8").replace(
            '"uses_instances_or_semantic_labels": false',
            '"uses_instances_or_semantic_labels": true',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query/label free"):
        _select_full_observation_coverage_views(
            scene_root=tmp_path,
            dataset_frame_ids=[0, 20, 40],
            candidates=[0, 2],
            maximum_views=2,
        )
