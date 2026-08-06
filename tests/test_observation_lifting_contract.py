from argparse import Namespace

import pytest

from radio_gs.field.observation_lifting_contract import (
    CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    apply_canonical_observation_contract,
    canonical_observation_contract,
    select_full_observation_coverage_ranked_dataset_indices,
    observation_contract_sha256,
    validate_observation_contract_metadata,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
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
        "per_view_normalization_stage": "pixel_feature_before_raster_lifting",
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


def _exact_marginal_metadata() -> dict:
    contract = canonical_observation_contract(
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME
    )
    return {
        "aggregation_mode": contract["aggregation_mode"],
        "registration_weight_mode": contract["registration_weight_mode"],
        "raster_view_fusion": contract["raster_view_fusion"],
        "normalize_each_view": True,
        "per_view_normalization_applied": True,
        "per_view_normalization_stage": "pixel_feature_before_raster_lifting",
        "depth_tolerance": 0.0,
        "relative_depth_tolerance": 0.0,
        "alpha_threshold": 0.0,
        "num_declared_views": 2,
        "selected_dataset_indices": [1, 5],
        "selected_frame_indices": [10, 50],
        "robust_mpr": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "shared_registration_responsibility": True,
        "registration_responsibility_cache_sha256": "a" * 64,
        "registration_responsibility_contract": {
            "registration_weight_mode": contract["registration_weight_mode"],
            "post_compositor_alpha_threshold": 0.0,
            "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
            "query_independent": True,
        },
        "observation_lifting_contract": contract,
        "observation_lifting_contract_sha256": observation_contract_sha256(contract),
    }


def test_exact_marginal_contract_is_explicit_and_fail_closed() -> None:
    args = Namespace(
        observation_contract=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
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
    assert args.aggregation_mode == "raster_marginal_responsibility"
    assert args.registration_weight_mode == (
        "exact_front_to_back_marginal_responsibility"
    )
    assert args.raster_view_fusion == "contribution_mean"
    assert args.normalize_each_view is True
    assert args.alpha_threshold == 0.0
    assert contract["view_selection"] == "uniform_temporal_deterministic"
    assert contract["feature_projection_order"] == "per_view_before_mpr"
    assert contract["responsibility_authority_schema"] == (
        SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
    )
    assert contract["responsibility_formula_sha256"] == (
        SPARSE_EXACT_MARGINAL_FORMULA_SHA256
    )
    assert (
        validate_observation_contract_metadata(
            _exact_marginal_metadata(),
            contract_name=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
        )
        == contract
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shared_registration_responsibility", False, "exact-marginal"),
        ("registration_responsibility_cache_sha256", "", "exact-marginal"),
        ("benchmark_images_opened", True, "exact-marginal"),
        ("alpha_threshold", 0.01, "canonical observation"),
    ],
)
def test_exact_marginal_contract_rejects_metadata_drift(
    field: str, value: object, message: str
) -> None:
    metadata = _exact_marginal_metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        validate_observation_contract_metadata(
            metadata,
            contract_name=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
        )


def test_exact_marginal_contract_rejects_formula_or_digest_drift() -> None:
    metadata = _exact_marginal_metadata()
    metadata["registration_responsibility_contract"] = {
        **metadata["registration_responsibility_contract"],
        "formula_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="exact-marginal"):
        validate_observation_contract_metadata(
            metadata,
            contract_name=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
        )

    metadata = _exact_marginal_metadata()
    metadata["observation_lifting_contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_observation_contract_metadata(
            metadata,
            contract_name=CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
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
    metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(
        contract
    )
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
