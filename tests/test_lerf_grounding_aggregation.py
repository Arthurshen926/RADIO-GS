import pytest
import torch

from radio_gs.scripts.eval_lerf_grounding import (
    aggregate_lerf_mode_metrics,
    apply_primitive_semantic_confidence,
    blend_primary_with_dominant_fallback,
    blend_primary_with_uncovered_fallback,
    blend_primary_first,
    blend_strongest_source,
    neutralize_invalid_primitive_scores_for_render,
    normalize_primitive_scores_by_valid_mass,
    monotonic_logit_calibration,
    validate_primitive_support_cache,
    validate_primitive_unary_cache,
    validate_primitive_posterior_cache,
    validate_primitive_posterior_identity_cache,
)


def test_monotonic_logit_calibration_is_identity_and_order_preserving() -> None:
    values = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    torch.testing.assert_close(monotonic_logit_calibration(values), values)
    calibrated = monotonic_logit_calibration(values, scale=0.7, bias=0.2)
    assert torch.equal(calibrated[[0, -1]], values[[0, -1]])
    assert bool((calibrated[1:] >= calibrated[:-1]).all())
    with pytest.raises(ValueError, match="calibration inputs"):
        monotonic_logit_calibration(values, scale=0.0)


def test_lerf_aggregation_separates_sample_scene_and_category_means() -> None:
    metrics = [
        {
            "loc_correct": 1,
            "loc_total": 2,
            "n_iou_samples": 2,
            "miou": 0.2,
            "per_category": {"a": {"miou": 0.1}, "b": {"miou": 0.3}},
        },
        {
            "loc_correct": 6,
            "loc_total": 8,
            "n_iou_samples": 8,
            "miou": 0.8,
            "per_category": {"a": {"miou": 0.7}, "b": {"miou": 0.9}},
        },
    ]

    aggregate = aggregate_lerf_mode_metrics(metrics)

    assert aggregate["localization_accuracy"] == pytest.approx(0.7)
    assert aggregate["sample_micro_miou"] == pytest.approx(0.68)
    assert aggregate["scene_macro_miou"] == pytest.approx(0.5)
    assert aggregate["category_macro_miou"] == pytest.approx(0.5)
    assert aggregate["sample_count"] == 10
    assert aggregate["scene_count"] == 2


def test_invalid_primitive_scores_are_neutral_during_alpha_compositing() -> None:
    scores = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    result = neutralize_invalid_primitive_scores_for_render(
        scores, torch.tensor([True, False, True])
    )

    assert torch.equal(result[0], scores[0])
    assert torch.equal(result[1], torch.zeros(2))
    assert torch.equal(result[2], scores[2])
    assert torch.equal(scores[1], torch.tensor([0.7, 0.3]))


def test_valid_mass_normalization_separates_score_from_coverage() -> None:
    rendered = torch.tensor(
        [
            [[0.20, 0.45]],
            [[0.10, 0.05]],
            [[0.25, 0.50]],
        ]
    )
    scores, coverage = normalize_primitive_scores_by_valid_mass(rendered)

    torch.testing.assert_close(
        scores,
        torch.tensor([[[0.8, 0.9]], [[0.4, 0.1]]]),
    )
    torch.testing.assert_close(coverage, torch.tensor([[0.25, 0.50]]))


def test_valid_mass_coverage_power_one_recovers_total_alpha_scores() -> None:
    rendered = torch.tensor(
        [
            [[0.20, 0.45]],
            [[0.10, 0.05]],
            [[0.25, 0.50]],
        ]
    )

    scores, coverage = normalize_primitive_scores_by_valid_mass(
        rendered, coverage_power=1.0
    )

    torch.testing.assert_close(scores, rendered[:-1])
    torch.testing.assert_close(coverage, rendered[-1])


def test_primitive_semantic_confidence_damps_query_support_rowwise() -> None:
    scores = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    result = apply_primitive_semantic_confidence(
        scores, torch.tensor([1.0, 0.5])
    )

    assert torch.equal(result[0], scores[0])
    assert torch.allclose(result[1], torch.tensor([0.3, 0.2]))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        apply_primitive_semantic_confidence(scores, torch.tensor([1.0, 1.1]))


def test_uncovered_fallback_blend_preserves_primary_and_fills_holes() -> None:
    primary = torch.tensor([[[0.7, 0.2]]])
    fallback = torch.tensor([[[0.4, 0.6]]])
    result = blend_primary_with_uncovered_fallback(
        primary, fallback, torch.tensor([[[1.0, 0.0]]])
    )

    assert torch.allclose(result, torch.tensor([[[0.7, 0.8]]]))


def test_dominant_fallback_blend_routes_by_query_independent_coverage() -> None:
    primary = torch.tensor([[[0.7, 0.2]]])
    fallback = torch.tensor([[[0.4, 0.6]]])
    result = blend_primary_with_dominant_fallback(
        primary,
        fallback,
        primary_coverage=torch.tensor([[[0.8, 0.1]]]),
        fallback_coverage=torch.tensor([[[0.1, 0.7]]]),
    )

    assert torch.allclose(result, torch.tensor([[[0.7, 0.8]]]))


def test_primary_first_only_completes_queries_without_positive_support() -> None:
    primary = torch.tensor([[[0.6, 0.0]], [[0.4, 0.1]]])
    fallback = torch.tensor([[[0.3, 0.2]], [[0.2, 0.5]]])
    result = blend_primary_first(
        primary, fallback, semantic_threshold=0.5
    )

    assert torch.equal(result[0], primary[0])
    assert torch.equal(result[1], primary[1] + fallback[1])


def test_strongest_source_preserves_or_completes_whole_queries() -> None:
    primary = torch.tensor([[[0.7, 0.0]], [[0.3, 0.1]]])
    fallback = torch.tensor([[[0.2, 0.1]], [[0.2, 0.5]]])
    result = blend_strongest_source(primary, fallback)

    assert torch.equal(result[0], primary[0])
    assert torch.equal(result[1], primary[1] + fallback[1])


def test_primitive_support_cache_enforces_solver_and_query_contract() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    payload = {
        "xyz": xyz.clone(),
        "valid": torch.tensor([True, False]),
        "features": torch.tensor([[0.8, 0.2], [0.0, 0.0]]),
        "metadata": {
            "query_names": ["bowl", "spoon"],
            "construction": "shared_3d_support_solver_probabilities",
        },
    }

    scores, valid = validate_primitive_support_cache(
        payload, xyz, ["bowl", "spoon"]
    )

    assert scores.shape == (2, 2)
    assert torch.equal(valid, torch.tensor([True, False]))

    payload["metadata"]["construction"] = "generic_negative_binary_relevancy"
    with pytest.raises(ValueError, match="shared_3d_support_solver_probabilities"):
        validate_primitive_support_cache(payload, xyz, ["bowl", "spoon"])


def test_primitive_support_cache_rejects_geometry_or_query_mismatch() -> None:
    xyz = torch.zeros(2, 3)
    payload = {
        "xyz": xyz.clone(),
        "valid": torch.ones(2, dtype=torch.bool),
        "features": torch.full((2, 1), 0.5),
        "metadata": {
            "query_names": ["bowl"],
            "construction": "shared_3d_support_solver_probabilities",
        },
    }
    with pytest.raises(ValueError, match="query order mismatch"):
        validate_primitive_support_cache(payload, xyz, ["spoon"])
    payload["metadata"]["query_names"] = ["bowl"]
    shifted = xyz.clone()
    shifted[0, 0] = 1.0
    with pytest.raises(ValueError, match="xyz mismatch"):
        validate_primitive_support_cache(payload, shifted, ["bowl"])


def test_primitive_unary_cache_accepts_only_independent_cosine() -> None:
    xyz = torch.zeros(2, 3)
    payload = {
        "xyz": xyz.clone(), "valid": torch.ones(2, dtype=torch.bool),
        "features": torch.tensor([[0.8], [-0.2]]),
        "metadata": {"query_names": ["bowl"],
                     "feature_space": "primitive_text_query_scores",
                     "scoring": "cosine"},
    }
    scores, valid = validate_primitive_unary_cache(payload, xyz, ["bowl"])
    assert scores.shape == (2, 1) and bool(valid.all())
    payload["metadata"]["scoring"] = "softmax_scene"
    with pytest.raises(ValueError, match="independent cosine"):
        validate_primitive_unary_cache(payload, xyz, ["bowl"])


def test_primitive_posterior_cache_enforces_typed_information_contract() -> None:
    xyz = torch.zeros(2, 3)
    payload = {
        "xyz": xyz.clone(),
        "valid": torch.ones(2, dtype=torch.bool),
        "query_scores": torch.tensor([[0.8], [0.2]]),
        "identity_query_scores": torch.tensor([[1.0], [0.4]]),
        "metadata": {
            "query_names": ["bowl"],
            "query_family": "text_object_extent",
            "typed_posterior": "official_sam3_siglip2_identity_extent_factorization_v1",
            "persistent_second_semantic_field": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "separate_identity_localization": True,
            "localization_authority": "field_siglip2_relevancy_identity",
        },
    }
    scores, valid = validate_primitive_posterior_cache(payload, xyz, ["bowl"])
    assert scores.shape == (2, 1) and bool(valid.all())
    identity = validate_primitive_posterior_identity_cache(payload, xyz, ["bowl"])
    assert identity is not None
    torch.testing.assert_close(identity, torch.tensor([[1.0], [0.4]]))
    payload["metadata"]["benchmark_masks_opened"] = True
    with pytest.raises(ValueError, match="forbidden"):
        validate_primitive_posterior_cache(payload, xyz, ["bowl"])
