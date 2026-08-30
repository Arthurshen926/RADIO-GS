import json

import torch

from radio_gs.v4.carrier import ProjectionTable
from radio_gs.v4.evaluation.lerf_development_pipeline import (
    _masked_descriptors,
    _load_dense_feature,
    _load_text_cache,
    _load_labels,
    _validate_source_authority,
    _validate_semantic_manifest,
    retain_top_query_tokens,
    prototype_max_token_posterior,
    accumulate_surface_features,
    accumulate_consistency_weighted_surface_features,
    update_surface_view_prototypes,
    surface_identity_token_posterior,
    surface_view_prototype_query_posterior,
    select_consistent_surface_view,
    conservative_token_geometry_completion,
    token_null_posterior,
)


def test_source_authority_validation_rejects_information_leak() -> None:
    authority = {
        "contract": "sam3-query-free-source-rgb-authority-v1",
        "scene": "scene",
        "images": [{"image_id": "frame_00001"}],
        "information_policy": {
            "benchmark_ground_truth_used": False,
            "query_text_used": False,
            "target_or_evaluation_rgb_used": True,
            "registered_source_rgb_only": True,
        },
    }
    try:
        _validate_source_authority(authority, "scene")
    except ValueError as error:
        assert "information policy" in str(error)
    else:
        raise AssertionError("leaking authority was accepted")


def test_semantic_manifest_validation_rejects_excluded_frame(tmp_path) -> None:
    manifest_path = tmp_path / "frame_manifest.json"
    feature_dir = tmp_path / "backbone"
    feature_dir.mkdir()
    manifest = {
        "features": {"backbone": {"dim": 1280, "grid": [46, 62], "subdir": "backbone"}},
        "frames": [{"frame_idx": 1}],
        "excluded_image_names": ["frame_00001.jpg"],
    }
    manifest_path.write_text(json.dumps(manifest))
    from radio_gs.v4.contracts.geometry_receipt import sha256_file
    authority = {"construction": {"frame_manifest": {"sha256": sha256_file(manifest_path)}}}
    try:
        _validate_semantic_manifest(
            manifest, manifest_path, authority, feature_dir.resolve(), [1]
        )
    except ValueError as error:
        assert "excluded frame" in str(error)
    else:
        raise AssertionError("excluded semantic frame was accepted")


def test_accumulate_surface_features_respects_sparse_projection_weights() -> None:
    projection = ProjectionTable(
        element_ids=torch.tensor([0, 0, 1]),
        pixel_ids=torch.tensor([0, 1, 1]),
        depths=torch.ones(3),
        weights=torch.tensor([1.0, 1.0, 2.0]),
        num_elements=2,
        height=1,
        width=2,
    )
    features = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]])
    feature_sum = torch.zeros(2, 2)
    feature_mass = torch.zeros(2)
    accumulate_surface_features(feature_sum, feature_mass, features, projection, channel_chunk_size=1)
    assert torch.equal(feature_sum, torch.tensor([[4.0, 6.0], [6.0, 8.0]]))
    assert torch.equal(feature_mass, torch.tensor([2.0, 2.0]))


def test_consistency_weighting_rejects_disagreeing_view_without_hiding_observation() -> None:
    projection = ProjectionTable(
        element_ids=torch.tensor([0, 1]), pixel_ids=torch.tensor([0, 1]),
        depths=torch.ones(2), weights=torch.ones(2),
        num_elements=2, height=1, width=2,
    )
    features = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    reference_sum = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    reference_mass = torch.tensor([2.0, 2.0])
    feature_sum, feature_mass = torch.zeros(2, 2), torch.zeros(2)
    reliability, observed = accumulate_consistency_weighted_surface_features(
        feature_sum, feature_mass, reference_sum, reference_mass, features, projection,
        agreement_floor=0.3, agreement_power=2.0,
    )
    assert torch.equal(observed, torch.tensor([True, True]))
    assert reliability[0] == 1 and reliability[1] == 0
    assert feature_mass[0] == 1 and feature_mass[1] == 0


def test_consistency_weighting_preserves_projection_mass_when_fully_reliable() -> None:
    projection = ProjectionTable(
        element_ids=torch.tensor([0, 0]), pixel_ids=torch.tensor([0, 1]),
        depths=torch.ones(2), weights=torch.tensor([2.0, 1.0]),
        num_elements=1, height=1, width=2,
    )
    features = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    feature_sum, feature_mass = torch.zeros(1, 2), torch.zeros(1)
    accumulate_consistency_weighted_surface_features(
        feature_sum, feature_mass, torch.tensor([[2.0, 1.0]]), torch.tensor([3.0]),
        features, projection,
        agreement_floor=0.3, agreement_power=2.0,
    )
    assert feature_mass[0] == 3
    assert torch.allclose(feature_sum[0], torch.tensor([2.0, 1.0]))


def test_surface_accumulation_rejects_out_of_range_projection() -> None:
    projection = ProjectionTable(
        element_ids=torch.tensor([2]), pixel_ids=torch.tensor([0]),
        depths=torch.ones(1), weights=torch.ones(1),
        num_elements=3, height=1, width=1,
    )
    try:
        accumulate_surface_features(torch.zeros(2, 1), torch.zeros(2), torch.ones(1, 1, 1), projection)
    except ValueError as error:
        assert "out-of-range surface element" in str(error)
    else:
        raise AssertionError("invalid sparse projection was accepted")


def test_view_prototypes_keep_strongest_observation_per_element() -> None:
    projection = ProjectionTable(
        element_ids=torch.tensor([0, 1]), pixel_ids=torch.tensor([0, 1]),
        depths=torch.ones(2), weights=torch.tensor([2.0, 1.0]),
        num_elements=2, height=1, width=2,
    )
    descriptors = torch.zeros(2, 1, 2, dtype=torch.float16)
    mass = torch.zeros(2, 1)
    update_surface_view_prototypes(
        descriptors, mass, torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), projection,
        channel_chunk_size=1,
    )
    weaker = ProjectionTable(
        element_ids=torch.tensor([0]), pixel_ids=torch.tensor([0]),
        depths=torch.ones(1), weights=torch.tensor([1.0]),
        num_elements=2, height=1, width=1,
    )
    update_surface_view_prototypes(
        descriptors, mass, torch.tensor([[[0.0]], [[1.0]]]), weaker,
        channel_chunk_size=1,
    )
    assert torch.equal(mass[:, 0], torch.tensor([2.0, 1.0]))
    assert torch.allclose(descriptors[0, 0].float(), torch.tensor([1.0, 0.0]))


def test_view_prototype_query_uses_best_retained_identity() -> None:
    descriptors = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mass = torch.ones(1, 2)
    probability = surface_view_prototype_query_posterior(
        descriptors, mass, torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]),
        temperature=0.1,
    )
    assert probability[0, 0] > 0.99


def test_view_prototype_query_ignores_empty_slots() -> None:
    descriptors = torch.tensor([
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ])
    mass = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    probability = surface_view_prototype_query_posterior(
        descriptors, mass, torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]),
        temperature=0.1,
    )
    assert probability[0, 0] == 0.5
    assert probability[0, 1] == 0


def test_consistent_view_selection_is_query_independent() -> None:
    descriptors = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 0.0]],
    ])
    mass = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    selected, observed = select_consistent_surface_view(
        descriptors, mass, torch.tensor([[0.1, 0.9], [1.0, 0.0]])
    )
    assert torch.allclose(selected, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert torch.equal(observed, torch.tensor([True, True]))


def test_dense_feature_loader_rejects_nonfinite_and_wrong_shape(tmp_path) -> None:
    nonfinite = tmp_path / "nonfinite.pt"
    torch.save(torch.tensor([[[float("nan")]]]), nonfinite)
    try:
        _load_dense_feature(nonfinite, expected_channels=1)
    except ValueError as error:
        assert "NaN or Inf" in str(error)
    else:
        raise AssertionError("non-finite semantic feature was accepted")
    wrong = tmp_path / "wrong.pt"
    torch.save(torch.zeros(2, 1, 1), wrong)
    try:
        _load_dense_feature(wrong, expected_channels=1)
    except ValueError as error:
        assert "channel mismatch" in str(error)
    else:
        raise AssertionError("wrong semantic feature shape was accepted")


def test_text_cache_requires_exact_unique_queries(tmp_path) -> None:
    path = tmp_path / "text.pt"
    torch.save({
        "text_encoder": "siglip2",
        "queries": ["cup", "bowl"],
        "embeddings": torch.stack([torch.ones(1536), -torch.ones(1536)]),
    }, path)
    loaded = _load_text_cache(path, ["bowl", "cup"], torch.device("cpu"))
    assert loaded.shape == (2, 1536)
    assert loaded[0, 0] < 0 and loaded[1, 0] > 0
    try:
        _load_text_cache(path, ["missing"], torch.device("cpu"))
    except ValueError as error:
        assert "exact queries" in str(error)
    else:
        raise AssertionError("inexact text cache was accepted")


def test_load_labels_accepts_single_point_list_polygon(tmp_path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "frame_00007.json").write_text(json.dumps({
        "info": {"height": 20, "width": 30},
        "objects": [{"category": "cup", "segmentation": [[1, 2], [8, 2], [8, 9]]}],
    }))
    annotations, categories, height, width = _load_labels(tmp_path, "scene")
    assert categories == ["cup"]
    assert annotations[7][0]["polygons"][0].shape == (3, 2)
    assert (height, width) == (20, 30)


def test_masked_descriptors_average_only_selected_pixels() -> None:
    features = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    masks = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]])
    result = _masked_descriptors(features, masks)
    assert torch.allclose(result, torch.eye(2))


def test_completion_preserves_observed_rows_and_assigns_at_most_one_token() -> None:
    centres = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0],
        [0.05, 0.0, 0.0], [3.0, 0.0, 0.0],
    ])
    membership = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
    ])
    completed, filled = conservative_token_geometry_completion(
        centres, membership, radius_multiplier=2.0, minimum_scale=0.05
    )
    assert torch.equal(completed[:4], membership[:4])
    assert bool(filled[4])
    assert not bool(filled[5])
    assert int((completed[4] > 0).sum()) == 1


def test_token_null_posterior_is_a_simplex_per_query() -> None:
    tokens = torch.eye(2)
    queries = torch.tensor([[1.0, 0.0], [-1.0, -1.0]])
    token_probability, null_probability = token_null_posterior(
        tokens, queries, temperature=0.1, null_similarity=0.0
    )
    assert torch.allclose(token_probability + null_probability, torch.ones(2, 2))
    assert token_probability[0, 0] > 0.99
    assert bool((null_probability[1] > token_probability[1]).all())


def test_retain_top_query_tokens_has_fixed_capacity() -> None:
    probability = torch.tensor([[0.1, 0.7, 0.4], [0.9, 0.2, 0.3]])
    retained = retain_top_query_tokens(probability, 2)
    assert torch.equal(retained, torch.tensor([[0.0, 0.7, 0.4], [0.9, 0.0, 0.3]]))


def test_prototype_pooling_preserves_a_strong_view_inside_token() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    token_ids = torch.tensor([0, 0, 1])
    query = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[0.0, -1.0]])
    positive, null = prototype_max_token_posterior(
        prototypes, token_ids, query, negatives, num_tokens=2, temperature=0.1
    )
    assert positive[0, 0] > 0.99
    assert positive[0, 0] + null[0, 0] == 1


def test_surface_identity_is_localized_before_token_pooling() -> None:
    descriptors = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.0]])
    observed = torch.tensor([True, True, True, False])
    membership = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    positive, null = surface_identity_token_posterior(
        descriptors, observed, membership,
        torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]),
        temperature=0.1, peak_count=2,
    )
    assert positive[0, 0] > 0.99
    assert positive[0, 1] < 0.01
    assert torch.allclose(positive + null, torch.ones_like(positive))
