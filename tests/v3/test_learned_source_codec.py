import torch

from radio_gs.v3.training.learned_source_codec import apply_codec, _principal_codec
from radio_gs.v3.evaluation.structured_source_capability import (
    _retrieval_diagnostics,
    _same_pixel_retrieval,
)
from radio_gs.v3.evaluation.visual_mapping_error_ladder import _direction_metrics
from radio_gs.v3.training.fit_render_metric_codec import _loss
from radio_gs.v3.training.native_visual_codec import GatedResidualVisualCodec, _matched_pairs
from radio_gs.v3.training.refine_native_visual_memory import _render_loss
from radio_gs.v3.training.build_multisource_correspondence_authority import (
    _support_overlap,
)
from radio_gs.v3.training.native_visual_set_codec import ObservationSetVisualCodec
from radio_gs.v3.training.materialize_native_visual_set_memory import _topk_all
from radio_gs.v3.training.build_source_overlap_graph import (
    _direct_support_map,
    _nearest_centroid,
    _select_graph_edges,
    _qualify_views,
)
from radio_gs.v3.evaluation.source_overlap_coverage import _union_coverage
from radio_gs.v3.training.build_row_propagation_authority import (
    _strongest_source_per_group,
    _voxel_keys,
    _assign_group_best,
)
from radio_gs.v3.evaluation.row_propagation_heldout import _tier_report
from radio_gs.v3.evaluation.row_propagation_render_heldout import _bucket_report
from radio_gs.v3.training.materialize_row_coverage_policy import _coverage_policy
from radio_gs.v3.evaluation.masked_structural_reconstruction import _masked_split
from radio_gs.v3.training.materialize_unknown_structural_initialization import (
    _interpolate_unknown,
)
from radio_gs.v3.training.materialize_unknown_aware_scene_state import (
    _reliability_scalars,
)


def test_apply_codec_centers_before_projection():
    values = torch.tensor([[2.0, 5.0], [4.0, 9.0]])
    mean = torch.tensor([1.0, 3.0])
    basis = torch.tensor([[1.0], [2.0]])
    assert torch.equal(apply_codec(values, mean, basis), torch.tensor([[5.0], [15.0]]))


def test_principal_codec_selects_dominant_axis():
    samples = torch.tensor([
        [-3.0, 0.1], [-2.0, -0.1], [2.0, 0.1], [3.0, -0.1]
    ])
    mean, basis, retained = _principal_codec(samples, 1, torch.device("cpu"))
    assert torch.allclose(mean, torch.tensor([0.0, 0.0]), atol=1e-6)
    assert abs(float(basis[0, 0])) > 0.99
    assert retained > 0.99


def test_same_pixel_retrieval_reports_identity_ceiling():
    target = torch.eye(4)
    top1, top5, margin = _same_pixel_retrieval(target, target, 4)
    assert top1 == 1.0
    assert top5 == 1.0
    assert margin == 1.0


def test_retrieval_diagnostics_decomposes_positive_and_hardest_negative():
    value = _retrieval_diagnostics(torch.eye(3), torch.eye(3), 3)
    assert value == {
        "top1": 1.0,
        "top5": 1.0,
        "mrr": 1.0,
        "positive_similarity": 1.0,
        "hardest_negative_similarity": 0.0,
        "positive_margin": 1.0,
    }


def test_ladder_bucket_metrics_keep_full_candidate_negatives():
    similarity = torch.eye(3)
    value = _direction_metrics(similarity, torch.tensor([True, False, True]))
    assert value["count"] == 2
    assert value["recall_at_1"] == 1.0
    assert value["hardest_negative_similarity"] == 0.0


def test_render_metric_loss_is_finite_for_exact_identity_episode():
    episode = {
        "features": torch.eye(3).half(),
        "inverse": torch.arange(3),
        "pixel_ids": torch.arange(3),
        "weights": torch.ones(3).half(),
        "target": torch.eye(3).half(),
        "num_pixels": 3,
    }
    loss = _loss(episode, torch.zeros(3), temperature=0.1)
    assert torch.isfinite(loss)
    assert float(loss) < 0.001


def test_gated_residual_visual_codec_outputs_unit_d320():
    model = GatedResidualVisualCodec(
        radio_dim=8, dino_dim=6, output_dim=4,
        radio_rank=3, dino_rank=2, hidden_dim=7,
    )
    embedding, radio, dino = model(torch.randn(5, 8), torch.randn(5, 6))
    assert embedding.shape == (5, 4)
    assert radio.shape == (5, 8)
    assert dino.shape == (5, 6)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_matched_pairs_are_cross_residue_same_gaussian():
    observations = [
        (torch.tensor([2, 5]), torch.tensor([10, 11])),
        (torch.tensor([2, 7]), torch.tensor([20, 21])),
    ]
    pairs = _matched_pairs(
        observations, [1, 2], pairs_per_view_pair=8, seed=0
    )
    assert torch.equal(pairs, torch.tensor([[0, 10, 1, 20, 2]]))


def test_renderer_refinement_loss_updates_only_visual_parameter():
    visual = torch.randn(7, 4, requires_grad=True)
    initial = visual.detach().clone()
    episode = {
        "rows": torch.tensor([1, 3, 5]),
        "inverse": torch.tensor([0, 1, 2]),
        "pixels": torch.tensor([0, 1, 1]),
        "weights": torch.tensor([1.0, 0.6, 0.4]),
        "target": torch.randn(2, 4),
        "num_pixels": 2,
    }
    loss, cosine, correspondence, anchor = _render_loss(
        visual, initial, episode, temperature=0.1, anchor_weight=0.1
    )
    loss.backward()
    assert visual.grad is not None
    assert torch.isfinite(torch.stack((loss, cosine, correspondence, anchor))).all()
    assert visual.grad[[0, 2, 4, 6]].abs().max() == 0


def test_multisource_support_overlap_allows_different_top_gaussian():
    left = torch.tensor([[1, 2, -1], [4, 5, 6]])
    right = torch.tensor([[2, 3, -1], [7, 8, 9]])
    assert torch.equal(_support_overlap(left, right), torch.tensor([True, False]))


def test_observation_set_codec_returns_one_unit_d320_per_set():
    model = ObservationSetVisualCodec()
    value, confidence = model.encode_set(
        torch.randn(2, 3, 1280), torch.randn(2, 3, 768),
        torch.tensor([[0.6, 0.3, 0.1], [0.8, 0.2, 0.0]]),
    )
    assert value.shape == (2, 320)
    assert confidence.shape == (2,)
    assert torch.allclose(value.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_set_materializer_selects_topk_weight_per_gaussian():
    shard = {
        "gaussian_ids": torch.tensor([1, 2, 1, 2, 1]),
        "pixel_ids": torch.tensor([10, 20, 11, 21, 12]),
        "base_weights": torch.tensor([0.2, 0.9, 0.8, 0.1, 0.4]),
    }
    identities, pixels, weights = _topk_all(shard, rows=4, top_k=2)
    assert torch.equal(identities, torch.tensor([1, 2]))
    assert torch.equal(pixels, torch.tensor([[11, 12], [20, 21]]))
    torch.testing.assert_close(weights, torch.tensor([[0.8, 0.4], [0.9, 0.1]]))


def test_direct_support_map_accumulates_shared_visibility_mass():
    left = torch.tensor([[2, 3], [7, -1]])
    left_weight = torch.tensor([[0.6, 0.4], [1.0, 0.0]])
    right = torch.tensor([[3, 9], [2, 3], [8, -1]])
    right_weight = torch.tensor([[0.8, 0.2], [0.7, 0.3], [1.0, 0.0]])
    match, score = _direct_support_map(left, left_weight, right, right_weight)
    assert torch.equal(match, torch.tensor([1, -1]))
    torch.testing.assert_close(score, torch.tensor([0.9, 0.0]))


def test_nearest_centroid_preserves_invalid_queries():
    query = torch.tensor([[0.1, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    reference = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    match, distance = _nearest_centroid(query, reference, chunk_size=1)
    assert torch.equal(match, torch.tensor([1, -1]))
    assert torch.allclose(distance[:1], torch.tensor([0.1]))
    assert torch.isinf(distance[1])


def test_overlap_graph_uses_symmetric_fixed_neighbor_union():
    overlap = torch.tensor([
        [0.0, 0.9, 0.1],
        [0.8, 0.0, 0.7],
        [0.2, 0.6, 0.0],
    ])
    assert _select_graph_edges(overlap, neighbors=1) == [(0, 1), (1, 2)]


def test_view_quality_gate_rejects_collapsed_top_support():
    geometry = {
        1: (torch.tensor([[1], [2], [3]]),),
        2: (torch.tensor([[7], [7], [7]]),),
    }
    accepted, reports = _qualify_views([1, 2], geometry, minimum_unique=2)
    assert accepted == [1]
    assert [row["accepted"] for row in reports] == [True, False]


def test_union_coverage_counts_a_pixel_once_across_neighbors():
    rows = torch.tensor([
        [1, 0, 2, 4, 0, 0, 1, 0, 0, 0, 0, 0, 1.0, 0.8],
        [1, 0, 5, 7, 0, 0, 1, 0, 0, 0, 0, 0, 1.0, 0.9],
        [1, 2, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2],
    ])
    report = _union_coverage(rows, [1], num_pixels=4)[0]
    assert report["direct_cycle_union"] == 0.25


def test_voxel_keys_keep_nearby_rows_together():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]])
    keys = _voxel_keys(xyz, 4)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_group_authority_uses_strongest_observed_row():
    value = _strongest_source_per_group(
        torch.tensor([3, 3, 8]), torch.tensor([2, 5, 7]),
        torch.tensor([0.0, 0.0, 0.2, 0.0, 0.0, 0.9, 0.0, 0.4]),
    )
    assert value == {3: 5, 8: 7}


def test_group_assignment_balances_distance_and_evidence():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
    source_row = torch.full((3,), -1, dtype=torch.long)
    tier = torch.full((3,), -1, dtype=torch.long)
    confidence = torch.zeros(3)
    mixture_source = torch.full((3, 2), -1, dtype=torch.long)
    mixture_weight = torch.zeros(3, 2)
    _assign_group_best(
        target_rows=torch.tensor([0]), target_groups=torch.tensor([4]),
        source_rows=torch.tensor([1, 2]), source_groups=torch.tensor([4, 4]),
        xyz=xyz, evidence_unit=torch.tensor([0.0, 0.8, 1.0]), distance_scale=0.2,
        assigned_before=torch.zeros(3, dtype=torch.bool), source_row=source_row,
        tier=tier, confidence=confidence, tier_value=2,
        mixture_source=mixture_source, mixture_weight=mixture_weight,
    )
    assert source_row[0] == 1
    assert set(mixture_source[0].tolist()) == {1, 2}
    assert torch.allclose(mixture_weight[0].sum(), torch.tensor(1.0))


def test_heldout_tier_report_keeps_fixed_quantiles():
    report = _tier_report(
        torch.tensor([0.2, 0.6, 0.8]), torch.tensor([2, 1, 2]), 2
    )
    assert report["count"] == 2
    assert abs(report["median"] - 0.2) < 1e-6
    assert abs(report["mean"] - 0.5) < 1e-6


def test_render_bucket_report_names_fixed_mass_bucket():
    report = _bucket_report(torch.tensor([0.4, 0.8]), torch.tensor([1, 0]), 1, "fine")
    assert report["bucket"] == "fine"
    assert report["count"] == 1


def test_failed_fine_gate_abstains_from_all_propagated_rows():
    assignments = torch.tensor([
        [0, 0, 0, 1.0], [1, 0, 1, 0.8], [2, 0, 3, 0.6]
    ])
    authority, confidence, unknown = _coverage_policy(
        assignments, assignments[:, 3], fine_gate_pass=False
    )
    assert torch.equal(authority, torch.tensor([True, False, False]))
    torch.testing.assert_close(confidence, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(unknown, 1 - confidence)


def test_masked_structural_split_is_deterministic_disjoint_subset():
    rows = torch.arange(100)
    first = _masked_split(rows)
    second = _masked_split(rows)
    assert torch.equal(first, second)
    assert int(first.sum()) == 20


def test_unknown_interpolation_changes_only_compact_visual_columns():
    memory = torch.zeros(3, 512)
    memory[0, :448] = 1
    memory[1, :448] = 3
    memory[:, 448:] = 7
    value = _interpolate_unknown(
        memory, torch.tensor([2]), torch.tensor([[0, 1]]),
        torch.tensor([[0.25, 0.75]]), torch.tensor([True]),
    )
    assert torch.all(value[2, :448] == 2.5)
    assert torch.equal(value[:, 448:], memory[:, 448:])
    assert torch.equal(value[:2], memory[:2])


def test_unknown_interpolation_can_target_semantic_block_only():
    memory = torch.zeros(2, 512)
    memory[0, 320:448] = 2
    value = _interpolate_unknown(
        memory, torch.tensor([1]), torch.tensor([[0]]), torch.tensor([[1.0]]),
        torch.tensor([True]), start=320, stop=448,
    )
    assert torch.all(value[1, :320] == 0)
    assert torch.all(value[1, 320:448] == 2)
    assert torch.all(value[1, 448:] == 0)


def test_unknown_aware_reliability_is_exactly_five_scalars():
    policy = {
        "visual_write_authority": torch.tensor([True, False]),
        "coverage_confidence": torch.tensor([1.0, 0.0]),
        "unknown_probability": torch.tensor([0.0, 1.0]),
    }
    propagation = {"assignments": torch.tensor([[1, 0, 2, 0.7]])}
    membership = {
        "num_rows": 2, "row_indices": torch.tensor([0, 1, 1]),
        "weights": torch.tensor([0.3, 0.4, 0.8]),
    }
    value = _reliability_scalars(policy, propagation, membership)
    assert value.shape == (2, 5)
    torch.testing.assert_close(value[:, 3], torch.tensor([0.0, 0.7]))
    torch.testing.assert_close(value[:, 4], torch.tensor([0.3, 0.8]))
