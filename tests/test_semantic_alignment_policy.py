import pytest
import torch
from torch import nn

from radio_gs.interfaces.semantic_alignment import (
    GlobalRegionSummaryBridge,
    GlobalSemanticBridgeManifest,
    SemanticAlignmentPolicy,
    SemanticAlignmentStage,
    SemanticOracleResult,
    project_dense_region_semantics,
)


def _result(stage, miou, loc):
    return SemanticOracleResult(
        stage=stage,
        dataset="oracle",
        miou=miou,
        localization_accuracy=loc,
        sample_count=10,
        protocol_hash="protocol",
    )


def test_policy_stops_at_official_spatial_when_sufficient():
    policy = SemanticAlignmentPolicy(0.4, 0.7)
    decision = policy.decide(
        _result(SemanticAlignmentStage.OFFICIAL_SPATIAL, 0.5, 0.8)
    )
    assert decision.selected_stage is SemanticAlignmentStage.OFFICIAL_SPATIAL


def test_policy_requires_official_crop_oracle_before_bridge():
    policy = SemanticAlignmentPolicy(0.4, 0.7)
    stage1 = _result(SemanticAlignmentStage.OFFICIAL_SPATIAL, 0.2, 0.5)
    with pytest.raises(RuntimeError, match="crop-summary"):
        policy.decide(stage1)


def test_bridge_manifest_is_fail_closed():
    policy = SemanticAlignmentPolicy(0.4, 0.7)
    stage1 = _result(SemanticAlignmentStage.OFFICIAL_SPATIAL, 0.2, 0.5)
    stage2 = _result(SemanticAlignmentStage.OFFICIAL_CROP_SUMMARY, 0.3, 0.6)
    invalid = GlobalSemanticBridgeManifest(
        checkpoint_sha256="bridge",
        training_scope="per_scene",
        frozen=True,
        uses_benchmark_test_vocabulary=False,
        uses_benchmark_scenes=False,
        training_dataset_manifest_sha256="train",
    )
    with pytest.raises(ValueError, match="global_cross_scene"):
        policy.decide(stage1, stage2=stage2, bridge_manifest=invalid)


def test_region_summary_bridge_is_permutation_invariant_and_not_a_text_head():
    torch.manual_seed(0)
    bridge = GlobalRegionSummaryBridge(input_dim=8, output_dim=8, hidden_dim=4)
    tokens = torch.randn(2, 5, 8)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    expected = bridge(tokens)
    permuted = bridge(tokens[:, permutation])

    assert expected.shape == (2, 8)
    torch.testing.assert_close(expected, permuted, atol=1e-6, rtol=1e-6)

    encoded, logits = bridge.encode_region_tokens(tokens)
    cached = bridge.summarize_preencoded_region(tokens, encoded, logits)
    torch.testing.assert_close(expected, cached, atol=1e-6, rtol=1e-6)


def test_dense_region_summary_matches_explicit_center_window():
    torch.manual_seed(1)
    bridge = GlobalRegionSummaryBridge(input_dim=8, output_dim=8, hidden_dim=4)
    feature_map = torch.randn(1, 8, 5, 5)

    dense = bridge.dense_square_regions(feature_map, (3,))
    explicit_tokens = feature_map[0, :, 1:4, 1:4].permute(1, 2, 0).reshape(9, 8)
    explicit = bridge(explicit_tokens)

    torch.testing.assert_close(dense[0, 0, :, 2, 2], explicit, atol=1e-6, rtol=1e-6)


def test_dense_semantic_projection_is_differentiable_and_multiscale():
    torch.manual_seed(2)
    bridge = GlobalRegionSummaryBridge(input_dim=8, output_dim=8, hidden_dim=4)
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    summary_head = nn.Linear(8, 6, bias=False)
    for parameter in summary_head.parameters():
        parameter.requires_grad_(False)
    feature_map = torch.randn(1, 8, 5, 5, requires_grad=True)

    projected = project_dense_region_semantics(
        bridge,
        summary_head,
        feature_map,
        kernel_sizes=(1, 3),
        projection_batch_size=7,
    )
    assert projected.shape == (1, 6, 5, 5)
    projected.sum().backward()
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()


def test_dense_semantic_projection_streaming_matches_stacked_reference():
    torch.manual_seed(3)
    bridge = GlobalRegionSummaryBridge(input_dim=8, output_dim=8, hidden_dim=4)
    summary_head = nn.Linear(8, 6, bias=False)
    for module in (bridge, summary_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    feature_map = torch.randn(2, 8, 5, 7, requires_grad=True)

    actual = project_dense_region_semantics(
        bridge,
        summary_head,
        feature_map,
        kernel_sizes=(1, 3, 5),
        projection_batch_size=11,
    )
    actual.square().sum().backward()
    actual_gradient = feature_map.grad.detach().clone()

    reference_input = feature_map.detach().clone().requires_grad_(True)
    summaries = bridge.dense_square_regions(reference_input, (1, 3, 5))
    batch, scales, channels, height, width = summaries.shape
    tokens = summaries.permute(0, 1, 3, 4, 2).reshape(-1, channels)
    chunks = []
    for start in range(0, tokens.shape[0], 11):
        projected = summary_head(tokens[start : start + 11, None])[:, 0]
        chunks.append(torch.nn.functional.normalize(projected.float(), dim=-1, eps=1e-8))
    descriptor = torch.cat(chunks).reshape(batch, scales, height, width, -1)
    reference = torch.nn.functional.normalize(
        descriptor.mean(dim=1), dim=-1, eps=1e-8
    ).permute(0, 3, 1, 2).contiguous()
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual_gradient, reference_input.grad, atol=3e-6, rtol=3e-6
    )


def test_dense_semantic_projection_rejects_empty_scales_and_invalid_batch_size():
    bridge = GlobalRegionSummaryBridge(input_dim=8, output_dim=8, hidden_dim=4)
    summary_head = nn.Linear(8, 6, bias=False)
    feature_map = torch.randn(1, 8, 3, 3)

    with pytest.raises(ValueError, match="at least one scale"):
        project_dense_region_semantics(bridge, summary_head, feature_map, kernel_sizes=())
    with pytest.raises(ValueError, match="must be positive"):
        project_dense_region_semantics(
            bridge, summary_head, feature_map, projection_batch_size=0
        )
