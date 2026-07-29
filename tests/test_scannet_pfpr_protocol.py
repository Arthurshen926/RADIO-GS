import numpy as np
import pytest
import torch
from PIL import Image

from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    validate_continuous_support_threshold,
)
from radio_gs.benchmarks.scannet_pfpr import score_dino_center
from radio_gs.benchmarks.scannet_pfpr.build_benchmark import (
    _assert_method_manifest_has_no_private_registration,
    _load_query_raster,
    query_raster_geometry,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import (
    ProtocolConfig,
    evaluate_ranked_locations,
    fixed_radius_nms,
    method_query_record,
    query_frame_exclusion_digest,
    validate_field_query_exclusion_commitment,
)
from radio_gs.benchmarks.scannet_pfpr.evaluate_predictions import evaluate
from radio_gs.benchmarks.scannet_pfpr.score_dino_center import (
    calibrate_query_and_field,
    center_spatial_descriptor,
    center_token_descriptor,
    validate_pfpr_observation_contract,
)
from radio_gs.benchmarks.scannet_pfpr.score_dino_center import (
    _fuse_query_prototype_scores,
    _vector_candidate_similarity,
    sample_spatial_descriptor_at_pixels,
)
from radio_gs.benchmarks.scannet_pfpr.audit_geometry_support import (
    MODE as PFPR_GEOMETRY_SUPPORT_MODE,
    public_geometry_support_record,
    validate_geometry_support_gate,
)
from radio_gs.interfaces.crop_spatial_alignment import GlobalCropSpatialAdapter
from radio_gs.scripts.build_crop_spatial_alignment_cache import _centers


def test_method_query_manifest_never_exposes_anchor_pose_or_depth() -> None:
    record = method_query_record(
        query_id="scene0011_01_q000",
        scene_id="scene0011_01",
        crop_rgb_path="/tmp/query.png",
        crop_rgb_sha256="crop-hash",
    )
    assert record["available_method_inputs"] == ["scene_id", "crop_rgb"]
    forbidden = {"pose", "depth", "anchor", "frame_id", "pixel"}
    assert not any(token in str(record).lower() for token in forbidden)


def test_pfpr_v2_uses_the_registered_depth_aligned_query_raster() -> None:
    """The crop center/FOV must match the RGB-D raster used to build the field."""

    depth_intrinsic = np.asarray(
        [[580.0, 0.0, 319.0], [0.0, 580.0, 239.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    color_intrinsic = np.asarray(
        [[1160.0, 0.0, 646.0], [0.0, 1160.0, 490.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    center, size = query_raster_geometry(
        depth_u=320,
        depth_v=240,
        depth_m=2.0,
        depth_intrinsic=depth_intrinsic,
        color_intrinsic=color_intrinsic,
        depth_size=(640, 480),
        color_size=(1296, 968),
        query_raster_contract="depth_aligned_rgb_v2",
    )
    assert center == (320, 240)
    assert size == (640, 480)


def test_pfpr_v2_resizes_method_visible_crop_source_to_registered_raster(tmp_path) -> None:
    path = tmp_path / "raw.png"
    Image.fromarray(np.full((4, 8, 3), 127, dtype=np.uint8)).save(path)
    image = _load_query_raster(
        path,
        target_size=(4, 2),
        query_raster_contract="depth_aligned_rgb_v2",
    )
    try:
        assert image.size == (4, 2)
    finally:
        image.close()


def test_pfpr_v2_field_must_commit_to_the_exact_public_query_frame_exclusion_set() -> None:
    digest = query_frame_exclusion_digest(["000020", "000002", "000020"])
    assert digest == query_frame_exclusion_digest([2, 20])
    validate_field_query_exclusion_commitment(
        "scannet-pfpr-small-v2", digest, digest
    )
    with pytest.raises(ValueError, match="exclusion"):
        validate_field_query_exclusion_commitment(
            "scannet-pfpr-small-v2", digest, "different"
        )


def test_pfpr_v2_method_manifest_allows_only_a_one_way_frame_commitment() -> None:
    safe = {
        "queries": [
            {
                "query_id": "q",
                "scene_id": "scene",
                "crop_rgb_path": "/tmp/q.png",
                "available_method_inputs": ["scene_id", "crop_rgb"],
            }
        ],
        "scene_domains": [
            {
                "scene_id": "scene",
                "excluded_query_source_frame_ids_sha256": "a" * 64,
            }
        ],
    }
    _assert_method_manifest_has_no_private_registration(safe)
    with pytest.raises(AssertionError, match="private registration"):
        _assert_method_manifest_has_no_private_registration(
            {
                **safe,
                "queries": [{**safe["queries"][0], "source_frame_id": "000020"}],
            }
        )


def test_fixed_radius_nms_returns_spatially_distinct_hypotheses() -> None:
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.04, 0.0, 0.0], [0.20, 0.0, 0.0]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    selected = fixed_radius_nms(xyz, scores, radius_m=0.10, maximum=3)
    np.testing.assert_array_equal(selected, [0, 2])


def test_full_sens_pfpr_uses_the_same_meaningful_support_gate() -> None:
    with pytest.raises(ValueError, match="meaningful continuous support"):
        validate_continuous_support_threshold(
            "scannet_full_observation_pfpr_queryheldout_v1", 1e-6
        )
    validate_continuous_support_threshold(
        "scannet_full_observation_pfpr_queryheldout_v1", 1e-2
    )


def test_pfpr_geometry_admission_reads_only_public_candidates_and_fails_closed() -> None:
    record = public_geometry_support_record(
        scene_id="scene",
        candidate_xyz=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        gaussian_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
        gaussian_covariance=torch.eye(3).reshape(1, 3, 3) * 0.01,
        gaussian_opacity=torch.ones(1),
        candidate_k=1,
        support_threshold=0.01,
        minimum_support_fraction=0.95,
        voxel_size_m=0.05,
    )
    payload = {
        "mode": PFPR_GEOMETRY_SUPPORT_MODE,
        "protocol": {
            "private_anchors_opened": False,
            "query_crop_pixels_opened": False,
            "instance_or_semantic_labels_opened": False,
            "test_set_calibration": False,
        },
        "scene_geometry_support": [record],
    }
    assert validate_geometry_support_gate(
        payload, scene_id="scene", minimum_support_fraction=0.95
    ) == pytest.approx(1.0)
    payload["protocol"]["private_anchors_opened"] = True
    with pytest.raises(ValueError, match="label/query free"):
        validate_geometry_support_gate(
            payload, scene_id="scene", minimum_support_fraction=0.95
        )


def test_recall_uses_anchor_distance_not_instance_identity() -> None:
    predicted = np.asarray([[0.08, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    metrics = evaluate_ranked_locations(
        predicted,
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        config=ProtocolConfig(),
    )
    assert metrics["recall_at_1_10cm"] is True
    assert metrics["recall_at_1_5cm"] is False
    assert metrics["first_correct_rank_10cm"] == 1


def test_evaluator_applies_public_nms_before_anchor_distance_metrics(tmp_path) -> None:
    benchmark = tmp_path / "benchmark"
    candidates = benchmark / "candidates"
    predictions = tmp_path / "predictions"
    candidates.mkdir(parents=True)
    predictions.mkdir()
    candidate_path = candidates / "scene.npy"
    np.save(
        candidate_path,
        np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.30, 0.0, 0.0]], dtype=np.float32),
    )
    (benchmark / "manifest.evaluator.json").write_text(
        '{"benchmark_version":"scannet-pfpr-small-v1","protocol_config":{},'
        '"scene_domains":[{"scene_id":"scene","candidate_xyz_path":"' + str(candidate_path) + '"}],'
        '"queries":[{"query_id":"q","scene_id":"scene","anchor_world_xyz":[0.30,0,0]}]}',
        encoding="utf-8",
    )
    np.save(predictions / "q.npy", np.asarray([0.9, 0.8, 0.7], dtype=np.float32))
    report = evaluate(benchmark, predictions, tmp_path / "report.json")
    assert report["metrics_query_micro"]["R@5_10cm"] == 1.0


def test_pfpr_query_descriptor_is_a_normalized_center_3x3_dino_summary() -> None:
    spatial = np.zeros((1, 2, 5, 5), dtype=np.float32)
    spatial[0, 0, 1:4, 1:4] = 1.0
    descriptor = center_spatial_descriptor(spatial)
    np.testing.assert_allclose(descriptor, [[1.0, 0.0]], atol=1e-6)


def test_pfpr_center_token_uses_only_the_anchor_aligned_spatial_token() -> None:
    spatial = np.zeros((1, 2, 5, 5), dtype=np.float32)
    spatial[0, 0, 1:4, 1:4] = 1.0
    spatial[0, 1, 2, 2] = 2.0
    descriptor = center_token_descriptor(spatial)
    np.testing.assert_allclose(
        descriptor, [[1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)]], atol=1e-6
    )


def test_pfpr_fixed_late_fusion_keeps_a_strong_prototype_without_metric_selection() -> None:
    scores = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    fused = _fuse_query_prototype_scores(scores, temperature=0.1)
    assert fused[0] > fused[1]
    assert 0.8 < float(fused[0]) < 0.9


def test_pfpr_vector_readout_normalizes_after_interpolation() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(2, 1, 1)
    precision = covariance.clone()
    field = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([1.0, 0.0])
    points = torch.tensor([[0.0, 0.0, 0.0]])
    indices = torch.tensor([[0, 1]], dtype=torch.long)
    score = _vector_candidate_similarity(
        xyz,
        covariance,
        field,
        query,
        points,
        precision=precision,
        opacity=torch.ones(2),
        candidate_indices=indices,
        coherence_sqrt=False,
    )
    torch.testing.assert_close(
        score, torch.tensor([1.0 / np.sqrt(2.0)], dtype=torch.float32)
    )


def test_pfpr_interleaved_coarse_to_fine_preserves_top1_and_refines_regions() -> None:
    xyz = np.asarray(
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.00, 0.0, 0.0],
            [1.05, 0.0, 0.0],
            [2.00, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    fine = np.asarray([0.95, 0.90, 0.20, 0.80, 0.10], dtype=np.float32)
    coarse = np.asarray([0.40, 0.30, 0.99, 0.98, 0.10], dtype=np.float32)
    ranked = score_dino_center._interleaved_coarse_to_fine_scores(
        xyz,
        fine,
        coarse,
        np.ones(len(xyz), dtype=np.bool_),
        region_radius_m=0.1,
        maximum_regions=4,
    )
    order = np.argsort(-ranked)
    assert int(order[0]) == 0
    assert int(order[1]) == 3
    assert ranked[2] < ranked[3]


def test_pfpr_interleave_can_anchor_rank_one_to_primary_evidence() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    ranked = score_dino_center._interleaved_coarse_to_fine_scores(
        xyz,
        fine_scores=np.asarray([0.4, 0.99, 0.2], dtype=np.float32),
        coarse_scores=np.asarray([0.3, 0.2, 0.9], dtype=np.float32),
        valid=np.ones(3, dtype=np.bool_),
        region_radius_m=0.1,
        maximum_regions=3,
        rank_one_scores=np.asarray([0.8, 0.1, 0.2], dtype=np.float32),
        rank_one_valid=np.asarray([True, False, True]),
    )

    order = np.argsort(-ranked)
    assert int(order[0]) == 0
    assert int(order[1]) == 1


def test_pfpr_primary_region_can_be_locally_refined_by_support() -> None:
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    ranked = score_dino_center._interleaved_coarse_to_fine_scores(
        xyz,
        fine_scores=np.asarray([0.5, 0.9, 0.7], dtype=np.float32),
        coarse_scores=np.asarray([0.4, 0.3, 0.8], dtype=np.float32),
        valid=np.ones(3, dtype=np.bool_),
        region_radius_m=0.1,
        maximum_regions=3,
        rank_one_scores=np.asarray([0.95, 0.2, 0.1], dtype=np.float32),
        rank_one_valid=np.ones(3, dtype=np.bool_),
        refine_rank_one_with_fine=True,
    )

    assert int(np.argmax(ranked)) == 1


def test_pfpr_primary_prefix_precedes_distant_support_regions() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    merged = score_dino_center._primary_prefix_then_support_scores(
        xyz,
        anchor_scores=np.asarray([0.1, 1.0, 0.0, 0.0, 0.0, 0.0]),
        primary_scores=np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.1]),
        support_scores=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 1.0]),
        primary_prefix=5,
        region_radius_m=0.1,
        maximum=6,
    )
    order = np.argsort(-merged)

    assert int(order[0]) == 1
    assert int(order[1]) == 2
    assert int(order[4]) == 5


def test_pfpr_field_space_calibration_is_query_independent_and_normalized() -> None:
    field = torch.tensor([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    query = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    raw_field, raw_query = calibrate_query_and_field(field, query, method="none")
    robust_field, robust_query = calibrate_query_and_field(
        field, query, method="diagonal_robust", sample_size=3
    )
    torch.testing.assert_close(raw_field.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(raw_query.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(robust_field.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(robust_query.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(raw_field, robust_field)


def test_pfpr_full_sens_source_contract_fails_closed_without_matching_digest() -> None:
    source = {
        "declared_source_contract": "scannet_full_observation_pfpr_queryheldout_v1",
        "field_source_contract_sha256": "a" * 64,
        "field_source_contract_version": "scannet_full_observation_pfpr_queryheldout_v1",
    }
    validate_pfpr_observation_contract(
        "scannet_full_observation_pfpr_queryheldout_v1", source
    )
    with np.testing.assert_raises_regex(ValueError, "matching source-contract version"):
        validate_pfpr_observation_contract(
            "scannet_full_observation_pfpr_queryheldout_v1",
            {**source, "field_source_contract_version": "field_only_dense_rgbd_v1"},
        )
    with np.testing.assert_raises_regex(ValueError, "source-contract digest"):
        validate_pfpr_observation_contract(
            "scannet_full_observation_pfpr_queryheldout_v1",
            {**source, "field_source_contract_sha256": ""},
        )


def test_teacher_oracle_samples_original_pixel_centers_without_registration_guess() -> None:
    # Each feature-grid location has an orthogonal descriptor.  The source
    # image is twice the spatial resolution, so these are exact cell centers.
    spatial = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]]
    )
    descriptor = sample_spatial_descriptor_at_pixels(
        spatial,
        np.asarray([2.5, 2.5], dtype=np.float32),
        image_width=4,
        image_height=4,
    )
    np.testing.assert_allclose(
        descriptor.cpu().numpy(), [[0.0, 1.0]], atol=1e-6
    )


def test_global_crop_spatial_adapter_starts_as_exact_normalized_identity() -> None:
    torch.manual_seed(3)
    adapter = GlobalCropSpatialAdapter(feature_dim=4, hidden_dim=2)
    values = torch.randn(3, 4)
    torch.testing.assert_close(
        adapter(values), torch.nn.functional.normalize(values, dim=-1), atol=1e-6, rtol=1e-6
    )


def test_pfpr_context_adapter_receives_only_crop_visible_center_and_context(
    tmp_path, monkeypatch
) -> None:
    """The optional bridge is query-side: both inputs come from the RGB crop."""

    image_path = tmp_path / "crop.png"
    Image.fromarray(np.full((128, 128, 3), 127, dtype=np.uint8)).save(image_path)

    class _Runtime:
        def encode_adaptor_images(self, images, _name, *, feature_fmt):
            assert feature_fmt == "NCHW"
            batch = int(images.shape[0])
            # The center descriptor is [1, 0], whereas the visible crop-wide
            # mean carries a different context vector.
            spatial = torch.zeros((batch, 2, 5, 5), device=images.device)
            spatial[:, 0, 1:4, 1:4] = 1.0
            spatial[:, 1, 0, :] = 2.0
            spatial[:, 1, 4, :] = 2.0
            spatial[:, 1, :, 0] = 2.0
            spatial[:, 1, :, 4] = 2.0
            return None, spatial

    class _CaptureAdapter:
        def __init__(self):
            self.center = None
            self.context = None

        def __call__(self, center, context):
            self.center = center.detach().cpu()
            self.context = context.detach().cpu()
            return center

    monkeypatch.setattr(
        score_dino_center.OfficialRadioRuntime,
        "load",
        staticmethod(lambda **_kwargs: _Runtime()),
    )
    adapter = _CaptureAdapter()
    descriptors = score_dino_center._encode_scene_queries(
        [{"crop_rgb_path": str(image_path)}],
        device=torch.device("cpu"),
        radio_repo="unused",
        radio_version="unused",
        batch_size=1,
        crop_context_adapter=adapter,  # type: ignore[arg-type]
    )
    assert adapter.center is not None and adapter.context is not None
    np.testing.assert_allclose(descriptors, adapter.center.numpy(), atol=1e-6)
    np.testing.assert_allclose(adapter.center.numpy(), [[1.0, 0.0]], atol=1e-6)
    assert not torch.allclose(adapter.center, adapter.context)
    torch.testing.assert_close(adapter.context.norm(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6)
    paired = score_dino_center._encode_scene_queries(
        [{"crop_rgb_path": str(image_path)}],
        device=torch.device("cpu"),
        radio_repo="unused",
        radio_version="unused",
        batch_size=1,
        crop_context_adapter=adapter,  # type: ignore[arg-type]
        preserve_raw_context_pair=True,
    )
    assert paired.shape == (1, 2, 2)
    np.testing.assert_allclose(paired[:, 0], adapter.center.numpy(), atol=1e-6)
    np.testing.assert_allclose(paired[:, 1], adapter.center.numpy(), atol=1e-6)


def test_pfpr_center_late_fusion_preserves_two_query_visible_prototypes(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "crop.png"
    Image.fromarray(np.full((128, 128, 3), 127, dtype=np.uint8)).save(image_path)

    class _Runtime:
        def encode_adaptor_images(self, images, _name, *, feature_fmt):
            spatial = torch.zeros((len(images), 2, 5, 5), device=images.device)
            spatial[:, 0] = 1.0
            spatial[:, 1, 2, 2] = 3.0
            return None, spatial

    monkeypatch.setattr(
        score_dino_center.OfficialRadioRuntime,
        "load",
        staticmethod(lambda **_kwargs: _Runtime()),
    )
    descriptors = score_dino_center._encode_scene_queries(
        [{"crop_rgb_path": str(image_path)}],
        device=torch.device("cpu"),
        radio_repo="unused",
        radio_version="unused",
        batch_size=1,
        query_pooling="center_late_fusion",
    )
    assert descriptors.shape == (1, 2, 2)
    assert not np.allclose(descriptors[:, 0], descriptors[:, 1])


def test_pfpr_vector_readout_and_fixed_late_fusion_are_explicit() -> None:
    xyz = torch.zeros((2, 3))
    covariance = torch.eye(3).repeat(2, 1, 1)
    field = torch.eye(2)
    query = torch.tensor([1.0, 0.0])
    points = torch.zeros((1, 3))
    indices = torch.tensor([[0, 1]])
    opacity = torch.ones(2)

    normalized = score_dino_center._vector_candidate_similarity(
        xyz,
        covariance,
        field,
        query,
        points,
        precision=covariance,
        opacity=opacity,
        candidate_indices=indices,
        coherence_sqrt=False,
    )
    coherence = score_dino_center._vector_candidate_similarity(
        xyz,
        covariance,
        field,
        query,
        points,
        precision=covariance,
        opacity=opacity,
        candidate_indices=indices,
        coherence_sqrt=True,
    )
    torch.testing.assert_close(normalized, torch.tensor([2**-0.5]))
    torch.testing.assert_close(
        coherence, normalized * torch.tensor(2**-0.25)
    )

    scores = torch.tensor([[0.2, 0.5], [0.4, 0.1]])
    fused = score_dino_center._fuse_query_prototype_scores(
        scores, temperature=0.1
    )
    expected = 0.1 * (
        torch.logsumexp(scores / 0.1, dim=0) - np.log(2.0)
    )
    torch.testing.assert_close(fused, expected)


def test_global_crop_alignment_centers_are_deterministic_and_crop_safe() -> None:
    first = _centers(
        scene="scene_a",
        frame="000000",
        width=320,
        height=240,
        crop_size=128,
        count=8,
        seed=7,
    )
    second = _centers(
        scene="scene_a",
        frame="000000",
        width=320,
        height=240,
        crop_size=128,
        count=8,
        seed=7,
    )
    np.testing.assert_array_equal(first, second)
    assert np.all((first[:, 0] >= 64) & (first[:, 0] <= 256))
    assert np.all((first[:, 1] >= 64) & (first[:, 1] <= 176))
