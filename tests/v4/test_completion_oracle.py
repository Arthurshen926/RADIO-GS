import copy
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from radio_gs.v4.completion import (
    OracleIdentityCompletionMLP,
    PartialObjectMembership,
    build_feature_cosine_similarity,
    build_pair_features,
    build_token_context,
    complete_unknown_only,
    completion_metrics,
)
from radio_gs.v4.carrier import Camera, SurfaceVoxelCarrier
from radio_gs.v4.completion.scannet import (
    CACHE_SCHEMA,
    MASK_DROPOUT_KEEP_PROBABILITY,
    MASK_DROPOUT_SALT,
    RADIO_BACKBONE_DIMENSION,
    RADIO_CHECKPOINT_SHA256,
    RADIO_PROJECTION_DIMENSION,
    RADIO_PROJECTION_SHA256,
    _canonical_json_sha256,
    _local_features_from_source_rgb,
    _local_features_from_source_rgb_and_radio,
    _mask_dropout_record,
    _positive_mask_support,
    _radio_feature_layout,
    _radio_projection_matrix,
    _render_valid_surface,
    _source_radio_paths,
    load_scene_cache,
)
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.train_scannet_completion_oracle import (
    RGB_RADIO_GEOMETRY_LAYOUT,
    _ablation_scope,
    _balanced_unknown_indices,
    _changed_factors_against_aligned_radio,
    _shuffle_radio_within_observation_strata,
    _sampling_audit,
)


def _partial():
    labels = torch.tensor([0, 0, 1, 1, -1])
    visible = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
    return labels, PartialObjectMembership.from_oracle_visibility(
        labels, visible, token_count=2
    )


def test_visibility_stratum_sampler_water_fills_without_duplicates():
    labels = torch.tensor(
        [0] * 23 + [1] * 40 + [-1] * 105,
        dtype=torch.long,
    )
    visible_but_unmasked = torch.zeros(labels.numel(), dtype=torch.bool)
    visible_but_unmasked[:3] = True
    visible_but_unmasked[23:43] = True
    visible_but_unmasked[63:68] = True
    never_visible = ~visible_but_unmasked
    runtime = {
        "labels": labels,
        "partial": SimpleNamespace(
            positive=torch.zeros(labels.numel(), 2, dtype=torch.bool),
            unknown=torch.ones(labels.numel(), 2, dtype=torch.bool),
        ),
        "unknown_strata": {
            "visible_but_unmasked": visible_but_unmasked,
            "never_visible": never_visible,
        },
        "payload": {"scene_id": "synthetic"},
    }
    indices = _balanced_unknown_indices(
        runtime,
        8,
        torch.Generator().manual_seed(17),
        sampling_mode="token_visibility_stratum_balanced",
    )
    audit = _sampling_audit(runtime, indices)
    assert indices.numel() == 32  # 8 per token plus 8 * min(K, 8) nulls.
    assert indices.unique().numel() == indices.numel()
    assert audit["selected_object_count"] == 16
    assert audit["selected_null_count"] == 16
    # Scarce visible examples are all retained, then only unused budget is
    # reallocated to the never-visible stratum.
    assert audit["visible_but_unmasked_object_count"] == 7
    assert audit["never_visible_object_count"] == 9
    assert audit["visible_but_unmasked_null_count"] == 5
    assert audit["never_visible_null_count"] == 11


def test_visibility_stratum_sampler_requires_sealed_strata():
    runtime = {
        "labels": torch.tensor([0, -1]),
        "partial": SimpleNamespace(
            positive=torch.zeros(2, 1, dtype=torch.bool),
            unknown=torch.ones(2, 1, dtype=torch.bool),
        ),
    }
    with pytest.raises(ValueError, match="requires sealed visibility strata"):
        _balanced_unknown_indices(
            runtime,
            1,
            torch.Generator().manual_seed(0),
            sampling_mode="token_visibility_stratum_balanced",
        )


def test_partial_membership_partitions_each_eligible_pair():
    labels, partial = _partial()
    state_count = partial.positive.int() + partial.negative.int() + partial.unknown.int()
    assert torch.equal(state_count, torch.ones(5, 2, dtype=torch.int32))
    assert bool(partial.negative[labels < 0].all())
    broken = partial.unknown.clone()
    broken[0, 0] = True
    with pytest.raises(ValueError, match="exactly"):
        PartialObjectMembership(partial.positive, partial.negative, broken, labels >= 0)


def test_completion_is_unknown_only_and_reserves_null_mass():
    _, partial = _partial()
    probability = torch.tensor([
        [0.5, 0.5], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9], [0.0, 0.0]
    ])
    membership, null = complete_unknown_only(
        partial, probability, completion_confidence_cap=0.9
    )
    assert membership[0].tolist() == pytest.approx([1.0, 0.0])
    assert membership[2].tolist() == pytest.approx([0.0, 1.0])
    assert membership[1].tolist() == pytest.approx([0.72, 0.18])
    assert membership[3].tolist() == pytest.approx([0.09, 0.81])
    assert float(null[1]) == pytest.approx(0.1)
    assert float(null[4]) == 1.0


def test_token_context_and_pair_scorer_do_not_need_integer_identity():
    _, partial = _partial()
    centres = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0],
        [2.1, 0.0, 0.0], [4.0, 0.0, 0.0],
    ])
    features = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]
    ])
    edges = torch.tensor([[0, 1, 1, 0, 2, 3, 3, 2], [1, 0, 0, 1, 3, 2, 2, 3]])
    context = build_token_context(
        centres, features, partial, edges, minimum_scale=0.04
    )
    pair = build_pair_features(
        centres, features, context, torch.tensor([1, 3]), minimum_scale=0.04
    )
    assert pair.shape[:2] == (2, 2)
    assert torch.isfinite(pair).all()
    scorer = OracleIdentityCompletionMLP(pair.shape[-1], hidden_dimension=16, dropout=0)
    assert scorer(pair).shape == (2, 2)


def test_categorical_null_logit_is_one_raw_shared_learned_logit():
    scorer = OracleIdentityCompletionMLP(3, hidden_dimension=4, dropout=0)
    for parameter in scorer.parameters():
        torch.nn.init.zeros_(parameter)
    for token_count in (2, 5):
        pair = torch.zeros(1, token_count, 3)
        logits = scorer.categorical_logits(pair)
        assert float(logits[0, -1].detach()) == 0.0
        assert float(torch.softmax(logits, -1)[0, -1].detach()) == pytest.approx(
            1 / (token_count + 1)
        )


def test_explicit_cosine_is_zero_when_missing_and_matches_unit_vectors():
    labels, partial = _partial()
    centres = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0],
        [2.1, 0.0, 0.0], [4.0, 0.0, 0.0],
    ])
    features = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]
    ])
    context = build_token_context(
        centres,
        features,
        partial,
        torch.empty(2, 0, dtype=torch.long),
        minimum_scale=0.04,
    )
    similarity = build_feature_cosine_similarity(
        features,
        context,
        torch.tensor([1, 3, 4]),
        feature_start=0,
        feature_stop=2,
    )
    torch.testing.assert_close(
        similarity,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )


def test_explicit_similarity_residual_changes_tokens_but_not_raw_null():
    scorer = OracleIdentityCompletionMLP(
        3,
        hidden_dimension=4,
        dropout=0,
        explicit_similarity_residual=True,
    )
    for name, parameter in scorer.named_parameters():
        if name != "similarity_log_scale":
            torch.nn.init.zeros_(parameter)
    pair = torch.zeros(1, 2, 3)
    logits = scorer.categorical_logits(
        pair, explicit_similarity=torch.tensor([[1.0, -1.0]])
    )
    assert float(logits[0, 0].detach()) == pytest.approx(1.0, abs=1e-6)
    assert float(logits[0, 1].detach()) == pytest.approx(-1.0, abs=1e-6)
    assert float(logits[0, 2].detach()) == 0
    with pytest.raises(ValueError, match="must align"):
        scorer.categorical_logits(pair)


def test_availability_dual_experts_hard_route_and_share_one_raw_null():
    scorer = OracleIdentityCompletionMLP(
        3,
        hidden_dimension=4,
        dropout=0,
        availability_conditioned_experts=True,
    )
    for parameter in scorer.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        scorer.visible_network[-1].bias.fill_(2.0)
    pair = torch.zeros(2, 2, 3)
    logits = scorer.categorical_logits(
        pair, source_available=torch.tensor([True, False])
    )
    torch.testing.assert_close(logits[0], torch.tensor([2.0, 2.0, 0.0]))
    torch.testing.assert_close(logits[1], torch.tensor([0.0, 0.0, 0.0]))
    assert [name for name, _ in scorer.named_parameters() if name == "null_logit"] == [
        "null_logit"
    ]
    with pytest.raises(ValueError, match="require sealed source availability"):
        scorer.categorical_logits(pair)


def test_rgb_h213_is_parameter_matched_to_radio_h128():
    rgb = OracleIdentityCompletionMLP(41, hidden_dimension=213, dropout=0)
    radio = OracleIdentityCompletionMLP(297, hidden_dimension=128, dropout=0)
    rgb_count = sum(parameter.numel() for parameter in rgb.parameters())
    radio_count = sum(parameter.numel() for parameter in radio.parameters())
    assert rgb_count == 55169
    assert radio_count == 55042
    assert abs(rgb_count - radio_count) / radio_count < 0.005


def test_radio_shuffle_preserves_values_but_destroys_element_alignment():
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(20, len(RGB_RADIO_GEOMETRY_LAYOUT), generator=generator)
    observed = torch.zeros(20, dtype=torch.bool)
    observed[:8] = True
    source_visible = torch.zeros(20, dtype=torch.bool)
    source_visible[:16] = True
    features[~source_visible, 4:68] = 0
    first, receipt = _shuffle_radio_within_observation_strata(
        features,
        scene_id="scene0123_00",
        membership_observed=observed,
        source_visible=source_visible,
        seed=31,
    )
    repeated, repeated_receipt = _shuffle_radio_within_observation_strata(
        features,
        scene_id="scene0123_00",
        membership_observed=observed,
        source_visible=source_visible,
        seed=31,
    )
    changed_seed, changed_receipt = _shuffle_radio_within_observation_strata(
        features,
        scene_id="scene0123_00",
        membership_observed=observed,
        source_visible=source_visible,
        seed=32,
    )
    assert torch.equal(first, repeated)
    assert receipt == repeated_receipt
    assert receipt["receipt_sha256"] != changed_receipt["receipt_sha256"]
    assert not torch.equal(first[:16, 4:68], features[:16, 4:68])
    assert not torch.equal(first, changed_seed)
    assert torch.equal(first[:, :4], features[:, :4])
    assert torch.equal(first[:, 68:], features[:, 68:])
    assert torch.equal(first[~source_visible, 4:68], features[~source_visible, 4:68])
    assert receipt["visible_radio_row_multiset_sha256_before"] == receipt[
        "visible_radio_row_multiset_sha256_after"
    ]
    assert all(row["fixed_point_count"] == 0 for row in receipt["strata"])
    assert all(row["moved_fraction"] == 1 for row in receipt["strata"])
    assert all(
        row["radio_row_multiset_sha256_before"]
        == row["radio_row_multiset_sha256_after"]
        for row in receipt["strata"]
    )
    with pytest.raises(ValueError, match="sealed F71"):
        _shuffle_radio_within_observation_strata(
            features[:, :7],
            scene_id="scene0123_00",
            membership_observed=observed,
            source_visible=source_visible,
            seed=31,
        )


def test_radio_shuffle_receipts_cover_singleton_and_empty_strata():
    features = torch.zeros(3, len(RGB_RADIO_GEOMETRY_LAYOUT))
    features[0, 4:68] = torch.arange(64)
    observed = torch.tensor([True, False, False])
    source_visible = torch.tensor([True, False, False])
    shuffled, receipt = _shuffle_radio_within_observation_strata(
        features,
        scene_id="scene0123_00",
        membership_observed=observed,
        source_visible=source_visible,
        seed=31,
    )
    assert torch.equal(shuffled, features)
    by_name = {row["stratum"]: row for row in receipt["strata"]}
    assert by_name["membership_observed"]["row_count"] == 1
    assert by_name["membership_observed"]["fixed_point_count"] == 1
    assert by_name["membership_observed"]["moved_fraction"] == 0
    assert by_name["membership_observed"]["singleton_fraction"] == 1
    assert by_name["visible_but_unmasked"]["row_count"] == 0
    assert by_name["visible_but_unmasked"]["singleton_fraction"] == 0


def test_ablation_receipt_does_not_call_capacity_matched_rgb_single_factor():
    base = SimpleNamespace(
        local_feature_mode="rgb_radio_geometry",
        unknown_sampling_mode="token_uniform",
        scoring_mode="mlp",
        radio_alignment_control="aligned",
        hidden_dimension=128,
    )
    assert _changed_factors_against_aligned_radio(base) == []
    shuffled = copy.copy(base)
    shuffled.radio_alignment_control = "shuffled_within_observation_stratum"
    assert _changed_factors_against_aligned_radio(shuffled) == [
        "radio_alignment_control"
    ]
    assert _ablation_scope(["radio_alignment_control"]) == (
        "single_factor:radio_alignment_control"
    )
    rgb = copy.copy(base)
    rgb.local_feature_mode = "rgb_geometry"
    rgb.hidden_dimension = 213
    assert _changed_factors_against_aligned_radio(rgb) == [
        "local_feature_mode", "hidden_dimension"
    ]
    assert _ablation_scope(["local_feature_mode", "hidden_dimension"]).startswith(
        "multi_factor:"
    )


def test_completion_metrics_verify_clamps_and_improvement():
    labels, partial = _partial()
    probability = torch.tensor([
        [0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0]
    ])
    membership, _ = complete_unknown_only(
        partial, probability, completion_confidence_cap=0.9
    )
    baseline = completion_metrics(partial.positive.float(), partial, labels)
    learned = completion_metrics(membership, partial, labels)
    assert learned["soft_3d_miou"] > baseline["soft_3d_miou"]
    assert learned["positive_clamp_max_error"] == 0
    assert learned["negative_clamp_max_error"] == 0


def test_completion_metrics_separate_unknown_precision_coverage_and_null():
    labels = torch.tensor([0, 0, 1, 1, -1, -1])
    visible = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool)
    partial = PartialObjectMembership.from_oracle_visibility(
        labels, visible, token_count=2
    )
    membership = torch.tensor([
        [1.0, 0.0],
        [0.8, 0.1],
        [0.0, 1.0],
        [0.6, 0.3],
        [0.7, 0.1],
        [0.1, 0.1],
    ])
    null = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.2, 0.8])
    metrics = completion_metrics(
        membership, partial, labels, null_probability=null, assignment_threshold=0.5
    )
    assert metrics["unknown_assignment_precision"] == pytest.approx(1 / 3)
    assert metrics["unknown_retained_object_coverage"] == pytest.approx(1.0)
    assert metrics["assigned_unknown_object_top1_accuracy"] == pytest.approx(0.5)
    assert metrics["unknown_correct_assignment_recall"] == pytest.approx(0.5)
    assert metrics["unknown_retained_set_null_recall"] == pytest.approx(0.5)
    assert metrics["full_k_plus_null_categorical_accuracy"] == pytest.approx(4 / 6)
    assert metrics["unknown_target_aware_token_mass_precision"] == pytest.approx(1.1 / 2.8)


def test_low_purity_element_is_excluded_from_evidence_and_metrics():
    labels = torch.tensor([0, 0, 1, -1])
    visible = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    eligible = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    partial = PartialObjectMembership.from_oracle_visibility(
        labels, visible, token_count=2, eligible_elements=eligible
    )
    assert not bool(
        partial.positive[1].any()
        or partial.negative[1].any()
        or partial.unknown[1].any()
    )
    probability = torch.tensor([
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.2, 0.3]
    ])
    null_probability = torch.tensor([0.0, 0.0, 0.0, 0.5])
    membership, null = complete_unknown_only(
        partial,
        probability,
        unknown_null_probability=null_probability,
        completion_confidence_cap=0.9,
    )
    assert not bool(membership[1].any())
    assert float(null[1]) == 1.0
    before = completion_metrics(membership, partial, labels, null_probability=null)
    altered = membership.clone()
    altered[1] = torch.tensor([100.0, 100.0])
    after = completion_metrics(altered, partial, labels, null_probability=null)
    assert after == before


def _small_carrier_and_camera(key="source"):
    carrier = SurfaceVoxelCarrier(
        torch.tensor([[0.0, 0.0, 1.0], [50.0, 50.0, 1.0]]),
        0.04,
        normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        maximum_splat_radius=1,
        surface_band_voxels=1.5,
        maximum_contributors_per_pixel=8,
    )
    camera = Camera(
        key,
        torch.tensor([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]),
        torch.eye(4),
        4,
        4,
    )
    return carrier, camera


def test_fixed_radio_projection_matches_preregistered_digest():
    matrix = _radio_projection_matrix()
    assert matrix.shape == (RADIO_BACKBONE_DIMENSION, RADIO_PROJECTION_DIMENSION)
    assert set(matrix.unique().tolist()) == {-0.125, 0.125}
    import hashlib

    assert hashlib.sha256(matrix.numpy().astype("<f4").tobytes()).hexdigest() == RADIO_PROJECTION_SHA256


def test_source_rgb_radio_features_zero_unavailable_and_do_not_open_heldout(tmp_path):
    source_rgb = tmp_path / "source.jpg"
    source_radio = tmp_path / "source.pt"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source_rgb)
    generator = torch.Generator().manual_seed(7)
    torch.save(torch.randn(1280, 4, 4, generator=generator).half(), source_radio)
    # This is deliberately not a Torch file; only explicitly passed source paths may open.
    (tmp_path / "heldout.pt").write_text("poisoned heldout feature")
    carrier, camera = _small_carrier_and_camera()
    baseline, baseline_available = _local_features_from_source_rgb(
        carrier, [camera], [source_rgb], carrier.normals
    )
    features, available = _local_features_from_source_rgb_and_radio(
        carrier, [camera], [source_rgb], [source_radio], carrier.normals
    )
    assert features.shape == (2, 71)
    assert torch.equal(available, baseline_available)
    assert torch.equal(features[:, :3], baseline[:, :3])
    assert torch.equal(features[:, 3] > 0.5, available)
    assert torch.equal(features[~available, :3], torch.zeros(1, 3))
    assert torch.equal(features[~available, 4:68], torch.zeros(1, 64))
    assert torch.allclose(features[available, 4:68].norm(dim=-1), torch.ones(1), atol=1e-5)


def _write_formal_radio_source_bundle(tmp_path, camera, source_rgb, *, saved_stem=None):
    scene_id = "formal_scene"
    scene = tmp_path / scene_id
    backbone = scene / "backbone"
    backbone.mkdir(parents=True)
    saved_stem = saved_stem or f"rgb_{camera.key}"
    feature = backbone / f"{saved_stem}.pt"
    torch.save(torch.ones(1280, camera.height, camera.width, dtype=torch.float16), feature)
    frame = {
        "source_rank": 0,
        "frame_idx": 0,
        "source_file": f"{camera.key}.jpg",
        "source_sha256": sha256_file(source_rgb),
        "saved_stem": saved_stem,
    }
    feature_signature = {
        "backbone": {
            "subdir": "backbone", "dim": 1280,
            "grid": [camera.height, camera.width], "dtype": "float16",
        },
        "summary": {"subdir": "summary", "dim": 2560, "dtype": "float32"},
        "adaptors": [],
    }
    resume_sha = "a" * 64
    bundle = {
        "schema_version": 1,
        "contract": "radio-feature-output-bundle-v1",
        "resume_contract_sha256": resume_sha,
        "frames": [{
            "frame": frame,
            "marker_relative_path": f".extract_frame_commits/{saved_stem}.json",
            "marker_sha256": "b" * 64,
            "feature_signature": feature_signature,
            "tensors": [
                {
                    "relative_path": f"backbone/{saved_stem}.pt",
                    "sha256": sha256_file(feature),
                    "dtype": "float16",
                    "shape": [1280, camera.height, camera.width],
                    "num_bytes": 1280 * camera.height * camera.width * 2,
                },
                {
                    "relative_path": f"summary/{saved_stem}.pt",
                    "sha256": "c" * 64,
                    "dtype": "float32",
                    "shape": [2560],
                    "num_bytes": 2560 * 4,
                },
            ],
        }],
    }
    manifest = {
        "scene": scene_id,
        "radio": {
            "version": "c-radio_v4-h",
            "checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
            "repo_hubconf_sha256": "d" * 64,
            "requested_adaptors": [],
        },
        "features": feature_signature,
        "num_frames": 1,
        "frames": [frame],
        "radio_input_resolution_hw": [camera.height * 16, camera.width * 16],
        "resolution_scale": 1.0,
        "sliding_window": False,
        "execution": {
            "resume_partial": True,
            "resume_contract": ".extract_resume_contract.json",
            "resume_contract_sha256": resume_sha,
            "committed_frame_validation": "same_fd_sha256_weights_only_dtype_shape_finite_v2",
        },
        "output_bundle": bundle,
        "output_bundle_sha256": _canonical_json_sha256(bundle),
    }
    (scene / "frame_manifest.json").write_text(json.dumps(manifest))
    return scene_id, feature, manifest


def test_formal_radio_source_resolver_binds_exact_inventory_and_never_loads_heldout(tmp_path):
    source_rgb = tmp_path / "source.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(source_rgb)
    _carrier, camera = _small_carrier_and_camera()
    scene_id, feature, manifest = _write_formal_radio_source_bundle(tmp_path, camera, source_rgb)
    (feature.parent / "rgb_heldout.pt").write_text("poison")
    manifest_path, paths, provenance = _source_radio_paths(
        tmp_path,
        scene_id,
        [camera],
        [source_rgb],
        height=camera.height,
        width=camera.width,
    )
    assert manifest_path.name == "frame_manifest.json"
    assert paths == [feature]
    assert provenance["radio_checkpoint_sha256"] == RADIO_CHECKPOINT_SHA256
    altered = copy.deepcopy(manifest)
    altered["frames"].append({**altered["frames"][0], "source_file": "heldout.jpg"})
    altered["num_frames"] = 2
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="exactly the selected"):
        _source_radio_paths(
            tmp_path, scene_id, [camera], [source_rgb],
            height=camera.height, width=camera.width,
        )


def test_formal_radio_source_resolver_accepts_numeric_stem_without_leading_zero(tmp_path):
    source_rgb = tmp_path / "source.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(source_rgb)
    _carrier, camera = _small_carrier_and_camera(key="000110")
    scene_id, feature, _manifest = _write_formal_radio_source_bundle(
        tmp_path, camera, source_rgb, saved_stem="rgb_110"
    )
    _manifest_path, paths, _provenance = _source_radio_paths(
        tmp_path,
        scene_id,
        [camera],
        [source_rgb],
        height=camera.height,
        width=camera.width,
    )
    assert paths == [feature]


def test_mask_support_is_deterministic_strict_subset_of_source_visibility():
    centres = torch.tensor([
        [0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]
    ])
    normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1)
    carrier = SurfaceVoxelCarrier(
        centres, 0.04, normals=normals, maximum_splat_radius=1,
        surface_band_voxels=1.5, maximum_contributors_per_pixel=8,
    )
    camera = Camera(
        "a", torch.tensor([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]),
        torch.eye(4), 4, 4,
    )
    source_visible = carrier.lift(torch.ones(4, 4, 1), camera).weight_sum > 0
    identities = torch.tensor([1, 1, 2, 2])
    first, records = _positive_mask_support(
        carrier, [camera], scene_id="scene0000_00",
        element_object_id=identities, object_ids=[1, 2],
    )
    second, repeated = _positive_mask_support(
        carrier, [camera], scene_id="scene0000_00",
        element_object_id=identities, object_ids=[1, 2],
    )
    assert torch.equal(first, second)
    assert records == repeated
    assert bool(first.any()) and bool((source_visible & ~first).any())
    assert not bool((first & ~source_visible).any())
    assert _mask_dropout_record("scene0000_00", "a", 1)["kept"] is True
    assert _mask_dropout_record("scene0000_00", "a", 2)["kept"] is False


def _v4_cache_payload():
    configuration = {
        "voxel_size": 0.04,
        "maximum_splat_radius": 1,
        "surface_band_voxels": 1.5,
        "maximum_contributors_per_pixel": 8,
        "minimum_voxel_instance_purity": 0.8,
        "minimum_observed_elements": 1,
        "observation_view_count": 1,
        "heldout_view_count": 1,
        "feature_height": 2,
        "feature_width": 3,
        "radio_backbone_dimension": 1280,
        "radio_projection_dimension": 64,
        "radio_projection_sha256": RADIO_PROJECTION_SHA256,
        "mask_dropout_keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
        "mask_dropout_salt": MASK_DROPOUT_SALT,
        "local_feature_layout": _radio_feature_layout(),
    }
    metadata = {
        **{
            key: configuration[key]
            for key in (
                "voxel_size", "maximum_splat_radius", "surface_band_voxels",
                "maximum_contributors_per_pixel", "minimum_voxel_instance_purity",
            )
        },
        "mesh_rgb_consumed": False,
        "heldout_rgb_opened": False,
        "heldout_target_authority": "original_mesh_vertex_instance_raycast",
        "heldout_target_uses_sparse_carrier": False,
        "observed_instance_membership_is_oracle_input": True,
        "observed_instance_membership_is_mask_supported_only": True,
        "unobserved_instance_membership_is_training_target_only": True,
        "full_instance_membership_is_training_target_only": False,
        "source_visibility_is_not_membership_observation": True,
        "source_radio_opened": True,
        "heldout_radio_opened": False,
        "radio_version": "c-radio_v4-h",
        "radio_checkpoint_sha256": RADIO_CHECKPOINT_SHA256,
        "radio_manifest_source_rgb_sha_bound": True,
        "radio_manifest_frame_count": 1,
        "radio_output_bundle_sha256": "e" * 64,
        "radio_resume_contract_sha256": "f" * 64,
        "radio_backbone_dimension": 1280,
        "radio_projection_dimension": 64,
        "radio_projection_method": "sha256_entry_rademacher_jl_v1",
        "radio_projection_salt": "radio_gs_v4_fixed_rademacher_jl64_v1",
        "radio_projection_sha256": RADIO_PROJECTION_SHA256,
        "radio_pixel_normalized_before_projection": True,
        "radio_pixel_normalized_after_projection": True,
        "radio_element_normalized_after_lift": True,
        "observation_frame_ids": ["source"],
        "source_radio_frame_ids": ["source"],
        "heldout_frame_ids": ["heldout"],
        "mask_dropout_method": "sha256_scene_frame_original_object_salt_v1",
        "mask_dropout_salt": MASK_DROPOUT_SALT,
        "mask_dropout_keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
        "mask_dropout_depends_on_validation_outcomes": False,
    }
    input_receipt = [
        {"role": "surface_mesh", "path": "mesh", "sha256": "0" * 64},
        {"role": "camera_transforms", "path": "transforms", "sha256": "1" * 64},
        {"role": "instance_segmentation", "path": "segments", "sha256": "2" * 64},
        {"role": "instance_aggregation", "path": "aggregation", "sha256": "3" * 64},
        {"role": "source_observation_rgb_0", "path": "source", "sha256": "4" * 64},
        {"role": "source_radio_frame_manifest", "path": "manifest", "sha256": "5" * 64},
        {
            "role": "source_observation_radio_backbone_0",
            "path": "radio", "sha256": "6" * 64,
        },
    ]
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    local_features = torch.zeros(2, 71)
    local_features[0, 3] = 1
    local_features[0, 4] = 1
    local_features[:, -3:] = normals
    observed = torch.tensor([True, False])
    available = torch.tensor([True, False])
    camera = {
        "key": "source", "intrinsic": torch.eye(3), "camera_to_world": torch.eye(4),
        "height": 2, "width": 3,
    }
    return {
        "schema": CACHE_SCHEMA,
        "scene_id": "scene0000_00",
        "centres": torch.tensor([[0.0, 0.0, 1.0], [50.0, 50.0, 1.0]]),
        "normals": normals,
        "local_features": local_features,
        "token_index": torch.tensor([0, -1]),
        "object_ids": [1],
        "observed_visible": observed.clone(),
        "appearance_available": available.clone(),
        "feature_available": available.clone(),
        "source_visible": available.clone(),
        "mask_supported": observed.clone(),
        "membership_observed": observed.clone(),
        "completion_valid": torch.tensor([True, False]),
        "mask_dropout_receipt": {
            "schema": "radio_gs.surface_object_memory_v4.source_mask_dropout.v1",
            "method": "sha256_scene_frame_original_object_salt_v1",
            "hash_input_order": ["scene_id", "frame_id", "original_object_id", "salt"],
            "salt": MASK_DROPOUT_SALT,
            "keep_probability": MASK_DROPOUT_KEEP_PROBABILITY,
            "candidate_object_ids": [1],
            "retained_object_ids": [1],
            "records": [_mask_dropout_record("scene0000_00", "source", 1)],
        },
        "configuration": configuration,
        "observation_cameras": [camera],
        "heldout_cameras": [{**camera, "key": "heldout"}],
        "heldout_mesh_target_rasters": [torch.zeros(2, 3, 1)],
        "surface_perfect_membership_ceiling": {
            "heldout_2d_soft_miou": 0.5, "token_view_count": 1, "per_view": [],
        },
        "geometry_receipt": {
            "inputs": input_receipt,
            "metadata": metadata,
            "source_rgb_opened": True,
            "target_rgb_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": True,
        },
        "input_receipt": input_receipt,
    }


def test_scene_cache_v4_fails_closed_on_feature_and_membership_tampering(tmp_path):
    path = tmp_path / "cache.pt"
    payload = _v4_cache_payload()
    torch.save(payload, path)
    assert load_scene_cache(path)["scene_id"] == "scene0000_00"

    unavailable = copy.deepcopy(payload)
    unavailable["local_features"][1, 4] = 0.25
    torch.save(unavailable, path)
    with pytest.raises(ValueError, match="exactly zero"):
        load_scene_cache(path)

    unsupported = copy.deepcopy(payload)
    unsupported["membership_observed"][0] = False
    unsupported["mask_supported"][0] = False
    unsupported["observed_visible"][0] = False
    torch.save(unsupported, path)
    with pytest.raises(ValueError, match="minimum mask-supported|deterministic dropout"):
        load_scene_cache(path)


def test_sparse_carrier_ceiling_denominator_includes_invalid_projection_mass():
    from radio_gs.v4.carrier import ProjectionTable

    table = ProjectionTable(
        element_ids=torch.tensor([0, 1]),
        pixel_ids=torch.tensor([0, 0]),
        depths=torch.tensor([1.0, 1.1]),
        weights=torch.tensor([1.0, 1.0]),
        num_elements=2,
        height=1,
        width=1,
    )

    class FixedCarrier:
        def project(self, _camera):
            return table

    camera = Camera("one", torch.eye(3), torch.eye(4), 1, 1)
    rendered = _render_valid_surface(
        FixedCarrier(), torch.ones(2, 1), torch.tensor([True, False]), camera
    )
    assert float(rendered[0, 0, 0]) == pytest.approx(0.5)
