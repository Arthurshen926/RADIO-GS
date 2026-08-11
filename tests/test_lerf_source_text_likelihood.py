from pathlib import Path
import hashlib

import torch

from radio_gs.querying.lerf_source_text_likelihood import (
    LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_V2_SCHEMA,
    POST_READOUT_ODDS_RESIDUAL_EPS_V4,
    POST_READOUT_ODDS_RESIDUAL_FORMULA_V4,
    POST_READOUT_ODDS_RESIDUAL_TRANSPORT_V4,
    POST_READOUT_PRIOR_PRESERVING_FORMULA_V3,
    POST_READOUT_PRIOR_PRESERVING_MIXTURE_V3,
    PRIOR_PRESERVING_MIXTURE_V2,
    build_lerf_source_text_likelihood_cache,
    compile_effective_probability,
    compile_post_readout_odds_residual,
    compile_post_readout_probability,
    legacy_canonical_field_coverage_reliability,
    state_dict_sha256,
    validate_lerf_source_text_likelihood_cache,
)
from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead
from radio_gs.querying.source_text_query_likelihood import (
    SOURCE_TEXT_CHECKPOINT_SCHEMA,
    source_text_likelihood_contract,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    canonical_negative_relevancy_query_scores,
    source_text_mapping_query_scores,
)


def _save_inputs(root: Path) -> tuple[Path, Path, Path, Path, str]:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    common = {
        "version": 4,
        "contract": "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4",
        "scale_ids": ["0.25", "0.45", "0.7"],
        "scale_radii_m": [0.25, 0.45, 0.7],
        "xyz": xyz,
        "valid": valid,
        "field_checkpoint_sha256": "a" * 64,
        "readout_checkpoint_sha256": "b" * 64,
        "renderer_geometry_checkpoint_sha256": "c" * 64,
    }
    positive = root / "positive.pt"
    negative = root / "negative.pt"
    torch.save(
        {
            **common,
            "query_ids": ["cup", "table"],
            "query_scores": torch.tensor(
                [
                    [[0.3, 0.1], [0.4, 0.0], [0.2, -0.1]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ).contiguous(),
        },
        positive,
    )
    torch.save(
        {
            **common,
            "query_ids": ["object", "things"],
            "query_scores": torch.tensor(
                [
                    [[0.0, -0.1], [0.1, -0.2], [0.0, -0.1]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ).contiguous(),
        },
        negative,
    )
    state = root / "state.pt"
    torch.save(
        {
            "schema": "radio_gs.factorized_primitive_state.v2",
            "schema_version": 2,
            "xyz": xyz,
            "valid": valid,
            "global_rows": torch.tensor([0]),
            "directional_dispersion": torch.tensor([0.2]),
            "observation_evidence": torch.tensor([0.8]),
            "visibility_purity_value": torch.tensor([0.5]),
            "visibility_purity_known": torch.tensor([True]),
            "metadata": {
                "field_checkpoint_sha256": "a" * 64,
                "query_independent": True,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        },
        state,
    )
    head = MonotoneQueryLikelihoodHead(affinity_channel_count=1)
    checkpoint = root / "head.pt"
    state_sha = state_dict_sha256(head.state_dict())
    torch.save(
        {
            "schema": SOURCE_TEXT_CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "head_class": "MonotoneQueryLikelihoodHead",
            "head_schema_version": head.schema_version,
            "state_dict": head.state_dict(),
            "contract": source_text_likelihood_contract(),
            "source_scene_ids": ["scene0001_00"],
            "source_access": {
                "official_scannet_train_scenes_only": True,
                "source_train_semantic_labels_opened": True,
                "development_labels_opened": False,
                "test_labels_opened": False,
                "lerf_queries_or_ground_truth_opened": False,
                "target_rgb_or_mask_opened": False,
                "benchmark_predictions_or_metrics_opened": False,
            },
        },
        checkpoint,
    )
    return positive, negative, state, checkpoint, state_sha


def test_default_mapping_is_bitwise_legacy() -> None:
    positive = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.0, -0.1]]])
    negative = torch.tensor([[[0.0, -0.2], [0.1, -0.1], [-0.1, -0.2]]])
    legacy = canonical_negative_relevancy_query_scores(
        positive, negative, logit_scale=10.0
    )
    default = source_text_mapping_query_scores(
        positive, negative, logit_scale=10.0
    )
    assert torch.equal(default, legacy)


def test_source_text_adapter_separates_q_c_and_exact_effective_probability(
    tmp_path: Path,
) -> None:
    positive, negative, state, checkpoint, state_sha = _save_inputs(tmp_path)
    payload = build_lerf_source_text_likelihood_cache(
        positive_score_cache=positive,
        negative_score_cache=negative,
        factorized_state=state,
        source_text_head_checkpoint=checkpoint,
        expected_head_state_sha256=state_sha,
    )
    assert payload["q"].shape == (2, 2)
    assert payload["c"].shape == (2,)
    assert payload["c"][0] > 0
    assert payload["c"][1] == 0
    assert torch.equal(
        payload["effective_probability"],
        0.5 + payload["c"][:, None] * (payload["q"] - 0.5),
    )
    assert torch.equal(
        payload["effective_probability"][1], torch.tensor([0.5, 0.5])
    )
    validated = validate_lerf_source_text_likelihood_cache(
        payload,
        expected_xyz=payload["xyz"],
        expected_valid=payload["valid"],
        expected_query_ids=["cup", "table"],
        expected_positive_score_cache=positive,
        expected_negative_score_cache=negative,
        expected_renderer_geometry_checkpoint_sha256="c" * 64,
    )
    assert torch.equal(validated.q, payload["q"])


def test_learned_mapping_repeats_only_for_frozen_scale_interface() -> None:
    positive = torch.zeros((2, 3, 2))
    negative = torch.zeros((2, 3, 4))
    learned = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    mapped = source_text_mapping_query_scores(
        positive,
        negative,
        logit_scale=10.0,
        learned_effective_probability=learned,
    )
    assert mapped.shape == (2, 3, 2)
    assert torch.equal(mapped[:, 0], learned)
    assert torch.equal(mapped[:, 1], learned)
    assert torch.equal(mapped[:, 2], learned)


def test_legacy_field_bridge_preserves_coverage_and_masks_unknown_purity() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyz_sha = hashlib.sha256(xyz.numpy().tobytes()).hexdigest()
    field = {
        "schema_version": 1,
        "architecture": {"num_gaussians": 2},
        "geometry_fingerprint": {"num_gaussians": 2, "xyz_sha256": xyz_sha},
        "reliability": torch.tensor([[0.6, 0.9, 1.0], [0.0, 0.0, 0.0]]),
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    coverage, reliability = legacy_canonical_field_coverage_reliability(
        field,
        expected_xyz=xyz,
        expected_valid=torch.tensor([True, False]),
        expected_field_checkpoint_sha256="a" * 64,
        observed_field_checkpoint_sha256="a" * 64,
    )
    assert torch.equal(coverage, torch.tensor([0.6, 0.0]))
    assert torch.allclose(reliability, torch.tensor([0.5, 0.0]))


def test_legacy_field_bridge_rejects_geometry_or_validity_drift() -> None:
    xyz = torch.zeros((2, 3))
    field = {
        "schema_version": 1,
        "architecture": {"num_gaussians": 2},
        "geometry_fingerprint": {"num_gaussians": 2, "xyz_sha256": "0" * 64},
        "reliability": torch.tensor([[0.6, 0.9, 1.0], [0.1, 0.1, 0.1]]),
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    try:
        legacy_canonical_field_coverage_reliability(
            field,
            expected_xyz=xyz,
            expected_valid=torch.tensor([True, False]),
            expected_field_checkpoint_sha256="a" * 64,
            observed_field_checkpoint_sha256="a" * 64,
        )
    except ValueError as error:
        assert "geometry differs" in str(error) or "carrier differs" in str(error)
    else:
        raise AssertionError("legacy bridge accepted drifted geometry/validity")


def test_prior_preserving_mixture_has_exact_boundary_identities() -> None:
    q = torch.tensor([[0.9, 0.1], [0.25, 0.75], [0.6, 0.4]])
    prior = torch.tensor([[0.2, 0.8], [0.8, 0.2], [0.55, 0.45]])
    confidence = torch.tensor([0.0, 1.0, 0.3])
    effective = compile_effective_probability(
        q,
        confidence,
        field_prior=prior,
        mode=PRIOR_PRESERVING_MIXTURE_V2,
    )
    assert torch.equal(effective[0], prior[0])
    assert torch.equal(effective[1], q[1])
    assert torch.equal(
        effective[2], (0.7 * prior[2] + 0.3 * q[2]).float()
    )
    assert bool(((effective >= 0) & (effective <= 1)).all())


def test_post_readout_v3_has_end_to_end_identity_and_locality() -> None:
    field = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    q = torch.tensor([[0.9, 0.1], [0.25, 0.75], [0.7, 0.3]])
    c = torch.tensor([0.0, 1.0, 0.25])
    final = compile_post_readout_probability(field, q, c)
    assert POST_READOUT_PRIOR_PRESERVING_MIXTURE_V3.endswith("_v3")
    assert "field_probability_final" in POST_READOUT_PRIOR_PRESERVING_FORMULA_V3
    assert torch.equal(final[0], field[0])
    assert torch.equal(final[1], q[1])
    assert torch.equal(final[2], (0.75 * field[2] + 0.25 * q[2]).float())
    assert torch.equal(final[c == 0], field[c == 0])
    assert torch.equal(final != field, (final != field) & (c[:, None] > 0))


def test_post_readout_v3_rejects_axes_range_and_nonfinite() -> None:
    field = torch.tensor([[0.2, 0.8]])
    q = torch.tensor([[0.9, 0.1]])
    for changed_q, changed_c in (
        (q[:, :1], torch.tensor([0.5])),
        (torch.tensor([[1.1, 0.1]]), torch.tensor([0.5])),
        (torch.tensor([[float("nan"), 0.1]]), torch.tensor([0.5])),
        (q, torch.tensor([1.1])),
    ):
        try:
            compile_post_readout_probability(field, changed_q, changed_c)
        except ValueError:
            pass
        else:
            raise AssertionError("post-readout v3 accepted malformed inputs")


def test_post_readout_v4_has_exact_neutral_identities_and_monotone_sign() -> None:
    legacy = torch.tensor(
        [[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]],
        dtype=torch.float32,
    )
    raw = torch.tensor(
        [[0.3, 0.7], [0.25, 0.75], [0.5, 0.5], [0.4, 0.6]],
        dtype=torch.float32,
    )
    q = torch.tensor(
        [[0.9, 0.1], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9]],
        dtype=torch.float32,
    )
    c = torch.tensor([1.0, 0.0, 0.75, 0.5], dtype=torch.float32)
    final = compile_post_readout_odds_residual(legacy, q, raw, c)

    assert POST_READOUT_ODDS_RESIDUAL_TRANSPORT_V4.endswith("_v4")
    assert "logit(q) - logit(p_field_raw)" in POST_READOUT_ODDS_RESIDUAL_FORMULA_V4
    assert POST_READOUT_ODDS_RESIDUAL_EPS_V4 == 1.0e-6
    assert torch.equal(final[0], legacy[0])  # zero/one odds are absorbing
    assert torch.equal(final[1], legacy[1])  # c=0 bitwise legacy
    assert torch.equal(final[2], legacy[2])  # q=p_raw bitwise legacy
    assert final[3, 0] < legacy[3, 0]  # negative evidence residual
    assert final[3, 1] > legacy[3, 1]  # positive evidence residual
    assert bool(((final >= 0) & (final <= 1)).all())


def test_post_readout_v4_fixed_epsilon_and_fail_closed_inputs() -> None:
    legacy = torch.tensor([[0.2, 0.8]])
    raw = torch.tensor([[0.3, 0.7]])
    q = torch.tensor([[0.9, 0.1]])
    c = torch.tensor([0.5])
    for changed_legacy, changed_q, changed_raw, changed_c in (
        (legacy[:, :1], q, raw, c),
        (legacy, q[:, :1], raw, c),
        (legacy, q, raw[:, :1], c),
        (legacy, torch.tensor([[float("nan"), 0.1]]), raw, c),
        (legacy, q, torch.tensor([[1.1, 0.7]]), c),
        (legacy, q, raw, torch.tensor([1.1])),
    ):
        try:
            compile_post_readout_odds_residual(
                changed_legacy,
                changed_q,
                changed_raw,
                changed_c,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("post-readout v4 accepted malformed inputs")
    try:
        compile_post_readout_odds_residual(legacy, q, raw, c, eps=1.0e-5)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("post-readout v4 accepted a changed epsilon")


def test_post_readout_v4_enforces_float32_storage_level_monotonicity() -> None:
    generator = torch.Generator().manual_seed(41)
    legacy = torch.rand((257, 9), generator=generator)
    raw = torch.rand((257, 9), generator=generator).clamp(1.0e-4, 1.0 - 1.0e-4)
    direction = torch.where(
        torch.rand((257, 9), generator=generator) > 0.5,
        torch.full((257, 9), float("inf")),
        torch.full((257, 9), float("-inf")),
    )
    q = torch.nextafter(raw, direction)
    c = torch.rand((257,), generator=generator) * 1.0e-4
    final = compile_post_readout_odds_residual(legacy, q, raw, c)
    positive = q > raw
    negative = q < raw
    assert bool((final[positive] >= legacy[positive]).all())
    assert bool((final[negative] <= legacy[negative]).all())


def test_prior_preserving_v2_cache_is_hash_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    positive, negative, state, checkpoint, state_sha = _save_inputs(tmp_path)
    payload = build_lerf_source_text_likelihood_cache(
        positive_score_cache=positive,
        negative_score_cache=negative,
        factorized_state=state,
        source_text_head_checkpoint=checkpoint,
        expected_head_state_sha256=state_sha,
        effective_probability_mode=PRIOR_PRESERVING_MIXTURE_V2,
    )
    assert payload["schema"] == LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_V2_SCHEMA
    assert payload["schema_version"] == 2
    assert torch.equal(
        payload["effective_probability"][1],
        payload["field_prior_probability"][1],
    )
    validate_lerf_source_text_likelihood_cache(
        payload,
        expected_xyz=payload["xyz"],
        expected_valid=payload["valid"],
        expected_query_ids=["cup", "table"],
        expected_positive_score_cache=positive,
        expected_negative_score_cache=negative,
        expected_renderer_geometry_checkpoint_sha256="c" * 64,
    )
    changed = dict(payload)
    changed["field_prior_probability"] = payload[
        "field_prior_probability"
    ].clone()
    changed["field_prior_probability"][0, 0] += 0.01
    try:
        validate_lerf_source_text_likelihood_cache(
            changed,
            expected_xyz=payload["xyz"],
            expected_valid=payload["valid"],
            expected_query_ids=["cup", "table"],
            expected_positive_score_cache=positive,
            expected_negative_score_cache=negative,
            expected_renderer_geometry_checkpoint_sha256="c" * 64,
        )
    except ValueError as error:
        assert "formula changed" in str(error) or "channel changed" in str(error)
    else:
        raise AssertionError("v2 cache accepted a changed field prior")
