from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts import materialize_lerf_genuine_mpr_raw_query_scores as raw


def _embedding(*nonzero_indices: int) -> torch.Tensor:
    value = torch.zeros(raw.FEATURE_DIMENSION, dtype=torch.float32)
    for index in nonzero_indices:
        value[index] = 1.0
    return value


def test_select_text_axis_is_exact_ordered_and_fp32() -> None:
    bank = {
        "queries": ["b", "a", "c"],
        "embeddings": torch.stack(
            [_embedding(1), _embedding(0), _embedding(2)]
        ),
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": "synthetic-siglip2",
    }
    selected, metadata = raw.select_text_axis(bank, ["c", "a"])
    assert selected.dtype == torch.float32
    assert torch.equal(selected[0], _embedding(2))
    assert torch.equal(selected[1], _embedding(0))
    assert metadata["ids"] == ["c", "a"]
    assert metadata["embedding_tensor_sha256"] == frozen.tensor_sha256_typed(
        selected
    )


def test_compute_raw_scores_is_fp32_replicated_and_invalid_zero() -> None:
    features = torch.zeros(3, raw.FEATURE_DIMENSION, dtype=torch.float16)
    features[0, 0] = 1.0
    features[2, 1] = 1.0
    valid = torch.tensor([True, False, True])
    text = torch.stack([_embedding(0), _embedding(1)])
    result = raw.compute_raw_scores(features, valid, text, chunk_size=1)
    assert result.shape == (3, raw.SCALE_COUNT, 2)
    assert result.dtype == torch.float32
    assert result.is_contiguous()
    assert torch.equal(result[:, 0], result[:, 1])
    assert torch.equal(result[:, 1], result[:, 2])
    assert torch.equal(result[0, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(result[1], torch.zeros(raw.SCALE_COUNT, 2))
    assert torch.equal(result[2, 0], torch.tensor([0.0, 1.0]))


def test_compute_raw_scores_rejects_nonzero_invalid_and_zero_valid() -> None:
    text = _embedding(0)[None]
    features = torch.zeros(2, raw.FEATURE_DIMENSION, dtype=torch.float16)
    features[0, 0] = 1.0
    with pytest.raises(ValueError, match="invalid MPR feature rows"):
        raw.compute_raw_scores(
            features, torch.tensor([False, True]), text, chunk_size=1
        )
    with pytest.raises(ValueError, match="valid MPR feature rows"):
        raw.compute_raw_scores(
            torch.zeros_like(features),
            torch.tensor([True, False]),
            text,
            chunk_size=1,
        )


def test_geometry_check_is_bitwise_and_dtype_strict() -> None:
    xyz = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    geometry = {"num_gaussians": 1, "xyz_sha256": "a" * 64}
    mpr = {"xyz": xyz, "geometry_fingerprint": geometry}
    raw._validate_geometry(
        mpr, {"xyz": xyz.clone(), "geometry_fingerprint": dict(geometry)}
    )
    with pytest.raises(ValueError, match="both be FP32"):
        raw._validate_geometry(
            mpr,
            {
                "xyz": xyz.double(),
                "geometry_fingerprint": dict(geometry),
            },
        )
    with pytest.raises(ValueError, match="bitwise"):
        raw._validate_geometry(
            mpr,
            {
                "xyz": xyz + torch.tensor([[0.0, 0.0, 1e-6]]),
                "geometry_fingerprint": dict(geometry),
            },
        )


def test_pair_output_paths_are_distinct_canonical_and_noclobber(
    tmp_path: Path,
) -> None:
    values = [str((tmp_path / name).resolve()) for name in ("p.pt", "p.json", "n.pt", "n.json")]
    args = SimpleNamespace(
        output_positive_cache=values[0],
        output_positive_report=values[1],
        output_negative_cache=values[2],
        output_negative_report=values[3],
    )
    assert set(raw._canonical_new_outputs(args)) == {
        "positive_cache",
        "positive_report",
        "negative_cache",
        "negative_report",
    }
    args.output_negative_report = args.output_positive_report
    with pytest.raises(ValueError, match="distinct"):
        raw._canonical_new_outputs(args)
    args.output_negative_report = values[3]
    Path(values[2]).touch()
    with pytest.raises(FileExistsError, match="must be new"):
        raw._canonical_new_outputs(args)


def test_authority_marks_genuine_descriptor_and_template_only_binding() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    geometry = {"num_gaussians": 2, "xyz_sha256": "1" * 64}
    mpr = {
        "xyz": xyz,
        "valid": valid,
        "geometry_fingerprint": geometry,
        "metadata": {
            "construction": "semantic_descriptor_raster_gaussian_top1_contribution_mean",
            "aggregation_mode": "raster_gaussian_top1",
            "registration_weight_mode": "alpha_depth",
            "raster_view_fusion": "contribution_mean",
            "num_declared_views": 2,
            "selected_frame_indices": [1, 3],
            "observation_lifting_contract": {"query_independent": True},
        },
    }
    sources = {
        role: {"path": f"/synthetic/{role}", "sha256": char * 64}
        for role, char in (
            ("field_checkpoint", "2"),
            ("readout_checkpoint", "3"),
            ("renderer_geometry_checkpoint", "4"),
            ("legacy_fp16_materializer_source", "5"),
        )
    }
    template_raw = {
        "authority": {
            "geometry_axis": {
                "num_gaussians": 2,
                "xyz_sha256": "1" * 64,
                "renderer_xyz_sha256": "1" * 64,
                "field_checkpoint_sha256": "2" * 64,
                "readout_checkpoint_sha256": "3" * 64,
                "renderer_geometry_checkpoint_sha256": "4" * 64,
            },
            "source_artifacts": sources,
        }
    }
    template = frozen.OursMultiscaleQueryScoreCache(
        query_scores=torch.zeros(2, 3, 1),
        valid=torch.tensor([True, True]),
        query_ids=("query",),
        scale_ids=("0.25", "0.45", "0.7"),
        scale_radii_m=(0.25, 0.45, 0.7),
        xyz_sha256="1" * 64,
        field_checkpoint_sha256="2" * 64,
        readout_checkpoint_sha256="3" * 64,
        renderer_geometry_checkpoint_sha256="4" * 64,
        score_semantics="raw_independent_normalized_cosine",
        score_formula=raw.SCORE_FORMULA,
        probability_route="",
        semantic_source_artifacts={},
    )
    record = {"path": "/synthetic/input", "sha256": "6" * 64}
    payload, authority = raw._build_payload(
        mpr=mpr,
        template_raw=template_raw,
        template=template,
        query_scores=torch.zeros(2, 3, 1),
        query_axis={"ids": ["query"], "order_sha256": "7" * 64},
        mpr_record=record,
        mpr_sidecar_record=record,
        text_bank_record=record,
        materializer_record=record,
        template_positive_record=record,
        template_negative_record=record,
        score_role="positive_benchmark_queries",
        features_sha256="8" * 64,
    )
    descriptor = authority["descriptor_axis"]
    assert descriptor["source"] == "sealed_genuine_official_crop_summary_mpr_features"
    assert descriptor["template_representation_inherited"] is False
    assert authority["score_role"] == "positive_benchmark_queries"
    assert payload["valid"].tolist() == [True, False]


def test_source_has_one_query_score_sha_key_and_audit_is_metric_closed() -> None:
    source = Path(raw.__file__).read_text(encoding="utf-8")
    assert source.count('"query_scores_sha256":') == 1
    audit = raw.access_audit()
    assert audit["benchmark_images_opened"] is False
    assert audit["benchmark_masks_opened"] is False
    assert audit["target_metrics_computed"] is False
    assert audit["gpu_used"] is False
