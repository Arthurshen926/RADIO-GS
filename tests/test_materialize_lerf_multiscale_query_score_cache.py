from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.evaluation.openclip_readout import cosine_logits
from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
)
from radio_gs.interfaces.surface_region_summary import (
    SURFACE_REGION_V3_GATED_RAW_PRIOR,
    SURFACE_SUMMARY_READOUT_V3,
    SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
    SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
)
from radio_gs.scripts import build_surface_region_semantic_cache as surface_builder
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    validate_ours_multiscale_query_score_cache,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    DESCRIPTOR_DIMENSION,
    DIRECT3D_CONTRACT,
    SCORE_SEMANTICS_2D,
    SHARED_AUTHORITY_CONTRACT,
    SIGLIP2_MODEL_NAME,
    SIGLIP2_TEXT_CANONICALIZATION,
    _direct_xyz_sha256,
    _readout_native_scales,
    materialize,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache_fp32 import (
    CACHE_VERSION as FP32_CACHE_VERSION,
    DIRECT3D_CONTRACT as FP32_DIRECT3D_CONTRACT,
    SHARED_AUTHORITY_CONTRACT as FP32_SHARED_AUTHORITY_CONTRACT,
    _compile_query_scores_fp32,
    materialize_fp32,
)
from radio_gs.scripts.materialize_lerf_streamed_multiscale_query_score_cache import (
    PROBABILITY_AUTHORITY_CONTRACT,
    PROBABILITY_DIRECT3D_CONTRACT,
    PROBABILITY_FEATURE_SPACE,
    PROBABILITY_SCORE_SEMANTICS,
    STREAMED_COMPLETION_REASON,
    STREAMED_CONSTRUCTION,
    STREAMED_FEATURE_SPACE,
    STREAMED_SCALE_AGGREGATION,
    materialize_streamed,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readout_authority_payload(
    contract: SurfaceRegionContractV2 | SurfaceRegionContractV3,
    *,
    schema_version: int,
    architecture_name: str,
    base_output_mode: str | None = None,
) -> dict[str, object]:
    architecture = {
        "name": architecture_name,
        "contract_sha256": contract.digest,
    }
    if base_output_mode is not None:
        architecture["base_output_mode"] = base_output_mode
    return {
        "schema_version": schema_version,
        "architecture": architecture,
        "provenance": {
            "region_contract": contract.to_dict(),
            "region_contract_sha256": contract.digest,
        },
    }


def test_readout_native_scales_accepts_strict_v2_and_v3_authorities() -> None:
    radii = (0.25, 0.45, 0.7)
    v2 = SurfaceRegionContractV2(radii_m=radii)
    v3 = SurfaceRegionContractV3(radii_m=radii)
    assert _readout_native_scales(
        _readout_authority_payload(
            v2,
            schema_version=3,
            architecture_name="surface_region_summary_readout_v2",
        )
    ) == radii
    assert _readout_native_scales(
        _readout_authority_payload(
            v3,
            schema_version=SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
            architecture_name=SURFACE_SUMMARY_READOUT_V3,
        )
    ) == radii
    assert _readout_native_scales(
        _readout_authority_payload(
            v3,
            schema_version=SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
            architecture_name=SURFACE_SUMMARY_READOUT_V3,
            base_output_mode=SURFACE_REGION_V3_GATED_RAW_PRIOR,
        )
    ) == radii


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("architecture", "architecture differs"),
        ("contract", "schema/contract version differs"),
        ("provenance_digest", "contract SHA256 differs"),
        ("architecture_digest", "contract SHA256 differs"),
        ("base_output_mode", "schema/base-output mode differs"),
    ],
)
def test_readout_native_scales_rejects_authority_mismatch(
    mutation: str, message: str
) -> None:
    contract: SurfaceRegionContractV2 | SurfaceRegionContractV3 = (
        SurfaceRegionContractV2()
        if mutation == "contract"
        else SurfaceRegionContractV3()
    )
    payload = _readout_authority_payload(
        contract,
        schema_version=SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
        architecture_name=SURFACE_SUMMARY_READOUT_V3,
    )
    if mutation == "architecture":
        payload["architecture"]["name"] = "surface_region_summary_readout_v2"
    elif mutation == "provenance_digest":
        payload["provenance"]["region_contract_sha256"] = "0" * 64
    elif mutation == "architecture_digest":
        payload["architecture"]["contract_sha256"] = "0" * 64
    elif mutation == "base_output_mode":
        payload["architecture"]["base_output_mode"] = (
            SURFACE_REGION_V3_GATED_RAW_PRIOR
        )
    with pytest.raises(ValueError, match=message):
        _readout_native_scales(payload)


def _write_inputs(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, False, True, True])
    global_rows = torch.where(valid)[0]
    native_radii = (0.25, 0.45, 0.7)
    readout_contract = SurfaceRegionContractV2(radii_m=native_radii)

    field = tmp_path / "field.pt"
    field.write_bytes(b"synthetic immutable canonical field checkpoint\n")
    field_sha = _sha(field)
    readout = tmp_path / "readout.pt"
    torch.save(
        {
            "schema_version": 3,
            "architecture": {
                "name": "surface_region_summary_readout_v2",
                "contract_sha256": readout_contract.digest,
            },
            "provenance": {
                "region_contract": readout_contract.to_dict(),
                "region_contract_sha256": readout_contract.digest,
            },
        },
        readout,
    )
    readout_sha = _sha(readout)
    renderer = tmp_path / "renderer.pt"
    torch.save({"model_state_dict": {"_xyz": xyz.clone()}}, renderer)
    renderer_sha = _sha(renderer)

    descriptors = torch.zeros(3, 3, DESCRIPTOR_DIMENSION, dtype=torch.float16)
    for row in range(descriptors.shape[0]):
        for scale in range(descriptors.shape[1]):
            descriptors[row, scale, (row + scale) % 4] = 1.0
    descriptor_payload = {
        "xyz": xyz,
        "valid": valid,
        "global_rows": global_rows,
        "features_by_scale": descriptors,
        "metadata": {
            "schema_version": 5,
            "feature_space": "official_siglip2_summary_descriptor_multiscale",
            "source": "canonical_radio_surface_region_readout",
            "query_set_invariant": True,
            "official_summary_head": True,
            "custom_text_projection": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "region_radii_m": list(native_radii),
            "region_contract": {"radii_m": list(native_radii)},
            "field_checkpoint": str(field.resolve()),
            "field_checkpoint_sha256": field_sha,
            "field_geometry_xyz_sha256": _direct_xyz_sha256(xyz),
            "readout_checkpoint": str(readout.resolve()),
            "readout_checkpoint_sha256": readout_sha,
            "official_radio_checkpoint_sha256": "b" * 64,
        },
    }
    descriptor = tmp_path / "surface_multiscale.pt"
    torch.save(descriptor_payload, descriptor)

    text_embeddings = torch.zeros(2, DESCRIPTOR_DIMENSION, dtype=torch.float32)
    text_embeddings[0, 0] = 2.0
    text_embeddings[1, 1] = 3.0
    text_payload = {
        "queries": ["red cup", "tea pot"],
        "query_ids": ["red cup", "tea pot"],
        "embeddings": text_embeddings,
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": SIGLIP2_MODEL_NAME,
        "text_canonicalization": SIGLIP2_TEXT_CANONICALIZATION,
    }
    text = tmp_path / "siglip2_queries.pt"
    torch.save(text_payload, text)
    return {
        "field": field,
        "field_sha": field_sha,
        "readout": readout,
        "readout_sha": readout_sha,
        "renderer": renderer,
        "renderer_sha": renderer_sha,
        "descriptor": descriptor,
        "descriptor_sha": _sha(descriptor),
        "descriptor_payload": descriptor_payload,
        "text": text,
        "text_sha": _sha(text),
        "text_payload": text_payload,
        "xyz": xyz,
        "valid": valid,
        "global_rows": global_rows,
        "descriptors": descriptors,
        "text_embeddings": text_embeddings,
        "native_radii": native_radii,
    }


def _run(inputs: dict[str, object], output: Path) -> dict[str, object]:
    return materialize(
        descriptor_cache=inputs["descriptor"],
        descriptor_cache_sha256=inputs["descriptor_sha"],
        text_query_cache=inputs["text"],
        text_query_cache_sha256=inputs["text_sha"],
        field_checkpoint=inputs["field"],
        field_checkpoint_sha256=inputs["field_sha"],
        readout_checkpoint=inputs["readout"],
        readout_checkpoint_sha256=inputs["readout_sha"],
        renderer_geometry_checkpoint=inputs["renderer"],
        renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
        output=output,
        chunk_size=2,
    )


def _run_fp32(inputs: dict[str, object], output: Path) -> dict[str, object]:
    return materialize_fp32(
        descriptor_cache=inputs["descriptor"],
        descriptor_cache_sha256=inputs["descriptor_sha"],
        text_query_cache=inputs["text"],
        text_query_cache_sha256=inputs["text_sha"],
        field_checkpoint=inputs["field"],
        field_checkpoint_sha256=inputs["field_sha"],
        readout_checkpoint=inputs["readout"],
        readout_checkpoint_sha256=inputs["readout_sha"],
        renderer_geometry_checkpoint=inputs["renderer"],
        renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
        output=output,
        chunk_size=2,
    )


def _write_streamed_scores(
    inputs: dict[str, object], path: Path
) -> tuple[Path, str, torch.Tensor]:
    xyz = inputs["xyz"]
    valid = inputs["valid"]
    rows = inputs["global_rows"]
    descriptors = inputs["descriptors"]
    text_embeddings = inputs["text_embeddings"]
    assert isinstance(xyz, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    assert isinstance(rows, torch.Tensor)
    assert isinstance(descriptors, torch.Tensor)
    assert isinstance(text_embeddings, torch.Tensor)
    scores = torch.zeros(
        len(xyz), 3, int(text_embeddings.shape[0]), dtype=torch.float16
    )
    for scale in range(3):
        scores[rows, scale] = cosine_logits(
            descriptors[:, scale], text_embeddings
        ).half()
    builder_path = Path(surface_builder.__file__).resolve()
    field = Path(inputs["field"]).resolve()
    readout = Path(inputs["readout"]).resolve()
    text = Path(inputs["text"]).resolve()
    payload = {
        "xyz": xyz,
        "valid": valid,
        "features": scores,
        "metadata": {
            "schema_version": 3,
            "feature_space": STREAMED_FEATURE_SPACE,
            "construction": STREAMED_CONSTRUCTION,
            "scoring": "raw_independent_normalized_cosine",
            "scale_aggregation": STREAMED_SCALE_AGGREGATION,
            "scale_count": 3,
            "scale_radii_m": list(inputs["native_radii"]),
            "query_names": list(inputs["text_payload"]["queries"]),
            "text_embedding_cache": str(text),
            "text_embedding_cache_sha256": inputs["text_sha"],
            "streaming_implementation": {
                "path": str(builder_path),
                "sha256": _sha(builder_path),
            },
            "semantic_cache_materialized": False,
            "completion": {
                "applied": False,
                "reason": STREAMED_COMPLETION_REASON,
            },
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": True,
            "semantic_provenance": {
                "source": "canonical_radio_surface_region_readout",
                "query_set_invariant": True,
                "official_summary_head": True,
                "custom_text_projection": False,
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
                "region_radii_m": list(inputs["native_radii"]),
                "field_checkpoint": str(field),
                "field_checkpoint_sha256": inputs["field_sha"],
                "field_geometry_xyz_sha256": _direct_xyz_sha256(xyz),
                "readout_checkpoint": str(readout),
                "readout_checkpoint_sha256": inputs["readout_sha"],
                "official_radio_checkpoint_sha256": "b" * 64,
            },
        },
    }
    torch.save(payload, path)
    return path, _sha(path), scores


def _run_streamed(
    inputs: dict[str, object], streamed: Path, streamed_sha: str, output: Path
) -> dict[str, object]:
    return materialize_streamed(
        streamed_score_cache=streamed,
        streamed_score_cache_sha256=streamed_sha,
        text_query_cache=inputs["text"],
        text_query_cache_sha256=inputs["text_sha"],
        field_checkpoint=inputs["field"],
        field_checkpoint_sha256=inputs["field_sha"],
        readout_checkpoint=inputs["readout"],
        readout_checkpoint_sha256=inputs["readout_sha"],
        renderer_geometry_checkpoint=inputs["renderer"],
        renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
        output=output,
    )


def _write_streamed_probabilities(
    inputs: dict[str, object], path: Path
) -> tuple[Path, str, torch.Tensor]:
    xyz = inputs["xyz"]
    valid = inputs["valid"]
    assert isinstance(xyz, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    scores = torch.zeros(len(xyz), 3, 2, dtype=torch.float16)
    scores[valid] = torch.tensor([0.25, 0.75], dtype=torch.float16)
    artifacts: dict[str, dict[str, str]] = {}
    for role in (
        "residual_codebook_checkpoint",
        "query_router_checkpoint",
        "generic_negative_text_cache",
    ):
        source = path.with_name(f"{role}.pt")
        source.write_bytes(f"immutable {role}\n".encode())
        artifacts[role] = {"path": str(source.resolve()), "sha256": _sha(source)}
    builder_path = Path(surface_builder.__file__).resolve()
    field = Path(inputs["field"]).resolve()
    readout = Path(inputs["readout"]).resolve()
    text = Path(inputs["text"]).resolve()
    provenance = {
        "source": "canonical_radio_surface_region_residual_codebook_query_router",
        "query_set_invariant": False,
        "representation_query_set_invariant": True,
        "query_router_query_dependent": True,
        "exact_frozen_v2_slot0_control": True,
        "official_summary_head": True,
        "custom_text_projection": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": True,
        "region_radii_m": list(inputs["native_radii"]),
        "field_checkpoint": str(field),
        "field_checkpoint_sha256": inputs["field_sha"],
        "field_geometry_xyz_sha256": _direct_xyz_sha256(xyz),
        "readout_checkpoint": str(readout),
        "readout_checkpoint_sha256": inputs["readout_sha"],
        "official_radio_checkpoint_sha256": "b" * 64,
        "query_router_score_contract": "canonical_negative_bernoulli_query_first",
        "query_router_logit_scale": 10.0,
        "slot_projection_contract": "four_independent_official_head_calls_Bx1x1280",
        "generic_negative_queries": ["object", "things", "stuff", "texture"],
    }
    for role, record in artifacts.items():
        provenance[role] = record["path"]
        provenance[f"{role}_sha256"] = record["sha256"]
    payload = {
        "xyz": xyz,
        "valid": valid,
        "features": scores,
        "metadata": {
            "schema_version": 4,
            "feature_space": PROBABILITY_FEATURE_SPACE,
            "construction": (
                "surface_residual_codebook_slotwise_head_then_query_router"
            ),
            "scoring": "canonical_negative_bernoulli_query_router_v1",
            "score_semantics": PROBABILITY_SCORE_SEMANTICS,
            "probability_route": "query_router_v1",
            "value_range": [0.0, 1.0],
            "logit_scale": 10.0,
            "generic_negative_queries": ["object", "things", "stuff", "texture"],
            "scale_aggregation": STREAMED_SCALE_AGGREGATION,
            "scale_count": 3,
            "scale_radii_m": list(inputs["native_radii"]),
            "query_names": list(inputs["text_payload"]["queries"]),
            "text_embedding_cache": str(text),
            "text_embedding_cache_sha256": inputs["text_sha"],
            "streaming_implementation": {
                "path": str(builder_path),
                "sha256": _sha(builder_path),
            },
            "semantic_cache_materialized": False,
            "completion": {
                "applied": False,
                "reason": STREAMED_COMPLETION_REASON,
            },
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": True,
            "semantic_provenance": provenance,
        },
    }
    torch.save(payload, path)
    return path, _sha(path), scores


def test_materializes_exact_n3q_direct3d_contract_and_shared_authority(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "scores.pt"

    report = _run(inputs, output)
    payload, output_sha, _ = load_torch_mapping(
        output, expected_sha256=report["query_score_cache"]["sha256"]
    )

    assert output_sha == sha256_file(output)
    assert payload["version"] == 2
    assert payload["contract"] == DIRECT3D_CONTRACT
    assert payload["query_ids"] == ["red cup", "tea pot"]
    assert payload["scale_ids"] == ["0.25", "0.45", "0.7"]
    assert payload["scale_radii_m"] == [0.25, 0.45, 0.7]
    assert payload["query_scores"].shape == (4, 3, 2)
    assert payload["query_scores"].dtype == torch.float16
    assert not bool(payload["query_scores"][1].any())

    expected = torch.zeros_like(payload["query_scores"])
    descriptors = inputs["descriptors"]
    text_embeddings = inputs["text_embeddings"]
    global_rows = inputs["global_rows"]
    assert isinstance(descriptors, torch.Tensor)
    assert isinstance(text_embeddings, torch.Tensor)
    assert isinstance(global_rows, torch.Tensor)
    for scale in range(3):
        expected[global_rows, scale] = cosine_logits(
            descriptors[:, scale], text_embeddings
        ).half()
    assert torch.equal(payload["query_scores"], expected)

    direct = validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=inputs["xyz"],
        expected_query_ids=("red cup", "tea pot"),
        expected_renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
    )
    assert direct.query_scores.shape == (4, 3, 2)
    assert direct.scale_ids == ("0.25", "0.45", "0.7")
    assert direct.scale_radii_m == (0.25, 0.45, 0.7)
    assert direct.field_checkpoint_sha256 == inputs["field_sha"]
    assert direct.renderer_geometry_checkpoint_sha256 == inputs["renderer_sha"]
    assert inputs["field_sha"] != inputs["renderer_sha"]

    authority = payload["authority"]
    assert authority["contract"] == SHARED_AUTHORITY_CONTRACT
    assert authority["scale_axis"] == [
        {"id": "0.25", "value": 0.25, "unit": "meter"},
        {"id": "0.45", "value": 0.45, "unit": "meter"},
        {"id": "0.7", "value": 0.7, "unit": "meter"},
    ]
    assert authority["query_axis"]["ids"] == ["red cup", "tea pot"]
    assert (
        authority["consumer_contracts"]["lerf2d_scalar_map_renderer"][
            "score_semantics"
        ]
        == SCORE_SEMANTICS_2D
    )
    constraints = authority["calibration_constraints"]
    assert set(constraints.values()) == {False}
    assert report["shared_renderer_authority"] == authority
    assert output.with_suffix(".pt.json").is_file()


def test_fp32_materializer_is_explicit_versioned_and_does_not_quantize(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    legacy_output = tmp_path / "legacy_fp16.pt"
    fp32_output = tmp_path / "explicit_fp32.pt"

    _run(inputs, legacy_output)
    report = _run_fp32(inputs, fp32_output)
    legacy = torch.load(legacy_output, map_location="cpu", weights_only=False)
    payload = torch.load(fp32_output, map_location="cpu", weights_only=False)

    assert legacy["version"] == 2
    assert legacy["contract"] == DIRECT3D_CONTRACT
    assert legacy["query_scores"].dtype == torch.float16
    assert payload["version"] == FP32_CACHE_VERSION
    assert payload["contract"] == FP32_DIRECT3D_CONTRACT
    assert payload["authority"]["contract"] == FP32_SHARED_AUTHORITY_CONTRACT
    assert payload["query_scores"].dtype == torch.float32
    assert payload["authority"]["score_dtype"] == "torch.float32"
    assert payload["authority"]["precision_contract"] == {
        "normalization_dtype": "torch.float32",
        "matmul_dtype": "torch.float32",
        "storage_dtype": "torch.float32",
        "post_matmul_quantization": False,
        "legacy_fp16_default_changed": False,
    }
    assert report["execution"]["explicit_fp32_opt_in"] is True
    assert report["execution"]["legacy_fp16_default_changed"] is False

    descriptors = inputs["descriptors"]
    global_rows = inputs["global_rows"]
    text_embeddings = inputs["text_embeddings"]
    xyz = inputs["xyz"]
    assert isinstance(descriptors, torch.Tensor)
    assert isinstance(global_rows, torch.Tensor)
    assert isinstance(text_embeddings, torch.Tensor)
    assert isinstance(xyz, torch.Tensor)
    expected = _compile_query_scores_fp32(
        descriptors,
        global_rows,
        text_embeddings,
        total_rows=int(xyz.shape[0]),
        chunk_size=2,
    )
    assert torch.equal(payload["query_scores"], expected)
    assert torch.equal(legacy["query_scores"], expected.half())

    validated = validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=xyz,
        expected_query_ids=("red cup", "tea pot"),
        expected_renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
    )
    assert torch.equal(validated.query_scores, expected)


def test_fp32_compiler_promotes_fp16_inputs_before_normalization_and_matmul() -> None:
    generator = torch.Generator().manual_seed(1701)
    descriptors = torch.randn(5, 3, 64, generator=generator).half()
    text = torch.randn(4, 64, generator=generator).half()
    rows = torch.arange(5, dtype=torch.int64)

    actual = _compile_query_scores_fp32(
        descriptors,
        rows,
        text,
        total_rows=5,
        chunk_size=3,
    )
    reference = torch.stack(
        [
            F.normalize(descriptors[:, scale].float(), dim=-1)
            @ F.normalize(text.float(), dim=-1).T
            for scale in range(3)
        ],
        dim=1,
    )
    fp16_quantized_operands = torch.stack(
        [
            (
                F.normalize(descriptors[:, scale].float(), dim=-1).half().float()
                @ F.normalize(text.float(), dim=-1).half().float().T
            )
            for scale in range(3)
        ],
        dim=1,
    )

    # GEMM may choose a different CPU kernel for the compiler's 3/2-row
    # chunks than for this five-row reference, while both remain FP32.
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    assert not torch.equal(actual, fp16_quantized_operands)
    assert float((actual - fp16_quantized_operands).abs().max()) > 1e-6


@pytest.mark.parametrize("mutation", ["dtype", "tensor", "authority_dtype"])
def test_fp32_validator_fails_closed_on_precision_drift(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _write_inputs(tmp_path / mutation)
    output = tmp_path / f"{mutation}.pt"
    _run_fp32(inputs, output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    if mutation == "dtype":
        payload["query_scores"] = payload["query_scores"].half()
        message = "contiguous torch.float32"
    elif mutation == "tensor":
        payload["query_scores"][0, 0, 0] += 1e-4
        message = "query-score SHA256 differs"
    else:
        payload["authority"]["score_dtype"] = "torch.float16"
        message = "authority score dtype differs"

    with pytest.raises(ValueError, match=message):
        validate_ours_multiscale_query_score_cache(
            payload,
            expected_xyz=inputs["xyz"],
            expected_query_ids=("red cup", "tea pot"),
            expected_renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
        )


def test_streamed_path_is_bitwise_equivalent_and_frozen_validator_compatible(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    streamed, streamed_sha, expected_scores = _write_streamed_scores(
        inputs, tmp_path / "streamed_scores.pt"
    )
    output = tmp_path / "streamed_authority.pt"
    report = _run_streamed(inputs, streamed, streamed_sha, output)
    payload, _, _ = load_torch_mapping(
        output, expected_sha256=report["query_score_cache"]["sha256"]
    )

    assert torch.equal(payload["query_scores"], expected_scores)
    assert payload["authority"]["descriptor_axis"] == {
        "dimension": DESCRIPTOR_DIMENSION,
        "materialized": False,
        "execution_representation": "streamed_scalar_scores_only",
        "valid_rows": int(inputs["valid"].sum()),
        "streamed_query_score_cache_sha256": streamed_sha,
        "readout_checkpoint_sha256": inputs["readout_sha"],
        "official_radio_checkpoint_sha256": "b" * 64,
    }
    direct = validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=inputs["xyz"],
        expected_query_ids=("red cup", "tea pot"),
        expected_renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
    )
    assert torch.equal(direct.query_scores.half(), expected_scores)

    standard = tmp_path / "standard_authority.pt"
    _run(inputs, standard)
    standard_payload, _, _ = load_torch_mapping(standard)
    assert torch.equal(standard_payload["query_scores"], payload["query_scores"])


def test_streamed_probability_path_is_explicit_v3_and_directly_consumable(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    streamed, streamed_sha, expected_scores = _write_streamed_probabilities(
        inputs, tmp_path / "streamed_probabilities.pt"
    )
    output = tmp_path / "probability_authority.pt"
    report = _run_streamed(inputs, streamed, streamed_sha, output)
    payload, _, _ = load_torch_mapping(output)

    assert payload["version"] == 3
    assert payload["contract"] == PROBABILITY_DIRECT3D_CONTRACT
    assert payload["authority"]["contract"] == PROBABILITY_AUTHORITY_CONTRACT
    assert payload["authority"]["score_semantics"] == PROBABILITY_SCORE_SEMANTICS
    assert torch.equal(payload["query_scores"], expected_scores)
    direct = validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=inputs["xyz"],
        expected_query_ids=("red cup", "tea pot"),
        expected_renderer_geometry_checkpoint_sha256=inputs["renderer_sha"],
    )
    assert direct.score_semantics == PROBABILITY_SCORE_SEMANTICS
    assert direct.probability_route == "query_router_v1"
    assert set(direct.semantic_source_artifacts) == {
        "residual_codebook_checkpoint",
        "query_router_checkpoint",
        "generic_negative_text_cache",
    }
    assert report["score_semantics"] == PROBABILITY_SCORE_SEMANTICS


def test_streamed_probability_rejects_values_outside_unit_interval(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    streamed, _, _ = _write_streamed_probabilities(
        inputs, tmp_path / "invalid_probabilities.pt"
    )
    payload = torch.load(streamed, map_location="cpu")
    payload["features"][0, 0, 0] = 1.25
    torch.save(payload, streamed)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _run_streamed(inputs, streamed, _sha(streamed), tmp_path / "invalid.pt")


def test_no_clobber_rejects_existing_cache_or_receipt(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "scores.pt"
    _run(inputs, output)

    with pytest.raises(FileExistsError, match="immutable output already exists"):
        _run(inputs, output)

    second = tmp_path / "second.pt"
    second.with_suffix(".pt.json").symlink_to(output.with_suffix(".pt.json"))
    with pytest.raises(FileExistsError, match="immutable output already exists"):
        _run(inputs, second)


def test_rejects_tampered_or_symlinked_source_cache(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    descriptor = inputs["descriptor"]
    assert isinstance(descriptor, Path)
    descriptor.write_bytes(descriptor.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _run(inputs, tmp_path / "tampered.pt")

    clean = _write_inputs(tmp_path / "symlink_case")
    link = tmp_path / "descriptor-link.pt"
    link.symlink_to(clean["descriptor"])
    clean["descriptor"] = link
    with pytest.raises(ValueError, match="symlink"):
        _run(clean, tmp_path / "symlinked.pt")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda descriptor, text: descriptor["metadata"].update(
                {"region_radii_m": [0.45, 0.25, 0.7]}
            ),
            "strictly increasing",
        ),
        (
            lambda descriptor, text: text.update(
                {"query_ids": ["tea pot", "red cup"]}
            ),
            "query_ids/order differs",
        ),
        (
            lambda descriptor, text: descriptor["metadata"].update(
                {"field_checkpoint_sha256": "c" * 64}
            ),
            "field_checkpoint SHA256 differs",
        ),
    ],
)
def test_rejects_scale_query_and_geometry_authority_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    descriptor_payload = inputs["descriptor_payload"]
    text_payload = inputs["text_payload"]
    assert isinstance(descriptor_payload, dict)
    assert isinstance(text_payload, dict)
    mutation(descriptor_payload, text_payload)
    descriptor = inputs["descriptor"]
    text = inputs["text"]
    assert isinstance(descriptor, Path)
    assert isinstance(text, Path)
    torch.save(descriptor_payload, descriptor)
    torch.save(text_payload, text)
    inputs["descriptor_sha"] = _sha(descriptor)
    inputs["text_sha"] = _sha(text)

    with pytest.raises(ValueError, match=message):
        _run(inputs, tmp_path / "drift.pt")


def test_rejects_descriptor_scale_order_that_differs_from_bound_readout(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    readout = inputs["readout"]
    descriptor = inputs["descriptor"]
    descriptor_payload = inputs["descriptor_payload"]
    assert isinstance(readout, Path)
    assert isinstance(descriptor, Path)
    assert isinstance(descriptor_payload, dict)

    drift_contract = SurfaceRegionContractV2(radii_m=(0.2, 0.4, 0.7))
    torch.save(
        {
            "schema_version": 3,
            "architecture": {
                "name": "surface_region_summary_readout_v2",
                "contract_sha256": drift_contract.digest,
            },
            "provenance": {
                "region_contract": drift_contract.to_dict(),
                "region_contract_sha256": drift_contract.digest,
            },
        },
        readout,
    )
    inputs["readout_sha"] = _sha(readout)
    descriptor_payload["metadata"]["readout_checkpoint_sha256"] = inputs[
        "readout_sha"
    ]
    torch.save(descriptor_payload, descriptor)
    inputs["descriptor_sha"] = _sha(descriptor)

    with pytest.raises(ValueError, match="differs from readout checkpoint"):
        _run(inputs, tmp_path / "readout-drift.pt")


def test_rejects_renderer_xyz_row_order_even_with_separately_bound_sha(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    renderer = inputs["renderer"]
    xyz = inputs["xyz"]
    assert isinstance(renderer, Path)
    assert isinstance(xyz, torch.Tensor)
    torch.save({"model_state_dict": {"_xyz": xyz[[1, 0, 2, 3]]}}, renderer)
    inputs["renderer_sha"] = _sha(renderer)

    with pytest.raises(ValueError, match="xyz/count/row-order differs"):
        _run(inputs, tmp_path / "renderer-row-drift.pt")


def test_public_cli_has_no_calibration_knobs() -> None:
    source = Path(
        "radio_gs/scripts/materialize_lerf_multiscale_query_score_cache.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--temperature"' not in source
    assert 'parser.add_argument("--threshold"' not in source
    assert 'parser.add_argument("--peak-normalize"' not in source
    assert 'parser.add_argument("--scale-aggregation"' not in source
