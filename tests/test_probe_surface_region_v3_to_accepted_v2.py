from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV3
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    DESCRIPTOR_DIMENSION,
    _direct_xyz_sha256,
    _validate_descriptor_cache,
)
from radio_gs.scripts.probe_surface_region_v3_to_accepted_v2 import (
    _full_output_paths,
    _load_text_axis,
    _response_comparison,
    _v3_to_v2_adapter_inputs,
    _write_full_outputs,
)
from radio_gs.utils.immutable_artifacts import sha256_file


def test_v3_to_v2_adapter_keeps_prefix_and_excludes_fill() -> None:
    geometry = torch.arange(2 * 4 * 16, dtype=torch.float32).reshape(2, 4, 16)
    token_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    support_fill = torch.tensor(
        [[False, True, False, False], [False, False, False, False]]
    )
    reliability = torch.tensor(
        [[[0.8], [0.4], [0.6], [0.0]], [[0.7], [0.5], [0.0], [0.0]]]
    )

    v2_geometry, v2_mask, v2_reliability = _v3_to_v2_adapter_inputs(
        geometry, token_mask, support_fill, reliability
    )

    assert v2_mask.tolist() == [
        [True, False, True, False],
        [True, True, False, False],
    ]
    assert torch.equal(v2_geometry[v2_mask], geometry[..., :14][v2_mask])
    assert not bool(v2_geometry[~v2_mask].any())
    assert torch.equal(v2_reliability[v2_mask], reliability[v2_mask])
    assert not bool(v2_reliability[~v2_mask].any())


def test_response_comparison_is_exact_for_identical_responses() -> None:
    generator = torch.Generator().manual_seed(7)
    response = torch.randn(200, 3, 5, generator=generator)

    report = _response_comparison(response, response.clone())

    assert report["pearson_flat"] == pytest.approx(1.0)
    assert report["pearson_per_scale_query_mean"] == pytest.approx(1.0)
    assert report["spearman_per_scale_query_mean"] == pytest.approx(1.0)
    assert report["top1pct_overlap_mean"] == pytest.approx(1.0)
    assert report["mean_absolute_error"] == 0.0
    assert report["root_mean_square_error"] == 0.0
    assert report["scale_argmax_agreement"] == 1.0


def test_load_text_axis_preserves_requested_order_and_normalizes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "text.pt"
    torch.save(
        {
            "queries": ["a", "b"],
            "embeddings": torch.tensor([[3.0, 0.0], [0.0, 7.0]]),
        },
        path,
    )

    result = _load_text_axis(path, ["b", "a"])

    assert torch.equal(result, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_full_fallback_descriptor_passes_official_materializer_validator(
    tmp_path: Path,
) -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, False, True])
    global_rows = torch.where(valid)[0]
    contract = SurfaceRegionContractV3(radii_m=(0.25, 0.45, 0.7))

    authorities = {}
    for name in (
        "field",
        "accepted_v2",
        "v3_checkpoint",
        "support_graph",
        "mpr_cache",
        "radio_checkpoint",
        "raw_resume_contract",
    ):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(f"immutable {name}\n".encode())
        authorities[name] = path

    v3_descriptor = tmp_path / "v3_descriptor.pt"
    torch.save(
        {
            "xyz": xyz,
            "valid": valid,
            "global_rows": global_rows,
            "features_by_scale": torch.ones(
                len(global_rows), 3, DESCRIPTOR_DIMENSION, dtype=torch.float16
            ),
            "metadata": {
                "schema_version": 5,
                "feature_space": (
                    "official_siglip2_summary_descriptor_multiscale"
                ),
                "source": "canonical_radio_surface_region_readout",
                "query_set_invariant": True,
                "official_summary_head": True,
                "custom_text_projection": False,
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
                "region_radii_m": list(contract.radii_m),
                "field_checkpoint": str(authorities["field"].resolve()),
                "field_checkpoint_sha256": sha256_file(authorities["field"]),
                "field_geometry_xyz_sha256": _direct_xyz_sha256(xyz),
                "readout_checkpoint": str(
                    authorities["v3_checkpoint"].resolve()
                ),
                "readout_checkpoint_sha256": sha256_file(
                    authorities["v3_checkpoint"]
                ),
                "official_radio_checkpoint_sha256": "b" * 64,
            },
        },
        v3_descriptor,
    )
    candidate = torch.randn(
        len(global_rows),
        3,
        DESCRIPTOR_DIMENSION,
        generator=torch.Generator().manual_seed(11),
    ).half()
    score_geometry = {
        "xyz": xyz,
        "valid": valid,
        "geometry_fingerprint": {
            "num_gaussians": len(xyz),
            "xyz_sha256": _direct_xyz_sha256(xyz),
        },
        "field_checkpoint_sha256": sha256_file(authorities["field"]),
        "renderer_geometry_checkpoint_sha256": "c" * 64,
    }
    paths = _full_output_paths(tmp_path / "full")

    _write_full_outputs(
        paths=paths,
        candidate=candidate,
        positive_scores=torch.randn(len(global_rows), 3, 2),
        negative_scores=torch.randn(len(global_rows), 3, 1),
        positive_ids=["one", "two"],
        negative_ids=["object"],
        contract=contract,
        global_rows=global_rows,
        output_valid=valid,
        primary_valid=None,
        v3_descriptor_path=v3_descriptor,
        accepted_v2_checkpoint=authorities["accepted_v2"],
        v3_checkpoint=authorities["v3_checkpoint"],
        support_graph=authorities["support_graph"],
        mpr_cache=authorities["mpr_cache"],
        radio_checkpoint=authorities["radio_checkpoint"],
        raw_resume_contract=authorities["raw_resume_contract"],
        score_geometry_authority=score_geometry,
    )
    payload = torch.load(
        paths["descriptor"], map_location="cpu", weights_only=False
    )

    validated = _validate_descriptor_cache(
        payload,
        field_checkpoint_path=authorities["field"].resolve(),
        field_checkpoint_sha256=sha256_file(authorities["field"]),
        readout_checkpoint_path=authorities["accepted_v2"].resolve(),
        readout_checkpoint_sha256=sha256_file(authorities["accepted_v2"]),
        readout_native_scales=contract.radii_m,
    )

    assert validated["row_storage"] == "sparse_valid_rows"
    assert validated["native_scales"] == contract.radii_m
    assert payload["metadata"]["source"] == (
        "canonical_radio_surface_region_readout"
    )
    assert payload["metadata"]["readout_checkpoint_sha256"] == sha256_file(
        authorities["accepted_v2"]
    )
    assert payload["metadata"]["surface_region_adapter"]["support_fill"] == (
        "excluded_from_v2_token_mask"
    )
