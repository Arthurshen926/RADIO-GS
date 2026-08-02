from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.evaluation.openclip_readout import cosine_logits
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
    materialize,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    field = tmp_path / "field.pt"
    field.write_bytes(b"synthetic immutable canonical field checkpoint\n")
    field_sha = _sha(field)
    readout = tmp_path / "readout.pt"
    torch.save(
        {
            "schema_version": 3,
            "provenance": {"region_contract": {"radii_m": list(native_radii)}},
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

    torch.save(
        {
            "schema_version": 3,
            "provenance": {
                "region_contract": {"radii_m": [0.2, 0.4, 0.7]}
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
