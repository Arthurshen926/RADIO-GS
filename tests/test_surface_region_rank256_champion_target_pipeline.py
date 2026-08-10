from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.interfaces import surface_region_rank256_champion as formal
from radio_gs.interfaces import (
    surface_region_v21b_reliability_conditioned_residual as model_interface,
)
from radio_gs.scripts import run_surface_region_rank256_champion_target as pipeline
from radio_gs.utils.immutable_artifacts import file_record


SHA = "0" * 64


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": SHA}


def _gate(tmp_path: Path) -> dict:
    return {
        "source_result": _record(tmp_path / "source.json"),
        "checkpoint": _record(tmp_path / "checkpoint.pt"),
        "normalization_authority": _record(tmp_path / "normalization.pt"),
        "source_promotion_authorized": True,
        "benchmark_opened": False,
    }


def test_target_builder_validates_source_before_target_files(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []

    def reject(*_args, **_kwargs):
        events.append("source")
        raise ValueError("no source PASS")

    monkeypatch.setattr(formal, "validate_champion_source", reject)
    args = argparse.Namespace(
        source_variant="v21b",
        source_result=str((tmp_path / "missing-source.json").resolve()),
        expected_source_result_sha256=SHA,
        output_authority=str((tmp_path / "authority.json").resolve()),
        target_descriptor_output=str((tmp_path / "descriptor.pt").resolve()),
        target_accepted_v2=str((tmp_path / "missing-accepted.pt").resolve()),
        target_adaptive_typed_context=str((tmp_path / "missing-adaptive.pt").resolve()),
        factorized_primitive_state=str((tmp_path / "missing-state.pt").resolve()),
        dataset_id="lerf",
        scene_id="figurines",
        geometry_checkpoint_sha256=SHA,
    )
    with pytest.raises(ValueError, match="no source PASS"):
        pipeline.build_target_authority(args)
    assert events == ["source"]
    assert not Path(args.output_authority).exists()


def test_target_builder_binds_rank256_source_records(tmp_path: Path, monkeypatch) -> None:
    gate = _gate(tmp_path)
    monkeypatch.setattr(pipeline, "_source", lambda _args: gate)
    for name in ("accepted.pt", "adaptive.pt", "state.pt"):
        (tmp_path / name).write_bytes(name.encode())
    args = argparse.Namespace(
        source_variant="v21b",
        source_result=gate["source_result"]["path"],
        expected_source_result_sha256=SHA,
        output_authority=str((tmp_path / "authority.json").resolve()),
        target_descriptor_output=str((tmp_path / "descriptor.pt").resolve()),
        target_accepted_v2=str((tmp_path / "accepted.pt").resolve()),
        target_adaptive_typed_context=str((tmp_path / "adaptive.pt").resolve()),
        factorized_primitive_state=str((tmp_path / "state.pt").resolve()),
        dataset_id="lerf",
        scene_id="figurines",
        geometry_checkpoint_sha256=SHA,
    )
    result = pipeline.build_target_authority(args)
    assert result["status"] == "rank256_target_authority_built"
    raw = __import__("json").loads(Path(args.output_authority).read_text())
    assert raw["source_variant"] == "v21b"
    assert raw["target_inputs"]["champion_checkpoint"] == gate["checkpoint"]
    assert raw["target_inputs"]["champion_normalization"] == gate["normalization_authority"]
    assert raw["access_audit"] == formal.target_access_audit()


def test_rank256_target_materializer_uses_reliability_conditioned_forward(
    tmp_path: Path, monkeypatch
) -> None:
    rows = 3
    base = torch.zeros(rows, 1536, dtype=torch.float32)
    base[:, 0] = 1.0
    context = torch.zeros(rows, 1280, dtype=torch.float32)
    context[:, 0] = 1.0
    raw_scalar = torch.zeros(rows, 18, dtype=torch.float32)
    statistics = torch.zeros(rows, 12, dtype=torch.float32)
    declared = torch.tensor([True, True, False])
    context[~declared] = 0
    accepted = {
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:test",
        "physical_space_authority": {"dataset_id": "lerf", "scene_id": "figurines", "geometry_checkpoint_sha256": SHA},
        "accepted_v2_e0": base,
        "accepted_base_valid": torch.ones(rows, dtype=torch.bool),
        "region_rows": torch.arange(rows).view(rows, 1),
        "token_mask": torch.ones(rows, 1, dtype=torch.bool),
        "anchor_index": torch.zeros(rows, dtype=torch.long),
        "canonical_region_indices": torch.arange(rows, dtype=torch.long),
        "region_fingerprints": [f"{index + 1:064x}" for index in range(rows)],
    }
    adaptive = {
        "region_row_ids": [f"row-{index}" for index in range(rows)],
        "pooled_context_radio_direction": context,
        "typed_context_statistics": statistics,
        "typed_context_valid": declared,
    }
    state = SimpleNamespace()
    input_records = {
        name: _record(tmp_path / f"{name}.pt")
        for name in (
            "target_accepted_v2", "target_adaptive_typed_context",
            "factorized_primitive_state", "champion_checkpoint",
            "champion_normalization",
        )
    }
    execution = {
        "source_variant": "v21b",
        "verified_source_gate": _gate(tmp_path),
        "verified_record": _record(tmp_path / "execution.json"),
        "target_inputs": input_records,
    }
    monkeypatch.setattr(pipeline, "validate_target_authority", lambda *_args, **_kwargs: execution)
    monkeypatch.setattr(pipeline, "_target_inputs", lambda _execution: {"records": input_records, "accepted": accepted, "adaptive": adaptive, "state": state})
    monkeypatch.setattr(pipeline.v21_target, "_validate_alignment", lambda _inputs: None)
    monkeypatch.setattr(pipeline, "aggregate_surface_region_full_scalars", lambda *_args: SimpleNamespace(summary=raw_scalar, use_full_scalar_mask=torch.ones(rows, dtype=torch.bool)))
    monkeypatch.setattr(pipeline.routing, "_pilot_routing", lambda *_args: (declared, torch.zeros(rows, dtype=torch.bool), declared))
    normalization = {"median": torch.zeros(30), "robust_scale": torch.ones(30)}
    model = model_interface.build_model_from_source_normalization(normalization)
    monkeypatch.setattr(formal, "load_champion_model", lambda *_args: (model, normalization, {}))
    output = tmp_path / "descriptor.pt"
    result = pipeline.materialize_target(
        argparse.Namespace(
            output=str(output.resolve()),
            execution_authority=str((tmp_path / "execution.json").resolve()),
            expected_execution_authority_sha256=SHA,
            batch_size=2,
        )
    )
    assert result["status"] == "rank256_target_descriptor_complete"
    payload = torch.load(output, map_location="cpu", weights_only=False)
    validated = formal.validate_target_descriptor(payload)
    assert torch.equal(validated["semantic_descriptor"], base)
    assert torch.equal(validated["active_update_mask"], declared)
    assert validated["fallback_bitwise_equal"] is True
    assert validated["reliability_score"].shape == (rows,)
    assert validated["angular_budget_radians"].shape == (rows,)
