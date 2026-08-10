from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from radio_gs.scripts import (
    build_lerf_v21_external_cache_execution_authority as external_builder,
)
from radio_gs.utils.immutable_artifacts import file_record


def _record(name: str) -> dict[str, str]:
    return {"path": f"/source/{name}", "sha256": name[0] * 64}


def _source_gate() -> dict:
    return {
        "source_promotion_authorized": True,
        "source_result": _record("result.json"),
        "execution_authority": _record("execution.json"),
        "checkpoint": _record("checkpoint.pt"),
        "normalization_authority": _record("normalization.pt"),
    }


from radio_gs.scripts import (
    build_surface_region_v21_query_execution_authority as query_builder,
)
from radio_gs.scripts import (
    build_surface_region_v21_target_execution_authority as target_builder,
)


def _args(tmp_path: Path, kind: str) -> argparse.Namespace:
    common = {
        "source_pilot_result": str((tmp_path / "source.json").resolve()),
        "expected_source_pilot_result_sha256": "a" * 64,
        "output_authority": str((tmp_path / f"{kind}.json").resolve()),
    }
    if kind == "target":
        return argparse.Namespace(
            **common,
            dataset_id="lerf",
            scene_id="figurines",
            geometry_checkpoint_sha256="9" * 64,
            target_accepted_v2=str((tmp_path / "accepted.pt").resolve()),
            target_adaptive_typed_context=str((tmp_path / "adaptive.pt").resolve()),
            factorized_primitive_state=str((tmp_path / "state.pt").resolve()),
            target_descriptor_output=str((tmp_path / "descriptor.pt").resolve()),
        )
    if kind == "query":
        return argparse.Namespace(
            **common,
            target_descriptor=str((tmp_path / "descriptor.pt").resolve()),
            positive_text_cache=str((tmp_path / "positive.pt").resolve()),
            query_relevance_output=str((tmp_path / "relevance.pt").resolve()),
        )
    return argparse.Namespace(
        **common,
        query_relevance_execution_authority=str(
            (tmp_path / "query_execution.json").resolve()
        ),
        query_relevance_authority=str((tmp_path / "relevance.pt").resolve()),
        comembership_feature_authority=str((tmp_path / "feature.pt").resolve()),
        comembership_inference_authority=str((tmp_path / "inference.pt").resolve()),
        renderer_geometry_checkpoint=str((tmp_path / "renderer.pth").resolve()),
        output_cache=str((tmp_path / "external.pt").resolve()),
        output_report=str((tmp_path / "external_report.json").resolve()),
    )


@pytest.mark.parametrize(
    ("module", "kind"),
    [
        (target_builder, "target"),
        (query_builder, "query"),
        (external_builder, "external"),
    ],
)
def test_builder_source_failure_precedes_every_target_stat_or_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module, kind: str
) -> None:
    touched = {"target": 0}

    def reject(*args, **kwargs):
        raise ValueError("source promotion rejected")

    def target_touch(*args, **kwargs):
        touched["target"] += 1
        raise AssertionError("target/query/code touched before source PASS")

    monkeypatch.setattr(module, "validate_source_pilot_chain", reject)
    for name in ("_canonical_existing", "_canonical_new", "file_record"):
        monkeypatch.setattr(module, name, target_touch)
    for name in ("load_json_object", "load_torch_mapping"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, target_touch)
    with pytest.raises(ValueError, match="source promotion rejected"):
        module.build(_args(tmp_path, kind))
    assert touched["target"] == 0


@pytest.mark.parametrize("module", [target_builder, query_builder, external_builder])
def test_builder_rejects_existing_and_noncanonical_outputs(
    tmp_path: Path, module
) -> None:
    existing = tmp_path / "exists.pt"
    existing.write_bytes(b"occupied")
    with pytest.raises(ValueError, match="new canonical"):
        module._canonical_new(str(existing.resolve()), label="output")
    noncanonical = str(tmp_path.resolve() / "nested" / ".." / "new.pt")
    with pytest.raises(ValueError, match="new canonical"):
        module._canonical_new(noncanonical, label="output")


def test_builder_clis_expose_only_authority_inputs_and_outputs() -> None:
    target_destinations = {
        action.dest for action in target_builder.build_parser()._actions
    }
    query_destinations = {
        action.dest for action in query_builder.build_parser()._actions
    }
    external_destinations = {
        action.dest for action in external_builder.build_parser()._actions
    }
    assert target_destinations == {
        "help",
        "source_pilot_result",
        "expected_source_pilot_result_sha256",
        "dataset_id",
        "scene_id",
        "geometry_checkpoint_sha256",
        "target_accepted_v2",
        "target_adaptive_typed_context",
        "factorized_primitive_state",
        "target_descriptor_output",
        "output_authority",
    }
    assert query_destinations == {
        "help",
        "source_pilot_result",
        "expected_source_pilot_result_sha256",
        "target_descriptor",
        "positive_text_cache",
        "query_relevance_output",
        "output_authority",
    }
    assert external_destinations == {
        "help",
        "source_pilot_result",
        "expected_source_pilot_result_sha256",
        "query_relevance_execution_authority",
        "query_relevance_authority",
        "comembership_feature_authority",
        "comembership_inference_authority",
        "renderer_geometry_checkpoint",
        "output_cache",
        "output_report",
        "output_authority",
    }


def test_target_builder_auto_binds_source_model_code_and_preregistration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, "target")
    for name in (
        "target_accepted_v2",
        "target_adaptive_typed_context",
        "factorized_primitive_state",
    ):
        Path(getattr(args, name)).write_bytes(name.encode("utf-8"))
    gate = _source_gate()
    monkeypatch.setattr(
        target_builder, "validate_source_pilot_chain", lambda *a, **k: gate
    )
    result = target_builder.build(args)
    authority = json.loads(Path(args.output_authority).read_text(encoding="utf-8"))
    assert result["authority"] == file_record(args.output_authority)
    assert authority["source_pilot_result"] == gate["source_result"]
    assert authority["target_inputs"]["v21_checkpoint"] == gate["checkpoint"]
    assert (
        authority["target_inputs"]["v21_normalization"]
        == gate["normalization_authority"]
    )
    assert authority["implementation"] == file_record(
        target_builder.TARGET_IMPLEMENTATION_PATH
    )
    assert authority["preregistration"] == file_record(
        target_builder.TARGET_PREREGISTRATION_PATH
    )


def test_query_builder_auto_binds_descriptor_positive_and_promoted_negative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, "query")
    Path(args.target_descriptor).write_bytes(b"descriptor")
    Path(args.positive_text_cache).write_bytes(b"positive")
    gate = _source_gate()
    descriptor = {
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:geometry-checkpoint-sha256:" + "9" * 64,
        "target_execution_authority": _record("target_execution.json"),
    }
    monkeypatch.setattr(
        query_builder, "validate_source_pilot_chain", lambda *a, **k: gate
    )
    monkeypatch.setattr(
        query_builder,
        "load_torch_mapping",
        lambda *a, **k: (descriptor, "d" * 64, Path(args.target_descriptor)),
    )
    monkeypatch.setattr(
        query_builder.target_formal,
        "validate_target_descriptor_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        query_builder.target_formal,
        "validate_target_execution_authority",
        lambda *a, **k: {
            "source_pilot_result": gate["source_result"],
            "target_inputs": {
                "v21_checkpoint": gate["checkpoint"],
                "v21_normalization": gate["normalization_authority"],
            },
        },
    )
    promoted_negative = _record("negative.pt")
    monkeypatch.setattr(
        query_builder, "_promoted_negative", lambda source: promoted_negative
    )
    result = query_builder.build(args)
    authority = json.loads(Path(args.output_authority).read_text(encoding="utf-8"))
    assert result["authority"] == file_record(args.output_authority)
    assert authority["target_descriptor"] == file_record(args.target_descriptor)
    assert authority["positive_text_cache"] == file_record(args.positive_text_cache)
    assert authority["canonical_negative_bank"] == promoted_negative
    assert authority["implementation"] == file_record(query_builder.IMPLEMENTATION_PATH)
    assert authority["preregistration"] == file_record(
        query_builder.PREREGISTRATION_PATH
    )


def test_external_builder_auto_binds_nested_authorities_renderer_and_preregistration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, "external")
    for name in (
        "query_relevance_execution_authority",
        "query_relevance_authority",
        "comembership_feature_authority",
        "comembership_inference_authority",
        "renderer_geometry_checkpoint",
    ):
        Path(getattr(args, name)).write_bytes(name.encode("utf-8"))
    gate = _source_gate()
    query_execution_record = file_record(args.query_relevance_execution_authority)
    descriptor = {
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:geometry-checkpoint-sha256:" + "9" * 64,
    }
    query_execution = {
        "source_pilot_result": gate["source_result"],
        "verified_source_gate": gate,
        "verified_descriptor": descriptor,
        "canonical_negative_bank": _record("negative.pt"),
        "target_descriptor": _record("descriptor.pt"),
        "positive_text_cache": _record("positive.pt"),
    }
    monkeypatch.setattr(
        external_builder, "validate_source_pilot_chain", lambda *a, **k: gate
    )
    monkeypatch.setattr(
        external_builder,
        "validate_query_execution_authority",
        lambda *a, **k: query_execution,
    )
    monkeypatch.setattr(
        external_builder,
        "load_torch_mapping",
        lambda *a, **k: ({}, "r" * 64, Path(args.query_relevance_authority)),
    )
    monkeypatch.setattr(
        external_builder,
        "validate_query_relevance_authority",
        lambda value: {"query_execution_authority": query_execution_record},
    )
    result = external_builder.build(args)
    authority = json.loads(Path(args.output_authority).read_text(encoding="utf-8"))
    assert result["authority"] == file_record(args.output_authority)
    assert authority["query_relevance_execution_authority"] == query_execution_record
    assert authority["renderer_geometry_checkpoint"] == file_record(
        args.renderer_geometry_checkpoint
    )
    assert authority["implementation"] == file_record(
        Path(external_builder.external.__file__).resolve()
    )
    assert authority["preregistration"] == file_record(
        external_builder.external.PREREGISTRATION
    )
