from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as core
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as v2,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2_lowmem as lowmem,
)
from radio_gs.scripts import (
    execute_lerf_source_only_global_ceiling_lowmem_lineage_compatibility as bridge,
)
from radio_gs.scripts import (
    select_lerf_source_only_global_reliability_ceiling as selector,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    write_frozen_json,
    write_torch_noclobber,
)


def _source(tmp_path: Path, scene: str, seed: int) -> bridge.SourceSpec:
    root = tmp_path / scene
    root.mkdir()
    input_names = {
        "base_descriptor", "responsibility_authority", "feature_manifest",
        "scene_config", "renderer_geometry_checkpoint", "official_radio_checkpoint",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
        "frozen_metric_config",
    }
    inputs = {}
    for name in sorted(input_names):
        path = root / f"{name}.bin"
        path.write_bytes(f"{scene}-{name}".encode())
        inputs[name] = file_record(path)
    teacher_path = root / "teacher.pt"
    authority_path = root / "execution.json"
    outputs = {
        "teacher_mean": str(teacher_path),
        "o1_positive": str(root / "o1_positive.pt"),
        "o1_negative": str(root / "o1_negative.pt"),
        "o2_positive": str(root / "o2_positive.pt"),
        "o2_negative": str(root / "o2_negative.pt"),
        "result": str(root / "result.json"),
    }
    authority = {
        "schema": v2.AUTHORITY_SCHEMA,
        "schema_version": v2.SCHEMA_VERSION,
        "status": "authorized_source_only_premetric_o1_o2_streaming",
        "scene_id": scene,
        "implementation": dict(lowmem.ENTRYPOINT_IMPLEMENTATION),
        "method_contract": lowmem.method_contract(),
        "method_contract_sha256": lowmem.METHOD_CONTRACT_SHA256,
        "feature_output_bundle_sha256": "a" * 64,
        "inputs": inputs,
        "outputs": outputs,
        "execution": {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "program_device": "cuda:0",
            "projection_batch_candidates": [128, 64],
            "pacing_seconds_per_projection_batch": 0.0,
            "thermal_poll_seconds": 300,
            "soft_pause_temperature_c": 0,
            "maximum_temperature_c": 88,
        },
        "query_free_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": core.access_audit(),
    }
    write_frozen_json(authority_path, authority)

    generator = torch.Generator().manual_seed(seed)
    descriptors = F.normalize(
        torch.randn(3, 4, 1536, generator=generator), dim=-1
    ).half()
    frame_ids = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    base = F.normalize(
        torch.randn(3, 3, 1536, generator=generator), dim=-1
    ).half()
    stats = lowmem.finalize_teacher_statistics_lowmem(
        descriptors, frame_ids, base, chunk_rows=2
    )
    agreement = stats[v2.VIEW_AGREEMENT_SCALAR]
    audit = stats[v2.LOO_AUDIT_FIELD]
    payload = {
        "schema": v2.MEAN_SCHEMA,
        "schema_version": v2.SCHEMA_VERSION,
        "scene_id": scene,
        "global_rows": torch.tensor([0, 1, 2]),
        "teacher_mean": stats["teacher_mean"],
        "teacher_valid": stats["teacher_valid"],
        "retained_view_count": stats["retained_view_count"],
        v2.VIEW_AGREEMENT_SCALAR: agreement,
        "producer": dict(lowmem.ENTRYPOINT_IMPLEMENTATION),
        "execution_authority": file_record(authority_path),
        "input_authority": {
            "base_descriptor": inputs["base_descriptor"],
            "responsibility_authority": inputs["responsibility_authority"],
            "feature_manifest": inputs["feature_manifest"],
            "official_radio_checkpoint": inputs["official_radio_checkpoint"],
        },
        "method_contract_sha256": lowmem.METHOD_CONTRACT_SHA256,
        "teacher_mean_sha256": core.tensor_sha256_typed(stats["teacher_mean"]),
        v2.VIEW_AGREEMENT_SHA256_FIELD: core.tensor_sha256_typed(agreement),
        v2.LOO_AUDIT_FIELD: audit,
        v2.LOO_AUDIT_SHA256_FIELD: canonical_json_sha256(audit),
        "access_audit": core.access_audit(),
    }
    lowmem.validate_teacher_payload_lowmem(payload)
    write_torch_noclobber(teacher_path, payload)
    record = file_record(teacher_path)
    return bridge.SourceSpec(scene, record["path"], record["sha256"])


def _prereg() -> dict[str, str]:
    return file_record(bridge.ORIGINAL_PREREGISTRATION_PATH)


def test_bridge_binds_frozen_lineage_and_exact_original_policy(tmp_path: Path) -> None:
    lineage = bridge.validate_local_lineage()
    assert lineage["original_selector_implementation"]["sha256"] == (
        bridge.ORIGINAL_SELECTOR_SHA256
    )
    assert lineage["lowmem_implementation"]["sha256"] == (
        bridge.LOWMEM_IMPLEMENTATION_SHA256
    )
    a = _source(tmp_path, "a", 4)
    b = _source(tmp_path, "b", 9)
    result = bridge.select_compatibility_candidate(
        [b, a], original_preregistration=_prereg()
    )
    assert result["schema"] == bridge.OUTPUT_SCHEMA
    assert result["source_scene_ids"] == ["a", "b"]
    assert result["source_count"] == 2
    eligible = [
        row["maximum_angle_radians"]
        for row in result["candidate_grid"] if row["eligible"]
    ]
    assert result["selection"]["global_maximum_angle_radians"] == max(
        eligible, default=0.15
    )
    assert result["created_after_source_results_for_lineage_compatibility_only"] is True
    assert result["selection_rule_preregistered_before_results"] is True
    assert result["metric_execution_authorized"] is False
    assert bridge.compatibility_contract()["original_selector_method_contract"][
        "selection"
    ] == "largest_eligible_angle_else_0.15"


def test_same_synthetic_audits_match_original_selector_field_for_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [_source(tmp_path, "a", 14), _source(tmp_path, "b", 19)]
    normalized = {
        spec.scene_id: bridge.load_lowmem_source(spec) for spec in specs
    }
    monkeypatch.setattr(
        bridge, "load_lowmem_source", lambda spec: deepcopy(normalized[spec.scene_id])
    )
    monkeypatch.setattr(
        selector, "load_source_summary", lambda spec: deepcopy(normalized[spec.scene_id])
    )
    original_specs = [
        selector.SourceSpec(
            spec.scene_id, spec.path, spec.sha256, "compact_source_summary_v1"
        )
        for spec in specs
    ]
    original = selector.select_global_ceiling(
        original_specs, preregistration=_prereg()
    )
    compatible = bridge.select_compatibility_candidate(
        list(reversed(specs)), original_preregistration=_prereg()
    )
    assert compatible["candidate_grid"] == original["candidate_grid"]
    assert compatible["selection"] == original["selection"]
    assert compatible["source_scene_ids"] == original["source_scene_ids"]


def test_payload_hash_and_preregistration_hash_fail_closed(tmp_path: Path) -> None:
    a = _source(tmp_path, "a", 1)
    b = _source(tmp_path, "b", 2)
    bad = bridge.SourceSpec(a.scene_id, a.path, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        bridge.select_compatibility_candidate(
            [bad, b], original_preregistration=_prereg()
        )
    prereg = _prereg()
    with pytest.raises(ValueError, match="SHA-256 differs"):
        bridge.select_compatibility_candidate(
            [a, b],
            original_preregistration={"path": prereg["path"], "sha256": "f" * 64},
        )


def test_tampered_lowmem_authority_lineage_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path, "a", 3)
    payload = torch.load(source.path, map_location="cpu", weights_only=True)
    authority_path = Path(payload["execution_authority"]["path"])
    authority = __import__("json").loads(authority_path.read_text())
    authority["method_contract_sha256"] = "0" * 64
    forged_authority = tmp_path / "forged_authority.json"
    write_frozen_json(forged_authority, authority)
    payload["execution_authority"] = file_record(forged_authority)
    forged_payload = tmp_path / "forged_lineage.pt"
    write_torch_noclobber(forged_payload, payload)
    record = file_record(forged_payload)
    with pytest.raises(ValueError, match="execution authority contract differs"):
        bridge.load_lowmem_source(
            bridge.SourceSpec("a", record["path"], record["sha256"])
        )


def test_tampered_source_audit_fails_even_with_recomputed_hash(tmp_path: Path) -> None:
    source = _source(tmp_path, "a", 8)
    payload = torch.load(source.path, map_location="cpu", weights_only=True)
    audit = deepcopy(payload[v2.LOO_AUDIT_FIELD])
    audit["target_candidate_authorized"] = True
    payload[v2.LOO_AUDIT_FIELD] = audit
    payload[v2.LOO_AUDIT_SHA256_FIELD] = canonical_json_sha256(audit)
    forged = tmp_path / "forged_audit.pt"
    write_torch_noclobber(forged, payload)
    record = file_record(forged)
    with pytest.raises(ValueError, match="LOO ceiling audit contract"):
        bridge.load_lowmem_source(
            bridge.SourceSpec("a", record["path"], record["sha256"])
        )


def test_duplicate_or_missing_scene_fails_closed(tmp_path: Path) -> None:
    a = _source(tmp_path, "a", 5)
    with pytest.raises(ValueError, match="at least two"):
        bridge.select_compatibility_candidate([a], original_preregistration=_prereg())
    with pytest.raises(ValueError, match="distinct"):
        bridge.select_compatibility_candidate([a, a], original_preregistration=_prereg())
