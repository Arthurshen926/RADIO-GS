from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import yaml

import radio_gs.scripts.aggregate_nvos_forward_beta_full8_nonexact as aggregator
import radio_gs.scripts.bind_nvos_forward_beta_v2_protocol_authority as binder
import radio_gs.scripts.nvos_forward_beta_scene_authority as scene_authority
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    canonical_json_sha256,
)
from radio_gs.scripts.nvos_forward_beta_scene_authority import (
    V1_PROFILE,
    V2_PROFILE,
    _validate_sized_file_record,
)
from radio_gs.scripts.stage_nvos_forward_beta_v2_snapshot import (
    AGGREGATOR_RELATIVE,
    AUTHORITY_RECEIPT_RELATIVE,
    CANDIDATE_RELATIVE,
    RELIABILITY_MANIFEST_RELATIVE,
    RUNNER_RELATIVE,
    SCENE_AUTHORITY_RELATIVE,
    STAGING_MANIFEST_RELATIVE,
    StagingError,
    _logical_reliability_record,
    validate_candidate_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / CANDIDATE_RELATIVE
RUNNER = ROOT / RUNNER_RELATIVE


def _candidate_payload() -> dict:
    return yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_namespace_rederives_exact_v2_contract() -> None:
    payload = _candidate_payload()
    method, digest = validate_candidate_payload(payload)
    bound_payload, bound_method, bound_digest, _ = binder.load_candidate_contract(
        CANDIDATE
    )

    assert bound_payload == payload
    assert bound_method == method
    assert bound_digest == digest == canonical_json_sha256(method)
    assert method["registered_forward_unary"]["mode"] == binder.FORWARD_MODE
    assert method["canonical_reliability_cache"] == binder.RELIABILITY_MARKER
    assert method["registered_forward_unary"][
        "semantic_precision_is_primary_for_nonanchors"
    ] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["method_namespace"].__setitem__(
                "registered_forward_unary", "beta_coverage_v1"
            ),
            "namespace differs",
        ),
        (
            lambda value: value.__setitem__(
                "v1_result_or_receipt_reuse_permitted", True
            ),
            "forbid v1",
        ),
        (
            lambda value: value["eligibility"].__setitem__(
                "main_result_eligible", True
            ),
            "must be false",
        ),
    ],
)
def test_candidate_drift_fails_closed(mutate, match: str) -> None:
    payload = deepcopy(_candidate_payload())
    mutate(payload)
    with pytest.raises(StagingError, match=match):
        validate_candidate_payload(payload)


def test_authority_binder_requires_query_independent_reliability_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "reliability.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    safe = {
        "artifact_type": binder.RELIABILITY_ARTIFACT_TYPE,
        "ordered_scenes": list(binder.ORDERED_SCENES),
        "safety_contract": {
            "query_independent": True,
            "uses_query": False,
            "uses_text": False,
            "uses_target_labels": False,
            "uses_target_masks": False,
            "uses_metric_feedback": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }
    monkeypatch.setattr(binder, "validate_manifest", lambda path: safe)
    payload, record = binder.validate_reliability_binding(manifest_path)
    assert payload == safe
    assert record["path"] == str(manifest_path.absolute())
    assert record["bytes"] == len(manifest_path.read_bytes())

    unsafe = deepcopy(safe)
    unsafe["safety_contract"]["uses_query"] = True
    monkeypatch.setattr(binder, "validate_manifest", lambda path: unsafe)
    with pytest.raises(binder.BetaV2AuthorityError, match="query-independent"):
        binder.validate_reliability_binding(manifest_path)


def test_logical_scene_source_binds_exact_pt_and_json_projection() -> None:
    payload = json.loads(
        (ROOT / RELIABILITY_MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )
    source = _logical_reliability_record(payload, scene="fern")
    row = payload["scenes"]["fern"]
    assert source == {
        **row["reliability_cache"],
        "metadata_path": row["build_report"]["path"],
        "metadata_sha256": row["build_report"]["sha256"],
    }
    assert set(source) == {
        "path",
        "bytes",
        "sha256",
        "metadata_path",
        "metadata_sha256",
    }


def test_v2_execution_artifacts_are_disjoint_from_v1() -> None:
    assert V2_PROFILE.candidate_id == "nvos-forward-beta-balanced-residual-v2"
    assert V2_PROFILE.forward_mode == "beta_balanced_residual_v2"
    assert V2_PROFILE.receipt_artifact == "nvos-forward-beta-v2-scene-receipt-v1"
    assert V2_PROFILE.command_artifact != V1_PROFILE.command_artifact
    assert V2_PROFILE.postcheck_artifact != V1_PROFILE.postcheck_artifact
    assert V2_PROFILE.receipt_artifact != V1_PROFILE.receipt_artifact
    assert str(AUTHORITY_RECEIPT_RELATIVE).endswith(
        "nvos_forward_beta_balanced_residual_v2_protocol_authority.json"
    )
    assert "balanced_residual_v2" in str(STAGING_MANIFEST_RELATIVE)
    assert "v2" in str(SCENE_AUTHORITY_RELATIVE)
    assert "v2" in str(AGGREGATOR_RELATIVE)


def test_v2_runner_binds_mode_reliability_and_separate_output() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'export FORWARD_BETA_VARIANT="v2"' in text
    assert 'export CANDIDATE_ID="nvos-forward-beta-balanced-residual-v2"' in text
    assert 'export FORWARD_MODE="beta_balanced_residual_v2"' in text
    assert "stage_nvos_forward_beta_v2_snapshot.py" in text
    assert "nvos_forward_beta_v2_scene_authority.py" in text
    assert "aggregate_nvos_forward_beta_v2_full8_nonexact.py" in text
    assert "nvos_forward_beta_balanced_residual_v2_full8" in text
    driver = (ROOT / "radio_gs/scripts/run_nvos_forward_beta_coverage_v1_queue.sh").read_text(
        encoding="utf-8"
    )
    assert '--canonical-reliability-cache "$reliability"' in driver
    assert 'RUNNER_AUTHORITY_PATH' in driver


def test_full_reliability_validation_is_not_repeated_per_scene_or_aggregate() -> None:
    scene_source = inspect.getsource(scene_authority.validate_run_manifest)
    aggregate_source = inspect.getsource(aggregator._validate_reliability_bindings)
    assert "validate_reliability_manifest_payload" in scene_source
    assert "verify_files=False" in scene_source
    assert "validate_reliability_manifest(" not in scene_source
    assert "validate_manifest_payload(stable_manifest, verify_files=False)" in (
        aggregate_source
    )


def test_v2_sized_file_record_accepts_lexical_parent_alias_and_checks_bytes(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    artifact = real_root / "reliability.json"
    artifact.write_bytes(b'{"schema_version": 1}\n')
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    lexical_artifact = alias_root / artifact.name
    encoded = artifact.read_bytes()
    record = {
        "path": str(lexical_artifact.absolute()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }

    assert _validate_sized_file_record(record, label="test reliability") == artifact

    wrong_size = {**record, "bytes": len(encoded) + 1}
    with pytest.raises(ValueError, match="byte count differs"):
        _validate_sized_file_record(wrong_size, label="test reliability")


def test_v2_sized_file_record_rejects_sha_drift_and_final_symlink(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "reliability.json"
    artifact.write_bytes(b'{"schema_version": 1}\n')
    encoded = artifact.read_bytes()
    record = {
        "path": str(artifact),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }

    with pytest.raises(ValueError, match="SHA-256 differs"):
        _validate_sized_file_record(
            {**record, "sha256": "0" * 64},
            label="test reliability",
        )

    link = tmp_path / "reliability-link.json"
    link.symlink_to(artifact)
    with pytest.raises(ValueError, match="symlink"):
        _validate_sized_file_record(
            {**record, "path": str(link)},
            label="test reliability",
        )
