from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import radio_gs.scripts.aggregate_nvos_forward_beta_full8_nonexact as base
import radio_gs.scripts.aggregate_nvos_forward_beta_v2_full8_nonexact as v2
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    STRICT_TASKS,
    build_authority,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _reliability_fixture(tmp_path: Path) -> tuple[dict, Path]:
    cache_scenes: dict[str, object] = {}
    sources: dict[str, object] = {}
    for scene in STRICT_TASKS:
        cache_path = tmp_path / "cache" / scene / "canonical_reliability.pt"
        report_path = tmp_path / "cache" / scene / "canonical_reliability.pt.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(f"cache-{scene}".encode("utf-8"))
        _write_json(report_path, {"scene": scene})
        cache_record = _record(cache_path)
        report_record = _record(report_path)
        cache_scenes[scene] = {
            "reliability_cache": cache_record,
            "build_report": report_record,
        }
        sources[scene] = {
            "field_checkpoint.pth": {"unrelated": "v2 source closure may coexist"},
            base.RELIABILITY_SOURCE_KEY: {
                **cache_record,
                "metadata_path": report_record["path"],
                "metadata_sha256": report_record["sha256"],
            },
        }
    reliability_manifest = {
        "ordered_scenes": list(STRICT_TASKS),
        "scenes": cache_scenes,
    }
    reliability_manifest_path = tmp_path / "reliability_manifest.json"
    _write_json(reliability_manifest_path, reliability_manifest)
    run_manifest_projection = {
        "reliability_cache_manifest": _record(reliability_manifest_path),
        "source_artifacts": sources,
    }
    return run_manifest_projection, reliability_manifest_path


def _method_manifest(*, candidate: str, mode: str) -> dict:
    method = {
        "support_mode": "canonical_support",
        "final_readout": "propagated",
        "selection_applied_to_main_output": False,
        "registered_forward_unary": {
            "mode": mode,
            "status": base.ELIGIBILITY,
            "strict_unseen_eligible": False,
            "selection_applied_to_main_output": False,
            "required_final_readout": "propagated",
            "scoring_adapter": deepcopy(base.EXPECTED_SCORING_CONTRACT),
        },
    }
    method_sha256 = base.canonical_json_sha256(method)
    authority = build_authority(
        candidate_method_sha256=method_sha256,
        scoring_contract=base.EXPECTED_SCORING_CONTRACT,
        repo_root=Path(__file__).resolve().parents[1],
    )
    return {
        "candidate": candidate,
        "eligibility": base.ELIGIBILITY,
        "scenes": list(STRICT_TASKS),
        "method_contract": method,
        "registered_forward_protocol_authority": authority,
        "registered_forward_protocol_authority_sha256": (
            base.canonical_json_sha256(authority)
        ),
    }


def test_v2_reliability_manifest_binds_all_pt_and_json_records(tmp_path: Path) -> None:
    run_manifest, reliability_manifest_path = _reliability_fixture(tmp_path)

    binding = base._validate_reliability_bindings(
        run_manifest,
        reliability_manifest_validator=lambda path: json.loads(
            Path(path).read_text(encoding="utf-8")
        ),
    )

    assert binding["manifest"] == _record(reliability_manifest_path)
    assert binding["logical_source_key"] == base.RELIABILITY_SOURCE_KEY
    assert list(binding["scene_bindings"]) == list(STRICT_TASKS)
    assert binding["all_scene_cache_and_metadata_records_match_authority"] is True


@pytest.mark.parametrize("field", ["sha256", "metadata_sha256"])
def test_v2_reliability_logical_source_drift_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    run_manifest, _reliability_manifest_path = _reliability_fixture(tmp_path)
    run_manifest["source_artifacts"]["fern"][base.RELIABILITY_SOURCE_KEY][field] = (
        "f" * 64
    )

    with pytest.raises(
        base.ForwardBetaAggregationError,
        match="fern: v2 reliability logical source differs",
    ):
        base._validate_reliability_bindings(
            run_manifest,
            reliability_manifest_validator=lambda path: json.loads(
                Path(path).read_text(encoding="utf-8")
            ),
        )


def test_v2_manifest_profile_rejects_v1_candidate_and_forward_mode() -> None:
    v1_candidate = _method_manifest(
        candidate="nvos-forward-beta-coverage-v1",
        mode=v2.V2_FORWARD_MODE,
    )
    with pytest.raises(base.ForwardBetaAggregationError, match="candidate differs"):
        base._validate_manifest(
            v1_candidate,
            expected_candidate=v2.V2_CANDIDATE_ID,
            expected_forward_mode=v2.V2_FORWARD_MODE,
        )

    v1_mode = _method_manifest(
        candidate=v2.V2_CANDIDATE_ID,
        mode="beta_coverage_v1",
    )
    with pytest.raises(base.ForwardBetaAggregationError, match="mode differs"):
        base._validate_manifest(
            v1_mode,
            expected_candidate=v2.V2_CANDIDATE_ID,
            expected_forward_mode=v2.V2_FORWARD_MODE,
        )


def test_v2_wrapper_supplies_closed_profile_and_dedicated_receipt_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_aggregate(**kwargs):
        captured.update(kwargs)
        return {"artifact_type": kwargs["artifact_type"]}

    monkeypatch.setattr(v2, "aggregate_forward_beta_full8", fake_aggregate)
    summary = v2.aggregate_forward_beta_v2_full8(
        run_manifest_path=tmp_path / "run.json",
        result_root=tmp_path / "results",
        receipt_root=tmp_path / "receipts",
    )

    assert captured["expected_candidate"] == v2.V2_CANDIDATE_ID
    assert captured["expected_forward_mode"] == v2.V2_FORWARD_MODE
    assert captured["require_reliability_bindings"] is True
    assert captured["receipt_validator"] is v2.validate_scene_receipt
    assert captured["artifact_type"] == v2.V2_ARTIFACT_TYPE
    assert summary["artifact_type"] == v2.V2_ARTIFACT_TYPE


def test_v2_profile_requires_explicit_v1_reuse_prohibition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _method_manifest(
        candidate=v2.V2_CANDIDATE_ID,
        mode=v2.V2_FORWARD_MODE,
    )
    monkeypatch.setattr(base, "_validate_reliability_bindings", lambda value: {})

    with pytest.raises(base.ForwardBetaAggregationError, match="forbid v1"):
        base._validate_manifest(
            manifest,
            expected_candidate=v2.V2_CANDIDATE_ID,
            expected_forward_mode=v2.V2_FORWARD_MODE,
            require_reliability_bindings=True,
        )
