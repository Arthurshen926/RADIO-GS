#!/usr/bin/env python3
"""Issue the benchmark-opening receipt without importing ScanNet code/data.

This process validates the already frozen Surface and text-response promotion
chain.  It intentionally has no dependency on ScanNet constants, evaluators,
labels, meshes, fields, graphs, or text caches.  A separate benchmark process
must validate the returned receipt digest before it may open those assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radio_gs.scripts import finalize_surface_text_response_promotion as text_finalizer
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "promoted_surface_text_benchmark_authority_receipt"
REQUIRED_SEEDS = (0, 1, 2)
FORBIDDEN_MODULE_PREFIXES = (
    "radio_gs.scannet_constants",
    "radio_gs.scripts.eval_scannet",
)
AUTHORITY_IMPLEMENTATION_SOURCES = (
    "radio_gs/scripts/finalize_scannet_surface_text_authority_gate.py",
    "radio_gs/scripts/finalize_surface_text_response_promotion.py",
    "radio_gs/scripts/finalize_surface_region_query_free_promotion.py",
    "radio_gs/scripts/eval_text_response_fidelity_gate.py",
    "radio_gs/evaluation/text_response_fidelity.py",
    "radio_gs/utils/immutable_artifacts.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path, label: str) -> dict[str, Any]:
    payload, _, _ = load_json_object(path, label=label)
    return payload


def _record_paths(records: object, label: str) -> list[Path]:
    _require(isinstance(records, list), f"{label} bindings must be a list")
    return [
        validate_file_record(record, label=f"{label}[{index}]")
        for index, record in enumerate(records)
    ]


def validate_text_response_authority(
    *,
    surface_manifest: Path,
    surface_completion: Path,
    audit_manifest: Path,
    audit_completion: Path,
) -> dict[str, Any]:
    surface_manifest = Path(surface_manifest)
    surface_completion = Path(surface_completion)
    audit_manifest = Path(audit_manifest)
    audit_completion = Path(audit_completion)
    audit = _json(audit_manifest, "text-response audit")
    completion = _json(audit_completion, "text-response audit completion")
    audit_path = validate_file_record(
        file_record(audit_manifest), label="text-response audit"
    )
    completion_path = validate_file_record(
        file_record(audit_completion), label="text-response audit completion"
    )
    _require(
        audit.get("artifact_type") == text_finalizer.STAGE_ARTIFACT_TYPE
        and audit.get("stage") == "audit"
        and audit.get("decision") == "promote_confirmed"
        and audit.get("main_result_eligible") is True
        and audit.get("benchmark_vocabulary_opened") is False
        and audit.get("benchmark_gate_status") == "closed",
        "text-response audit is not accepted for a main-result benchmark",
    )
    _require(
        completion.get("artifact_type") == text_finalizer.COMPLETION_ARTIFACT_TYPE
        and completion.get("stage") == "audit"
        and completion.get("status") == "complete"
        and completion.get("decision") == "promote_confirmed"
        and completion.get("main_result_eligible") is True
        and completion.get("benchmark_vocabulary_opened") is False
        and Path(str(completion.get("stage_manifest", ""))).resolve() == audit_path
        and completion.get("stage_manifest_sha256") == sha256_file(audit_path),
        "text-response audit completion is not accepted",
    )
    _require(
        audit.get("required_seeds") == list(REQUIRED_SEEDS),
        "accepted audit does not freeze seeds 0/1/2",
    )
    plan_record = audit.get("plan")
    plan_path = validate_file_record(plan_record, label="text-response plan")
    plan = text_finalizer.validate_plan(plan_path)
    _require(plan["sha256"] == plan_record["sha256"], "audit plan SHA differs")
    plan_payload = plan["payload"]
    surface_binding = plan_payload.get("surface_promotion", {})
    surface_manifest_path = validate_file_record(
        file_record(surface_manifest), label="Surface promotion manifest"
    )
    surface_completion_path = validate_file_record(
        file_record(surface_completion), label="Surface promotion completion"
    )
    _require(
        Path(str(surface_binding.get("manifest", ""))).resolve()
        == surface_manifest_path
        and surface_binding.get("manifest_sha256")
        == sha256_file(surface_manifest_path)
        and Path(str(surface_binding.get("completion", ""))).resolve()
        == surface_completion_path
        and surface_binding.get("completion_sha256")
        == sha256_file(surface_completion_path),
        "accepted audit belongs to another Surface promotion",
    )

    bindings = audit.get("bindings")
    _require(isinstance(bindings, Mapping), "accepted audit lacks evidence bindings")
    descriptor_bindings = bindings.get("descriptors")
    report_bindings = bindings.get("reports")
    _require(
        isinstance(descriptor_bindings, Mapping)
        and isinstance(report_bindings, Mapping),
        "accepted audit descriptor/report bindings are incomplete",
    )
    dev = audit.get("dev_dependency")
    _require(isinstance(dev, Mapping), "accepted audit lacks frozen dev dependency")
    result = text_finalizer.finalize_stage(
        stage="audit",
        plan_path=plan_path,
        control_descriptors=_record_paths(
            descriptor_bindings.get("control"), "audit control descriptors"
        ),
        candidate_descriptors=_record_paths(
            descriptor_bindings.get("candidate"), "audit candidate descriptors"
        ),
        control_reports=_record_paths(
            report_bindings.get("control"), "audit control reports"
        ),
        candidate_reports=_record_paths(
            report_bindings.get("candidate"), "audit candidate reports"
        ),
        gate_path=validate_file_record(bindings.get("gate"), label="audit gate"),
        text_bank_path=validate_file_record(
            bindings.get("text_bank"), label="audit text bank"
        ),
        text_bank_manifest_path=validate_file_record(
            bindings.get("text_bank_manifest"), label="audit text bank manifest"
        ),
        output=audit_path,
        completion=completion_path,
        dev_manifest=validate_file_record(
            {
                "path": dev.get("manifest"),
                "sha256": dev.get("manifest_sha256"),
            },
            label="frozen dev manifest",
        ),
        dev_completion=validate_file_record(
            {
                "path": dev.get("completion"),
                "sha256": dev.get("completion_sha256"),
            },
            label="frozen dev completion",
        ),
    )
    _require(
        result.get("decision") == "promote_confirmed"
        and result.get("main_result_eligible") is True,
        "strict audit recomputation did not confirm promotion",
    )

    response_rows = plan_payload.get("candidate")
    _require(
        isinstance(response_rows, list) and len(response_rows) == 3,
        "accepted plan does not bind three response checkpoints",
    )
    readouts: dict[int, dict[str, Any]] = {}
    for row in response_rows:
        _require(isinstance(row, Mapping), "response checkpoint binding is invalid")
        seed = row.get("seed")
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed in REQUIRED_SEEDS
            and seed not in readouts,
            "response checkpoints do not exactly cover seeds 0/1/2",
        )
        checkpoint = file_record(row.get("checkpoint"))
        sidecar = file_record(row.get("sidecar"))
        _require(
            checkpoint["sha256"] == row.get("checkpoint_sha256")
            and sidecar["sha256"] == row.get("sidecar_sha256"),
            f"response seed {seed} checkpoint/sidecar authority differs",
        )
        readouts[int(seed)] = {
            "seed": int(seed),
            "checkpoint": checkpoint,
            "sidecar": sidecar,
        }
    _require(set(readouts) == set(REQUIRED_SEEDS), "response seed set is incomplete")
    return {
        "selected_candidate": str(plan_payload["selected_candidate"]),
        "method_id": str(plan_payload["candidate_method_id"]),
        "surface_manifest": file_record(surface_manifest_path),
        "surface_completion": file_record(surface_completion_path),
        "text_audit_manifest": file_record(audit_path),
        "text_audit_completion": file_record(completion_path),
        "text_plan": file_record(plan_path),
        "readouts": [readouts[seed] for seed in REQUIRED_SEEDS],
    }


def _forbidden_loaded_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )


def issue_receipt(
    *,
    surface_manifest: Path,
    surface_completion: Path,
    audit_manifest: Path,
    audit_completion: Path,
    output: Path,
) -> dict[str, Any]:
    authority = validate_text_response_authority(
        surface_manifest=surface_manifest,
        surface_completion=surface_completion,
        audit_manifest=audit_manifest,
        audit_completion=audit_completion,
    )
    forbidden = _forbidden_loaded_modules()
    _require(
        not forbidden,
        f"authority gate imported forbidden benchmark modules: {forbidden}",
    )
    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "accepted_before_benchmark_open",
        "required_seeds": list(REQUIRED_SEEDS),
        "benchmark_data_opened": False,
        "forbidden_benchmark_modules_loaded": forbidden,
        "promotion_authority": authority,
        "authority_inputs": {
            "surface_manifest": authority["surface_manifest"],
            "surface_completion": authority["surface_completion"],
            "text_audit_manifest": authority["text_audit_manifest"],
            "text_audit_completion": authority["text_audit_completion"],
        },
        "implementation_sources": [
            {"relative_path": relative, **file_record(repo_root / relative)}
            for relative in AUTHORITY_IMPLEMENTATION_SOURCES
        ],
    }
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        observed, _, receipt = load_json_object(
            output,
            label="existing benchmark authority receipt",
        )
        _require(observed == payload, "existing authority receipt differs")
    else:
        receipt = write_frozen_json(output, payload)
    return {
        "receipt": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "status": payload["status"],
        "benchmark_data_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-manifest", required=True, type=Path)
    parser.add_argument("--surface-completion", required=True, type=Path)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--audit-completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            issue_receipt(
                surface_manifest=args.surface_manifest,
                surface_completion=args.surface_completion,
                audit_manifest=args.audit_manifest,
                audit_completion=args.audit_completion,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
