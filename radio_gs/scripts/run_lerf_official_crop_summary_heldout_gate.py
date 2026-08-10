#!/usr/bin/env python3
"""Plan or execute the preregistered official crop-summary heldout audit.

The wrapper deliberately has no frame-selection fallback.  In particular,
Ramen is always evaluated on the four source-heldout frames 2,45,87,130; the
historical MPR ``excluded_frame_ids`` field is never consulted because it also
contains target frames.  Planning is CPU-only and does not open tensor bytes.
Execution must be explicitly requested and calls the existing sibling audit,
not a benchmark evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Mapping

import torch

from radio_gs.scripts.seal_lerf_official_crop_summary_bundle import (
    RAMEN_SOURCE_HELDOUT_FRAME_IDS,
    SCHEMA as RESEAL_SCHEMA,
    file_sha256,
    load_preregistration,
    validate_preregistered_implementation,
    validate_scene_contract,
)


SCHEMA = "radio_gs.lerf_official_crop_summary_heldout_gate_authority.v1"
SCHEMA_VERSION = 1
AUDIT_MODULE = "radio_gs.scripts.audit_mpr_view_consistency"


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if not record["path"] or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError(f"{label} is invalid")
    return record


def _verified_record(
    value: object,
    *,
    label: str,
    hash_bytes: bool,
) -> dict[str, str]:
    record = _record(value, label=label)
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if hash_bytes:
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"{label} SHA-256 differs")
    return {"path": str(path), "sha256": record["sha256"]}


def _load_seal(
    path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, object], dict[str, str]]:
    source = Path(path).expanduser().resolve()
    expected = str(expected_sha256)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("reseal manifest requires a lowercase trusted SHA-256")
    actual = file_sha256(source)
    if actual != expected:
        raise ValueError("reseal manifest SHA-256 differs")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != RESEAL_SCHEMA:
        raise ValueError("official crop-summary reseal schema differs")
    if value.get("mode") != "content_addressed_immutable_reseal":
        raise ValueError("heldout execution requires a completed content reseal")
    if value.get("tensor_content_hashes_computed") is not True:
        raise ValueError("heldout execution requires tensor content hashes")
    return value, {"path": str(source), "sha256": actual}


def build_gate_authority(
    preregistration: Mapping[str, object],
    preregistration_record: Mapping[str, str],
    *,
    scene: str,
    reseal: Mapping[str, object],
    reseal_record: Mapping[str, str],
    result_output: str | Path,
    device: str,
    repo_root: str | Path,
    verify_large_inputs: bool,
) -> dict[str, object]:
    wrapper_record = validate_preregistered_implementation(
        preregistration, "gate_wrapper"
    )
    contract, selected, heldout, forbidden, authority_record = validate_scene_contract(
        preregistration, scene
    )
    if scene == "ramen" and tuple(heldout) != RAMEN_SOURCE_HELDOUT_FRAME_IDS:
        raise ValueError("Ramen source-heldout frame contract differs")
    if reseal.get("scene") != scene:
        raise ValueError("reseal scene differs")
    if reseal.get("preregistration") != dict(preregistration_record):
        raise ValueError("reseal preregistration authority differs")
    if reseal.get("selected_view_authority") != authority_record:
        raise ValueError("reseal selected-view authority differs")
    if reseal.get("selected_frame_ids") != selected:
        raise ValueError("reseal selected-frame IDs differ")
    if reseal.get("source_heldout_frame_ids") != heldout:
        raise ValueError("reseal source-heldout frame IDs differ")
    if reseal.get("forbidden_target_frame_ids") != forbidden:
        raise ValueError("reseal forbidden target-frame IDs differ")
    frame_records = reseal.get("frame_records")
    if not isinstance(frame_records, list):
        raise ValueError("reseal frame records are missing")
    sealed_frame_ids = sorted(
        int(record["frame_id"])
        for record in frame_records
        if isinstance(record, dict) and "frame_id" in record
    )
    if sealed_frame_ids != sorted(selected + heldout):
        raise ValueError("reseal frame records differ from selected+heldout source frames")
    if set(sealed_frame_ids).intersection(forbidden):
        raise ValueError("reseal contains forbidden target frames")

    config = _verified_record(
        contract.get("config"), label="heldout config", hash_bytes=True
    )
    geometry = _verified_record(
        contract.get("geometry_checkpoint"),
        label="geometry checkpoint",
        hash_bytes=verify_large_inputs,
    )
    mpr = _verified_record(
        contract.get("genuine_mpr"),
        label="genuine official crop-summary MPR",
        hash_bytes=verify_large_inputs,
    )
    audit = _verified_record(
        preregistration.get("heldout_audit_implementation"),
        label="heldout audit implementation",
        hash_bytes=True,
    )
    root = Path(repo_root).expanduser().resolve()
    launcher = (root / "radio_gs/scripts/run_repo_python.sh").resolve()
    if not launcher.is_file():
        raise ValueError(f"repository Python launcher is missing: {launcher}")
    result = Path(result_output).expanduser().resolve()
    if result.suffix != ".pt":
        raise ValueError("heldout gate result output must have a .pt suffix")
    if not str(device).strip():
        raise ValueError("heldout gate device must be explicit")

    frame_argument = ",".join(str(frame_id) for frame_id in heldout)
    if not frame_argument:
        raise ValueError("heldout wrapper refuses an empty frame selection")
    command = [
        str(launcher),
        "-m",
        AUDIT_MODULE,
        "heldout",
        "--config",
        config["path"],
        "--geometry-checkpoint",
        geometry["path"],
        "--mpr-cache",
        mpr["path"],
        "--output",
        str(result),
        "--device",
        str(device),
        "--frame-ids",
        frame_argument,
    ]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "preregistration": dict(preregistration_record),
        "gate_wrapper_implementation": wrapper_record,
        "reseal": dict(reseal_record),
        "selected_view_authority": authority_record,
        "inputs": {
            "config": config,
            "geometry_checkpoint": geometry,
            "genuine_mpr": mpr,
            "heldout_audit_implementation": audit,
        },
        "source_heldout_frame_ids": heldout,
        "forbidden_target_frame_ids": forbidden,
        "explicit_frame_argument": frame_argument,
        "command": command,
        "result_output": str(result),
        "protocol": {
            "source_only": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_metric_execution_authorized": False,
            "historical_mpr_excluded_frame_fallback_allowed": False,
            "performance_role": "diagnostic_readiness_gate_not_target_promotion",
            "required_result_checks": [
                "exact_preregistered_source_heldout_frame_ids",
                "every_frame_has_positive_visible_pixels",
                "every_frame_has_positive_registered_rows",
                "every_frame_mean_pixel_cosine_is_finite",
            ],
        },
        "large_input_hashes_verified": bool(verify_large_inputs),
    }


def validate_gate_result(
    authority: Mapping[str, object], result_path: str | Path
) -> dict[str, object]:
    source = Path(result_path).expanduser().resolve()
    if source != Path(str(authority["result_output"])).expanduser().resolve():
        raise ValueError("heldout gate result path differs from sealed authority")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("gate validation requires weights_only=True support") from exc
    if not isinstance(payload, dict):
        raise ValueError("heldout gate result must contain an object")
    expected = list(authority["source_heldout_frame_ids"])
    actual = [int(value) for value in payload.get("selected_frame_ids", [])]
    if actual != expected:
        raise ValueError("heldout result frame IDs differ from preregistration")
    per_view = payload.get("per_view")
    if not isinstance(per_view, list) or len(per_view) != len(expected):
        raise ValueError("heldout result per-view records are incomplete")
    if [int(record.get("frame_id", -1)) for record in per_view] != expected:
        raise ValueError("heldout result per-view order differs")
    for record in per_view:
        if int(record.get("visible_pixels", 0)) <= 0:
            raise ValueError("heldout result has no visible pixels")
        if int(record.get("registered_rows", 0)) <= 0:
            raise ValueError("heldout result has no registered rows")
        cosine = float(record.get("mean_pixel_cosine", float("nan")))
        if not torch.isfinite(torch.tensor(cosine)).item():
            raise ValueError("heldout result cosine is non-finite")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("heldout result protocol is missing")
    if protocol.get("benchmark_masks_opened") is not False:
        raise ValueError("heldout result opened benchmark masks")
    if protocol.get("text_queries_opened") is not False:
        raise ValueError("heldout result opened text queries")
    return {
        "passed": True,
        "scene": authority["scene"],
        "selected_frame_ids": actual,
        "num_views": len(actual),
        "mean_pixel_cosine": sum(
            float(record["mean_pixel_cosine"]) * int(record["visible_pixels"])
            for record in per_view
        )
        / sum(int(record["visible_pixels"]) for record in per_view),
        "role": "diagnostic_readiness_gate_not_target_promotion",
    }


def execute_gate(authority: Mapping[str, object]) -> dict[str, object]:
    result = Path(str(authority["result_output"]))
    if result.exists() or result.with_suffix(result.suffix + ".json").exists():
        raise FileExistsError(f"refusing to replace heldout gate result: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(list(authority["command"]), check=True)
    return validate_gate_result(authority, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--scene", required=True, choices=("ramen", "teatime"))
    parser.add_argument("--reseal", required=True)
    parser.add_argument("--expected-reseal-sha256", required=True)
    parser.add_argument("--result-output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--authority-output", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the GPU heldout renderer after sealing the execution authority.",
    )
    parser.add_argument(
        "--skip-large-input-hash-validation",
        action="store_true",
        help="Planning only: defer MPR/checkpoint byte hashing. Forbidden with --execute.",
    )
    args = parser.parse_args()
    if args.execute and args.skip_large_input_hash_validation:
        parser.error("--execute requires full large-input hash validation")
    preregistration, preregistration_record = load_preregistration(
        args.preregistration, args.expected_preregistration_sha256
    )
    reseal, reseal_record = _load_seal(args.reseal, args.expected_reseal_sha256)
    repo_root = Path(__file__).resolve().parents[2]
    authority = build_gate_authority(
        preregistration,
        preregistration_record,
        scene=args.scene,
        reseal=reseal,
        reseal_record=reseal_record,
        result_output=args.result_output,
        device=args.device,
        repo_root=repo_root,
        verify_large_inputs=not args.skip_large_input_hash_validation,
    )
    authority_path = Path(args.authority_output).expanduser().resolve()
    if authority_path.exists():
        raise FileExistsError(f"refusing to replace gate authority: {authority_path}")
    from radio_gs.scripts.seal_lerf_official_crop_summary_bundle import _atomic_json

    _atomic_json(authority_path, authority)
    summary: dict[str, object] = {
        "authority_output": str(authority_path),
        "scene": args.scene,
        "source_heldout_frame_ids": authority["source_heldout_frame_ids"],
        "executed": False,
    }
    if args.execute:
        summary["gate"] = execute_gate(authority)
        summary["executed"] = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
