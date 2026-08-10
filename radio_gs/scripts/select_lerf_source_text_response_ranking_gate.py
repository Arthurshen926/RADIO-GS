#!/usr/bin/env python3
"""Audit assets or select a LERF source text-response/ranking candidate.

Inventory mode only inspects paths and small JSON authorities.  Gate mode
consumes compact per-scene summaries produced under the preregistered
source-heldout protocol; it never loads descriptor maps or runs a renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

from radio_gs.evaluation.lerf_source_text_response_ranking import (
    paired_source_gate,
    validate_scene_summary,
)


PREREGISTRATION_SCHEMA = (
    "radio_gs.lerf_source_text_response_ranking_preregistration.v1"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(
    path: str | Path, expected_sha256: str, *, label: str
) -> tuple[dict[str, object], dict[str, str]]:
    source = Path(path).expanduser().resolve()
    expected = str(expected_sha256)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label} requires a lowercase trusted SHA-256")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, {"path": str(source), "sha256": actual}


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if not record["path"] or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError(f"{label} differs")
    return record


def validate_preregistration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("source text-response preregistration must be an object")
    preregistration = dict(value)
    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("schema_version") != 1
        or preregistration.get("status")
        != "sealed_before_source_response_materialization_or_gate"
    ):
        raise ValueError("source text-response preregistration differs")
    if preregistration.get("target_metric_execution_authorized") is not False:
        raise ValueError("source text-response preregistration authorizes target metric")
    if preregistration.get("target_results_used_for_gate_design") is not False:
        raise ValueError("source text-response gate design is target contaminated")
    implementation = _record(
        preregistration.get("selector_implementation"),
        label="source text-response selector implementation",
    )
    if file_sha256(implementation["path"]) != implementation["sha256"]:
        raise ValueError("source text-response selector implementation changed")
    for key, label in (
        ("frame_evaluator_implementation", "frame evaluator"),
        ("shared_response_metric_implementation", "shared response metric"),
    ):
        record = _record(preregistration.get(key), label=label)
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"source text-response {label} implementation changed")
    scenes = preregistration.get("source_scenes")
    if not isinstance(scenes, Mapping) or len(scenes) < 2:
        raise ValueError("source text-response preregistration requires >=2 scenes")
    if list(scenes) != sorted(scenes):
        raise ValueError("source text-response scene order differs")
    for scene_id, scene in scenes.items():
        if not isinstance(scene, Mapping):
            raise ValueError(f"source text-response scene contract differs: {scene_id}")
        frames = scene.get("source_heldout_frame_ids")
        if (
            not isinstance(frames, list)
            or not frames
            or frames != sorted(set(frames))
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames)
        ):
            raise ValueError(f"source-heldout frame contract differs: {scene_id}")
        forbidden = scene.get("forbidden_target_frame_ids")
        if not isinstance(forbidden, list) or set(frames).intersection(forbidden):
            raise ValueError(f"source/target frame separation differs: {scene_id}")
    query_bank = preregistration.get("query_bank")
    if not isinstance(query_bank, Mapping) or query_bank.get("query_split") != "dev":
        raise ValueError("source text-response query bank differs")
    if int(query_bank.get("queries", -1)) != 101:
        raise ValueError("source text-response query count differs")
    decision = preregistration.get("decision_rule")
    if not isinstance(decision, Mapping) or decision.get("one_global_policy") is not True:
        raise ValueError("source text-response decision rule differs")
    return preregistration


def audit_asset_inventory(preregistration: Mapping[str, object]) -> dict[str, object]:
    """Inspect names/metadata only; never open a tensor artifact."""

    checked = validate_preregistration(preregistration)
    query_bank = checked["query_bank"]
    bank_path = Path(str(query_bank["path"]))
    bank_manifest = Path(str(query_bank["manifest_path"]))
    scenes: list[dict[str, object]] = []
    for scene_id, contract in checked["source_scenes"].items():
        tensor_dir = Path(str(contract["raw_teacher_tensor_dir"]))
        expected = [int(value) for value in contract["source_heldout_frame_ids"]]
        present = [frame for frame in expected if (tensor_dir / f"rgb_{frame}.pt").is_file()]
        reseal = Path(str(contract["required_reseal_manifest_path"]))
        scenes.append(
            {
                "scene_id": scene_id,
                "raw_teacher_tensor_dir_present": tensor_dir.is_dir(),
                "source_heldout_tensor_names_present": present,
                "source_heldout_tensor_name_coverage": len(present) / len(expected),
                "formal_content_reseal_present": reseal.is_file(),
                "control_response_summary_present": False,
                "candidate_response_summary_present": False,
            }
        )
    ready = (
        bank_path.is_file()
        and bank_manifest.is_file()
        and all(scene["formal_content_reseal_present"] for scene in scenes)
        and all(scene["source_heldout_tensor_name_coverage"] == 1.0 for scene in scenes)
    )
    return {
        "schema": "radio_gs.lerf_source_text_response_asset_inventory.v1",
        "schema_version": 1,
        "query_bank": {
            "artifact_present": bank_path.is_file(),
            "artifact_size_bytes": bank_path.stat().st_size if bank_path.is_file() else None,
            "manifest_present": bank_manifest.is_file(),
            "manifest_size_bytes": (
                bank_manifest.stat().st_size if bank_manifest.is_file() else None
            ),
            "tensor_content_opened": False,
        },
        "scenes": scenes,
        "existing_assets_ready_for_summary_materialization": ready,
        "compact_control_or_candidate_summaries_exist": False,
        "gate_executable_now": False,
        "shortest_missing_chain": [
            "complete P1 per-scene content reseal without tensor reencoding",
            "seal one candidate-specific source-response execution authority",
            "render control and candidate descriptor maps on exact legal source-heldout frames",
            "stream target-blind dev-bank responses through evaluate_source_frame",
            "write compact per-scene summaries, then run this selector",
        ],
        "target_or_benchmark_metric_opened": False,
    }


def _parse_scene_paths(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use scene=/absolute/path.json")
        scene, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not scene or scene in result or not path.is_file():
            raise ValueError(f"{label} scene/path differs: {value}")
        result[scene] = path
    return result


def _load_scene_summaries(
    values: Sequence[str], *, label: str
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    paths = _parse_scene_paths(values, label=label)
    summaries: list[dict[str, object]] = []
    records: list[dict[str, str]] = []
    for scene_id in sorted(paths):
        path = paths[scene_id]
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        summary = validate_scene_summary(value)
        if summary["scene_id"] != scene_id:
            raise ValueError(f"{label} scene identity differs")
        summaries.append(summary)
        records.append(
            {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    return summaries, records


def select(
    preregistration: Mapping[str, object],
    preregistration_record: Mapping[str, str],
    *,
    control_values: Sequence[str],
    candidate_values: Sequence[str],
) -> dict[str, object]:
    checked = validate_preregistration(preregistration)
    controls, control_records = _load_scene_summaries(
        control_values, label="control summary"
    )
    candidates, candidate_records = _load_scene_summaries(
        candidate_values, label="candidate summary"
    )
    required = list(checked["source_scenes"])
    gate = paired_source_gate(
        controls, candidates, required_scene_ids=required
    )
    for summary in controls + candidates:
        contract = checked["source_scenes"][summary["scene_id"]]
        if summary["source_heldout_frame_ids"] != contract[
            "source_heldout_frame_ids"
        ]:
            raise ValueError("source text-response summary frame authority differs")
        if summary["query_bank"] != checked["query_bank"]:
            raise ValueError("source text-response summary query bank differs")
        if summary["frame_evaluator_implementation"] != checked[
            "frame_evaluator_implementation"
        ]:
            raise ValueError("source text-response summary evaluator differs")
    return {
        **gate,
        "preregistration": dict(preregistration_record),
        "input_summaries": {
            "control": control_records,
            "candidate": candidate_records,
        },
        "metric_execution_authorized": False,
        "metric_executed": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--control-summary", action="append", required=True)
    gate_parser.add_argument("--candidate-summary", action="append", required=True)
    gate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preregistration, preregistration_record = _load_json(
        args.preregistration,
        args.expected_preregistration_sha256,
        label="source text-response preregistration",
    )
    if args.command == "inventory":
        result = audit_asset_inventory(preregistration)
    else:
        result = select(
            preregistration,
            preregistration_record,
            control_values=args.control_summary,
            candidate_values=args.candidate_summary,
        )
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to replace source gate result: {output}")
        _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
