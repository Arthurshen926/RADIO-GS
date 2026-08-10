#!/usr/bin/env python3
"""Select one frozen LERF candidate on the reserved audit-90 query bank.

This is a CPU-only confirmation sibling.  It deliberately does not share the
development selector's preregistration schema: the input preregistration must
bind the disjoint ``audit`` split with exactly 90 queries.  The scientific
decision is delegated unchanged to :func:`paired_source_gate`.
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
    "radio_gs.lerf_source_text_response_ranking_audit_confirmation_preregistration.v1"
)
PREREGISTRATION_STATUS = (
    "sealed_candidate_specific_after_dev_pass_before_reserved_audit_open"
)
QUERY_SPLIT = "audit"
QUERY_COUNT = 90


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if (
        not Path(record["path"]).is_absolute()
        or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    ):
        raise ValueError(f"{label} differs")
    return record


def _query_bank(value: object) -> dict[str, object]:
    keys = {
        "path",
        "sha256",
        "manifest_path",
        "manifest_sha256",
        "query_split",
        "queries",
        "embedding_tensor_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("audit confirmation query-bank schema differs")
    bank = dict(value)
    if bank["query_split"] != QUERY_SPLIT or bank["queries"] != QUERY_COUNT:
        raise ValueError("audit confirmation requires the reserved audit-90 query bank")
    for key in ("path", "manifest_path"):
        if not Path(str(bank[key])).is_absolute():
            raise ValueError(f"audit confirmation query-bank {key} differs")
    for key in ("sha256", "manifest_sha256", "embedding_tensor_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(bank[key])):
            raise ValueError(f"audit confirmation query-bank {key} differs")
    return bank


def validate_preregistration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("audit confirmation preregistration must be an object")
    preregistration = dict(value)
    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("schema_version") != 1
        or preregistration.get("status") != PREREGISTRATION_STATUS
    ):
        raise ValueError("audit confirmation preregistration schema differs")
    if preregistration.get("one_shot_confirmation") is not True:
        raise ValueError("audit confirmation must be one shot")
    if preregistration.get("development_selection_complete") is not True:
        raise ValueError("audit confirmation requires completed development selection")
    if preregistration.get("target_metric_execution_authorized") is not False:
        raise ValueError("audit confirmation preregistration authorizes target metric")
    if preregistration.get("target_results_used_for_gate_design") is not False:
        raise ValueError("audit confirmation gate design is target contaminated")

    for key, label in (
        ("selector_implementation", "selector"),
        ("frame_evaluator_implementation", "frame evaluator"),
        ("shared_response_metric_implementation", "shared response metric"),
    ):
        record = _record(preregistration.get(key), label=label)
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"audit confirmation {label} implementation changed")

    scenes = preregistration.get("source_scenes")
    if not isinstance(scenes, Mapping) or len(scenes) < 2:
        raise ValueError("audit confirmation requires at least two source scenes")
    if list(scenes) != sorted(scenes):
        raise ValueError("audit confirmation source-scene order differs")
    for scene_id, contract in scenes.items():
        if not isinstance(contract, Mapping):
            raise ValueError(f"audit confirmation scene contract differs: {scene_id}")
        frames = contract.get("source_heldout_frame_ids")
        forbidden = contract.get("forbidden_target_frame_ids")
        if (
            not isinstance(frames, list)
            or not frames
            or frames != sorted(set(frames))
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames)
        ):
            raise ValueError(f"audit confirmation frame contract differs: {scene_id}")
        if not isinstance(forbidden, list) or set(frames).intersection(forbidden):
            raise ValueError(f"audit confirmation source/target separation differs: {scene_id}")

    preregistration["query_bank"] = _query_bank(preregistration.get("query_bank"))
    decision = preregistration.get("decision_rule")
    required_scenes = list(scenes)
    if (
        not isinstance(decision, Mapping)
        or decision.get("one_global_policy") is not True
        or decision.get("required_source_scenes") != required_scenes
        or decision.get("strict_pooled_improvements")
        != [
            "response_mae_lower",
            "ranking_spearman_mean_higher",
            "top_decile_overlap_mean_higher",
        ]
        or decision.get("every_scene_nonregression")
        != [
            "response_mae",
            "response_profile_cosine_mean",
            "ranking_spearman_mean",
            "ranking_spearman_p05",
            "top_decile_overlap_mean",
            "top_decile_overlap_p05",
        ]
        or decision.get("tolerance") != 0.0
        or decision.get("fallback") != "unchanged_control"
        or decision.get("per_scene_or_per_query_thresholds") is not False
    ):
        raise ValueError("audit confirmation decision rule differs")
    return preregistration


def _load_json(
    path: str | Path, expected_sha256: str, *, label: str
) -> tuple[dict[str, object], dict[str, str]]:
    source = Path(path).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
        raise ValueError(f"{label} requires a lowercase trusted SHA-256")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, {"path": str(source), "sha256": digest}


def _parse_scene_paths(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use scene=/absolute/path.json")
        scene_id, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not scene_id or scene_id in result or not path.is_file():
            raise ValueError(f"{label} scene/path differs: {value}")
        result[scene_id] = path
    return result


def _load_scene_summaries(
    values: Sequence[str], *, label: str
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    summaries: list[dict[str, object]] = []
    records: list[dict[str, str]] = []
    for scene_id, path in sorted(_parse_scene_paths(values, label=label).items()):
        raw = path.read_bytes()
        summary = validate_scene_summary(json.loads(raw.decode("utf-8")))
        if summary["scene_id"] != scene_id:
            raise ValueError(f"{label} scene identity differs")
        summaries.append(summary)
        records.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
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
        control_values, label="control audit summary"
    )
    candidates, candidate_records = _load_scene_summaries(
        candidate_values, label="candidate audit summary"
    )
    required = list(checked["source_scenes"])
    for summary in controls + candidates:
        scene_id = str(summary["scene_id"])
        if scene_id not in checked["source_scenes"]:
            raise ValueError("audit confirmation summary scene differs")
        if summary["source_heldout_frame_ids"] != checked["source_scenes"][scene_id][
            "source_heldout_frame_ids"
        ]:
            raise ValueError("audit confirmation summary frame authority differs")
        if summary["query_bank"] != checked["query_bank"]:
            raise ValueError("audit confirmation summary query bank differs")
        if summary["frame_evaluator_implementation"] != checked[
            "frame_evaluator_implementation"
        ]:
            raise ValueError("audit confirmation summary evaluator differs")

    gate = paired_source_gate(controls, candidates, required_scene_ids=required)
    if gate["query_bank"]["query_split"] != QUERY_SPLIT:
        raise AssertionError("paired source gate returned a non-audit query split")
    if gate["query_bank"]["queries"] != QUERY_COUNT:
        raise AssertionError("paired source gate returned a non-audit90 query count")
    if gate["protocol"]["target_metric_execution_authorized"] is not False:
        raise AssertionError("paired source gate authorized a target metric")
    return {
        **gate,
        "confirmation_preregistration": dict(preregistration_record),
        "confirmation_query_split": QUERY_SPLIT,
        "confirmation_query_count": QUERY_COUNT,
        "one_shot_confirmation": True,
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
    parser.add_argument("--control-summary", action="append", required=True)
    parser.add_argument("--candidate-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preregistration, preregistration_record = _load_json(
        args.preregistration,
        args.expected_preregistration_sha256,
        label="audit confirmation preregistration",
    )
    result = select(
        preregistration,
        preregistration_record,
        control_values=args.control_summary,
        candidate_values=args.candidate_summary,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace audit confirmation result: {output}")
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PREREGISTRATION_SCHEMA",
    "PREREGISTRATION_STATUS",
    "QUERY_COUNT",
    "QUERY_SPLIT",
    "file_sha256",
    "select",
    "validate_preregistration",
]
