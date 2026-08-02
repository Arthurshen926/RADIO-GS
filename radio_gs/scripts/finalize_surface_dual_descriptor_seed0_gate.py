#!/usr/bin/env python3
"""Finalize the dual-descriptor seed-0 gate from materializer evidence.

The training program is intentionally unable to satisfy its own point/render
check.  This finalizer reopens the immutable candidate, independently binds
the materializer report and every file named by it, converts the report's
minimal replay claim into the exact evidence schema accepted by
``build_pilot_gate``, and publishes a no-clobber final decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.evaluation.text_response_fidelity import (
    canonical_json_sha256,
    tensor_sha256,
)
from radio_gs.scripts.materialize_surface_dual_descriptor import (
    ADAPTER_ARTIFACT_TYPE,
    ARTIFACT_TYPE as DESCRIPTOR_ARTIFACT_TYPE,
    SCALAR_ARTIFACT_TYPE,
    _validate_replay_weights,
    _validate_compositor_manifest,
)
from radio_gs.scripts.train_surface_region_dual_descriptor_residual_pilot import (
    build_pilot_gate,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_dual_descriptor_seed0_gate_finalization"
EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "candidate_adapter_state_dict_sha256",
        "independent_materializer_replay",
        "frozen_scalar_compositor_replay",
        "point_render_replay_max_abs_error",
    }
)


def _require_record(raw: object, *, label: str) -> dict[str, str]:
    path = validate_file_record(raw, label=label)
    return {"path": str(path), "sha256": str(raw["sha256"])}  # type: ignore[index]


def _load_candidate(path: Path) -> tuple[dict[str, Any], dict[str, str], Mapping[str, Any]]:
    payload, digest, source = load_torch_mapping(
        path, map_location="cpu", label="dual-descriptor seed-0 candidate"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != ADAPTER_ARTIFACT_TYPE
        or payload.get("training_complete") is not True
        or payload.get("gate_status")
        != "training_complete_pending_point_render_replay"
        or payload.get("pilot_advance_gate_passed") is not False
        or payload.get("continuation_authorized") is not False
        or payload.get("seed1_executed") is not False
    ):
        raise ValueError("candidate is not the pending immutable seed-0 pilot")
    history = payload.get("history")
    best_epoch = payload.get("best_epoch")
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 0 <= best_epoch < len(history)
        or not isinstance(history[best_epoch], Mapping)
        or history[best_epoch].get("epoch") != best_epoch
        or history[best_epoch].get("adapter_state_dict_sha256")
        != payload.get("adapter_state_dict_sha256")
    ):
        raise ValueError("candidate selected-history binding differs")
    report_path = source.with_suffix(source.suffix + ".json")
    report, report_digest, report_source = load_json_object(
        report_path, label="dual-descriptor seed-0 candidate report"
    )
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type") != f"{ADAPTER_ARTIFACT_TYPE}_report"
        or Path(str(report.get("output", ""))).resolve() != source
        or report.get("checkpoint_sha256") != digest
        or report.get("best_epoch") != best_epoch
        or report.get("adapter_state_dict_sha256")
        != payload.get("adapter_state_dict_sha256")
        or report.get("selected_history_record") != history[best_epoch]
    ):
        raise ValueError("candidate JSON report binding differs")
    return (
        payload,
        {"path": str(source), "sha256": digest},
        {
            "selected": history[best_epoch],
            "report": {"path": str(report_source), "sha256": report_digest},
        },
    )


def _load_materializer_report(
    path: Path,
    *,
    candidate_record: Mapping[str, str],
    adapter_state_sha256: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    report, digest, source = load_json_object(
        path, label="dual-descriptor materializer report"
    )
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type")
        != "surface_dual_descriptor_materialization_report"
        or report.get("target_blind") is not True
        or report.get("benchmark_targets_or_metrics_used") is not False
        or report.get("device") != "cpu"
        or report.get("point_render_replay_passed") is not True
        or report.get("formal_point_render_replay_evidence_eligible") is not True
        or report.get("replay_weights_schema_version") != 2
        or report.get("replay_operator_layout") != "sparse_triplets_v2"
        or report.get("official_token_bitwise_equal") is not True
        or report.get("official_descriptor_bitwise_equal") is not True
        or report.get("adapter_checkpoint") != candidate_record
    ):
        raise ValueError("materializer report is not a target-blind candidate replay")
    for key in (
        "descriptor_cache",
        "scalar_cache",
        "adapter_checkpoint_report",
        "base_checkpoint",
        "radio_checkpoint",
        "compositor_weights",
    ):
        _require_record(report.get(key), label=f"materializer {key}")
    input_records = report.get("input_caches")
    training = report.get("adapter_training_caches")
    if (
        not isinstance(input_records, list)
        or not input_records
        or not isinstance(training, Mapping)
        or set(training) != {"train", "validation"}
    ):
        raise ValueError("materializer cache bindings differ")
    for index, record in enumerate(input_records):
        _require_record(record, label=f"materializer input cache {index}")
    for split in ("train", "validation"):
        records = training[split]
        if not isinstance(records, list) or not records:
            raise ValueError("materializer training cache bindings differ")
        for index, record in enumerate(records):
            _require_record(record, label=f"materializer {split} cache {index}")
    compositor = report.get("scalar_compositor_manifest")
    compositor_path = _require_record(
        {"path": compositor.get("path"), "sha256": compositor.get("sha256")}
        if isinstance(compositor, Mapping)
        else compositor,
        label="materializer scalar compositor manifest",
    )
    validated_compositor = _validate_compositor_manifest(
        Path(compositor_path["path"])
    )
    if (
        not isinstance(compositor, Mapping)
        or compositor.get("selected_variant")
        != validated_compositor["selected_variant"]
    ):
        raise ValueError("materializer selected compositor binding differs")
    evidence = report.get("point_render_replay_evidence")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != EVIDENCE_KEYS
        or evidence.get("candidate_adapter_state_dict_sha256")
        != adapter_state_sha256
        or float(evidence.get("point_render_replay_max_abs_error", -1.0))
        != float(report.get("point_render_replay_max_abs_error", -2.0))
    ):
        raise ValueError("materializer point/render evidence binding differs")
    # ``build_pilot_gate`` owns the semantic validation of these six fields.
    return dict(report), {"path": str(source), "sha256": digest}, dict(evidence)


def _recompute_materializer_replay(
    report: Mapping[str, Any],
    *,
    candidate_record: Mapping[str, str],
) -> dict[str, Any]:
    descriptor_record = report["descriptor_cache"]
    scalar_record = report["scalar_cache"]
    descriptor, _, descriptor_path = load_torch_mapping(
        descriptor_record["path"],
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="finalizer descriptor cache",
    )
    scalar, _, scalar_path = load_torch_mapping(
        scalar_record["path"],
        expected_sha256=scalar_record["sha256"],
        map_location="cpu",
        label="finalizer scalar replay cache",
    )
    primitive_ids = descriptor.get("primitive_ids")
    semantic = descriptor.get("semantic_descriptors")
    content = descriptor.get("descriptor_cache_content")
    descriptor_provenance = descriptor.get("provenance")
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("artifact_type") != DESCRIPTOR_ARTIFACT_TYPE
        or not isinstance(primitive_ids, list)
        or not primitive_ids
        or len(set(primitive_ids)) != len(primitive_ids)
        or not isinstance(semantic, torch.Tensor)
        or semantic.device.type != "cpu"
        or semantic.dtype != torch.float32
        or semantic.shape != (len(primitive_ids), 1536)
        or not bool(torch.isfinite(semantic).all())
        or not isinstance(content, Mapping)
        or not isinstance(descriptor_provenance, Mapping)
        or content.get("adapter_checkpoint") != candidate_record
        or descriptor.get("scalar_cache") != scalar_record
        or descriptor.get("descriptor_cache_content_sha256")
        != scalar.get("descriptor_cache_content_sha256")
    ):
        raise ValueError("descriptor/scalar materialization binding differs")
    norms = semantic.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0):
        raise ValueError("finalizer descriptor rows are not normalized")
    if (
        content.get("semantic_descriptors_sha256") != tensor_sha256(semantic)
        or descriptor_provenance.get("adapter_checkpoint") != candidate_record
    ):
        raise ValueError("descriptor internal candidate/digest binding differs")
    scalar_contract = scalar.get("contract")
    if (
        scalar.get("schema_version") != 1
        or scalar.get("artifact_type") != SCALAR_ARTIFACT_TYPE
        or scalar.get("primitive_ids") != primitive_ids
        or scalar.get("scalar_compositor_manifest")
        != report["scalar_compositor_manifest"]
        or scalar.get("compositor_weights") != report["compositor_weights"]
        or scalar.get("contract") != report["production_contract"]
        or not isinstance(scalar_contract, Mapping)
        or scalar_contract.get("replay_weights_schema_version") != 2
        or scalar_contract.get("replay_operator_layout")
        != "sparse_triplets_v2"
    ):
        raise ValueError("scalar replay internal authority differs")
    compositor_record = report["scalar_compositor_manifest"]
    compositor = _validate_compositor_manifest(Path(compositor_record["path"]))
    input_records = report["input_caches"]
    operator, queries, weights_record = _validate_replay_weights(
        Path(report["compositor_weights"]["path"]),
        primitive_ids=primitive_ids,
        compositor=compositor,
        input_cache_records=input_records,
    )
    if (
        operator.get("schema_version") != 2
        or operator.get("layout") != "sparse_triplets_v2"
        or weights_record != report["compositor_weights"]
        or not isinstance(scalar.get("query_bank"), torch.Tensor)
        or not torch.equal(scalar["query_bank"], queries)
    ):
        raise ValueError("finalizer requires the exact sparse-v2 replay authority")
    primitive_scores = semantic @ queries.T
    render_rows = operator["render_row_index"]
    primitive_rows = operator["primitive_row_index"]
    weights = operator["weights"]
    point_then_render = torch.zeros(
        operator["num_render_rows"], queries.shape[0], dtype=torch.float32
    )
    point_then_render.index_add_(
        0,
        render_rows,
        weights[:, None] * primitive_scores[primitive_rows],
    )
    rendered_semantic = torch.zeros(
        operator["num_render_rows"], semantic.shape[1], dtype=torch.float32
    )
    rendered_semantic.index_add_(
        0,
        render_rows,
        weights[:, None] * semantic[primitive_rows],
    )
    render_then_query = rendered_semantic @ queries.T
    for key, expected in (
        ("primitive_scalar_scores", primitive_scores),
        ("point_then_render_scores", point_then_render),
        ("render_then_query_scores", render_then_query),
    ):
        observed = scalar.get(key)
        if not isinstance(observed, torch.Tensor) or not torch.equal(observed, expected):
            raise ValueError(f"finalizer recomputed {key} differs")
    replay_error = float((point_then_render - render_then_query).abs().max().item())
    if not math.isfinite(replay_error) or replay_error > 1e-6:
        raise ValueError("finalizer independently recomputed replay exceeds 1e-6")
    query_sha = tensor_sha256(queries)
    primitive_score_sha = tensor_sha256(primitive_scores)
    point_render_sha = canonical_json_sha256(
        {
            "query_bank_sha256": query_sha,
            "primitive_scalar_scores_sha256": primitive_score_sha,
            "point_then_render_scores_sha256": tensor_sha256(point_then_render),
            "render_then_query_scores_sha256": tensor_sha256(render_then_query),
            "point_render_replay_max_abs_error": replay_error,
        }
    )
    if (
        scalar.get("query_bank_sha256") != query_sha
        or scalar.get("primitive_scalar_scores_sha256") != primitive_score_sha
        or scalar.get("point_then_render_scores_sha256")
        != tensor_sha256(point_then_render)
        or scalar.get("render_then_query_scores_sha256")
        != tensor_sha256(render_then_query)
        or scalar.get("scalar_score_replay_sha256") != point_render_sha
        or report.get("scalar_score_replay_sha256") != point_render_sha
        or float(scalar.get("point_render_replay_max_abs_error", -1.0))
        != replay_error
        or float(report.get("point_render_replay_max_abs_error", -1.0))
        != replay_error
    ):
        raise ValueError("finalizer replay digest/error binding differs")
    return {
        "replay_error": replay_error,
        "descriptor_cache": {"path": str(descriptor_path), **descriptor_record},
        "scalar_cache": {"path": str(scalar_path), **scalar_record},
        "replay_weights_schema_version": 2,
        "replay_operator_layout": "sparse_triplets_v2",
        "scalar_score_replay_sha256": point_render_sha,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"seed-0 finalization output must be new: {output}")
    candidate, candidate_record, candidate_authority = _load_candidate(
        Path(args.adapter_checkpoint)
    )
    materializer, materializer_record, evidence = _load_materializer_report(
        Path(args.materializer_report),
        candidate_record=candidate_record,
        adapter_state_sha256=str(candidate["adapter_state_dict_sha256"]),
    )
    recomputed_replay = _recompute_materializer_replay(
        materializer,
        candidate_record=candidate_record,
    )
    if evidence["point_render_replay_max_abs_error"] != recomputed_replay[
        "replay_error"
    ]:
        raise ValueError("materializer evidence differs from independent recomputation")
    selected = candidate_authority["selected"]
    gate = build_pilot_gate(
        selected,
        point_render_replay_evidence=evidence,
    )
    if gate.get("finalization_status") != "finalized":
        raise RuntimeError("materializer evidence did not finalize the seed-0 gate")
    passed = gate.get("passed") is True
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "candidate_checkpoint": candidate_record,
        "candidate_report": candidate_authority["report"],
        "candidate_adapter_state_dict_sha256": candidate[
            "adapter_state_dict_sha256"
        ],
        "selected_epoch": int(candidate["best_epoch"]),
        "materializer_report": materializer_record,
        "materialized_descriptor_cache": materializer["descriptor_cache"],
        "materialized_scalar_cache": materializer["scalar_cache"],
        "scalar_compositor_manifest": materializer[
            "scalar_compositor_manifest"
        ],
        "compositor_weights": materializer["compositor_weights"],
        "point_render_replay_evidence": evidence,
        "independent_finalizer_recomputation": recomputed_replay,
        "seed0_single_conjunction_gate": gate,
        "pilot_advance_gate_passed": passed,
        "continuation_authorized": passed,
        "additional_seed_or_architecture_authorized": passed,
        "seed1_executed": False,
        "external_benchmarks_opened": False,
        "metric_continuation": False,
        "decision": "advance" if passed else "stop_seed0_gate_failed",
    }
    write_frozen_json(output, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--materializer-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = finalize(build_arg_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
