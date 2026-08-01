#!/usr/bin/env python3
"""Confirm canonical-v5 only from externally anchored query-free artifacts."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from radio_gs.evaluation.capability_fidelity import select_query_free_compositor
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


def _paths(raw: str) -> list[Path]:
    paths = [Path(value) for value in str(raw).replace(",", " ").split() if value]
    if len(paths) < 2:
        raise ValueError("cross-scene confirmation requires at least two scene reports")
    return paths


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not path or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} has an invalid path or SHA-256")
    return {"path": path, "sha256": digest}


def _finite_json(value: object, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, label=f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, label=f"{label}.{key}")
        return
    raise ValueError(f"{label} contains an unsupported value")


def _load_json_record(record: object, *, label: str) -> tuple[dict, Path]:
    normalized = _record(record, label=label)
    value, _digest, source = load_json_object(
        normalized["path"],
        expected_sha256=normalized["sha256"],
        label=label,
    )
    _finite_json(value, label=label)
    return value, source


def _validate_audit(
    audit: dict,
    *,
    manifest: dict,
    checkpoint: Path,
    checkpoint_sha256: str,
    held_out: dict,
    label: str,
) -> None:
    if set(audit) != {
        "schema_version",
        "audit",
        "protocol",
        "artifacts",
        "aggregate",
        "per_frame",
    } or audit.get("schema_version") != 1 or audit.get("audit") != "canonical_capability_fidelity_v1":
        raise ValueError(f"{label} audit schema differs")
    if audit.get("aggregate") != held_out:
        raise ValueError(f"{label} metrics differ from its audit")
    protocol = audit.get("protocol")
    artifacts = audit.get("artifacts")
    aggregate = audit.get("aggregate")
    per_frame = audit.get("per_frame")
    if not all(isinstance(value, dict) for value in (protocol, artifacts, aggregate)):
        raise ValueError(f"{label} audit provenance is missing")
    expected_aggregate_keys = {
        "raw_radio",
        "official_dino_v3",
        "official_sam3",
        "support_fraction_on_visible",
        "supported_visible_pixels",
        "total_visible_pixels",
    }
    if set(aggregate) != expected_aggregate_keys:
        raise ValueError(f"{label} aggregate schema differs")
    for space in ("raw_radio", "official_dino_v3", "official_sam3"):
        metrics = aggregate.get(space)
        if not isinstance(metrics, dict) or set(metrics) != {
            "pixels",
            "mean_cosine",
            "p05_cosine",
            "local_relation",
        }:
            raise ValueError(f"{label} {space} metric schema differs")
        relation = metrics.get("local_relation")
        if not isinstance(relation, dict) or set(relation) != {
            "pairs",
            "affinity_mae",
            "affinity_pearson",
            "teacher_boundary_margin",
            "predicted_boundary_margin",
            "boundary_margin_retention",
        }:
            raise ValueError(f"{label} {space} relation schema differs")
    if (
        not isinstance(per_frame, list)
        or [row.get("frame_id") for row in per_frame if isinstance(row, dict)]
        != protocol.get("frame_ids")
    ):
        raise ValueError(f"{label} audit per-frame coverage differs")
    support = float(aggregate["support_fraction_on_visible"])
    supported = aggregate["supported_visible_pixels"]
    visible = aggregate["total_visible_pixels"]
    if (
        not 0 <= support <= 1
        or not isinstance(supported, int)
        or not isinstance(visible, int)
        or supported < 0
        or visible < supported
    ):
        raise ValueError(f"{label} audit support summary is invalid")
    audit_contract = manifest.get("fixed_audit_contract", {})
    if (
        protocol.get("held_out_from_mpr") is not True
        or protocol.get("frame_ids") != manifest.get("fidelity_frame_ids")
        or protocol.get("benchmark_masks_opened") is not False
        or protocol.get("text_queries_opened") is not False
        or protocol.get("capability_map_source") != "official_extracted"
        or protocol.get("alpha_threshold") != audit_contract.get("alpha_threshold")
        or protocol.get("support_eps") != audit_contract.get("support_eps")
        or protocol.get("boundary_quantile") != audit_contract.get("boundary_quantile")
        or protocol.get("residual_mode") != audit_contract.get("residual_mode")
        or Path(str(artifacts.get("field_checkpoint", ""))).resolve() != checkpoint.resolve()
        or artifacts.get("field_checkpoint_sha256") != checkpoint_sha256
        or artifacts.get("config_sha256") != manifest.get("config_sha256")
        or artifacts.get("resolved_config_sha256") != manifest.get("resolved_config_sha256")
        or artifacts.get("geometry_checkpoint_sha256") != manifest.get("geometry_checkpoint_sha256")
        or artifacts.get("radio_checkpoint_sha256") != manifest.get("radio_checkpoint_sha256")
        or artifacts.get("feature_output_bundle_sha256")
        != manifest.get("feature_extraction_safety_contract", {}).get(
            "final_output_bundle_sha256"
        )
        or artifacts.get("view_residual_checkpoint") != ""
        or artifacts.get("boundary_residual_checkpoint") != ""
    ):
        raise ValueError(f"{label} audit differs from its run manifest")
    for source_name in ("dino_v3", "sam3"):
        actual = artifacts.get("official_capability_sources", {}).get(source_name)
        expected = manifest.get("official_capability_sources", {}).get(source_name)
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise ValueError(f"{label} lacks {source_name} source provenance")
        for key in (
            "frame_manifest_sha256",
            "output_bundle_sha256",
            "radio_checkpoint_sha256",
            "radio_checkpoint_provenance",
            "radio_checkpoint_load_contract",
            "scene",
            "image_dir",
            "frame_indices_sha256",
            "execution",
        ):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"{label} {source_name} source differs: {key}")


def _source_invariant(source: dict) -> dict:
    execution = source.get("feature_extraction_execution", {})
    return {
        "adaptor_name": source.get("adaptor_name"),
        "native_grid": source.get("native_grid"),
        "radio_version": source.get("radio_version"),
        "radio_checkpoint_sha256": source.get("radio_checkpoint_sha256"),
        "radio_checkpoint_provenance": source.get("radio_checkpoint_provenance"),
        "radio_checkpoint_load_contract": source.get(
            "radio_checkpoint_load_contract"
        ),
        "execution": source.get("execution"),
        "radio_source_tree_sha256": source.get("radio_source_tree_sha256"),
        "runtime_fingerprint_sha256": source.get("runtime_fingerprint_sha256"),
        "feature_execution_contract": {
            key: execution.get(key)
            for key in (
                "resume_partial",
                "atomic_tensor_commit",
                "atomic_manifest_commit",
                "committed_frame_validation",
                "invalid_or_missing_frame_policy",
                "radio_thermal_pacing_seconds_per_image",
                "pacing_order",
            )
        },
    }


def _frozen_contract(manifest: dict) -> dict:
    feature_safety = dict(manifest.get("feature_extraction_safety_contract", {}))
    feature_safety.pop("final_output_bundle_sha256", None)
    sources = manifest.get("official_capability_sources", {})
    return {
        "epochs": manifest.get("epochs"),
        "seed": manifest.get("seed"),
        "fixed_training_contract": manifest.get("fixed_training_contract"),
        "fixed_audit_contract": manifest.get("fixed_audit_contract"),
        "fixed_selection_contract": manifest.get("fixed_selection_contract"),
        "implementation_sources": manifest.get("implementation_sources"),
        "implementation_source_tree": manifest.get("implementation_source_tree"),
        "runner_sha256": manifest.get("runner_sha256"),
        "radio_checkpoint_sha256": manifest.get("radio_checkpoint_sha256"),
        "feature_extraction_safety_contract": feature_safety,
        "thermal_safety_contract": manifest.get("thermal_safety_contract"),
        "continuous_stage_safety_contract": manifest.get(
            "continuous_stage_safety_contract"
        ),
        "official_capability_contracts": {
            name: _source_invariant(dict(sources.get(name, {})))
            for name in ("dino_v3", "sam3")
        },
    }


def confirm(
    paths: list[Path],
    *,
    trusted_registry: str | Path,
    expected_registry_sha256: str,
) -> dict:
    if len(paths) < 2:
        raise ValueError("cross-scene confirmation requires at least two scene reports")
    registry, registry_sha256, registry_path = load_json_object(
        trusted_registry,
        expected_sha256=expected_registry_sha256,
        label="external canonical-v5 authority registry",
    )
    if set(registry) != {"schema_version", "contract", "screens"} or (
        registry.get("schema_version") != 1
        or registry.get("contract") != "canonical-v5-external-trusted-registry-v1"
        or not isinstance(registry.get("screens"), list)
    ):
        raise ValueError("external authority registry schema differs")
    registry_by_report: dict[Path, dict] = {}
    for entry in registry["screens"]:
        if not isinstance(entry, dict) or set(entry) != {
            "scene",
            "report",
            "run_manifest",
            "completion_bundle",
            "candidates",
        }:
            raise ValueError("external authority registry screen entry differs")
        report_record = _record(entry["report"], label="registry report")
        key = Path(report_record["path"]).resolve()
        if key in registry_by_report:
            raise ValueError("external authority registry repeats a report")
        registry_by_report[key] = entry
    requested = [path.resolve() for path in paths]
    if len(set(requested)) != len(requested) or set(requested) != set(registry_by_report):
        raise ValueError("scene reports differ from the external authority registry")

    rows: list[dict] = []
    frozen_contract = None
    scenes: set[str] = set()
    candidate_names: set[str] | None = None
    for requested_path in requested:
        authority = registry_by_report[requested_path]
        report, report_path = _load_json_record(
            authority["report"], label="trusted canonical-v5 scene report"
        )
        manifest, manifest_path = _load_json_record(
            authority["run_manifest"], label="trusted canonical-v5 run manifest"
        )
        completion, completion_path = _load_json_record(
            authority["completion_bundle"],
            label="trusted canonical-v5 completion bundle",
        )
        if report_path.resolve() != requested_path:
            raise ValueError("registry report path identity differs")
        if (
            report.get("schema_version") != 1
            or report.get("benchmark_queries_opened") is not False
            or report.get("benchmark_masks_opened") is not False
            or report.get("run_manifest_sha256")
            != _record(authority["run_manifest"], label="registry manifest")["sha256"]
            or Path(str(report.get("run_manifest", ""))).resolve()
            != manifest_path.resolve()
        ):
            raise ValueError(f"{report_path} is not an externally anchored query-free v5 screen")
        if (
            manifest.get("screen") != "canonical-v5-query-free-capacity"
            or manifest.get("benchmark_queries_opened") is not False
            or manifest.get("benchmark_masks_opened") is not False
            or not isinstance(manifest.get("implementation_source_tree"), dict)
            or not str(manifest["implementation_source_tree"].get("tree_sha256", ""))
        ):
            raise ValueError(f"{manifest_path} is not a frozen v5 manifest")
        dino_source = manifest.get("official_capability_sources", {}).get("dino_v3", {})
        sam_source = manifest.get("official_capability_sources", {}).get("sam3", {})
        scene = str(dino_source.get("scene", ""))
        if (
            not scene
            or scene != str(authority["scene"])
            or sam_source.get("scene") != scene
            or scene in scenes
        ):
            raise ValueError("v5 confirmation scenes are missing, repeated, or unbound")
        scenes.add(scene)

        authority_candidates = authority.get("candidates")
        if not isinstance(authority_candidates, dict) or not authority_candidates:
            raise ValueError("external authority registry has no candidates")
        completion_expected = {
            "schema_version": 1,
            "contract": "canonical-v5-capacity-screen-completion-v1",
            "screen": "canonical-v5-query-free-capacity",
            "scene": scene,
            "report": _record(authority["report"], label="registry report"),
            "run_manifest": _record(authority["run_manifest"], label="registry manifest"),
            "candidates": authority_candidates,
            "feature_output_bundle_sha256": manifest.get(
                "feature_extraction_safety_contract", {}
            ).get("final_output_bundle_sha256"),
            "implementation_source_tree_sha256": manifest.get(
                "implementation_source_tree", {}
            ).get("tree_sha256"),
            "benchmark_queries_opened": False,
            "benchmark_masks_opened": False,
        }
        if completion != completion_expected:
            raise ValueError(f"{completion_path} completion bundle differs from registry/run")

        selection = report.get("query_free_selection", {})
        selection_contract = manifest.get("fixed_selection_contract")
        expected_selection_keys = {
            "baseline",
            "max_mean_dense_drop",
            "max_p05_dense_drop",
            "max_unsupported_fraction",
            "min_relation_gain",
            "objective",
        }
        if (
            not isinstance(selection_contract, dict)
            or set(selection_contract) != expected_selection_keys
            or report.get("fixed_selection_contract") != selection_contract
            or selection.get("baseline_variant") != selection_contract["baseline"]
            or selection.get("thresholds")
            != {
                key: selection_contract[key]
                for key in (
                    "max_mean_dense_drop",
                    "max_p05_dense_drop",
                    "max_unsupported_fraction",
                    "min_relation_gain",
                )
            }
        ):
            raise ValueError(f"{report_path} selection contract is not manifest-bound")
        candidate_rows = report.get("candidates")
        if not isinstance(candidate_rows, list) or not candidate_rows:
            raise ValueError(f"{report_path} has no candidates")
        variants: dict[str, dict] = {}
        for candidate in candidate_rows:
            if not isinstance(candidate, dict):
                raise ValueError(f"{report_path} has an invalid candidate record")
            name = str(candidate.get("name", ""))
            held_out = candidate.get("held_out_fidelity")
            trusted = authority_candidates.get(name)
            if (
                not name
                or name in variants
                or not isinstance(held_out, dict)
                or not isinstance(trusted, dict)
                or set(trusted) != {"checkpoint", "audit"}
            ):
                raise ValueError(f"{report_path} has invalid externally anchored candidates")
            checkpoint_record = _record(trusted["checkpoint"], label=f"{scene}/{name} checkpoint")
            audit_record = _record(trusted["audit"], label=f"{scene}/{name} audit")
            checkpoint_path = Path(checkpoint_record["path"])
            if (
                Path(str(candidate.get("checkpoint", ""))).resolve() != checkpoint_path.resolve()
                or candidate.get("checkpoint_sha256") != checkpoint_record["sha256"]
                or Path(str(candidate.get("audit", ""))).resolve()
                != Path(audit_record["path"]).resolve()
                or candidate.get("audit_sha256") != audit_record["sha256"]
            ):
                raise ValueError(f"{report_path} candidate {name} differs from registry")
            field, field_payload = load_canonical_field_checkpoint(
                checkpoint_path,
                map_location="cpu",
                expected_sha256=checkpoint_record["sha256"],
            )
            if (
                field_payload.get("feature_output_bundle_sha256")
                != manifest["feature_extraction_safety_contract"][
                    "final_output_bundle_sha256"
                ]
            ):
                raise ValueError(f"{scene}/{name} field belongs to another feature bundle")
            del field
            audit, _audit_path = _load_json_record(
                audit_record, label=f"{scene}/{name} fidelity audit"
            )
            _validate_audit(
                audit,
                manifest=manifest,
                checkpoint=checkpoint_path,
                checkpoint_sha256=checkpoint_record["sha256"],
                held_out=held_out,
                label=f"{scene}/{name}",
            )
            variants[name] = held_out
        current_candidates = set(variants)
        frozen_candidates = set(
            manifest.get("fixed_training_contract", {}).get("candidates", {})
        )
        if current_candidates != set(authority_candidates) or current_candidates != frozen_candidates:
            raise ValueError(f"{report_path} candidate set differs from frozen authority")
        recomputed = select_query_free_compositor(
            variants,
            baseline=selection_contract["baseline"],
            max_mean_dense_drop=selection_contract["max_mean_dense_drop"],
            max_p05_dense_drop=selection_contract["max_p05_dense_drop"],
            max_unsupported_fraction=selection_contract["max_unsupported_fraction"],
            min_relation_gain=selection_contract["min_relation_gain"],
        )
        if selection != recomputed:
            raise ValueError(f"{report_path} selection differs from CPU replay")
        if candidate_names is None:
            candidate_names = current_candidates
        elif candidate_names != current_candidates:
            raise ValueError("v5 scene reports have different candidate sets")
        contract = _frozen_contract(manifest)
        if frozen_contract is None:
            frozen_contract = contract
        elif frozen_contract != contract:
            raise ValueError("v5 scene reports use different frozen contracts")
        rows.append(
            {
                "scene": scene,
                "report": str(report_path),
                "report_sha256": _record(authority["report"], label="registry report")["sha256"],
                "run_manifest": str(manifest_path),
                "run_manifest_sha256": _record(authority["run_manifest"], label="registry manifest")["sha256"],
                "completion_bundle": str(completion_path),
                "completion_bundle_sha256": _record(
                    authority["completion_bundle"], label="registry completion"
                )["sha256"],
                "feature_output_bundle_sha256": manifest[
                    "feature_extraction_safety_contract"
                ]["final_output_bundle_sha256"],
                "official_capability_sources": manifest[
                    "official_capability_sources"
                ],
                "selected_variant": recomputed.get("selected_variant"),
                "selection_status": recomputed.get("selection_status"),
                "promotion_allowed": recomputed.get("promotion_allowed") is True,
            }
        )
    selected = [row["selected_variant"] for row in rows]
    baseline = str(frozen_contract["fixed_selection_contract"]["baseline"])
    if any(value is None for value in selected):
        status, confirmed = "cross_scene_gate_failed_no_promotion", None
    elif len(set(selected)) != 1:
        status, confirmed = "cross_scene_selection_inconsistent_no_promotion", None
    elif selected[0] == baseline:
        status, confirmed = "cross_scene_baseline_retained", baseline
    elif not all(row["promotion_allowed"] for row in rows):
        status, confirmed = "cross_scene_candidate_not_promotion_eligible", None
    else:
        status, confirmed = "cross_scene_query_free_candidate_confirmed", selected[0]
    # The stable-descriptor readers close each file after validating it.  Reopen
    # every externally authorized artifact once more at the transaction
    # boundary so a replacement between scene iterations cannot be hidden in
    # the newly published confirmation.
    for screen_index, authority in enumerate(registry["screens"]):
        for key in ("report", "run_manifest", "completion_bundle"):
            validate_file_record(
                authority[key],
                label=f"registry screen {screen_index} {key}",
            )
        for name, records in authority["candidates"].items():
            for key in ("checkpoint", "audit"):
                validate_file_record(
                    records[key],
                    label=f"registry screen {screen_index} {name} {key}",
                )
    validate_file_record(
        {"path": str(registry_path), "sha256": registry_sha256},
        label="external canonical-v5 authority registry final recheck",
    )
    return {
        "schema_version": 2,
        "confirmation_status": status,
        "confirmed_variant": confirmed,
        "scenes": sorted(scenes),
        "scene_reports": rows,
        "trusted_registry": str(registry_path),
        "trusted_registry_sha256": registry_sha256,
        "frozen_contract": frozen_contract,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "next_gate": (
            "evaluate frozen benchmark protocols without changing capacity or rules"
            if status == "cross_scene_query_free_candidate_confirmed"
            else "retain the control and do not open benchmark queries for tuning"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-reports", required=True)
    parser.add_argument("--trusted-registry", required=True)
    parser.add_argument("--trusted-registry-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = confirm(
        _paths(args.scene_reports),
        trusted_registry=args.trusted_registry,
        expected_registry_sha256=args.trusted_registry_sha256,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"confirmation output already exists: {output}")
    write_frozen_json(output, report)
    print(report)


if __name__ == "__main__":
    main()
