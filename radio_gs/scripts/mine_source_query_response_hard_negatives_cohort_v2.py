"""Mine one frozen-cohort scene's source-only response hard negatives.

This V2 entrypoint keeps the V1 scene0001 producer and authority immutable.  It
reuses the frozen pure mining algorithm, but derives scene membership and split
from the caller-SHA-bound 24-train/8-validation cohort before opening any scene
or text-bank authority.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation import source_query_response_hard_negatives as algorithm
from radio_gs.evaluation.source_query_response_hard_negatives import (
    MATRIX_BLOCK_ROWS,
    PER_SOURCE_K,
    TEACHER_COSINE_MAX,
    TEACHER_COSINE_MIN,
    TEACHER_RESPONSE_TEMPERATURE,
    build_multiview_teacher_targets,
    build_negative_authority,
    evaluate_source_query_response,
    mine_scene_global_hard_negatives,
    validate_negative_authority,
)
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import mine_source_query_response_hard_negatives as v1_miner
from radio_gs.scripts import train_surface_region_full_scalar_residual as cohort_api
from radio_gs.scripts.eval_text_response_fidelity_gate import (
    load_text_embedding_bank,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    load_fit_text_embedding_bank,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
)


PREREGISTRATION = Path(__file__).resolve().parents[2] / (
    "paper/artifacts/clean_scene_source_query_response_hard_negative_cohort_v2_"
    "preregistration_20260806.json"
)
PREREGISTRATION_ARTIFACT = (
    "clean_scene_source_query_response_hard_negative_cohort_v2_preregistration"
)
PREREGISTRATION_SCHEMA_VERSION = 2
V1_MINER_SHA256 = "06ba0089063a02eba29b5dc25fa930df686debc6aa0555427e725245c758840b"
ALGORITHM_SHA256 = "45fc9adae25dc90decf8c211de2cc6bac933f8a1576d6b8c84eee6c31515c184"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bound_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _repository_root() / path
    return path.resolve()


def _require_exact_record(
    value: object,
    *,
    label: str,
    require_authority_sha256: bool = False,
) -> dict[str, str]:
    required = {"path", "sha256"}
    if require_authority_sha256:
        required.add("authority_sha256")
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError(f"{label} record differs")
    digest = v1_miner._require_sha(value["sha256"], label=label)
    record = {"path": str(_bound_path(value["path"])), "sha256": digest}
    if require_authority_sha256:
        record["authority_sha256"] = v1_miner._require_sha(
            value["authority_sha256"], label=f"{label} content authority"
        )
    return record


def validate_preregistration(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("cohort V2 preregistration must be a mapping")
    prereg = dict(value)
    authorization = prereg.get("authorization", {})
    if (
        prereg.get("artifact") != PREREGISTRATION_ARTIFACT
        or prereg.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION
        or prereg.get("status")
        != "sealed_before_cohort_v2_miner_implementation_or_execution"
        or authorization.get("cohort_scene_source_only_mining_authorized") is not True
        or authorization.get("model_training_authorized") is not False
        or authorization.get("benchmark_execution_authorized") is not False
        or authorization.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("cohort V2 preregistration header differs")

    cohort_record = _require_exact_record(
        prereg.get("immutable_cohort"),
        label="immutable cohort",
        require_authority_sha256=True,
    )
    if (
        prereg["immutable_cohort"].get("source_train_scenes") != 24
        or prereg["immutable_cohort"].get("source_validation_scenes") != 8
        or prereg["immutable_cohort"].get("scene_and_physical_space_disjoint")
        is not True
    ):
        raise ValueError("cohort V2 preregistration split contract differs")

    frozen = prereg.get("frozen_algorithm", {})
    algorithm_record = _require_exact_record(frozen, label="frozen algorithm")
    if (
        algorithm_record["sha256"] != ALGORITHM_SHA256
        or frozen.get("teacher_response_temperature")
        != TEACHER_RESPONSE_TEMPERATURE
        or frozen.get("teacher_cosine_minimum_inclusive") != TEACHER_COSINE_MIN
        or frozen.get("teacher_cosine_maximum_inclusive") != TEACHER_COSINE_MAX
        or frozen.get("per_source_k") != PER_SOURCE_K
        or frozen.get("matrix_block_rows") != MATRIX_BLOCK_ROWS
        or frozen.get("cross_scene_similarity_matrix") is not False
        or frozen.get("per_scene_hyperparameters") is not False
    ):
        raise ValueError("cohort V2 frozen algorithm differs")
    if (
        algorithm_record["path"] != str(Path(algorithm.__file__).resolve())
        or sha256_file(algorithm_record["path"]) != ALGORITHM_SHA256
    ):
        raise ValueError("cohort V2 algorithm implementation changed")

    baseline = prereg.get("immutable_v1_baseline", {})
    v1_preregistration = _require_exact_record(
        baseline.get("preregistration"), label="V1 preregistration"
    )
    if (
        sha256_file(v1_preregistration["path"])
        != v1_preregistration["sha256"]
    ):
        raise ValueError("V1 preregistration is not byte-identical")
    v1_record = _require_exact_record(baseline.get("miner"), label="V1 miner")
    if (
        v1_record["sha256"] != V1_MINER_SHA256
        or v1_record["path"] != str(Path(v1_miner.__file__).resolve())
        or sha256_file(v1_record["path"]) != V1_MINER_SHA256
    ):
        raise ValueError("V1 miner is not byte-identical")
    scene0001 = _require_exact_record(
        baseline.get("scene0001_authority"), label="V1 scene0001 authority"
    )
    if sha256_file(scene0001["path"]) != scene0001["sha256"]:
        raise ValueError("V1 scene0001 authority is not byte-identical")

    access = prereg.get("source_access", {})
    if (
        access.get("generic_target_blind_text_bank_opened") is not True
        or any(
            access.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_labels_opened",
                "benchmark_masks_opened",
                "benchmark_queries_opened",
                "target_heldout_opened",
                "target_metrics_computed",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("cohort V2 preregistration source access differs")
    prereg["verified_cohort_record"] = cohort_record
    return prereg


def resolve_declared_scene(cohort: Mapping[str, Any], scene_id: str) -> str:
    """Return the frozen split for exactly one cohort scene."""

    scene = str(scene_id)
    train = [str(item) for item in cohort.get("source_train_scene_ids", [])]
    validation = [
        str(item) for item in cohort.get("source_validation_scene_ids", [])
    ]
    memberships = int(scene in train) + int(scene in validation)
    if memberships != 1:
        raise ValueError("scene_id is not a unique member of the frozen clean cohort")
    return "source_train" if scene in train else "source_validation"


def preflight_declared_scene(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, str], str]:
    """Resolve authorization before opening any scene or text-bank authority."""

    prereg_value, prereg_sha, prereg_path = load_json_object(
        PREREGISTRATION,
        expected_sha256=args.expected_preregistration_sha256,
        label="cohort V2 hard-negative preregistration",
    )
    prereg = validate_preregistration(prereg_value)
    expected_cohort = prereg["verified_cohort_record"]
    declared_cohort = {
        "path": str(_bound_path(args.cohort_authority)),
        "sha256": str(args.expected_cohort_authority_sha256),
    }
    if declared_cohort != {
        "path": expected_cohort["path"],
        "sha256": expected_cohort["sha256"],
    }:
        raise ValueError("caller cohort differs from cohort V2 preregistration")
    cohort, cohort_record = cohort_api.load_cohort_authority(
        declared_cohort["path"], expected_sha256=declared_cohort["sha256"]
    )
    if cohort.get("authority_sha256") != expected_cohort["authority_sha256"]:
        raise ValueError("cohort content authority differs from preregistration")
    split = resolve_declared_scene(cohort, args.scene_id)
    return (
        prereg,
        {"path": str(prereg_path), "sha256": prereg_sha},
        cohort,
        cohort_record,
        split,
    )


def execution_audit(scene_id: str, cohort_split: str) -> dict[str, Any]:
    if cohort_split not in {"source_train", "source_validation"}:
        raise ValueError("cohort split differs")
    return {
        "scene_id": str(scene_id),
        "cohort_split": cohort_split,
        "device": "cpu",
        "one_scene_at_a_time": True,
        "cross_scene_similarity_matrix": False,
        "benchmark_data_opened": False,
        "benchmark_metrics_computed": False,
        "generic_target_blind_text_bank_opened": True,
        "benchmark_text_queries_opened": False,
        "text_queries_opened": False,
        "text_queries_opened_field_semantics": (
            "legacy field meaning benchmark text queries only"
        ),
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if str(args.device) != "cpu":
        raise ValueError("cohort V2 source-only mining is frozen to CPU")
    output = Path(args.output).expanduser().resolve()
    report_output = Path(args.report_output).expanduser().resolve()
    if output.exists() or report_output.exists():
        raise FileExistsError("output and report must both be new paths")

    prereg, prereg_record, cohort, cohort_record, split = (
        preflight_declared_scene(args)
    )
    accepted_path = v1_miner._required_file(
        args.accepted_v2,
        args.expected_accepted_v2_sha256,
        label="AcceptedV2 authority",
    )
    teacher_path = v1_miner._required_file(
        args.teacher,
        args.expected_teacher_sha256,
        label="official multiview teacher authority",
    )
    fit_path = v1_miner._required_file(
        args.fit_text_bank,
        args.expected_fit_text_bank_sha256,
        label="fit text bank",
    )
    fit_manifest = v1_miner._required_file(
        args.fit_text_bank_manifest,
        args.expected_fit_text_bank_manifest_sha256,
        label="fit text bank manifest",
    )
    dev_path = v1_miner._required_file(
        args.dev_text_bank,
        args.expected_dev_text_bank_sha256,
        label="dev text bank",
    )
    dev_manifest = v1_miner._required_file(
        args.dev_text_bank_manifest,
        args.expected_dev_text_bank_manifest_sha256,
        label="dev text bank manifest",
    )

    accepted_value = v1_miner._load_mapping(
        accepted_path,
        args.expected_accepted_v2_sha256,
        label="AcceptedV2 authority",
    )
    teacher_value = v1_miner._load_mapping(
        teacher_path,
        args.expected_teacher_sha256,
        label="official multiview teacher authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_value)
    teacher = shard.validate_teacher_observation_authority(teacher_value)
    shard.validate_teacher_accepted_sampling_alignment(teacher, accepted)
    if accepted["scene_id"] != args.scene_id or teacher["scene_id"] != args.scene_id:
        raise ValueError("scene identity differs from the cohort declaration")
    if (
        teacher["input_authority"]["accepted_region_authority_file_sha256"]
        != args.expected_accepted_v2_sha256
    ):
        raise ValueError("teacher is not caller-bound to this AcceptedV2 file")
    for authority in (accepted, teacher):
        access = authority.get("source_access")
        if not isinstance(access, Mapping) or any(
            access.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_labels_opened",
                "benchmark_masks_opened",
                "benchmark_queries_opened",
                "target_heldout_opened",
                "text_queries_opened",
            )
        ):
            raise ValueError("scene authority is not sealed source-only")

    fit_bank = load_fit_text_embedding_bank(fit_path, fit_manifest)
    dev_bank = load_text_embedding_bank(dev_path, dev_manifest, "dev")
    text_authorities = prereg["target_blind_text_authorities"]
    for label, bank, bank_path, manifest_path, expected in (
        (
            "fit",
            fit_bank,
            fit_path,
            fit_manifest,
            text_authorities["mining_fit_embeddings"],
        ),
        (
            "dev",
            dev_bank,
            dev_path,
            dev_manifest,
            text_authorities["diagnostic_dev_embeddings"],
        ),
    ):
        if (
            bank_path != _bound_path(expected["path"])
            or manifest_path != _bound_path(expected["manifest_path"])
            or str(bank["file_sha256"]) != expected["sha256"]
            or str(bank["manifest_sha256"]) != expected["manifest_sha256"]
        ):
            raise ValueError(f"{label} text bank differs from preregistration")
    if fit_bank["vocabulary_sha256"] != dev_bank["vocabulary_sha256"]:
        raise ValueError("fit/dev text banks do not share the frozen vocabulary")
    fit_text = fit_bank["embeddings"].float().cpu().contiguous()
    dev_text = dev_bank["embeddings"].float().cpu().contiguous()
    if fit_text.shape[0] != 806 or dev_text.shape[0] != 101:
        raise ValueError("fit/dev query counts differ from preregistration")

    region_count = int(accepted["accepted_v2_e0"].shape[0])
    teacher_consensus_fit, teacher_fit_response, view_counts = (
        build_multiview_teacher_targets(
            teacher["pair_descriptors"],
            teacher["pair_region_indices"],
            fit_text,
            region_count=region_count,
        )
    )
    teacher_consensus_dev, teacher_dev_response, dev_view_counts = (
        build_multiview_teacher_targets(
            teacher["pair_descriptors"],
            teacher["pair_region_indices"],
            dev_text,
            region_count=region_count,
        )
    )
    if not torch.equal(view_counts, dev_view_counts) or not torch.equal(
        teacher_consensus_fit, teacher_consensus_dev
    ):
        raise ValueError("teacher consensus unexpectedly depends on the query bank")
    diagnostic = evaluate_source_query_response(
        accepted["accepted_v2_e0"],
        teacher_dev_response,
        dev_text,
        scale_indices=accepted["scale_indices"],
        view_counts=view_counts,
    )
    channels = mine_scene_global_hard_negatives(
        teacher_consensus_fit,
        teacher_fit_response,
        accepted["region_rows"],
        accepted["token_mask"],
        scale_indices=accepted["scale_indices"],
    )
    fit_record = v1_miner._bank_record(fit_bank, split="fit")
    dev_record = v1_miner._bank_record(dev_bank, split="dev")
    input_authority = {
        "preregistration": prereg_record,
        "cohort": {
            **cohort_record,
            "authority_sha256": cohort["authority_sha256"],
            "split": split,
        },
        "accepted_v2": {
            "path": str(accepted_path),
            "sha256": args.expected_accepted_v2_sha256,
            "channel_sha256": accepted["channel_sha256"],
        },
        "official_multiview_siglip2_teacher": {
            "path": str(teacher_path),
            "sha256": args.expected_teacher_sha256,
            "channel_sha256": teacher["channel_sha256"],
        },
        "fit_text_bank": fit_record,
        "dev_text_bank": dev_record,
        "implementation": {
            "algorithm": {
                "path": str(Path(algorithm.__file__).resolve()),
                "sha256": sha256_file(Path(algorithm.__file__).resolve()),
            },
            "producer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "immutable_v1_miner": {
                "path": str(Path(v1_miner.__file__).resolve()),
                "sha256": sha256_file(Path(v1_miner.__file__).resolve()),
            },
        },
    }
    authority = build_negative_authority(
        scene_id=args.scene_id,
        canonical_region_indices=accepted["canonical_region_indices"],
        region_fingerprints=accepted["region_fingerprints"],
        channels=channels,
        input_authority=input_authority,
    )
    validate_negative_authority(
        authority,
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
    )
    v1_miner._atomic_torch_save(authority, output)
    output_sha = sha256_file(output)
    reloaded_value = v1_miner._load_mapping(
        output, output_sha, label="reloaded cohort V2 hard-negative authority"
    )
    reloaded = validate_negative_authority(
        reloaded_value,
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
    )
    if reloaded["content_authority_sha256"] != authority["content_authority_sha256"]:
        raise ValueError("reloaded hard-negative content authority differs")

    report = {
        "artifact": "clean_scene_source_query_response_hard_negative_cohort_v2_results",
        "schema_version": 2,
        "status": "declared_cohort_scene_source_only_authority_complete",
        "scene_id": args.scene_id,
        "cohort_split": split,
        "input_authority": input_authority,
        "generic_text_bank_audit": {
            "fit": fit_record,
            "dev": dev_record,
            "fit_dev_query_disjoint": True,
            "canonical_family": "target_blind_imagenet1k_primary_v1",
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_vocabulary_for_construction": False,
            "generic_target_blind_text_bank_opened": True,
            "benchmark_text_queries_opened": False,
            "text_queries_opened": False,
        },
        "teacher_targets": {
            "temperature": TEACHER_RESPONSE_TEMPERATURE,
            "consensus_descriptor_sha256": tensor_sha256(teacher_consensus_fit),
            "fit_response_sha256": tensor_sha256(teacher_fit_response),
            "dev_response_sha256": tensor_sha256(teacher_dev_response),
            "view_count_distribution": v1_miner._distribution(view_counts.double()),
        },
        "response_diagnostic": diagnostic,
        "hard_negative_contract": {
            "teacher_cosine_interval_inclusive": [
                TEACHER_COSINE_MIN,
                TEACHER_COSINE_MAX,
            ],
            "per_source_k": PER_SOURCE_K,
            "matrix_block_rows": MATRIX_BLOCK_ROWS,
            "dense_cross_scene_matrix": False,
            "spatially_distinct": "zero_shared_active_primitive_tokens",
        },
        "hard_negative_metrics": v1_miner._negative_metrics(
            authority["channels"], regions=region_count
        ),
        "output": {
            "path": str(output),
            "sha256": output_sha,
            "size_bytes": output.stat().st_size,
            "content_authority_sha256": authority["content_authority_sha256"],
        },
        "execution": execution_audit(args.scene_id, split),
    }
    v1_miner._atomic_json_save(report, report_output)
    return output, report_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)
    parser.add_argument("--accepted-v2", required=True)
    parser.add_argument("--expected-accepted-v2-sha256", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--expected-fit-text-bank-sha256", required=True)
    parser.add_argument("--fit-text-bank-manifest", required=True)
    parser.add_argument("--expected-fit-text-bank-manifest-sha256", required=True)
    parser.add_argument("--dev-text-bank", required=True)
    parser.add_argument("--expected-dev-text-bank-sha256", required=True)
    parser.add_argument("--dev-text-bank-manifest", required=True)
    parser.add_argument("--expected-dev-text-bank-manifest-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser


def main() -> None:
    output, report = run(build_parser().parse_args())
    print(json.dumps({"output": str(output), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
