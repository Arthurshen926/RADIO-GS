"""Mine source-only generic-query hard negatives for one sealed clean scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.evaluation import source_query_response_hard_negatives as interface
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
from radio_gs.scripts.eval_text_response_fidelity_gate import (
    load_text_embedding_bank,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    load_fit_text_embedding_bank,
)
from radio_gs.utils.immutable_artifacts import load_torch_payload, sha256_file


PREREGISTRATION = Path(
    "paper/artifacts/clean_scene_source_query_response_hard_negative_preregistration_20260806.json"
)


def _require_sha(value: str, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _required_file(path: str | Path, expected_sha256: str, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    expected = _require_sha(expected_sha256, label=label)
    if sha256_file(source) != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return source


def _load_mapping(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    value, _, _ = load_torch_payload(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label=label,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(value)


def _distribution(value: torch.Tensor) -> dict[str, float | int]:
    array = torch.as_tensor(value).detach().double().cpu().numpy()
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("reported distribution must be finite and nonempty")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _bank_record(bank: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "path": str(bank["path"]),
        "sha256": str(bank["file_sha256"]),
        "manifest_path": str(bank["manifest_path"]),
        "manifest_sha256": str(bank["manifest_sha256"]),
        "queries": int(
            bank.get("query_count", len(bank.get("query_ids", [])))
        ),
        "ordered_records_sha256": str(bank["ordered_records_sha256"]),
        "vocabulary_sha256": str(bank["vocabulary_sha256"]),
        "embedding_tensor_sha256": str(bank["embedding_tensor_sha256"]),
        "embedding_semantic_sha256": str(bank["embedding_semantic_sha256"]),
        "algorithm_version": str(
            bank.get("algorithm_version", "imagenet1k-primary-v1")
        ),
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
    }


def _negative_metrics(channels: Mapping[str, torch.Tensor], *, regions: int) -> dict[str, Any]:
    anchors = channels["anchor_region_indices"]
    codes = channels["source_codes"]
    teacher_selected = (codes & 1) > 0
    response_selected = (codes & 2) > 0
    counts = torch.bincount(anchors, minlength=regions)
    code_counts = Counter(int(value) for value in codes.tolist())
    return {
        "directed_pairs": int(anchors.numel()),
        "anchors": regions,
        "anchors_with_any_negative": int((counts > 0).sum()),
        "negatives_per_anchor": _distribution(counts.double()),
        "source_code_counts": {
            "teacher_similarity_only": int(code_counts[1]),
            "response_nearest_only": int(code_counts[2]),
            "both": int(code_counts[3]),
        },
        "teacher_cosine_all_pairs": _distribution(channels["teacher_cosines"]),
        "teacher_cosine_band_selected": _distribution(
            channels["teacher_cosines"][teacher_selected]
        ),
        "response_profile_cosine_all_pairs": _distribution(
            channels["response_profile_cosines"]
        ),
        "response_profile_cosine_selected": _distribution(
            channels["response_profile_cosines"][response_selected]
        ),
        "cross_scale_pair_fraction": float(
            (
                channels["anchor_scale_indices"]
                != channels["negative_scale_indices"]
            )
            .double()
            .mean()
        ),
        "zero_shared_active_token_pairs": int(anchors.numel()),
    }


def _atomic_torch_save(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if str(args.device) != "cpu":
        raise ValueError("this source-only diagnostic is frozen to CPU")
    output = Path(args.output).expanduser().resolve()
    report_output = Path(args.report_output).expanduser().resolve()
    if output.exists() or report_output.exists():
        raise FileExistsError("output and report must both be new paths")
    prereg = _required_file(
        PREREGISTRATION,
        args.expected_preregistration_sha256,
        label="preregistration",
    )
    accepted_path = _required_file(
        args.accepted_v2,
        args.expected_accepted_v2_sha256,
        label="AcceptedV2 authority",
    )
    teacher_path = _required_file(
        args.teacher,
        args.expected_teacher_sha256,
        label="official multiview teacher authority",
    )
    fit_path = _required_file(
        args.fit_text_bank,
        args.expected_fit_text_bank_sha256,
        label="fit text bank",
    )
    fit_manifest = _required_file(
        args.fit_text_bank_manifest,
        args.expected_fit_text_bank_manifest_sha256,
        label="fit text bank manifest",
    )
    dev_path = _required_file(
        args.dev_text_bank,
        args.expected_dev_text_bank_sha256,
        label="dev text bank",
    )
    dev_manifest = _required_file(
        args.dev_text_bank_manifest,
        args.expected_dev_text_bank_manifest_sha256,
        label="dev text bank manifest",
    )

    accepted_value = _load_mapping(
        accepted_path,
        args.expected_accepted_v2_sha256,
        label="AcceptedV2 authority",
    )
    teacher_value = _load_mapping(
        teacher_path,
        args.expected_teacher_sha256,
        label="official multiview teacher authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_value)
    teacher = shard.validate_teacher_observation_authority(teacher_value)
    shard.validate_teacher_accepted_sampling_alignment(teacher, accepted)
    if accepted["scene_id"] != args.scene_id or teacher["scene_id"] != args.scene_id:
        raise ValueError("scene identity differs from the CLI authority")
    if teacher["input_authority"]["accepted_region_authority_file_sha256"] != args.expected_accepted_v2_sha256:
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
    fit_record = _bank_record(fit_bank, split="fit")
    dev_record = _bank_record(dev_bank, split="dev")
    input_authority = {
        "preregistration": {
            "path": str(prereg),
            "sha256": sha256_file(prereg),
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
            "interface": {
                "path": str(Path(interface.__file__).resolve()),
                "sha256": sha256_file(Path(interface.__file__).resolve()),
            },
            "producer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
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
    _atomic_torch_save(authority, output)
    output_sha = sha256_file(output)
    reloaded_value = _load_mapping(
        output, output_sha, label="reloaded hard-negative authority"
    )
    reloaded = validate_negative_authority(
        reloaded_value,
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
    )
    if reloaded["content_authority_sha256"] != authority["content_authority_sha256"]:
        raise ValueError("reloaded hard-negative content authority differs")

    report = {
        "artifact": "clean_scene_source_query_response_hard_negative_results",
        "schema_version": 1,
        "status": "source_only_diagnostic_and_negative_authority_complete",
        "scene_id": args.scene_id,
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
            "text_queries_opened_field_semantics": (
                "legacy field meaning benchmark text queries only"
            ),
        },
        "teacher_targets": {
            "temperature": TEACHER_RESPONSE_TEMPERATURE,
            "consensus_descriptor_sha256": tensor_sha256(
                teacher_consensus_fit
            ),
            "fit_response_sha256": tensor_sha256(teacher_fit_response),
            "dev_response_sha256": tensor_sha256(teacher_dev_response),
            "view_count_distribution": _distribution(view_counts.double()),
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
        "hard_negative_metrics": _negative_metrics(
            authority["channels"], regions=region_count
        ),
        "output": {
            "path": str(output),
            "sha256": output_sha,
            "size_bytes": output.stat().st_size,
            "content_authority_sha256": authority[
                "content_authority_sha256"
            ],
        },
        "execution": {
            "device": "cpu",
            "benchmark_data_opened": False,
            "benchmark_metrics_computed": False,
            "generic_target_blind_text_bank_opened": True,
            "benchmark_text_queries_opened": False,
            "text_queries_opened": False,
            "text_queries_opened_field_semantics": (
                "legacy field meaning benchmark text queries only"
            ),
            "scene0002_skipped": True,
            "scene0002_reason": "official teacher and AcceptedV2 authority absent at preregistration",
        },
    }
    _atomic_json_save(report, report_output)
    return output, report_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
