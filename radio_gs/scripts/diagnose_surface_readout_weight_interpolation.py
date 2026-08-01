#!/usr/bin/env python3
"""CPU-only, no-selection Pareto diagnostic for Surface readout interpolation.

The interpolation grid is frozen in source.  Surface and target-blind ``fit``
metrics form the query-free diagnostic Pareto front.  Held-out ``dev`` is
loaded and reported only after that grid is fixed; it is explicitly excluded
from Pareto dominance, alpha selection, promotion, and checkpoint creation.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import evaluate_response_fidelity
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import materialize_surface_text_response_descriptors as materializer
from radio_gs.scripts.eval_text_response_fidelity_gate import (
    load_text_embedding_bank,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    compute_query_free_response_selection_metrics,
    load_fit_text_embedding_bank,
)
from radio_gs.scripts.train_surface_region_summary_readout import _targets
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_surface_region_summary_readout_v2,
    sha256_file,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_readout_weight_interpolation_pareto_diagnostic"
FIXED_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
REQUIRED_SEEDS = (0, 1, 2)
COMMON_TRAINING_FIELDS = (
    "hidden_dim",
    "epochs",
    "patience",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "token_weight",
    "relation_weight",
    "reliability_attention_mode",
    "context_pooling_mode",
    "canonical_noise_degrees",
    "canonical_noise_calibration",
    "seed",
)
PARETO_OBJECTIVES = (
    ("surface.summary_token_cosine", "maximize"),
    ("surface.mean_descriptor_cosine", "maximize"),
    ("surface.all_view_descriptor_cosine", "maximize"),
    ("surface.relation_fidelity", "maximize"),
    ("fit.text_support_top1_agreement", "maximize"),
    ("fit.text_response_smooth_l1", "minimize"),
    ("fit.descriptor_relation_smooth_l1", "minimize"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_regular_file(raw: str | Path, *, label: str) -> Path:
    value = Path(raw)
    _require(value.is_absolute(), f"{label} must be an absolute path")
    resolved = value.resolve(strict=True)
    _require(value == resolved, f"{label} must be canonical and non-symlinked")
    _require(resolved.is_file(), f"{label} must be a regular file")
    return resolved


def _cpu_only_preflight() -> None:
    # Requiring an explicit empty CUDA namespace prevents an accidental GPU
    # context even if a future dependency starts probing CUDA during import.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(
        visible in {"", "-1"},
        "set CUDA_VISIBLE_DEVICES='' (or -1) for this CPU-only diagnostic",
    )


def _normalized_cache_provenance(value: object) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "checkpoint cache provenance is missing")
    result = dict(value)
    result.pop("cache_bindings", None)
    return result


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    value, _, source = load_json_object(path, label=label)
    _require(source == path, f"{label} immutable loader resolved another path")
    return value


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _control_checkpoint_authority_index(
    manifest_path: Path,
) -> dict[Path, dict[str, Any]]:
    """Read externally declared control hashes before opening checkpoints."""

    manifest = _json_object(manifest_path, label="control attention screen")
    _require(
        manifest.get("artifact_type")
        == "surface_c1024_attention_pooling_postcache_continuation"
        and manifest.get("selected_variant") == "joint_attention_v1"
        and manifest.get("selection_status") == "joint_attention_retained"
        and manifest.get("promotion_gate_passed") is False,
        "control attention screen does not freeze the selected joint variant",
    )
    variants = manifest.get("variants")
    selected = variants.get("joint_attention_v1") if isinstance(variants, Mapping) else None
    rows = selected.get("seeds") if isinstance(selected, Mapping) else None
    _require(isinstance(rows, list) and len(rows) == 3, "control screen seed index differs")
    result: dict[Path, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "control screen contains an invalid seed row")
        seed = row.get("seed")
        checkpoint = row.get("checkpoint")
        _require(
            seed in REQUIRED_SEEDS
            and isinstance(checkpoint, Mapping)
            and set(checkpoint) == {"path", "sha256"}
            and _is_sha256(checkpoint.get("sha256")),
            "control screen checkpoint binding is invalid",
        )
        path = _canonical_regular_file(checkpoint["path"], label="bound control checkpoint")
        _require(path not in result, "control screen repeats a checkpoint")
        result[path] = {"seed": int(seed), "sha256": str(checkpoint["sha256"])}
    _require(
        {record["seed"] for record in result.values()} == set(REQUIRED_SEEDS),
        "control screen does not exactly cover seeds 0/1/2",
    )
    return result


def _candidate_checkpoint_authority_index(
    run_manifest: Path,
) -> tuple[Path, dict[Path, dict[str, Any]]]:
    """Use the run-bound completion as the external candidate SHA authority."""

    completion_path = _canonical_regular_file(
        run_manifest.parent / "text_response_distill.complete",
        label="candidate distill completion",
    )
    completion = _json_object(completion_path, label="candidate distill completion")
    run_binding = completion.get("run_manifest")
    _require(
        completion.get("schema_version") == 2
        and completion.get("artifact_type")
        == "surface_region_text_response_distill_completion"
        and completion.get("status") == "complete_three_seed_guarded_authority"
        and isinstance(run_binding, Mapping)
        and run_binding
        == {"path": str(run_manifest), "sha256": sha256_file(run_manifest)},
        "candidate completion does not bind the supplied run manifest",
    )
    rows = completion.get("seeds")
    _require(isinstance(rows, list) and len(rows) == 3, "candidate completion seed index differs")
    result: dict[Path, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "candidate completion contains an invalid seed row")
        seed = row.get("seed")
        checkpoint = row.get("checkpoint")
        report = row.get("report")
        _require(
            seed in REQUIRED_SEEDS
            and isinstance(checkpoint, Mapping)
            and set(checkpoint) == {"path", "sha256"}
            and _is_sha256(checkpoint.get("sha256"))
            and isinstance(report, Mapping)
            and set(report) == {"path", "sha256"}
            and _is_sha256(report.get("sha256")),
            "candidate completion checkpoint/report binding is invalid",
        )
        path = _canonical_regular_file(checkpoint["path"], label="bound candidate checkpoint")
        report_path = _canonical_regular_file(report["path"], label="bound candidate report")
        _require(
            report_path == path.with_suffix(path.suffix + ".json")
            and sha256_file(report_path) == report["sha256"],
            "candidate completion report path/SHA differs",
        )
        _require(path not in result, "candidate completion repeats a checkpoint")
        result[path] = {
            "seed": int(seed),
            "sha256": str(checkpoint["sha256"]),
            "report": {"path": str(report_path), "sha256": str(report["sha256"])},
        }
    _require(
        {record["seed"] for record in result.values()} == set(REQUIRED_SEEDS),
        "candidate completion does not exactly cover seeds 0/1/2",
    )
    return completion_path, result


def _validate_common_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_seed: int,
    cache_meta: Mapping[str, Any],
    radio_path: Path,
    radio_sha256: str,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    model, payload, checkpoint_sha, source = load_surface_region_summary_readout_v2(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
    )
    _require(source == path, "checkpoint immutable loader resolved another path")
    provenance = payload.get("provenance")
    config = payload.get("training_config")
    architecture = payload.get("architecture")
    _require(isinstance(provenance, Mapping), "checkpoint lacks provenance")
    _require(isinstance(config, Mapping), "checkpoint lacks training_config")
    _require(isinstance(architecture, Mapping), "checkpoint lacks architecture")
    _require(provenance.get("uses_benchmark_scenes") is False, "checkpoint used benchmark scenes")
    _require(
        provenance.get("uses_benchmark_test_vocabulary") is False,
        "checkpoint used benchmark vocabulary",
    )
    _require(provenance.get("scene_disjoint") is True, "checkpoint is not scene disjoint")
    _require(
        provenance.get("custom_text_projection") is False,
        "checkpoint uses a custom text projection",
    )
    seed = config.get("seed")
    _require(
        seed == expected_seed and seed in REQUIRED_SEEDS,
        "checkpoint seed differs from its external authority",
    )
    _require(
        provenance.get("random_seed_contract")
        == {
            "seed": seed,
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        },
        "checkpoint random-seed contract differs",
    )
    _require(
        architecture.get("name") == "surface_region_summary_readout_v2"
        and architecture.get("contract_sha256")
        == cache_meta["region_contract_sha256"],
        "checkpoint architecture/cache contract differs",
    )
    _require(
        provenance.get("region_contract_sha256")
        == cache_meta["region_contract_sha256"]
        and provenance.get("region_contract") == cache_meta["region_contract"],
        "checkpoint/cache SurfaceRegion contract differs",
    )
    validation = provenance.get("validation")
    _require(
        _normalized_cache_provenance(validation)
        == _normalized_cache_provenance(cache_meta["checkpoint_validation"]),
        "checkpoint binds different validation caches",
    )
    train = provenance.get("train")
    _require(isinstance(train, Mapping), "checkpoint lacks train provenance")
    _require(
        not (set(train.get("scenes", [])) & set(cache_meta["scenes"])),
        "checkpoint train/validation scenes overlap",
    )
    _require(
        radio_sha256 == cache_meta["radio_checkpoint_sha256"]
        and str(config.get("radio_checkpoint")) == str(radio_path),
        "RADIO checkpoint/cache/readout provenance differs",
    )
    model.cpu().eval().requires_grad_(False)
    return model, payload, checkpoint_sha


def _validate_control_checkpoint(
    path: Path,
    *,
    binding_manifest: Path,
    expected_sha256: str,
    expected_seed: int,
    cache_meta: Mapping[str, Any],
    radio_path: Path,
    radio_sha256: str,
) -> dict[str, Any]:
    model, payload, checkpoint_sha = _validate_common_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_seed=expected_seed,
        cache_meta=cache_meta,
        radio_path=radio_path,
        radio_sha256=radio_sha256,
    )
    _require(
        not isinstance(payload["provenance"].get("text_response_distillation"), Mapping),
        "control checkpoint unexpectedly contains text-response distillation",
    )
    seed = int(payload["training_config"]["seed"])
    report_path = path.with_suffix(path.suffix + ".json")
    authority = materializer._validate_attention_postcache_binding(
        binding_manifest,
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha,
        report_path=report_path,
        seed=seed,
        cache_meta=cache_meta,
    )
    report = materializer._validate_legacy_report(
        report_path,
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint=payload,
        cache_meta=cache_meta,
    )
    return {
        "model": model,
        "payload": payload,
        "seed": seed,
        "path": path,
        "sha256": checkpoint_sha,
        "report": report,
        "authority": authority,
    }


def _validate_candidate_checkpoint(
    path: Path,
    *,
    run_manifest: Path,
    expected_sha256: str,
    expected_seed: int,
    cache_meta: Mapping[str, Any],
    radio_path: Path,
    radio_sha256: str,
    fit_bank: Mapping[str, Any],
) -> dict[str, Any]:
    model, payload, checkpoint_sha = _validate_common_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_seed=expected_seed,
        cache_meta=cache_meta,
        radio_path=radio_path,
        radio_sha256=radio_sha256,
    )
    seed = int(payload["training_config"]["seed"])
    provenance = payload["provenance"]
    distillation = provenance.get("text_response_distillation")
    _require(isinstance(distillation, Mapping), "candidate lacks text-response distillation")
    expected_fit = distillation.get("fit_text_bank")
    _require(isinstance(expected_fit, Mapping), "candidate lacks fit-bank provenance")
    _require(
        expected_fit.get("artifact_path") == str(fit_bank["path"])
        and expected_fit.get("artifact_sha256") == fit_bank["artifact_sha256"]
        and expected_fit.get("manifest_path") == str(fit_bank["manifest_path"])
        and expected_fit.get("manifest_sha256") == fit_bank["manifest_sha256"],
        "candidate binds a different fit text bank",
    )
    report_path = path.with_suffix(path.suffix + ".json")
    bound_run = materializer._validate_distill_run_manifest(
        provenance.get("distill_run_manifest"),
        checkpoint_path=path,
        report_path=report_path,
        seed=seed,
        cache_meta=cache_meta,
        radio_path=radio_path,
        radio_sha256=radio_sha256,
    )
    _require(
        bound_run["path"] == str(run_manifest)
        and bound_run["sha256"] == sha256_file(run_manifest),
        "candidate binds another distill run manifest",
    )
    report = materializer._validate_checkpoint_report(
        report_path,
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint=payload,
        cache_meta=cache_meta,
        run_manifest=bound_run,
    )
    return {
        "model": model,
        "payload": payload,
        "seed": seed,
        "path": path,
        "sha256": checkpoint_sha,
        "report": report,
        "authority": bound_run,
    }


def assert_interpolable(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless two same-seed endpoint payloads share one state space."""

    control_config = control.get("training_config")
    candidate_config = candidate.get("training_config")
    _require(
        isinstance(control_config, Mapping) and isinstance(candidate_config, Mapping),
        "interpolation endpoints lack training configs",
    )
    for field in COMMON_TRAINING_FIELDS:
        _require(
            control_config.get(field) == candidate_config.get(field),
            f"endpoint training field {field} differs",
        )
    _require(
        control.get("architecture") == candidate.get("architecture"),
        "endpoint architectures differ",
    )
    control_state = control.get("state_dict")
    candidate_state = candidate.get("state_dict")
    _require(
        isinstance(control_state, Mapping) and isinstance(candidate_state, Mapping),
        "endpoint state_dict is missing",
    )
    _require(
        list(control_state) == list(candidate_state),
        "endpoint state_dict key order differs",
    )
    tensor_records = []
    for name in control_state:
        left = control_state[name]
        right = candidate_state[name]
        _require(
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor),
            f"state {name} is not tensor-valued",
        )
        _require(left.device.type == right.device.type == "cpu", f"state {name} is not on CPU")
        _require(left.shape == right.shape, f"state {name} shape differs")
        _require(left.dtype == right.dtype, f"state {name} dtype differs")
        _require(left.is_floating_point(), f"state {name} is not floating point")
        _require(
            bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),
            f"state {name} is non-finite",
        )
        tensor_records.append(
            {"name": name, "shape": list(left.shape), "dtype": str(left.dtype)}
        )
    return {
        "architecture": copy.deepcopy(control["architecture"]),
        "seed": int(control_config["seed"]),
        "tensor_count": len(tensor_records),
        "tensors": tensor_records,
    }


def interpolate_state_dict(
    control: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    value = float(alpha)
    _require(math.isfinite(value) and 0.0 <= value <= 1.0, "alpha must be in [0,1]")
    _require(list(control) == list(candidate), "state_dict keys differ")
    return {
        name: control[name].mul(1.0 - value).add(candidate[name], alpha=value)
        for name in control
    }


def _relation_smooth_l1(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> float:
    student = F.normalize(torch.as_tensor(student).float(), dim=-1, eps=1e-8)
    teacher = F.normalize(torch.as_tensor(teacher).float(), dim=-1, eps=1e-8)
    _require(student.shape == teacher.shape and student.ndim == 2, "relation inputs differ")
    count = int(student.shape[0])
    _require(count > 0, "relation inputs are empty")
    total = 0.0
    for start in range(0, count, int(chunk_size)):
        predicted = student[start : start + int(chunk_size)] @ student.T
        target = teacher[start : start + int(chunk_size)] @ teacher.T
        total += float(F.smooth_l1_loss(predicted, target, reduction="sum"))
    return total / float(count * count)


@torch.inference_mode()
def _evaluate_alpha(
    model: torch.nn.Module,
    head: torch.nn.Module,
    data: Mapping[str, Any],
    *,
    fit_embeddings: torch.Tensor,
    dev_bank: Mapping[str, Any],
    scene_ids: Sequence[str],
    region_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    token_cosines: list[float] = []
    descriptor_cosines: list[float] = []
    all_view_cosines: list[float] = []
    students: list[torch.Tensor] = []
    teachers: list[torch.Tensor] = []
    row_count = len(data["radio_features"])
    for start in range(0, row_count, int(batch_size)):
        rows = torch.arange(start, min(start + int(batch_size), row_count))
        target_token, teacher, all_descriptors, teacher_mask = _targets(data, rows)
        predicted_token = model(
            data["radio_features"][rows],
            data["geometry"][rows],
            anchor_index=data["anchor_index"][rows],
            token_mask=data["token_mask"][rows],
            reliability=data["reliability"][rows],
        )
        student = F.normalize(
            head(predicted_token[:, None])[:, 0].float(),
            dim=-1,
            eps=1e-8,
        ).cpu()
        token_cosines.extend(
            F.cosine_similarity(predicted_token.cpu(), target_token, dim=-1).tolist()
        )
        descriptor_cosines.extend(
            F.cosine_similarity(student, teacher, dim=-1).tolist()
        )
        pair = torch.einsum("bd,bvd->bv", student, all_descriptors)
        all_view_cosines.extend(pair[teacher_mask].tolist())
        students.append(student)
        teachers.append(teacher)
    student = torch.cat(students).contiguous()
    teacher = torch.cat(teachers).contiguous()
    surface = {
        "summary_token_cosine": sum(token_cosines) / len(token_cosines),
        "mean_descriptor_cosine": sum(descriptor_cosines) / len(descriptor_cosines),
        "all_view_descriptor_cosine": sum(all_view_cosines) / len(all_view_cosines),
        "relation_fidelity": 1.0 - _relation_smooth_l1(student, teacher),
    }
    surface["selection_score"] = 0.5 * (
        surface["mean_descriptor_cosine"]
        + surface["all_view_descriptor_cosine"]
    )
    fit = compute_query_free_response_selection_metrics(
        student,
        teacher,
        fit_embeddings,
        scene_ids=scene_ids,
    )
    dev_full = evaluate_response_fidelity(
        student,
        teacher,
        dev_bank["embeddings"],
        scene_ids=scene_ids,
        region_ids=region_ids,
        query_ids=dev_bank["query_ids"],
    )
    dev = {
        "usage": "posthoc_only_excluded_from_pareto_and_selection",
        "counts": dev_full["counts"],
        "aggregate": dev_full["aggregate"],
        "student_response_sha256": dev_full["student_response_sha256"],
        "teacher_response_sha256": dev_full["teacher_response_sha256"],
    }
    return {"surface": surface, "fit": fit, "dev_posthoc": dev}


def _nested_metric(row: Mapping[str, Any], path: str) -> float:
    value: object = row
    for name in path.split("."):
        _require(isinstance(value, Mapping), f"metric path {path} is missing")
        value = value.get(name)
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"metric {path} is non-finite",
    )
    return float(value)


def pareto_front_alphas(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return the no-dev nondominated alpha grid in original fixed order."""

    front: list[float] = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            weakly_better = True
            strictly_better = False
            for metric, direction in PARETO_OBJECTIVES:
                left = _nested_metric(row, metric)
                right = _nested_metric(other, metric)
                if direction == "maximize":
                    weakly_better &= right >= left
                    strictly_better |= right > left
                else:
                    weakly_better &= right <= left
                    strictly_better |= right < left
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(float(row["alpha"]))
    return front


def _mean_metric(rows: Sequence[Mapping[str, Any]], path: str) -> float:
    return sum(_nested_metric(row, path) for row in rows) / len(rows)


def _aggregate_seed_rows(per_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for alpha in FIXED_ALPHAS:
        matches = [
            point
            for seed_row in per_seed
            for point in seed_row["points"]
            if float(point["alpha"]) == alpha
        ]
        _require(len(matches) == len(REQUIRED_SEEDS), "alpha does not cover all seeds")
        surface_fields = tuple(matches[0]["surface"])
        fit_fields = tuple(matches[0]["fit"])
        dev_fields = tuple(matches[0]["dev_posthoc"]["aggregate"])
        result.append(
            {
                "alpha": alpha,
                "surface": {
                    name: _mean_metric(matches, f"surface.{name}")
                    for name in surface_fields
                },
                "fit": {
                    name: _mean_metric(matches, f"fit.{name}")
                    for name in fit_fields
                },
                "dev_posthoc": {
                    "usage": "posthoc_only_excluded_from_pareto_and_selection",
                    "aggregate_seed_mean": {
                        name: _mean_metric(matches, f"dev_posthoc.aggregate.{name}")
                        for name in dev_fields
                    },
                },
            }
        )
    return result


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    _cpu_only_preflight()
    _require(int(args.batch_size) > 0, "batch_size must be positive")
    validation_paths = [
        _canonical_regular_file(value, label="validation cache")
        for value in args.validation_cache
    ]
    radio_path = _canonical_regular_file(args.radio_checkpoint, label="RADIO checkpoint")
    control_manifest = _canonical_regular_file(
        args.control_binding_manifest,
        label="control binding manifest",
    )
    candidate_manifest = _canonical_regular_file(
        args.candidate_run_manifest,
        label="candidate run manifest",
    )
    fit_path = _canonical_regular_file(args.fit_text_bank, label="fit text bank")
    fit_manifest = _canonical_regular_file(
        args.fit_text_bank_manifest,
        label="fit text bank manifest",
    )
    dev_path = _canonical_regular_file(args.dev_text_bank, label="dev text bank")
    dev_manifest = _canonical_regular_file(
        args.dev_text_bank_manifest,
        label="dev text bank manifest",
    )
    controls = [
        _canonical_regular_file(value, label="control checkpoint")
        for value in args.control_checkpoint
    ]
    candidates = [
        _canonical_regular_file(value, label="candidate checkpoint")
        for value in args.candidate_checkpoint
    ]
    _require(
        len(controls) == len(candidates) == len(REQUIRED_SEEDS),
        "control/candidate checkpoints must each exactly cover three seeds",
    )
    control_authority = _control_checkpoint_authority_index(control_manifest)
    candidate_completion, candidate_authority = _candidate_checkpoint_authority_index(
        candidate_manifest
    )
    _require(
        set(controls) == set(control_authority),
        "supplied controls differ from the exact attention-screen checkpoint set",
    )
    _require(
        set(candidates) == set(candidate_authority),
        "supplied candidates differ from the exact distill-completion checkpoint set",
    )

    # Caches, head, fit bank, and dev bank are each loaded exactly once and
    # reused for every seed/alpha evaluation.
    data, cache_meta = materializer._load_validation_caches(
        validation_paths,
        include_summary_tokens=True,
    )
    radio_sha = sha256_file(radio_path)
    _require(
        radio_sha == cache_meta["radio_checkpoint_sha256"],
        "RADIO checkpoint differs from validation-cache provenance",
    )
    fit_bank = load_fit_text_embedding_bank(fit_path, fit_manifest)
    fit_bank = {
        **fit_bank,
        "path": fit_path,
        "artifact_sha256": sha256_file(fit_path),
        "manifest_path": fit_manifest,
        "manifest_sha256": sha256_file(fit_manifest),
    }
    dev_bank = load_text_embedding_bank(dev_path, dev_manifest, "dev")
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).cpu().eval()
    head.requires_grad_(False)

    control_by_seed: dict[int, dict[str, Any]] = {}
    for path in controls:
        external = control_authority[path]
        record = _validate_control_checkpoint(
            path,
            binding_manifest=control_manifest,
            expected_sha256=external["sha256"],
            expected_seed=external["seed"],
            cache_meta=cache_meta,
            radio_path=radio_path,
            radio_sha256=radio_sha,
        )
        _require(record["seed"] not in control_by_seed, "duplicate control seed")
        control_by_seed[record["seed"]] = record
    candidate_by_seed: dict[int, dict[str, Any]] = {}
    for path in candidates:
        external = candidate_authority[path]
        record = _validate_candidate_checkpoint(
            path,
            run_manifest=candidate_manifest,
            expected_sha256=external["sha256"],
            expected_seed=external["seed"],
            cache_meta=cache_meta,
            radio_path=radio_path,
            radio_sha256=radio_sha,
            fit_bank=fit_bank,
        )
        _require(record["seed"] not in candidate_by_seed, "duplicate candidate seed")
        candidate_by_seed[record["seed"]] = record
    _require(
        set(control_by_seed) == set(candidate_by_seed) == set(REQUIRED_SEEDS),
        "endpoint checkpoints do not exactly cover seeds 0/1/2",
    )

    per_seed = []
    for seed in REQUIRED_SEEDS:
        control = control_by_seed[seed]
        candidate = candidate_by_seed[seed]
        compatibility = assert_interpolable(control["payload"], candidate["payload"])
        model = control["model"]
        points = []
        for alpha in FIXED_ALPHAS:
            state = interpolate_state_dict(
                control["payload"]["state_dict"],
                candidate["payload"]["state_dict"],
                alpha,
            )
            model.load_state_dict(state, strict=True)
            metrics = _evaluate_alpha(
                model,
                head,
                data,
                fit_embeddings=fit_bank["embeddings"],
                dev_bank=dev_bank,
                scene_ids=cache_meta["scene_ids"],
                region_ids=cache_meta["region_ids"],
                batch_size=int(args.batch_size),
            )
            points.append({"alpha": alpha, **metrics})
        per_seed.append(
            {
                "seed": seed,
                "control": {
                    "path": str(control["path"]),
                    "sha256": control["sha256"],
                    "report": control["report"],
                    "authority": control["authority"],
                },
                "candidate": {
                    "path": str(candidate["path"]),
                    "sha256": candidate["sha256"],
                    "report": candidate["report"],
                    "authority": candidate["authority"],
                },
                "interpolation_compatibility": compatibility,
                "points": points,
            }
        )
    aggregate = _aggregate_seed_rows(per_seed)
    pareto = pareto_front_alphas(aggregate)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "diagnostic_only_no_alpha_selected_no_checkpoint_created",
        "device": "cpu",
        "fixed_alphas": list(FIXED_ALPHAS),
        "interpolation": {
            "formula": "theta_alpha=(1-alpha)*theta_control+alpha*theta_candidate",
            "pairing": "same_seed_only",
            "state_contract": "exact_architecture_key_order_shape_dtype_finite_fp",
            "endpoint_meaning": {"alpha_0": "control", "alpha_1": "candidate"},
        },
        "selection_contract": {
            "alpha_grid_fixed_before_heldout_dev_load": True,
            "selected_alpha": None,
            "checkpoint_emitted": False,
            "promotion_decision_emitted": False,
            "pareto_sources": ["query_free_surface_validation", "target_blind_fit"],
            "dev_usage": "posthoc_only_excluded_from_pareto_selection_training_and_promotion",
            "audit_opened": False,
        },
        "input_bindings": {
            "validation_caches": cache_meta["caches"],
            "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
            "control_binding_manifest": {
                "path": str(control_manifest),
                "sha256": sha256_file(control_manifest),
            },
            "candidate_run_manifest": {
                "path": str(candidate_manifest),
                "sha256": sha256_file(candidate_manifest),
            },
            "candidate_completion": {
                "path": str(candidate_completion),
                "sha256": sha256_file(candidate_completion),
            },
            "fit_text_bank": {
                "path": str(fit_path),
                "sha256": fit_bank["artifact_sha256"],
                "manifest_path": str(fit_manifest),
                "manifest_sha256": fit_bank["manifest_sha256"],
                "query_count": fit_bank["query_count"],
                "embedding_tensor_sha256": fit_bank["embedding_tensor_sha256"],
            },
            "dev_text_bank": {
                "path": str(dev_path),
                "sha256": dev_bank["file_sha256"],
                "manifest_path": str(dev_manifest),
                "manifest_sha256": dev_bank["manifest_sha256"],
                "query_count": len(dev_bank["query_ids"]),
                "embedding_tensor_sha256": dev_bank["embedding_tensor_sha256"],
            },
        },
        "pareto_contract": {
            "objectives": [
                {"metric": metric, "direction": direction}
                for metric, direction in PARETO_OBJECTIVES
            ],
            "dev_metrics_included": False,
            "front_alphas": pareto,
            "interpretation": "nondominated_diagnostic_set_not_a_selection",
        },
        "per_seed": per_seed,
        "aggregate_seed_mean": aggregate,
    }
    payload["diagnostic_contract_sha256"] = canonical_json_sha256(
        {
            "artifact_type": ARTIFACT_TYPE,
            "fixed_alphas": list(FIXED_ALPHAS),
            "pareto_objectives": list(PARETO_OBJECTIVES),
            "selection_contract": payload["selection_contract"],
        }
    )
    output = Path(args.output)
    _require(output.is_absolute(), "output must be an absolute path")
    write_frozen_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", action="append", required=True)
    parser.add_argument("--control-checkpoint", action="append", required=True)
    parser.add_argument("--candidate-checkpoint", action="append", required=True)
    parser.add_argument("--control-binding-manifest", required=True)
    parser.add_argument("--candidate-run-manifest", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--fit-text-bank-manifest", required=True)
    parser.add_argument("--dev-text-bank", required=True)
    parser.add_argument("--dev-text-bank-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = diagnose(args)
    print(
        {
            "output": str(Path(args.output).resolve()),
            "status": payload["status"],
            "pareto_front_alphas": payload["pareto_contract"]["front_alphas"],
        }
    )


if __name__ == "__main__":
    main()
