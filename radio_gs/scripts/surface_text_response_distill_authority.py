#!/usr/bin/env python3
"""Fail-closed authority artifacts for Surface text-response distillation.

The shell runner deliberately delegates immutable-state decisions here.  This
module never starts CUDA work: it owns no-follow locks, freezes and rechecks the
complete input/runtime closure, and validates each guarded seed terminal from
SHA-bound artifacts.  There is no generic pickle fallback.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import io
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from radio_gs.scripts.finalize_gpu_guard_receipt import TELEMETRY_COLUMNS, validate_receipt
from radio_gs.scripts.surface_gpu1_lock_supervisor import (
    GPU1_SINGLETON_ADDRESS,
    GPU1_SINGLETON_PROTOCOL,
    SINGLETON_FD_ENV,
    SINGLETON_PROTOCOL_ENV,
    _open_kernel_singleton,
    _singleton_protocol,
    verify_inherited_singleton,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    sha256_file,
    stable_descriptor_load,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 3
ARTIFACT_TYPE = "surface_region_text_response_distill_run"
SEED_TERMINAL_ARTIFACT_TYPE = "surface_text_response_distill_seed_terminal"
COMPLETION_ARTIFACT_TYPE = "surface_region_text_response_distill_completion"
GPU_CHECK_ARTIFACT_TYPE = "physical_gpu1_preflight"
REQUIRED_SEEDS = (0, 1, 2)
ALLOWED_CANDIDATES = {
    "control_c256_geometric",
    "context_c1024_geometric",
    "context_c1024_uniform",
    "core_c1024_geometric",
}
LEGACY_EPOCH_SELECTION = (
    "surface_control_0p002_fit_scene_robust_0p005_then_response_error_v3"
)
EPOCH_SELECTION = (
    "surface_control_0p002_fit_scene_robust_0p005_accepted_anchor_"
    "fixed_1over2048_then_response_error_v4"
)
PROPOSAL_ALPHA_NUMERATOR = 1
PROPOSAL_ALPHA_DENOMINATOR = 2048
HISTORY_HASH_CHAIN_ALGORITHM = (
    "sha256_canonical_json_previous_plus_record_without_selection_score_v1"
)
PROPOSAL_LOSS_MEASUREMENT_STATE = "raw_proposal_before_micro_projection"
PROPOSAL_LOSS_FIELDS = (
    "total",
    "token",
    "descriptor",
    "relation",
    "independent_response",
    "scene_response",
    "scene_profile",
    "scene_ranking",
)
LEGACY_FLAT_PROPOSAL_LOSS_FIELDS = {
    "total": "loss",
    "token": "token_loss",
    "descriptor": "descriptor_loss",
    "relation": "relation_loss",
    "independent_response": "independent_response_loss",
    "scene_response": "scene_response_loss",
    "scene_profile": "scene_profile_loss",
    "scene_ranking": "scene_ranking_loss",
}
PROPOSAL_STATE_MACHINE = {
    "name": "accepted_anchor_fixed_micro_ray_fresh_adamw_v1",
    "proposal_source": "current_accepted_anchor",
    "proposal_optimizer": "fresh_adamw_complete_epoch",
    "alpha_numerator": PROPOSAL_ALPHA_NUMERATOR,
    "alpha_denominator": PROPOSAL_ALPHA_DENOMINATOR,
    "trial_interpolation": "anchor+alpha*(raw-anchor)",
    "validation_evaluations_per_proposal": 1,
    "acceptance": "response_selection_feasible_is_exactly_true",
    "feasible_nonbest_action": "accept_as_next_anchor",
    "infeasible_action": "restore_exact_preproposal_anchor",
    "optimizer_moments": "reset_before_every_proposal",
    "best_selection": "existing_v3_robust_lexicographic_rank_control_and_trials",
    "patience": "consecutive_proposals_without_global_best_update",
    "persistent_generator": "advanced_across_proposals_never_rolled_back",
    "backtracking": "none_fixed_alpha_single_trial",
    "proposal_loss_accounting": {
        "measurement_state": PROPOSAL_LOSS_MEASUREMENT_STATE,
        "fields": list(PROPOSAL_LOSS_FIELDS),
        "legacy_flat_mirror": dict(LEGACY_FLAT_PROPOSAL_LOSS_FIELDS),
    },
}
SURFACE_CONTROL_NONINFERIORITY_TOLERANCE = 0.002
FIT_RESPONSE_NONINFERIORITY_TOLERANCE = 0.005
SURFACE_CONTROL_METRICS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)
FIT_RESPONSE_QUALITY_METRICS = (
    "text_response_profile_cosine_mean",
    "text_response_profile_cosine_p05",
    "text_response_ranking_spearman_mean",
    "text_response_ranking_spearman_p05",
    "text_response_top_decile_overlap_mean",
    "text_response_top_decile_overlap_p05",
)
FIT_RESPONSE_SCENE_QUALITY_METRICS = (
    "profile_cosine_mean",
    "profile_cosine_p05",
    "ranking_spearman_mean",
    "ranking_spearman_p05",
    "top_decile_overlap_mean",
    "top_decile_overlap_p05",
)
FIT_RESPONSE_SCENE_ERROR_METRICS = ("smooth_l1", "mae")
CALIBRATION_ALGORITHM_VERSION = (
    "per-seed-surface-warmstart-dual-response-pairwise-gradient-budget-v3"
)
INDEPENDENT_RESPONSE_LOSS = "independent_normalized_cosine_response_smooth_l1"
SCENE_RESPONSE_LOSS = (
    "scene_wise_text_response_weighted_profile_pairwise_gap_smooth_l1"
)
SCENE_PROFILE_LOSS = "scene_wise_centered_text_response_profile_cosine_distance"
SCENE_PAIRWISE_GAP_LOSS = "scene_wise_text_response_pairwise_gap_smooth_l1"
SCENE_RESPONSE_OBJECTIVE = {
    "name": SCENE_RESPONSE_LOSS,
    "profile_loss": SCENE_PROFILE_LOSS,
    "profile_weight": 0.2,
    "ranking_loss": SCENE_PAIRWISE_GAP_LOSS,
    "ranking_weight": 1.0,
    "tie_tolerance": 1e-6,
    "pairwise_gap_normalization": "per_scene_query_teacher_response_span",
}
RESPONSE_BRANCH_GRADIENT_RATIO = 0.25
TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND = 0.5
FROZEN_GRADIENT_DESIGN_DIAGNOSTIC_LEXICAL_PATH = (
    "/root/RADIO-GS/output/optimization_20260801/"
    "warmstart_gradient_pairwise_balanced_bound_gpu1_seed0_v6/result.json"
)
FROZEN_GRADIENT_DESIGN_DIAGNOSTIC = {
    "path": (
        "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
        "warmstart_gradient_pairwise_balanced_bound_gpu1_seed0_v6/result.json"
    ),
    "sha256": "5e5b149f154324001283058e9105198072c4b089f10cfdbade217903a4ea6c35",
}
GRADIENT_DIAGNOSTIC_SCENES = (
    "scene0024_00",
    "scene0038_01",
    "scene0049_00",
    "scene0054_00",
)
GRADIENT_DIAGNOSTIC_ROWS = 48
GRADIENT_DIAGNOSTIC_IMPLEMENTATION = {
    "path": "/root/RADIO-GS/radio_gs/scripts/diagnose_warmstart_response_gradients.py",
    "sha256": "c541299a1c12889177600295e45818d5f44450fd7a03da0de0de917895e44bac",
}
GRADIENT_DIAGNOSTIC_TRAINING_IMPLEMENTATION = {
    "path": "/root/RADIO-GS/radio_gs/scripts/train_surface_region_text_response_distill.py",
    "sha256": "13c958e1ef0f38154a7e76707b499a6c55e3b32271f95e4a030ea8e9678848f8",
}
GRADIENT_DIAGNOSTIC_LOSS_IMPLEMENTATION = {
    "path": "/root/RADIO-GS/radio_gs/losses/direct_point_query_logit_distill_loss.py",
    "sha256": "870a00530ee1dc2e671734413d393956a56ae8b11de3bb38ecf3a7bb1fde8977",
}
CONTEXT_POOLING_MODE = "joint_attention_v1"
ATTENTION_BINDING_MODE = "attention_postcache_joint_v1"
TRAINING_CONTRACT = {
    "hidden_dim": 256,
    "epochs": 60,
    "patience": 10,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "token_weight": 0.25,
    "relation_weight": 0.1,
    "reliability_attention_mode": "log_prior",
    "context_pooling_mode": CONTEXT_POOLING_MODE,
    "canonical_noise_degrees": 0.0,
    "canonical_noise_calibration": "",
    "seeds": list(REQUIRED_SEEDS),
    "response_lambda_source": "per_seed_exact_surface_warmstart_gradient_budget",
    "response_branch_gradient_target_ratio": RESPONSE_BRANCH_GRADIENT_RATIO,
    "total_response_gradient_ratio_upper_bound": (
        TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND
    ),
    "response_gradient_bound_scope": (
        "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
    ),
    "response_losses": [
        INDEPENDENT_RESPONSE_LOSS,
        SCENE_RESPONSE_LOSS,
    ],
    "scene_response_objective": dict(SCENE_RESPONSE_OBJECTIVE),
    "scene_tie_tolerance": 1e-6,
    "training_batching": "shuffle_complete_scene_groups_no_partial_scenes_v1",
    "max_complete_scene_batch_rows": 64,
    "proposal_state_machine": dict(PROPOSAL_STATE_MACHINE),
    "epoch_selection": EPOCH_SELECTION,
    "surface_control_initialization": "exact_seed_checkpoint_state_dict",
    "surface_control_noninferiority_tolerance": (
        SURFACE_CONTROL_NONINFERIORITY_TOLERANCE
    ),
}
LEGACY_TRAINING_CONTRACT = {
    key: value
    for key, value in {
        **TRAINING_CONTRACT,
        "epoch_selection": LEGACY_EPOCH_SELECTION,
    }.items()
    if key != "proposal_state_machine"
}
REGISTERED_LEGACY_MANIFEST = {
    "path": (
        "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
        "surface_text_response_pairwise_balanced_robust_c1024_gpu1only_v1/"
        "run_manifest.json"
    ),
    "sha256": "ccdb44d233850a72c590cc8c5ca5dfffe134666dad22897a9b0bb90c90658c80",
}
PYTHON_ENTRYPOINTS = (
    "radio_gs/scripts/surface_text_response_distill_authority.py",
    "radio_gs/scripts/train_surface_region_text_response_distill.py",
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py",
    "radio_gs/scripts/finalize_gpu_guard_receipt.py",
    "radio_gs/scripts/finalize_surface_text_response_promotion.py",
)
SHELL_SOURCES = (
    "radio_gs/scripts/run_surface_region_text_response_distill.sh",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
    "radio_gs/scripts/run_repo_python.sh",
)
IMPLEMENTATION_SOURCES = (
    "radio_gs/scripts/run_surface_region_text_response_distill.sh",
    "radio_gs/scripts/surface_text_response_distill_authority.py",
    "radio_gs/scripts/train_surface_region_text_response_distill.py",
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py",
    "radio_gs/scripts/finalize_gpu_guard_receipt.py",
    "radio_gs/scripts/finalize_surface_text_response_promotion.py",
    "radio_gs/scripts/train_surface_region_summary_readout.py",
    "radio_gs/scripts/surface_attention_pooling_screen.py",
    "radio_gs/losses/direct_point_query_logit_distill_loss.py",
    "radio_gs/interfaces/surface_region_summary.py",
    "radio_gs/models/siglip_projection.py",
    "radio_gs/scripts/build_target_blind_siglip2_embedding_artifact.py",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
    "radio_gs/scripts/run_repo_python.sh",
    "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
    "radio_gs/utils/immutable_artifacts.py",
)
RUNTIME_PACKAGES = ("numpy", "Pillow", "scipy", "timm", "torch", "torchvision")
RUNTIME_MODULES = (
    "radio_gs",
    "radio_gs.interfaces.surface_region_summary",
    "radio_gs.losses.direct_point_query_logit_distill_loss",
    "radio_gs.models.siglip_projection",
    "radio_gs.scripts.train_surface_region_summary_readout",
    "radio_gs.scripts.surface_attention_pooling_screen",
    "radio_gs.scripts.train_surface_region_text_response_distill",
    "radio_gs.utils.immutable_artifacts",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "RADIO_GS_REPO_ROOT",
    "RADIO_GS_DRIVER_LIBRARY",
    "RADIO_GS_LD_LIBRARY_PATH",
    "RADIO_GS_SITE_PACKAGES",
)
KERNEL_FAULT_PATTERN = re.compile(
    r"(?:\bNVRM\b.*\bXid\b|fallen off|PCIe.*(?:error|fatal)|GPU.*lost PCIe)",
    re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_LOCK_ROOT = Path("/root/RADIO-GS/output")
LOCK_ROOT_BINDING_VERSION = "canonical_output_root_binding_v1"
LOCK_ROOT_BINDING_ENV = "TEXT_RESPONSE_DISTILL_LOCK_ROOT_BINDING_SHA256"
OUTPUT_DIRECTORY_NAMES = (
    "locks",
    "readouts",
    "logs",
    "audits",
    "receipts",
    "telemetry",
)
TRAINING_COMMAND_CONTRACT_VERSION = "complete_distill_train_argv_v3"
TRAINING_COMMAND_ARGUMENT_FIELDS = {
    "train_caches",
    "validation_caches",
    "fit_text_bank",
    "fit_text_bank_manifest",
    "calibration_manifests",
    "run_manifest",
    "radio_checkpoint",
    "output_root",
}
JOURNAL_HEADER_PATTERN = re.compile(
    r"^surface_text_response_seed=(?P<seed>[0-9]+)"
    r"\tstart_epoch=(?P<start>[0-9]+)\tend_epoch=(?P<end>[0-9]+)$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regular_file(path: Path, *, label: str) -> Path:
    source = Path(os.path.abspath(os.fspath(path)))
    info = os.stat(source, follow_symlinks=False)
    _require(
        stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
        f"{label} is not a non-symlink regular file: {source}",
    )
    return source


def _module_file(repo_root: Path, module_name: str) -> Path | None:
    if not module_name.startswith("radio_gs"):
        return None
    stem = repo_root.joinpath(*module_name.split("."))
    source = stem.with_suffix(".py")
    if source.is_file():
        return source
    package = stem / "__init__.py"
    return package if package.is_file() else None


def _module_name(repo_root: Path, source: Path) -> str:
    parts = list(source.resolve().relative_to(repo_root.resolve()).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def discover_repo_python_closure(
    repo_root: str | Path,
    entrypoints: Iterable[str] = PYTHON_ENTRYPOINTS,
) -> tuple[str, ...]:
    """Return the complete static in-repository import closure."""

    root = Path(repo_root).resolve()
    queue = [root / relative for relative in entrypoints]
    discovered: set[Path] = set()
    while queue:
        source = _regular_file(queue.pop(), label="distill closure source").resolve()
        if source in discovered:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot parse distill closure source: {source}") from exc
        discovered.add(source)
        relative = source.relative_to(root)
        for parent in relative.parents:
            if not parent.parts:
                continue
            initializer = root / parent / "__init__.py"
            if initializer.is_file() and initializer.resolve() not in discovered:
                # Package initializers execute before the imported submodule;
                # parse their imports too instead of only inventorying them.
                queue.append(initializer)
        module_name = _module_name(root, source)
        package_name = module_name if source.name == "__init__.py" else module_name.rpartition(".")[0]
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                raw = node.module or ""
                if node.level:
                    try:
                        resolved = importlib.util.resolve_name("." * node.level + raw, package_name)
                    except (ImportError, ValueError) as exc:
                        raise ValueError(f"cannot resolve closure import: {source}") from exc
                else:
                    resolved = raw
                if resolved:
                    candidates.append(resolved)
                    candidates.extend(
                        f"{resolved}.{alias.name}" for alias in node.names if alias.name != "*"
                    )
            for candidate in candidates:
                dependency = _module_file(root, candidate)
                if dependency is not None and dependency.resolve() not in discovered:
                    queue.append(dependency)
    return tuple(sorted(str(path.relative_to(root)) for path in discovered))


def _runtime_fingerprint(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    packages: dict[str, str] = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    imports: dict[str, dict[str, str]] = {}
    for name in RUNTIME_MODULES:
        module = importlib.import_module(name)
        raw = getattr(module, "__file__", None)
        _require(isinstance(raw, str) and bool(raw), f"runtime module lacks source: {name}")
        source = _regular_file(Path(raw), label="runtime import").resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"runtime import escaped repository: {name} -> {source}") from exc
        imports[name] = {
            "path": str(source),
            "relative_path": str(relative),
            "sha256": sha256_file(source),
        }
    executable = _regular_file(Path(sys.executable), label="Python executable").resolve()
    cudnn = torch.backends.cudnn.version()
    return {
        "repository_import_root": str(root),
        "imported_modules": imports,
        "python_executable": file_record(executable),
        "python_version": sys.version,
        "python_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": int(cudnn) if cudnn is not None else None,
        "environment": {key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS},
    }


def build_runtime_closure(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    python_sources = discover_repo_python_closure(root)
    relative_paths = sorted(set(python_sources) | set(SHELL_SOURCES))
    repository_sources = {
        "python_entrypoints": list(PYTHON_ENTRYPOINTS),
        "shell_sources": list(SHELL_SOURCES),
        "files": {
            relative: sha256_file(
                _regular_file(root / relative, label="distill repository source")
            )
            for relative in relative_paths
        },
    }
    repository_sources["digest"] = canonical_json_sha256(repository_sources)
    payload = {
        "schema_version": 1,
        "repository_sources": repository_sources,
        "runtime_fingerprint": _runtime_fingerprint(root),
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def _directory_identity(info: os.stat_result, *, label: str) -> dict[str, int]:
    _require(stat.S_ISDIR(info.st_mode), f"{label} is not a directory")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "owner_uid": int(info.st_uid),
        "owner_gid": int(info.st_gid),
        "permissions": int(stat.S_IMODE(info.st_mode)),
    }


def _entry_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "owner_uid": int(info.st_uid),
        "owner_gid": int(info.st_gid),
        "permissions": int(stat.S_IMODE(info.st_mode)),
    }


def inspect_canonical_lock_root(lock_root: Path | None = None) -> dict[str, Any]:
    """Inspect the one permitted output-root symlink without creating anything.

    The repository's deployed ``output`` entry is an intentional absolute
    symlink into ``/mnt/pool``.  Only that exact lexical root may be a symlink;
    descendants are opened separately with ``O_NOFOLLOW``.  Re-reading the
    entry around a descriptor-backed target check closes the ordinary
    unlink/repoint race, while the returned identity is frozen in the run
    manifest and inherited supervisor contract.
    """

    lexical = Path(
        os.path.abspath(os.fspath(CANONICAL_LOCK_ROOT if lock_root is None else lock_root))
    )
    _require(
        lexical == CANONICAL_LOCK_ROOT,
        "GPU1 authority lock must be /root/RADIO-GS/output/.physical_gpu1.lock",
    )
    parent = lexical.parent
    parent_info = os.stat(parent, follow_symlinks=False)
    _require(
        stat.S_ISDIR(parent_info.st_mode)
        and not stat.S_ISLNK(parent_info.st_mode)
        and stat.S_IMODE(parent_info.st_mode) & 0o022 == 0,
        f"canonical output parent is not a controlled real directory: {parent}",
    )
    before = os.stat(lexical, follow_symlinks=False)
    is_link = stat.S_ISLNK(before.st_mode)
    _require(
        is_link or stat.S_ISDIR(before.st_mode),
        f"canonical output root is not a directory/symlink: {lexical}",
    )
    link_target: str | None = None
    if is_link:
        link_target = os.readlink(lexical)
        _require(
            before.st_nlink == 1
            and before.st_uid == parent_info.st_uid
            and Path(link_target).is_absolute(),
            "canonical output root symlink is not controlled and absolute",
        )
    resolved = lexical.resolve(strict=True)
    if link_target is not None:
        _require(
            Path(os.path.abspath(link_target)) == resolved,
            "canonical output root must be the only symlink in its target chain",
        )
    resolved_info = os.stat(resolved, follow_symlinks=False)
    _require(
        stat.S_ISDIR(resolved_info.st_mode)
        and not stat.S_ISLNK(resolved_info.st_mode),
        f"canonical output target is not a real directory: {resolved}",
    )
    _require(
        stat.S_IMODE(resolved_info.st_mode) & 0o022 == 0,
        f"canonical output target is group/world writable: {resolved}",
    )
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(resolved, flags)
    try:
        descriptor_info = os.fstat(descriptor)
        followed_info = os.stat(lexical)
        after = os.stat(lexical, follow_symlinks=False)
        parent_after = os.stat(parent, follow_symlinks=False)
        _require(
            _entry_identity(after) == _entry_identity(before)
            and stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
            and (not is_link or os.readlink(lexical) == link_target),
            "canonical output root entry changed during inspection",
        )
        _require(
            _directory_identity(parent_after, label="canonical output parent")
            == _directory_identity(parent_info, label="canonical output parent"),
            "canonical output parent identity changed during inspection",
        )
        descriptor_identity = _directory_identity(
            descriptor_info,
            label="canonical output descriptor",
        )
        _require(
            descriptor_identity
            == _directory_identity(followed_info, label="canonical output path")
            == _directory_identity(resolved_info, label="canonical output target"),
            "canonical output resolved target identity differs",
        )
    finally:
        os.close(descriptor)
    return {
        "version": LOCK_ROOT_BINDING_VERSION,
        "lexical_path": str(lexical),
        "entry_type": "controlled_symlink" if is_link else "directory",
        "lexical_parent": str(parent),
        "lexical_parent_identity": _directory_identity(
            parent_info,
            label="canonical output parent",
        ),
        "root_entry_identity": _entry_identity(before),
        "symlink_change_time_ns": int(before.st_ctime_ns) if is_link else None,
        "symlink_target": link_target,
        "resolved_path": str(resolved),
        "resolved_directory_identity": descriptor_identity,
    }


def validate_canonical_lock_root_binding(
    expected: object,
    *,
    lock_root: Path | None = None,
) -> dict[str, Any]:
    observed = inspect_canonical_lock_root(lock_root)
    _require(
        isinstance(expected, Mapping) and dict(expected) == observed,
        "canonical output root symlink/target identity changed",
    )
    return observed


def _secure_output_directory(
    path: Path,
    *,
    root_binding: Mapping[str, Any],
    create: bool,
) -> dict[str, Any]:
    """Open/create one output directory below the bound root, never following children."""

    lexical_root = Path(str(root_binding.get("lexical_path", "")))
    resolved_root = Path(str(root_binding.get("resolved_path", "")))
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = target.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"secure directory escapes repository output: {target}") from exc
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(resolved_root, flags)
    try:
        _require(
            _directory_identity(os.fstat(descriptor), label="bound output root")
            == root_binding.get("resolved_directory_identity"),
            "bound output root descriptor identity changed",
        )
        for component in relative.parts:
            _require(
                component not in {"", ".", ".."},
                f"invalid secure output component: {component!r}",
            )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                _require(create, f"required output directory is missing: {target}")
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise ValueError(
                        f"refuse symlink/non-directory output component: {component}"
                    ) from exc
            except OSError as exc:
                raise ValueError(
                    f"refuse symlink/non-directory output component: {component}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        identity = _directory_identity(
            os.fstat(descriptor),
            label=f"secure output directory {target}",
        )
    finally:
        os.close(descriptor)
    return {
        "lexical_path": str(target),
        "resolved_path": str(resolved_root.joinpath(*relative.parts)),
        "identity": identity,
    }


def _acquire_nofollow_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("authority lock requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        path_info = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(info.st_mode)
            and info.st_nlink == 1
            and (info.st_dev, info.st_ino) == (path_info.st_dev, path_info.st_ino),
            f"authority lock identity differs: {path}",
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"authority lock is already held: {path}") from exc
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def verify_readonly_source_snapshot(repo_root: Path) -> Path:
    """Reject mutable or aliased producer sources before acquiring GPU1."""

    lexical_root = Path(os.path.abspath(os.fspath(repo_root)))
    try:
        root = lexical_root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError("source snapshot root is missing") from exc
    _require(
        lexical_root == root,
        "source snapshot root must not traverse a symlink",
    )

    def validate_node(path: Path, *, expect_directory: bool = False) -> None:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise ValueError(f"source snapshot entry is unavailable: {path}") from exc
        _require(
            not stat.S_ISLNK(info.st_mode),
            f"source snapshot contains a symlink: {path}",
        )
        if expect_directory:
            _require(
                stat.S_ISDIR(info.st_mode),
                "source snapshot root is not a real directory",
            )
        else:
            _require(
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
                f"source snapshot contains a non-file entry: {path}",
            )
            if stat.S_ISREG(info.st_mode):
                _require(
                    info.st_nlink == 1,
                    f"source snapshot contains a multiply linked file: {path}",
                )
        _require(
            stat.S_IMODE(info.st_mode) & 0o222 == 0,
            f"source snapshot contains a writable entry: {path}",
        )

    validate_node(root, expect_directory=True)
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        for name in sorted([*directory_names, *file_names]):
            validate_node(Path(current) / name)
    return root


def run_locked(
    *, repo_root: Path, lock_root: Path, output_root: Path, command: Sequence[str]
) -> int:
    root = verify_readonly_source_snapshot(repo_root)
    canonical_lock_root = Path(os.path.abspath(os.fspath(lock_root)))
    _require(
        canonical_lock_root == CANONICAL_LOCK_ROOT,
        "GPU1 authority lock must be /root/RADIO-GS/output/.physical_gpu1.lock",
    )
    root_binding = inspect_canonical_lock_root(canonical_lock_root)
    output = _secure_output_directory(
        output_root,
        root_binding=root_binding,
        create=True,
    )
    locks = _secure_output_directory(
        Path(output["lexical_path"]) / "locks",
        root_binding=root_binding,
        create=True,
    )
    global_lock = Path(root_binding["resolved_path"]) / ".physical_gpu1.lock"
    run_lock = Path(locks["resolved_path"]) / "text_response_distill.run.lock"
    descriptors: list[int] = []
    try:
        descriptors.append(_acquire_nofollow_lock(global_lock))
        descriptors.append(_acquire_nofollow_lock(run_lock))
        descriptors.append(_open_kernel_singleton(GPU1_SINGLETON_ADDRESS))
        argv = [str(value) for value in command]
        if argv and argv[0] == "--":
            argv.pop(0)
        _require(argv and all(argv), "locked authority command is empty")
        environment = dict(os.environ)
        environment["TEXT_RESPONSE_DISTILL_GLOBAL_LOCK_FD"] = str(descriptors[0])
        environment["TEXT_RESPONSE_DISTILL_RUN_LOCK_FD"] = str(descriptors[1])
        environment[LOCK_ROOT_BINDING_ENV] = canonical_json_sha256(root_binding)
        environment[SINGLETON_FD_ENV] = str(descriptors[2])
        environment[SINGLETON_PROTOCOL_ENV] = _singleton_protocol(
            GPU1_SINGLETON_ADDRESS
        )
        for descriptor in descriptors:
            os.set_inheritable(descriptor, True)
        return int(
            subprocess.run(
                argv,
                env=environment,
                pass_fds=tuple(descriptors),
                check=False,
            ).returncode
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _verify_inherited_lock(descriptor: int, expected_path: Path) -> None:
    _require(descriptor >= 3, "authority lock descriptor is invalid")
    expected = Path(os.path.abspath(os.fspath(expected_path)))
    info = os.fstat(descriptor)
    path_info = os.stat(expected, follow_symlinks=False)
    _require(
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and not stat.S_ISLNK(path_info.st_mode)
        and (info.st_dev, info.st_ino) == (path_info.st_dev, path_info.st_ino),
        f"inherited authority lock does not own {expected}",
    )
    # Re-locking an inherited open-file description succeeds without releasing
    # the supervisor's lock.  An arbitrary environment variable or unrelated
    # descriptor therefore cannot authorize the runner.
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError(f"inherited authority lock is not held: {expected}") from exc


def verify_inherited_locks(
    *,
    lock_root: Path,
    output_root: Path,
    global_descriptor: int,
    run_descriptor: int,
    singleton_descriptor: int,
) -> dict[str, Any]:
    canonical = Path(os.path.abspath(os.fspath(lock_root)))
    _require(
        canonical == CANONICAL_LOCK_ROOT,
        "GPU1 authority lock must be /root/RADIO-GS/output/.physical_gpu1.lock",
    )
    root_binding = inspect_canonical_lock_root(canonical)
    _require(
        os.environ.get(LOCK_ROOT_BINDING_ENV)
        == canonical_json_sha256(root_binding),
        "inherited canonical output root binding differs",
    )
    output = Path(os.path.abspath(os.fspath(output_root)))
    output_binding = _secure_output_directory(
        output,
        root_binding=root_binding,
        create=False,
    )
    locks = _secure_output_directory(
        output / "locks",
        root_binding=root_binding,
        create=False,
    )
    global_path = canonical / ".physical_gpu1.lock"
    run_path = output / "locks/text_response_distill.run.lock"
    _verify_inherited_lock(
        int(global_descriptor),
        Path(root_binding["resolved_path"]) / ".physical_gpu1.lock",
    )
    _verify_inherited_lock(
        int(run_descriptor),
        Path(locks["resolved_path"]) / "text_response_distill.run.lock",
    )
    singleton = verify_inherited_singleton(
        int(singleton_descriptor),
        GPU1_SINGLETON_ADDRESS,
    )
    return {
        "global_lock": str(global_path),
        "run_lock": str(run_path),
        "canonical_output_root_binding": root_binding,
        "output_root_binding": output_binding,
        "kernel_singleton": singleton,
    }


def _normalize_bus(value: str) -> str:
    text = str(value).strip().lower()
    parts = text.split(":")
    return ":".join(parts[-2:])


def gpu_check_payload(
    *,
    phase: str,
    gpu_uuid: str,
    nvidia_bus_id: str,
    proc_bus_id: str,
    pci_prefix: str,
    compute_owners: Sequence[str],
    observed_epoch: int,
) -> dict[str, Any]:
    owners = [str(value).strip() for value in compute_owners if str(value).strip()]
    prefix = str(pci_prefix).strip().lower()
    _require(str(gpu_uuid).startswith("GPU-") and len(str(gpu_uuid)) > 8, "invalid GPU1 UUID")
    _require(
        bool(nvidia_bus_id)
        and _normalize_bus(nvidia_bus_id) == _normalize_bus(proc_bus_id),
        "physical GPU1 PCI bus identity differs",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{32}", prefix) is not None
        and set(prefix) != {"f"},
        "physical GPU1 PCIe configuration space is not responding",
    )
    _require(not owners, f"physical GPU1 already has compute owners: {owners}")
    _require(
        isinstance(observed_epoch, int)
        and not isinstance(observed_epoch, bool)
        and observed_epoch > 0,
        "physical GPU1 observation epoch is invalid",
    )
    return {
        "schema_version": 1,
        "artifact_type": GPU_CHECK_ARTIFACT_TYPE,
        "status": "physical_gpu1_idle_and_pcie_responsive",
        "phase": str(phase),
        "observed_epoch": observed_epoch,
        "gpu_identity": {
            "physical_index": 1,
            "uuid": str(gpu_uuid),
            "pci_bus_id": str(nvidia_bus_id),
        },
        "proc_pci_bus_id": str(proc_bus_id),
        "pci_config_prefix_hex": prefix,
        "compute_owners": [],
    }


def record_gpu_check(args: argparse.Namespace) -> dict[str, Any]:
    payload = gpu_check_payload(
        phase=args.phase,
        gpu_uuid=args.gpu_uuid,
        nvidia_bus_id=args.gpu_bus_id,
        proc_bus_id=args.proc_bus_id,
        pci_prefix=args.pci_prefix,
        compute_owners=args.compute_owner,
        observed_epoch=args.observed_epoch,
    )
    if args.run_manifest is not None:
        manifest, _, _ = _manifest(Path(args.run_manifest))
        _require(
            manifest.get("gpu_identity") == payload["gpu_identity"],
            "GPU1 identity changed from the distill run manifest",
        )
    write_frozen_json(args.output, payload)
    return {"gpu_check": file_record(args.output), "gpu_identity": payload["gpu_identity"]}


def _resolve_command_argument(value: str, *, working_directory: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = working_directory / path
    return path.resolve()


def _cache_records(
    pattern: str,
    *,
    expected_count: int,
    working_directory: Path | None = None,
) -> list[dict[str, str]]:
    import glob

    raw_pattern = str(pattern)
    if working_directory is not None and not os.path.isabs(raw_pattern):
        raw_pattern = str(working_directory / raw_pattern)
    paths = [Path(value).resolve() for value in sorted(glob.glob(raw_pattern))]
    _require(len(paths) == expected_count, f"cache pattern requires exactly {expected_count} shards")
    _require(len(set(paths)) == len(paths), "cache pattern contains duplicate shards")
    return [file_record(path) for path in paths]


def _seed_path_map(values: Sequence[str], *, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in values:
        seed_text, separator, path_text = str(raw).partition("=")
        _require(separator == "=" and seed_text.isdigit() and path_text, f"invalid {label}")
        seed = int(seed_text)
        _require(seed in REQUIRED_SEEDS and seed not in result, f"duplicate {label} seed")
        result[seed] = Path(path_text).resolve()
    _require(set(result) == set(REQUIRED_SEEDS), f"{label} must cover seeds 0/1/2")
    return result


def _training_command_arguments(args: argparse.Namespace) -> dict[str, Any]:
    calibrations = _seed_path_map(
        args.calibration_manifest, label="calibration manifest"
    )
    return {
        "train_caches": str(args.train_caches),
        "validation_caches": str(args.validation_caches),
        "fit_text_bank": os.fspath(args.fit_text_bank),
        "fit_text_bank_manifest": os.fspath(args.fit_text_bank_manifest),
        "calibration_manifests": {
            str(seed): os.fspath(calibrations[seed]) for seed in REQUIRED_SEEDS
        },
        "run_manifest": os.fspath(args.run_manifest),
        "radio_checkpoint": os.fspath(args.radio_checkpoint),
        "output_root": os.fspath(args.output_root),
    }


def _training_argv(
    *,
    repo_root: Path,
    arguments: Mapping[str, Any],
    output_row: Mapping[str, Any],
    surface_control: Mapping[str, Any],
    seed: int,
) -> list[str]:
    _require(
        set(arguments) == TRAINING_COMMAND_ARGUMENT_FIELDS,
        "distill training command argument fields differ",
    )
    _require(output_row.get("seed") == seed, f"distill seed-{seed} output row differs")
    _require(
        surface_control.get("seed") == seed
        and isinstance(surface_control.get("checkpoint"), Mapping),
        f"distill seed-{seed} Surface control differs",
    )
    control_checkpoint = dict(surface_control["checkpoint"])
    validate_file_record(
        control_checkpoint,
        label=f"distill seed-{seed} Surface control checkpoint",
    )
    checkpoint_name = Path(str(output_row.get("checkpoint", ""))).name
    _require(bool(checkpoint_name), f"distill seed-{seed} checkpoint name is empty")
    command_checkpoint = Path(str(arguments["output_root"])) / "readouts" / checkpoint_name
    _require(
        command_checkpoint.resolve() == Path(str(output_row["checkpoint"])).resolve(),
        f"distill seed-{seed} command output differs from its frozen output",
    )
    return [
        "bash",
        str(repo_root / "radio_gs/scripts/run_repo_python.sh"),
        "radio_gs/scripts/train_surface_region_text_response_distill.py",
        "train",
        "--train-caches",
        str(arguments["train_caches"]),
        "--validation-caches",
        str(arguments["validation_caches"]),
        "--fit-text-bank",
        str(arguments["fit_text_bank"]),
        "--fit-text-bank-manifest",
        str(arguments["fit_text_bank_manifest"]),
        "--calibration-manifest",
        str(arguments["calibration_manifests"][str(seed)]),
        "--run-manifest",
        str(arguments["run_manifest"]),
        "--surface-control-checkpoint",
        str(control_checkpoint["path"]),
        "--surface-control-checkpoint-sha256",
        str(control_checkpoint["sha256"]),
        "--output",
        str(command_checkpoint),
        "--hidden-dim",
        "256",
        "--epochs",
        "60",
        "--patience",
        "10",
        "--batch-size",
        "16",
        "--learning-rate",
        "2e-4",
        "--weight-decay",
        "1e-4",
        "--token-weight",
        "0.25",
        "--relation-weight",
        "0.1",
        "--reliability-attention-mode",
        "log_prior",
        "--context-pooling-mode",
        CONTEXT_POOLING_MODE,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--radio-checkpoint",
        str(arguments["radio_checkpoint"]),
    ]


def _build_training_command_contract(
    *,
    repo_root: Path,
    arguments: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    surface_promotion: Mapping[str, Any],
) -> dict[str, Any]:
    controls = _surface_controls_by_seed(surface_promotion)
    commands = []
    for seed in REQUIRED_SEEDS:
        rows = [row for row in outputs if row.get("seed") == seed]
        _require(len(rows) == 1, f"distill training command lacks seed-{seed} output")
        argv = _training_argv(
            repo_root=repo_root,
            arguments=arguments,
            output_row=rows[0],
            surface_control=controls[seed],
            seed=seed,
        )
        commands.append(
            {"seed": seed, "argv": argv, "argv_sha256": canonical_json_sha256(argv)}
        )
    return {
        "version": TRAINING_COMMAND_CONTRACT_VERSION,
        "working_directory": str(repo_root),
        "arguments": dict(arguments),
        "commands": commands,
    }


def validate_training_command_contract(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    contract = manifest.get("training_command_contract")
    _require(
        isinstance(contract, Mapping)
        and set(contract) == {"version", "working_directory", "arguments", "commands"}
        and contract.get("version") == TRAINING_COMMAND_CONTRACT_VERSION,
        "distill training command contract differs",
    )
    working_directory = Path(str(contract.get("working_directory", ""))).resolve()
    source_root = Path(
        str(manifest.get("authority_contract", {}).get("source_snapshot_root", ""))
    ).resolve()
    _require(
        working_directory == source_root,
        "distill training command working directory differs from the source snapshot",
    )
    raw_arguments = contract.get("arguments")
    _require(
        isinstance(raw_arguments, Mapping)
        and set(raw_arguments) == TRAINING_COMMAND_ARGUMENT_FIELDS
        and all(
            isinstance(raw_arguments[field], str) and raw_arguments[field]
            for field in TRAINING_COMMAND_ARGUMENT_FIELDS - {"calibration_manifests"}
        )
        and isinstance(raw_arguments.get("calibration_manifests"), Mapping)
        and set(raw_arguments["calibration_manifests"])
        == {str(seed) for seed in REQUIRED_SEEDS}
        and all(
            isinstance(value, str) and value
            for value in raw_arguments["calibration_manifests"].values()
        ),
        "distill training command arguments differ",
    )
    arguments = {
        str(key): (
            {str(seed): str(path) for seed, path in value.items()}
            if key == "calibration_manifests"
            else str(value)
        )
        for key, value in raw_arguments.items()
    }
    authority_contract = manifest.get("authority_contract")
    _require(
        isinstance(authority_contract, Mapping)
        and Path(arguments["output_root"]).is_absolute()
        and arguments["output_root"]
        == str(authority_contract.get("output_root", "")),
        "distill command output root differs from the frozen authority root",
    )
    _require(
        _cache_records(
            arguments["train_caches"],
            expected_count=4,
            working_directory=working_directory,
        )
        == manifest.get("train_caches")
        and _cache_records(
            arguments["validation_caches"],
            expected_count=2,
            working_directory=working_directory,
        )
        == manifest.get("validation_caches"),
        "distill training cache argv differs from frozen cache bindings",
    )
    individual_bindings = {
        "fit_text_bank": manifest.get("fit_text_bank", {}).get("artifact"),
        "fit_text_bank_manifest": manifest.get("fit_text_bank", {}).get("manifest"),
        "radio_checkpoint": manifest.get("radio_checkpoint"),
    }
    for field, binding in individual_bindings.items():
        _require(isinstance(binding, Mapping), f"distill {field} binding differs")
        _require(
            file_record(
                _resolve_command_argument(
                    arguments[field], working_directory=working_directory
                )
            )
            == dict(binding),
            f"distill {field} argv differs from its frozen file binding",
        )
    calibrations = manifest.get("calibrations")
    _require(
        isinstance(calibrations, list) and len(calibrations) == len(REQUIRED_SEEDS),
        "distill calibration bindings differ",
    )
    by_seed = {
        row.get("seed"): row for row in calibrations if isinstance(row, Mapping)
    }
    _require(set(by_seed) == set(REQUIRED_SEEDS), "distill calibration seeds differ")
    for seed in REQUIRED_SEEDS:
        _require(
            file_record(
                _resolve_command_argument(
                    arguments["calibration_manifests"][str(seed)],
                    working_directory=working_directory,
                )
            )
            == by_seed[seed].get("manifest"),
            f"distill seed-{seed} calibration argv differs",
        )
    _require(
        _resolve_command_argument(
            arguments["run_manifest"], working_directory=working_directory
        )
        == manifest_path.resolve(),
        "distill training argv binds another run manifest",
    )
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, list), "distill training output index differs")
    expected = _build_training_command_contract(
        repo_root=source_root,
        arguments=arguments,
        outputs=outputs,
        surface_promotion=manifest.get("surface_promotion", {}),
    )
    _require(dict(contract) == expected, "distill training argv contract was not reproduced")
    return expected


def expected_training_argv(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    seed: int,
) -> list[str]:
    contract = validate_training_command_contract(manifest, manifest_path=manifest_path)
    matches = [row for row in contract["commands"] if row.get("seed") == seed]
    _require(len(matches) == 1, f"distill training argv lacks seed-{seed}")
    return list(matches[0]["argv"])


def _attention_surface_binding(
    *, surface: Path, candidate: str, train: list[dict], validation: list[dict]
) -> dict[str, Any]:
    from radio_gs.scripts import surface_attention_pooling_screen as pooling

    _require(
        candidate == "context_c1024_geometric",
        "attention-postcache text distillation requires context_c1024_geometric",
    )
    paths = {
        "run_manifest": surface / "run_manifest.json",
        "cache_pairing": surface / "cache_pairing.json",
        "attention_pooling_screen": surface / "attention_pooling_screen.json",
        "screen_completion": surface / "screen.complete",
        "runtime_closure_final": surface / "runtime_closure_final.json",
    }
    manifest, manifest_sha, manifest_path = load_json_object(
        paths["run_manifest"], label="Surface attention-postcache run manifest"
    )
    pairing, pairing_sha, pairing_path = load_json_object(
        paths["cache_pairing"], label="Surface attention-postcache cache pairing"
    )
    screen, _, _ = load_json_object(
        paths["attention_pooling_screen"],
        label="Surface attention-postcache pooling screen",
    )
    closure, _, _ = load_json_object(
        paths["runtime_closure_final"],
        label="Surface attention-postcache final closure",
    )
    _require(
        manifest.get("screen") == "surface-c1024-attention-postcache-continuation-v1"
        and manifest.get("continuation_contract", {}).get("mode")
        == "post_cache_only_no_parent_mutation_v1"
        and manifest.get("continuation_contract", {}).get("cache_writes_forbidden")
        is True
        and manifest.get("cache_bundle") == pairing.get("rows")
        and pairing.get("status")
        == "single_c1024_cache_exact_teacher_replay_verified"
        and pairing.get("run_manifest")
        == {"path": str(manifest_path), "sha256": manifest_sha}
        and pairing.get("benchmark_queries_opened") is False
        and pairing.get("benchmark_masks_opened") is False,
        "attention-postcache manifest/cache pairing differs",
    )
    parent_record = manifest["continuation_contract"].get("parent_run_manifest")
    expected_manifest_record = {"path": str(manifest_path), "sha256": manifest_sha}
    expected_pairing_record = {"path": str(pairing_path), "sha256": pairing_sha}
    _require(
        screen.get("artifact_type")
        == "surface_c1024_attention_pooling_postcache_continuation"
        and screen.get("run_manifest") == expected_manifest_record
        and screen.get("cache_pairing_report") == expected_pairing_record
        and screen.get("parent_run_manifest") == parent_record
        and pairing.get("parent_run_manifest") == parent_record
        and screen.get("selected_variant") == CONTEXT_POOLING_MODE
        and screen.get("selection_status") == "joint_attention_retained"
        and screen.get("promotion_gate_passed") is False
        and screen.get("benchmark_queries_opened") is False
        and screen.get("benchmark_masks_opened") is False,
        "attention-postcache screen did not freeze the joint query-free winner",
    )
    runtime = manifest.get("runtime_closure")
    inventory = closure.get("attempt_inventory")
    _require(
        isinstance(runtime, Mapping)
        and closure.get("artifact_type") == "surface_region_runtime_closure_audit"
        and closure.get("status") == "runtime_closure_verified"
        and closure.get("phase") == "final_before_completion"
        and closure.get("full_checkpoint_rehashed") is True
        and Path(str(closure.get("run_manifest", ""))).resolve() == manifest_path
        and closure.get("run_manifest_sha256") == manifest_sha
        and closure.get("runtime_closure_digest") == runtime.get("digest")
        and isinstance(inventory, Mapping)
        and inventory.get("run_manifest") == expected_manifest_record
        and inventory.get("digest") == screen.get("child_attempt_inventory_digest"),
        "attention-postcache final runtime closure differs",
    )
    try:
        completion_text = paths["screen_completion"].read_text(
            encoding="utf-8"
        ).strip()
        completion_time = datetime.fromisoformat(completion_text)
    except (OSError, ValueError) as error:
        raise ValueError("attention-postcache screen completion differs") from error
    _require(
        completion_time.tzinfo is not None
        and completion_time.utcoffset() is not None,
        "attention-postcache screen completion lacks timezone",
    )

    cache_rows = pairing.get("rows")
    _require(
        isinstance(cache_rows, list) and len(cache_rows) == 6,
        "attention-postcache pairing requires six cache rows",
    )
    by_role_shard: dict[tuple[str, int], Mapping[str, Any]] = {}
    selected_sidecars: list[dict[str, str]] = []
    for row in cache_rows:
        _require(isinstance(row, Mapping), "attention-postcache cache row is invalid")
        key = (str(row.get("role")), int(row.get("shard", -1)))
        _require(key not in by_role_shard, "attention-postcache cache row is duplicated")
        by_role_shard[key] = row
        validate_file_record(row.get("c1024", {}), label=f"Surface c1024 cache {key}")
        validate_file_record(
            row.get("c1024_sidecar", {}), label=f"Surface c1024 sidecar {key}"
        )
        selected_sidecars.append(dict(row["c1024_sidecar"]))
    expected_keys = {
        *(("train", shard) for shard in range(4)),
        *(("validation", shard) for shard in range(2)),
    }
    _require(
        set(by_role_shard) == expected_keys,
        "attention-postcache cache roles/shards differ",
    )
    selected_caches = [
        dict(by_role_shard[("train", shard)]["c1024"]) for shard in range(4)
    ] + [
        dict(by_role_shard[("validation", shard)]["c1024"]) for shard in range(2)
    ]
    _require(
        selected_caches == train + validation,
        "distill caches differ from the frozen attention-postcache cache bundle",
    )

    variants = screen.get("variants")
    _require(
        isinstance(variants, Mapping)
        and set(variants) == {
            pooling.JOINT_CONTEXT_POOLING,
            pooling.SEPARATE_CONTEXT_POOLING,
        },
        "attention-postcache variants differ",
    )
    reconstructed: dict[str, dict[str, Any]] = {}
    stage_terminals: dict[str, dict[str, str]] = {}
    selected_readouts: list[dict[str, str]] = []
    for variant in (pooling.JOINT_CONTEXT_POOLING, pooling.SEPARATE_CONTEXT_POOLING):
        observed_variant = variants[variant]
        _require(isinstance(observed_variant, Mapping), "attention variant is invalid")
        seed_rows = observed_variant.get("seeds")
        _require(
            isinstance(seed_rows, list)
            and len(seed_rows) == 3
            and {row.get("seed") for row in seed_rows if isinstance(row, Mapping)}
            == {0, 1, 2},
            f"attention-postcache {variant} seeds differ",
        )
        ordered = sorted(seed_rows, key=lambda row: int(row["seed"]))
        normalized_rows = []
        for row in ordered:
            seed = int(row["seed"])
            checkpoint_record = dict(row.get("checkpoint", {}))
            checkpoint_path = validate_file_record(
                checkpoint_record,
                label=f"Surface attention {variant} seed-{seed} checkpoint",
            )
            report, _, _ = load_json_object(
                checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"),
                label=f"Surface attention {variant} seed-{seed} report",
            )
            validation_metrics = report.get("validation")
            _require(
                report.get("checkpoint_sha256") == checkpoint_record["sha256"]
                and report.get("best_epoch") == row.get("best_epoch")
                and math.isclose(
                    float(report.get("best_selection_score")),
                    float(row.get("best_selection_score")),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and isinstance(validation_metrics, Mapping)
                and all(
                    math.isclose(
                        float(validation_metrics[key]),
                        float(row.get("validation", {}).get(key)),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    for key in pooling.ALL_VALIDATION_COMPONENTS
                ),
                f"attention-postcache {variant} seed-{seed} report differs",
            )
            normalized = {
                "seed": seed,
                "checkpoint": checkpoint_record,
                "best_epoch": int(row["best_epoch"]),
                "best_selection_score": float(row["best_selection_score"]),
                "validation": {
                    key: float(row["validation"][key])
                    for key in pooling.ALL_VALIDATION_COMPONENTS
                },
            }
            normalized_rows.append(normalized)
            stage_terminals[f"readout_{variant}_seed{seed}"] = checkpoint_record
            if variant == CONTEXT_POOLING_MODE:
                selected_readouts.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint_record,
                        "best_epoch": normalized["best_epoch"],
                        "best_selection_score": normalized[
                            "best_selection_score"
                        ],
                        "validation": normalized["validation"],
                    }
                )
        reconstructed[variant] = {
            "seeds": normalized_rows,
            "mean_selection_score": sum(
                row["best_selection_score"] for row in normalized_rows
            )
            / len(normalized_rows),
            "mean_validation": {
                key: sum(row["validation"][key] for row in normalized_rows)
                / len(normalized_rows)
                for key in pooling.ALL_VALIDATION_COMPONENTS
            },
        }
        _require(
            math.isclose(
                reconstructed[variant]["mean_selection_score"],
                float(observed_variant.get("mean_selection_score")),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and all(
                math.isclose(
                    reconstructed[variant]["mean_validation"][key],
                    float(observed_variant.get("mean_validation", {}).get(key)),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for key in pooling.ALL_VALIDATION_COMPONENTS
            ),
            f"attention-postcache {variant} aggregate differs",
        )
    decision = pooling.promotion_decision(
        reconstructed, manifest.get("selection_contract", {})
    )
    _require(
        decision == {
            key: variants[pooling.SEPARATE_CONTEXT_POOLING].get(key)
            for key in decision
        }
        and decision.get("eligible_for_query_free_promotion") is False,
        "attention-postcache promotion decision differs",
    )

    receipts = screen.get("child_attempt_receipts")
    attempts = inventory.get("attempts")
    _require(
        isinstance(receipts, list)
        and len(receipts) == 6
        and isinstance(attempts, list)
        and len(attempts) == 6,
        "attention-postcache receipt inventory differs",
    )
    expected_receipts = []
    for attempt in sorted(attempts, key=lambda row: str(row.get("stage"))):
        _require(
            isinstance(attempt, Mapping)
            and attempt.get("stage") in stage_terminals
            and attempt.get("attempt_index") == 1
            and attempt.get("result") == "completed"
            and attempt.get("command_status") == 0
            and attempt.get("run_manifest", expected_manifest_record)
            == expected_manifest_record,
            "attention-postcache attempt inventory differs",
        )
        receipt = dict(attempt.get("receipt", {}))
        receipt_path = validate_file_record(
            receipt, label=f"Surface attention receipt {attempt.get('stage')}"
        )
        receipt_payload, _, _ = load_json_object(
            receipt_path, label=f"Surface attention receipt {attempt.get('stage')}"
        )
        _require(
            receipt_payload.get("stage") == attempt.get("stage")
            and receipt_payload.get("attempt_index") == 1
            and receipt_payload.get("result") == "completed"
            and receipt_payload.get("command_status") == 0
            and receipt_payload.get("run_manifest") == expected_manifest_record
            and receipt_payload.get("terminal")
            == stage_terminals[str(attempt.get("stage"))],
            "attention-postcache receipt terminal differs",
        )
        expected_receipts.append(receipt)
    _require(
        receipts == expected_receipts,
        "attention-postcache screen receipt list differs from final closure",
    )
    return {
        "binding_mode": ATTENTION_BINDING_MODE,
        **{name: file_record(path) for name, path in paths.items()},
        "selected_variant": CONTEXT_POOLING_MODE,
        "selected_readouts": selected_readouts,
        "selected_cache_sidecars": selected_sidecars,
    }


def _legacy_surface_binding(
    *, surface: Path, candidate: str, train: list[dict], validation: list[dict]
) -> dict[str, Any]:
    paths = {
        "run_manifest": surface / "run_manifest.json",
        "cache_pairing": surface / "cache_pairing.json",
        "query_free_screen": surface / "query_free_screen.json",
        "screen_completion": surface / "screen.complete",
        "promotion_manifest": surface / "query_free_promotion_bundle.json",
        "promotion_completion": surface / "query_free_promotion.complete.json",
    }
    promotion, promotion_sha, promotion_path = load_json_object(
        paths["promotion_manifest"], label="Surface promotion manifest"
    )
    completion, _, _ = load_json_object(
        paths["promotion_completion"], label="Surface promotion completion"
    )
    _require(
        promotion.get("selected_candidate") == candidate
        and completion.get("selected_candidate") == candidate
        and completion.get("promotion_manifest_sha256") == promotion_sha
        and Path(str(completion.get("promotion_manifest", ""))).resolve() == promotion_path
        and completion.get("main_result_eligible") is False,
        "distill candidate is not the frozen query-free selection",
    )
    raw_caches = promotion.get("bindings", {}).get("caches")
    _require(isinstance(raw_caches, list), "Surface promotion lacks cache bindings")
    selected = [
        {"path": str(Path(str(row.get("path", ""))).resolve()), "sha256": row.get("sha256")}
        for row in raw_caches
        if isinstance(row, Mapping) and row.get("candidate") == candidate
    ]
    for index, record in enumerate(selected):
        validate_file_record(record, label=f"Surface selected cache {index}")
    _require(selected == train + validation, "distill caches differ from the frozen query-free bundle")
    return {name: file_record(path) for name, path in paths.items()}


def _surface_binding(
    *, surface_root: Path, candidate: str, train: list[dict], validation: list[dict]
) -> dict[str, Any]:
    surface = Path(surface_root).resolve()
    attention = surface / "attention_pooling_screen.json"
    legacy = surface / "query_free_promotion_bundle.json"
    _require(
        attention.is_file() != legacy.is_file(),
        "Surface root must expose exactly one legacy or attention-postcache authority",
    )
    if attention.is_file():
        return _attention_surface_binding(
            surface=surface,
            candidate=candidate,
            train=train,
            validation=validation,
        )
    return _legacy_surface_binding(
        surface=surface,
        candidate=candidate,
        train=train,
        validation=validation,
    )


def _surface_controls_by_seed(
    surface_promotion: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Return exact same-seed attention controls from the frozen Surface binding."""

    rows = surface_promotion.get("selected_readouts")
    _require(
        surface_promotion.get("binding_mode") == ATTENTION_BINDING_MODE
        and surface_promotion.get("selected_variant") == CONTEXT_POOLING_MODE
        and isinstance(rows, list)
        and len(rows) == len(REQUIRED_SEEDS),
        "text-response warm start requires the frozen attention controls",
    )
    controls: dict[int, dict[str, Any]] = {}
    expected_fields = {
        "seed",
        "checkpoint",
        "best_epoch",
        "best_selection_score",
        "validation",
    }
    for raw in rows:
        _require(
            isinstance(raw, Mapping) and set(raw) == expected_fields,
            "Surface control index fields differ",
        )
        seed = raw.get("seed")
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed in REQUIRED_SEEDS
            and seed not in controls,
            "Surface control seed index differs",
        )
        checkpoint = raw.get("checkpoint")
        validate_file_record(
            checkpoint, label=f"Surface control seed-{seed} checkpoint"
        )
        validation = raw.get("validation")
        _require(
            isinstance(validation, Mapping)
            and set(validation) == set(SURFACE_CONTROL_METRICS),
            f"Surface control seed-{seed} validation differs",
        )
        for field in SURFACE_CONTROL_METRICS:
            _finite(
                validation[field],
                label=f"Surface control seed-{seed} validation {field}",
            )
        _require(
            isinstance(raw.get("best_epoch"), int)
            and not isinstance(raw.get("best_epoch"), bool)
            and int(raw["best_epoch"]) > 0,
            f"Surface control seed-{seed} best epoch differs",
        )
        _finite(
            raw.get("best_selection_score"),
            label=f"Surface control seed-{seed} best score",
        )
        controls[int(seed)] = dict(raw)
    _require(
        set(controls) == set(REQUIRED_SEEDS),
        "Surface controls do not cover seeds 0/1/2",
    )
    return controls


def _seed_outputs(output_root: Path, candidate: str) -> list[dict[str, Any]]:
    root = output_root.resolve()
    rows = []
    for seed in REQUIRED_SEEDS:
        checkpoint = root / "readouts" / f"{candidate}_text_response_seed{seed}.pt"
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "report": str(checkpoint.with_suffix(checkpoint.suffix + ".json")),
                "training_log": str(root / "logs" / f"train_seed{seed}.log"),
                "audit_report": str(root / "audits" / f"audit_seed{seed}.json"),
                "guard_command": str(root / "receipts" / f"seed{seed}.command.json"),
                "guard_telemetry": str(root / "telemetry" / f"seed{seed}.csv"),
                "guard_receipt": str(root / "receipts" / f"seed{seed}.guard.json"),
                "kernel_journal": str(root / "receipts" / f"seed{seed}.kernel.log"),
                "gpu_preflight": str(root / "receipts" / f"seed{seed}.gpu_pre.json"),
                "gpu_postflight": str(root / "receipts" / f"seed{seed}.gpu_post.json"),
                "terminal": str(root / "receipts" / f"seed{seed}.complete.json"),
            }
        )
    return rows


def _thermal_contract(args: argparse.Namespace, guard: Path) -> dict[str, Any]:
    value = {
        "physical_gpu": 1,
        "maximum_temperature_c": int(args.gpu_max_temp_c),
        "maximum_start_temperature_c": int(args.gpu_start_max_temp_c),
        "maximum_power_limit_w": float(args.gpu_max_power_limit_w),
        "poll_seconds": int(args.gpu_poll_seconds),
        "soft_pause_temperature_c": int(args.gpu_soft_pause_temp_c),
        "soft_resume_temperature_c": int(args.gpu_soft_resume_temp_c),
        "peer_gpu": (
            None if args.gpu_peer_index is None else int(args.gpu_peer_index)
        ),
        "peer_pause_temperature_c": int(args.gpu_peer_pause_temp_c),
        "peer_resume_temperature_c": int(args.gpu_peer_resume_temp_c),
        "peer_quiet_seconds_before_launch": int(args.gpu_peer_quiet_seconds),
        "peer_max_power_w": float(args.gpu_peer_max_power_w),
        "peer_max_memory_mib": int(args.gpu_peer_max_memory_mib),
        "peer_max_utilization_pct": int(args.gpu_peer_max_util_pct),
        "peer_activity_action": str(args.gpu_peer_activity_action),
        "owner_pid_namespace_mode": str(args.gpu_owner_pid_namespace_mode),
        "guard": file_record(guard),
    }
    _require(
        0 < value["maximum_start_temperature_c"] < value["maximum_temperature_c"]
        and value["poll_seconds"] >= 1,
        "invalid hard thermal contract",
    )
    _require(
        value["owner_pid_namespace_mode"]
        == "exclusive-singleton-after-clear-v1",
        "invalid GPU owner PID namespace contract",
    )
    _require(
        (
            value["soft_pause_temperature_c"] == 0
            and value["soft_resume_temperature_c"] == 0
        )
        or 0
        < value["soft_resume_temperature_c"]
        < value["soft_pause_temperature_c"]
        < value["maximum_temperature_c"],
        "invalid own-GPU soft thermal contract",
    )
    if value["peer_gpu"] is None:
        _require(
            value["peer_pause_temperature_c"] == 0
            and value["peer_resume_temperature_c"] == 0
            and value["peer_quiet_seconds_before_launch"] == 0
            and value["peer_max_power_w"] == 0.0
            and value["peer_max_memory_mib"] == 0
            and value["peer_max_utilization_pct"] == 100,
            "peer thresholds require an explicit peer GPU",
        )
    else:
        _require(
            0
            <= value["peer_resume_temperature_c"]
            < value["peer_pause_temperature_c"]
            and 0 <= value["peer_max_utilization_pct"] <= 100
            and value["peer_activity_action"] in {"pause", "terminate"},
            "invalid peer-GPU thermal contract",
        )
    return value


def _implementation_sources(repo_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(repo_root / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _output_directory_bindings(
    output: Path,
    *,
    root_binding: Mapping[str, Any],
    create: bool,
) -> dict[str, dict[str, Any]]:
    return {
        name: _secure_output_directory(
            output / name,
            root_binding=root_binding,
            create=create,
        )
        for name in OUTPUT_DIRECTORY_NAMES
    }


def validate_authority_path_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = manifest.get("authority_contract")
    _require(isinstance(contract, Mapping), "distill authority path contract differs")
    lexical_root = Path(
        os.path.abspath(os.fspath(contract.get("main_output_root", "")))
    )
    _require(
        lexical_root == CANONICAL_LOCK_ROOT,
        "distill manifest does not use the canonical main output root",
    )
    root_binding = validate_canonical_lock_root_binding(
        contract.get("main_output_root_binding"),
        lock_root=lexical_root,
    )
    output = Path(os.path.abspath(os.fspath(contract.get("output_root", ""))))
    output_binding = _secure_output_directory(
        output,
        root_binding=root_binding,
        create=False,
    )
    _require(
        contract.get("output_root_binding") == output_binding,
        "distill output root identity changed",
    )
    expected_directories = _output_directory_bindings(
        output,
        root_binding=root_binding,
        create=False,
    )
    _require(
        contract.get("output_directory_bindings") == expected_directories,
        "distill output directory identity changed",
    )
    _require(
        contract.get("global_gpu_lock")
        == str(lexical_root / ".physical_gpu1.lock")
        and contract.get("output_run_lock")
        == str(output / "locks/text_response_distill.run.lock")
        and contract.get("root_path_protocol") == LOCK_ROOT_BINDING_VERSION,
        "distill authority lock paths changed",
    )
    return {
        "main_output_root_binding": root_binding,
        "output_root_binding": output_binding,
        "output_directory_bindings": expected_directories,
    }


def validate_gradient_design_diagnostic(
    value: object,
    *,
    surface_control: Mapping[str, Any],
    train_caches: Sequence[Mapping[str, Any]],
    radio_checkpoint: Mapping[str, Any],
    fit_text_bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently reproduce the frozen schema-2 gradient derivations."""

    _require(isinstance(value, Mapping), "gradient diagnostic is not an object")
    payload = dict(value)
    _require(
        set(payload)
        == {
            "schema_version",
            "artifact_type",
            "device",
            "rows",
            "scenes",
            "scene_response_objective",
            "losses",
            "gradient_l2",
            "equal_surface_gradient_lambdas",
            "weighted_component_gradient_l2_upper_bounds",
            "component_balance",
            "bindings",
        }
        and payload.get("schema_version") == 2
        and payload.get("artifact_type")
        == "warmstart_surface_text_response_gradient_diagnostic"
        and payload.get("device") == "cuda:0"
        and payload.get("rows") == GRADIENT_DIAGNOSTIC_ROWS
        and payload.get("scenes") == list(GRADIENT_DIAGNOSTIC_SCENES)
        and payload.get("scene_response_objective")
        == SCENE_RESPONSE_OBJECTIVE,
        "gradient diagnostic schema-2 contract differs",
    )
    bindings = payload.get("bindings")
    _require(
        isinstance(bindings, Mapping)
        and set(bindings)
        == {
            "surface_control",
            "radio_checkpoint",
            "train_caches",
            "fit_text_bank",
            "fit_text_bank_manifest",
            "implementation",
            "training_implementation",
            "loss_implementation",
        }
        and bindings.get("surface_control") == dict(surface_control)
        and bindings.get("radio_checkpoint") == dict(radio_checkpoint)
        and bindings.get("train_caches")
        == [dict(record) for record in train_caches]
        and bindings.get("fit_text_bank") == fit_text_bank.get("artifact")
        and bindings.get("fit_text_bank_manifest")
        == fit_text_bank.get("manifest")
        and bindings.get("implementation")
        == GRADIENT_DIAGNOSTIC_IMPLEMENTATION
        and bindings.get("training_implementation")
        == GRADIENT_DIAGNOSTIC_TRAINING_IMPLEMENTATION
        and bindings.get("loss_implementation")
        == GRADIENT_DIAGNOSTIC_LOSS_IMPLEMENTATION,
        "gradient diagnostic immutable bindings differ",
    )
    for implementation, label in (
        (GRADIENT_DIAGNOSTIC_IMPLEMENTATION, "diagnostic"),
        (GRADIENT_DIAGNOSTIC_TRAINING_IMPLEMENTATION, "training"),
        (GRADIENT_DIAGNOSTIC_LOSS_IMPLEMENTATION, "loss"),
    ):
        _require(
            file_record(implementation["path"]) == implementation,
            f"gradient diagnostic {label} implementation SHA-256 differs",
        )

    losses = payload.get("losses")
    gradients = payload.get("gradient_l2")
    equal_lambdas = payload.get("equal_surface_gradient_lambdas")
    bounds = payload.get("weighted_component_gradient_l2_upper_bounds")
    balance = payload.get("component_balance")
    _require(
        isinstance(losses, Mapping)
        and set(losses)
        == {
            "surface",
            "token",
            "descriptor",
            "relation",
            "independent_response",
            "scene_response",
            "scene_profile",
            "scene_ranking",
        }
        and isinstance(gradients, Mapping)
        and set(gradients)
        == {
            "surface",
            "independent_response",
            "scene_response",
            "scene_profile",
            "scene_ranking",
        }
        and isinstance(equal_lambdas, Mapping)
        and set(equal_lambdas)
        == {
            "independent_response",
            "scene_response",
            "scene_profile",
            "scene_ranking",
        }
        and isinstance(bounds, Mapping)
        and set(bounds) == {"scene_profile", "scene_ranking", "triangle_sum"}
        and isinstance(balance, Mapping)
        and set(balance)
        == {
            "raw_equalizing_profile_weight",
            "frozen_profile_weight",
            "weighted_profile_to_ranking_gradient_ratio",
            "derivation",
        },
        "gradient diagnostic numeric field topology differs",
    )
    loss_values = {
        field: _finite(losses.get(field), label=f"diagnostic loss {field}")
        for field in losses
    }
    gradient_values = {
        field: _finite(
            gradients.get(field), label=f"diagnostic gradient norm {field}"
        )
        for field in gradients
    }
    lambda_values = {
        field: _finite(
            equal_lambdas.get(field), label=f"diagnostic equal lambda {field}"
        )
        for field in equal_lambdas
    }
    bound_values = {
        field: _finite(bounds.get(field), label=f"diagnostic bound {field}")
        for field in bounds
    }
    _require(
        all(value >= 0.0 for value in loss_values.values())
        and all(value > 0.0 for value in gradient_values.values())
        and all(value > 0.0 for value in lambda_values.values())
        and all(value > 0.0 for value in bound_values.values()),
        "gradient diagnostic numeric values are invalid",
    )

    profile_weight = float(SCENE_RESPONSE_OBJECTIVE["profile_weight"])
    ranking_weight = float(SCENE_RESPONSE_OBJECTIVE["ranking_weight"])
    expected_scene_loss = (
        profile_weight * loss_values["scene_profile"]
        + ranking_weight * loss_values["scene_ranking"]
    )
    _require(
        math.isclose(
            loss_values["scene_response"],
            expected_scene_loss,
            rel_tol=5e-8,
            abs_tol=1e-9,
        ),
        "gradient diagnostic composite scene loss was not reproduced",
    )
    surface_gradient = gradient_values["surface"]
    _require(
        all(
            math.isclose(
                lambda_values[field],
                surface_gradient / gradient_values[field],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for field in (
                "independent_response",
                "scene_response",
                "scene_profile",
                "scene_ranking",
            )
        ),
        "gradient diagnostic equal-surface lambdas were not reproduced",
    )
    expected_profile_bound = profile_weight * gradient_values["scene_profile"]
    expected_ranking_bound = ranking_weight * gradient_values["scene_ranking"]
    expected_triangle_bound = expected_profile_bound + expected_ranking_bound
    _require(
        math.isclose(
            bound_values["scene_profile"],
            expected_profile_bound,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            bound_values["scene_ranking"],
            expected_ranking_bound,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            bound_values["triangle_sum"],
            expected_triangle_bound,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and gradient_values["scene_response"]
        <= expected_triangle_bound + 1e-12,
        "gradient diagnostic weighted component bounds were not reproduced",
    )
    raw_equalizing_weight = (
        ranking_weight
        * gradient_values["scene_ranking"]
        / gradient_values["scene_profile"]
    )
    weighted_ratio = expected_profile_bound / expected_ranking_bound
    _require(
        math.isclose(
            _finite(
                balance.get("raw_equalizing_profile_weight"),
                label="diagnostic raw equalizing profile weight",
            ),
            raw_equalizing_weight,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and _finite(
            balance.get("frozen_profile_weight"),
            label="diagnostic frozen profile weight",
        )
        == profile_weight
        and math.isclose(
            _finite(
                balance.get("weighted_profile_to_ranking_gradient_ratio"),
                label="diagnostic weighted component ratio",
            ),
            weighted_ratio,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and balance.get("derivation")
        == (
            "fit_only_seed0_fixed_calibration_batch_rounded_near_unit_"
            "weighted_component_gradient_ratio"
        ),
        "gradient diagnostic component balance was not reproduced",
    )
    return payload


def _calibration_bindings(
    args: argparse.Namespace,
    *,
    surface_promotion: Mapping[str, Any],
    train: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifests = _seed_path_map(
        args.calibration_manifest, label="calibration manifest"
    )
    audits = _seed_path_map(args.calibration_audit, label="calibration audit")
    diagnostic_record = file_record(args.gradient_diagnostic)
    _require(
        os.path.abspath(os.fspath(args.gradient_diagnostic))
        == FROZEN_GRADIENT_DESIGN_DIAGNOSTIC_LEXICAL_PATH
        and diagnostic_record == FROZEN_GRADIENT_DESIGN_DIAGNOSTIC
        and diagnostic_record["sha256"] == str(args.gradient_diagnostic_sha256),
        "formal schema-2 gradient design diagnostic binding differs",
    )
    diagnostic, _, _ = load_json_object(
        args.gradient_diagnostic, label="formal gradient design diagnostic"
    )
    controls = _surface_controls_by_seed(surface_promotion)
    validate_gradient_design_diagnostic(
        diagnostic,
        surface_control=controls[0]["checkpoint"],
        train_caches=train,
        radio_checkpoint=file_record(args.radio_checkpoint),
        fit_text_bank={
            "artifact": file_record(args.fit_text_bank),
            "manifest": file_record(args.fit_text_bank_manifest),
        },
    )
    rows: list[dict[str, Any]] = []
    expected_objective = {
        "surface_objective": (
            "token_weight*(1-cosine_summary_token)"
            "+masked_mean_one_minus_all_view_cosine"
            "+relation_weight*smooth_l1_descriptor_relation"
        ),
        "token_weight": TRAINING_CONTRACT["token_weight"],
        "relation_weight": TRAINING_CONTRACT["relation_weight"],
        "independent_response_loss": INDEPENDENT_RESPONSE_LOSS,
        "scene_response_loss": SCENE_RESPONSE_LOSS,
        "scene_response_objective": dict(SCENE_RESPONSE_OBJECTIVE),
        "scene_tie_tolerance": SCENE_RESPONSE_OBJECTIVE["tie_tolerance"],
        "branch_gradient_target_ratio": RESPONSE_BRANCH_GRADIENT_RATIO,
        "combined_response_gradient_ratio_upper_bound": (
            TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND
        ),
        "upper_bound_derivation": (
            "triangle_inequality_sum_of_two_branch_l2_budgets"
        ),
        "gradient_bound_scope": (
            "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
        ),
        "training_batching": "shuffle_complete_scene_groups_no_partial_scenes_v1",
        "max_complete_scene_batch_rows": TRAINING_CONTRACT[
            "max_complete_scene_batch_rows"
        ],
    }
    for seed in REQUIRED_SEEDS:
        payload, payload_sha, payload_path = load_json_object(
            manifests[seed], label=f"seed-{seed} calibration manifest"
        )
        control = payload.get("surface_control")
        _require(
            payload.get("schema_version") == 2
            and payload.get("artifact_type")
            == "surface_text_response_gradient_calibration"
            and payload.get("algorithm_version")
            == CALIBRATION_ALGORITHM_VERSION
            and payload.get("objective_contract") == expected_objective
            and payload.get("seed") == seed
            and isinstance(control, Mapping)
            and control.get("seed") == seed
            and {key: control.get(key) for key in ("path", "sha256")}
            == controls[seed]["checkpoint"]
            and control.get("train_caches") == train
            and control.get("validation_caches") == validation
            and payload.get("design_diagnostic", {}).get("path")
            == diagnostic_record["path"]
            and payload.get("design_diagnostic", {}).get("sha256")
            == diagnostic_record["sha256"],
            f"seed-{seed} calibration/control binding differs",
        )
        audit, audit_sha, audit_path = load_json_object(
            audits[seed], label=f"seed-{seed} calibration CPU audit"
        )
        _require(
            audit.get("status") == "calibration_verified"
            and audit.get("seed") == seed
            and audit.get("calibration_manifest") == str(payload_path)
            and audit.get("calibration_manifest_sha256") == payload_sha
            and audit.get("response_lambdas")
            == payload.get("gradient_contract", {}).get("response_lambdas")
            and audit.get("device") == "cpu",
            f"seed-{seed} calibration CPU audit differs",
        )
        rows.append(
            {
                "seed": seed,
                "manifest": {"path": str(payload_path), "sha256": payload_sha},
                "audit": {"path": str(audit_path), "sha256": audit_sha},
                "surface_control": dict(control),
                "response_lambdas": dict(audit["response_lambdas"]),
            }
        )
    return rows, diagnostic_record


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    output = Path(os.path.abspath(os.fspath(args.output_root)))
    lock_root = Path(os.path.abspath(os.fspath(args.lock_root)))
    _require(
        lock_root == CANONICAL_LOCK_ROOT,
        "GPU1 authority lock must be /root/RADIO-GS/output/.physical_gpu1.lock",
    )
    root_binding = inspect_canonical_lock_root(lock_root)
    output_binding = _secure_output_directory(
        output,
        root_binding=root_binding,
        create=False,
    )
    output_directories = _output_directory_bindings(
        output,
        root_binding=root_binding,
        create=False,
    )
    candidate = str(args.candidate)
    _require(candidate in ALLOWED_CANDIDATES, f"unknown frozen SurfaceRegion candidate: {candidate}")
    train = _cache_records(args.train_caches, expected_count=4, working_directory=repo)
    validation = _cache_records(
        args.validation_caches, expected_count=2, working_directory=repo
    )
    initial_check, _, _ = load_json_object(args.initial_gpu_preflight, label="initial GPU1 preflight")
    _require(
        initial_check.get("artifact_type") == GPU_CHECK_ARTIFACT_TYPE
        and initial_check.get("phase") == "initial_manifest_binding",
        "initial GPU1 preflight differs",
    )
    gpu_identity = initial_check.get("gpu_identity")
    _require(isinstance(gpu_identity, Mapping), "initial GPU1 preflight lacks identity")
    runtime_closure = build_runtime_closure(repo)
    outputs = _seed_outputs(Path(output_binding["resolved_path"]), candidate)
    surface_promotion = _surface_binding(
        surface_root=Path(args.surface_root),
        candidate=candidate,
        train=train,
        validation=validation,
    )
    _surface_controls_by_seed(surface_promotion)
    calibrations, gradient_diagnostic = _calibration_bindings(
        args,
        surface_promotion=surface_promotion,
        train=train,
        validation=validation,
    )
    command_arguments = _training_command_arguments(args)
    _require(
        command_arguments["output_root"] == str(output),
        "distill command output root is not the lexical authority output root",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "authority_status": "query_free_three_seed_gpu1_run_frozen",
        "candidate": candidate,
        "surface_promotion": surface_promotion,
        "train_caches": train,
        "validation_caches": validation,
        "fit_text_bank": {
            "artifact": file_record(args.fit_text_bank),
            "manifest": file_record(args.fit_text_bank_manifest),
        },
        "radio_checkpoint": file_record(args.radio_checkpoint),
        "calibrations": calibrations,
        "gradient_design_diagnostic": gradient_diagnostic,
        "initial_gpu_preflight": file_record(args.initial_gpu_preflight),
        "gpu_identity": dict(gpu_identity),
        "outputs": outputs,
        "training_contract": dict(TRAINING_CONTRACT),
        "training_command_contract": _build_training_command_contract(
            repo_root=repo,
            arguments=command_arguments,
            outputs=outputs,
            surface_promotion=surface_promotion,
        ),
        "thermal_safety_contract": _thermal_contract(args, Path(args.thermal_guard)),
        "implementation_sources": _implementation_sources(repo),
        "runtime_closure": runtime_closure,
        "authority_contract": {
            "global_gpu_lock": str(
                (lock_root / ".physical_gpu1.lock")
            ),
            "output_run_lock": str(output / "locks/text_response_distill.run.lock"),
            "source_snapshot_root": str(repo),
            "main_output_root": str(lock_root),
            "main_output_root_binding": root_binding,
            "output_root": str(output),
            "output_root_binding": output_binding,
            "output_directory_bindings": output_directories,
            "root_path_protocol": LOCK_ROOT_BINDING_VERSION,
            "lock_protocol": "nofollow_regular_nlink1_flock_exclusive_nonblocking_v1",
            "global_gpu_kernel_singleton_protocol": GPU1_SINGLETON_PROTOCOL,
            "global_gpu_kernel_singleton_inherited_fd_verified": True,
            "input_loading": "stable_descriptor_hash_load_rehash_no_unsafe_fallback_v1",
            "seed_resume": "skip_only_exact_guarded_terminal_v1",
            "closure_verification": "before_and_after_every_seed_v1",
            "kernel_fault_gate": "bound_journal_interval_no_xid_pcie_fault_v1",
        },
    }
    return manifest


def create_or_verify_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.run_manifest).resolve()
    if path.exists() or path.is_symlink():
        observed, digest, source = load_json_object(path, label="distill run manifest")
        selection_contract = _manifest_selection_contract(
            observed, digest=digest, source=source
        )
        if selection_contract == LEGACY_EPOCH_SELECTION:
            _manifest(path)
            return {
                "status": "verified",
                "manifest": {"path": str(source), "sha256": digest},
            }
        expected = build_manifest(args)
        _require(observed == expected, "distill run manifest/runtime closure changed")
        return {"status": "verified", "manifest": {"path": str(source), "sha256": digest}}
    expected = build_manifest(args)
    write_frozen_json(path, expected)
    return {"status": "created", "manifest": file_record(path)}


def _manifest_selection_contract(
    value: Mapping[str, Any], *, digest: str, source: Path
) -> str:
    training_contract = value.get("training_contract")
    if training_contract == TRAINING_CONTRACT:
        return EPOCH_SELECTION
    registered_legacy = (
        training_contract == LEGACY_TRAINING_CONTRACT
        and str(source.resolve()) == REGISTERED_LEGACY_MANIFEST["path"]
        and digest == REGISTERED_LEGACY_MANIFEST["sha256"]
    )
    _require(
        registered_legacy,
        "distill run manifest method contract is neither current nor the exact "
        "registered formal legacy manifest",
    )
    return LEGACY_EPOCH_SELECTION


def _manifest(path: Path) -> tuple[dict[str, Any], str, Path]:
    value, digest, source = load_json_object(path, label="distill run manifest")
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == ARTIFACT_TYPE,
        "distill run manifest schema/artifact contract differs",
    )
    _manifest_selection_contract(value, digest=digest, source=source)
    validate_authority_path_contract(value)
    validate_training_command_contract(value, manifest_path=source)
    return value, digest, source


def _seed_row(manifest: Mapping[str, Any], seed: int) -> dict[str, Any]:
    rows = manifest.get("outputs")
    _require(isinstance(rows, list) and len(rows) == len(REQUIRED_SEEDS), "distill output index differs")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("seed") == seed]
    _require(len(matches) == 1, f"distill output index lacks seed {seed}")
    return dict(matches[0])


def _calibration_row(manifest: Mapping[str, Any], seed: int) -> dict[str, Any]:
    rows = manifest.get("calibrations")
    _require(
        isinstance(rows, list) and len(rows) == len(REQUIRED_SEEDS),
        "distill calibration index differs",
    )
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("seed") == seed
    ]
    _require(len(matches) == 1, f"distill calibration index lacks seed {seed}")
    return dict(matches[0])


def validate_receipt_training_command(
    command: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    seed: int,
) -> int:
    expected_argv = expected_training_argv(
        manifest,
        manifest_path=manifest_path,
        seed=seed,
    )
    prepared_epoch = command.get("prepared_epoch")
    _require(
        command.get("run_manifest")
        == {"path": str(manifest_path.resolve()), "sha256": manifest_sha256}
        and command.get("seed") == seed
        and command.get("scene") == manifest.get("candidate")
        and command.get("gpu_identity") == manifest.get("gpu_identity")
        and command.get("argv") == expected_argv
        and command.get("argv_sha256") == canonical_json_sha256(expected_argv)
        and isinstance(prepared_epoch, int)
        and not isinstance(prepared_epoch, bool)
        and prepared_epoch > 0,
        f"seed-{seed} guard command differs from the frozen complete training argv",
    )
    return prepared_epoch


def classify_seed(manifest_path: Path, seed: int) -> str:
    manifest, _, _ = _manifest(manifest_path)
    row = _seed_row(manifest, seed)
    terminal = Path(row["terminal"])
    paths = [Path(str(value)) for key, value in row.items() if key != "seed"]
    if terminal.exists() or terminal.is_symlink():
        validate_seed_terminal(manifest_path, seed)
        return "complete"
    present = [str(path) for path in paths if path.exists() or path.is_symlink()]
    _require(not present, f"seed-{seed} has partial/no-terminal artifacts: {present}")
    return "pending"


def _finite(value: object, *, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _recompute_v3_response_epoch_selection(history: object) -> tuple[int, float]:
    _require(isinstance(history, list) and bool(history), "response history is empty")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        _require(isinstance(raw, Mapping), f"response history row {index} is invalid")
        row = dict(raw)
        _require(
            row.get("epoch") == index,
            "response history must start at control epoch 0 and be contiguous",
        )
        _require(
            row.get("scene_response_objective") == SCENE_RESPONSE_OBJECTIVE,
            "response history scene-response objective differs",
        )
        if index:
            for field in (
                "independent_response_loss",
                "scene_response_loss",
                "scene_profile_loss",
                "scene_ranking_loss",
            ):
                _require(
                    _finite(row.get(field), label=f"history {field}") >= 0.0,
                    f"history {field} must be non-negative",
                )
        rows.append(row)

    def scene_metrics(
        row: Mapping[str, Any],
        *,
        expected_scenes: set[str] | None,
        label: str,
    ) -> dict[str, dict[str, float]]:
        raw_scenes = row.get("text_response_scene_metrics")
        _require(
            isinstance(raw_scenes, Mapping) and bool(raw_scenes),
            f"{label} text_response_scene_metrics must be non-empty",
        )
        normalized: dict[str, dict[str, float]] = {}
        expected_fields = {
            *FIT_RESPONSE_SCENE_QUALITY_METRICS,
            *FIT_RESPONSE_SCENE_ERROR_METRICS,
        }
        for scene, raw_metrics in raw_scenes.items():
            _require(
                isinstance(scene, str)
                and bool(scene)
                and scene not in normalized
                and isinstance(raw_metrics, Mapping)
                and set(raw_metrics) == expected_fields,
                f"{label} fit scene metrics are malformed",
            )
            normalized[scene] = {
                field: _finite(
                    raw_metrics.get(field),
                    label=f"{label} scene {scene} {field}",
                )
                for field in (
                    *FIT_RESPONSE_SCENE_QUALITY_METRICS,
                    *FIT_RESPONSE_SCENE_ERROR_METRICS,
                )
            }
        _require(
            expected_scenes is None or set(normalized) == expected_scenes,
            f"{label} fit scene IDs drifted from epoch 0",
        )
        expected_worst = {
            "text_response_scene_worst_smooth_l1": max(
                metrics["smooth_l1"] for metrics in normalized.values()
            ),
            "text_response_scene_worst_mae": max(
                metrics["mae"] for metrics in normalized.values()
            ),
            **{
                f"text_response_scene_worst_{field}": min(
                    metrics[field] for metrics in normalized.values()
                )
                for field in FIT_RESPONSE_SCENE_QUALITY_METRICS
            },
        }
        _require(
            all(
                math.isclose(
                    _finite(row.get(field), label=f"{label} {field}"),
                    value,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                for field, value in expected_worst.items()
            ),
            f"{label} fit scene worst-case summaries differ",
        )
        return normalized

    control_surface = {
        field: _finite(rows[0].get(field), label=f"history control {field}")
        for field in SURFACE_CONTROL_METRICS
    }
    control_fit_quality = {
        field: _finite(rows[0].get(field), label=f"history control {field}")
        for field in FIT_RESPONSE_QUALITY_METRICS
    }
    control_scenes = scene_metrics(
        rows[0], expected_scenes=None, label="history fit-response control"
    )
    ranked: list[tuple[int, tuple[float, ...]]] = []
    for row in rows:
        fields = (
            "surface_selection_score",
            "text_support_top1_agreement",
            "text_response_smooth_l1",
            "text_response_mae",
            "descriptor_relation_smooth_l1",
            *FIT_RESPONSE_QUALITY_METRICS,
            *SURFACE_CONTROL_METRICS,
        )
        values = {
            field: _finite(row.get(field), label=f"history {field}")
            for field in fields
        }
        current_scenes = scene_metrics(
            row,
            expected_scenes=set(control_scenes),
            label=f"history epoch {row['epoch']}",
        )
        surface_deltas = {
            field: values[field] - control_surface[field]
            for field in SURFACE_CONTROL_METRICS
        }
        surface_feasible = all(
            delta >= -SURFACE_CONTROL_NONINFERIORITY_TOLERANCE
            or math.isclose(
                delta,
                -SURFACE_CONTROL_NONINFERIORITY_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for delta in surface_deltas.values()
        )
        aggregate_quality_deltas = {
            field: values[field] - control_fit_quality[field]
            for field in FIT_RESPONSE_QUALITY_METRICS
        }
        aggregate_fit_feasible = all(
            delta >= -FIT_RESPONSE_NONINFERIORITY_TOLERANCE
            or math.isclose(
                delta,
                -FIT_RESPONSE_NONINFERIORITY_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for delta in aggregate_quality_deltas.values()
        )
        scene_deltas: dict[str, dict[str, float]] = {}
        per_scene_fit_feasible = True
        for scene in sorted(control_scenes):
            scene_deltas[scene] = {
                field: current_scenes[scene][field] - control_scenes[scene][field]
                for field in (
                    *FIT_RESPONSE_SCENE_QUALITY_METRICS,
                    *FIT_RESPONSE_SCENE_ERROR_METRICS,
                )
            }
            quality_feasible = all(
                scene_deltas[scene][field]
                >= -FIT_RESPONSE_NONINFERIORITY_TOLERANCE
                or math.isclose(
                    scene_deltas[scene][field],
                    -FIT_RESPONSE_NONINFERIORITY_TOLERANCE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field in FIT_RESPONSE_SCENE_QUALITY_METRICS
            )
            error_feasible = all(
                scene_deltas[scene][field]
                <= FIT_RESPONSE_NONINFERIORITY_TOLERANCE
                or math.isclose(
                    scene_deltas[scene][field],
                    FIT_RESPONSE_NONINFERIORITY_TOLERANCE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field in FIT_RESPONSE_SCENE_ERROR_METRICS
            )
            per_scene_fit_feasible = (
                per_scene_fit_feasible and quality_feasible and error_feasible
            )
        fit_feasible = aggregate_fit_feasible and per_scene_fit_feasible
        feasible = surface_feasible and fit_feasible
        expected_fit_deltas = {
            "aggregate_quality": aggregate_quality_deltas,
            "per_scene": scene_deltas,
        }
        expected_improvement = {
            "smooth_l1": _finite(
                rows[0].get("text_response_smooth_l1"),
                label="history control text_response_smooth_l1",
            )
            - values["text_response_smooth_l1"],
            "mae": _finite(
                rows[0].get("text_response_mae"),
                label="history control text_response_mae",
            )
            - values["text_response_mae"],
        }
        _require(
            row.get("surface_control_deltas") == surface_deltas
            and row.get("surface_control_feasible") is surface_feasible
            and row.get("surface_control_tolerance")
            == SURFACE_CONTROL_NONINFERIORITY_TOLERANCE,
            "response history Surface-control feasibility was not reproduced",
        )
        _require(
            row.get("fit_response_control_deltas") == expected_fit_deltas
            and row.get("fit_response_aggregate_control_feasible")
            is aggregate_fit_feasible
            and row.get("fit_response_per_scene_control_feasible")
            is per_scene_fit_feasible
            and row.get("fit_response_control_feasible") is fit_feasible
            and row.get("fit_response_control_tolerance")
            == FIT_RESPONSE_NONINFERIORITY_TOLERANCE
            and row.get("response_selection_feasible") is feasible
            and row.get(
                "fit_response_error_improvement_control_minus_candidate"
            )
            == expected_improvement,
            "response history fit-scene feasibility was not reproduced",
        )
        if feasible:
            ranked.append(
                (
                    int(row["epoch"]),
                    (
                        -values["text_response_smooth_l1"],
                        -values["text_response_mae"],
                        values["text_response_ranking_spearman_p05"],
                        values["text_response_ranking_spearman_mean"],
                        values["text_response_profile_cosine_p05"],
                        values["text_response_profile_cosine_mean"],
                        values["text_response_top_decile_overlap_p05"],
                        values["text_response_top_decile_overlap_mean"],
                        values["text_support_top1_agreement"],
                        -values["descriptor_relation_smooth_l1"],
                        values["surface_selection_score"],
                    ),
                )
            )
    _require(bool(ranked), "response history has no robust-feasible epoch")
    best_epoch, _ = max(ranked, key=lambda value: value[1])
    best_score = float(rows[best_epoch]["surface_selection_score"])
    for row in rows:
        expected = best_score if int(row["epoch"]) == best_epoch else -1.0
        _require(
            math.isclose(
                _finite(
                    row.get("selection_score"),
                    label="history selection_score",
                ),
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "response history selection_score was not independently reproduced",
        )
    return best_epoch, best_score


def _response_epoch_rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -_finite(row.get("text_response_smooth_l1"), label="history rank smooth_l1"),
        -_finite(row.get("text_response_mae"), label="history rank mae"),
        _finite(
            row.get("text_response_ranking_spearman_p05"),
            label="history rank ranking_spearman_p05",
        ),
        _finite(
            row.get("text_response_ranking_spearman_mean"),
            label="history rank ranking_spearman_mean",
        ),
        _finite(
            row.get("text_response_profile_cosine_p05"),
            label="history rank profile_cosine_p05",
        ),
        _finite(
            row.get("text_response_profile_cosine_mean"),
            label="history rank profile_cosine_mean",
        ),
        _finite(
            row.get("text_response_top_decile_overlap_p05"),
            label="history rank top_decile_overlap_p05",
        ),
        _finite(
            row.get("text_response_top_decile_overlap_mean"),
            label="history rank top_decile_overlap_mean",
        ),
        _finite(
            row.get("text_support_top1_agreement"),
            label="history rank text_support_top1_agreement",
        ),
        -_finite(
            row.get("descriptor_relation_smooth_l1"),
            label="history rank descriptor_relation_smooth_l1",
        ),
        _finite(
            row.get("surface_selection_score"),
            label="history rank surface_selection_score",
        ),
    )


def _recompute_history_hash_chain(history: Sequence[Mapping[str, Any]]) -> str:
    _require(bool(history), "response history hash chain is empty")
    previous_sha256: str | None = None
    for index, row in enumerate(history):
        _require(isinstance(row, Mapping), f"history hash-chain row {index} is invalid")
        payload = dict(row)
        payload.pop("history_hash_chain", None)
        payload.pop("selection_score", None)
        digest = canonical_json_sha256(
            {"previous_sha256": previous_sha256, "record": payload}
        )
        expected = {
            "algorithm": HISTORY_HASH_CHAIN_ALGORITHM,
            "previous_sha256": previous_sha256,
            "sha256": digest,
        }
        _require(
            row.get("history_hash_chain") == expected,
            f"response history hash chain differs at row {index}",
        )
        previous_sha256 = digest
    assert previous_sha256 is not None
    return previous_sha256


def _replay_v4_proposal_state_machine(
    history: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    _require(
        isinstance(epochs, int)
        and not isinstance(epochs, bool)
        and epochs > 0
        and isinstance(patience, int)
        and not isinstance(patience, bool)
        and patience > 0,
        "proposal state-machine epochs/patience contract differs",
    )
    _require(
        bool(history) and len(history) <= epochs + 1,
        "proposal state-machine history length differs",
    )
    control = history[0]
    _require(isinstance(control, Mapping), "proposal state-machine control row is invalid")
    control_hash = control.get("anchor_state_dict_sha256_after_proposal")
    _require(
        control.get("epoch") == 0
        and control.get("initialization") == "frozen_surface_control_checkpoint"
        and control.get("state_machine_role") == "frozen_control_initial_anchor"
        and control.get("accepted") is True
        and control.get("rejected") is False
        and control.get("anchor_epoch_after_proposal") == 0
        and SHA256_PATTERN.fullmatch(str(control_hash)) is not None
        and control.get("best_updated") is True
        and control.get("best_epoch_after_proposal") == 0
        and control.get("best_state_dict_sha256_after_proposal") == control_hash
        and control.get("patience_stale_after_proposal") == 0
        and control.get("patience_stop_after_proposal") is False
        and "proposal" not in control
        and "proposal_losses" not in control
        and "loss_measurement_state" not in control,
        "proposal state-machine control row differs",
    )
    anchor_epoch = 0
    anchor_hash = str(control_hash)
    best_epoch = 0
    best_hash = str(control_hash)
    best_rank = _response_epoch_rank(control)
    stale = 0
    accepted_count = 0
    rejected_count = 0
    proposal_fields = {
        "index",
        "source_anchor_epoch",
        "anchor_state_dict_sha256",
        "raw_state_dict_sha256",
        "trial_state_dict_sha256",
        "alpha_numerator",
        "alpha_denominator",
        "optimizer_state_reset",
        "validation_evaluations",
        "backtracking",
        "persistent_generator",
    }
    for index, row in enumerate(history[1:], start=1):
        _require(isinstance(row, Mapping), f"proposal state-machine row {index} is invalid")
        proposal = row.get("proposal")
        _require(
            row.get("epoch") == index
            and row.get("state_machine_role") == "fixed_micro_ray_trial"
            and isinstance(proposal, Mapping)
            and set(proposal) == proposal_fields
            and proposal.get("index") == index
            and proposal.get("source_anchor_epoch") == anchor_epoch
            and proposal.get("anchor_state_dict_sha256") == anchor_hash
            and SHA256_PATTERN.fullmatch(
                str(proposal.get("raw_state_dict_sha256", ""))
            )
            is not None
            and SHA256_PATTERN.fullmatch(
                str(proposal.get("trial_state_dict_sha256", ""))
            )
            is not None
            and proposal.get("alpha_numerator") == PROPOSAL_ALPHA_NUMERATOR
            and proposal.get("alpha_denominator") == PROPOSAL_ALPHA_DENOMINATOR
            and proposal.get("optimizer_state_reset") is True
            and proposal.get("validation_evaluations") == 1
            and proposal.get("backtracking") == "none_fixed_alpha_single_trial"
            and proposal.get("persistent_generator")
            == "advanced_never_rolled_back",
            f"proposal state-machine row {index} differs",
        )
        proposal_losses = row.get("proposal_losses")
        _require(
            row.get("loss_measurement_state")
            == PROPOSAL_LOSS_MEASUREMENT_STATE
            and isinstance(proposal_losses, Mapping)
            and tuple(proposal_losses) == PROPOSAL_LOSS_FIELDS,
            f"proposal loss-accounting row {index} differs",
        )
        for field, flat_field in LEGACY_FLAT_PROPOSAL_LOSS_FIELDS.items():
            value = _finite(
                proposal_losses.get(field),
                label=f"proposal loss {field} at row {index}",
            )
            _require(
                value >= 0.0 and row.get(flat_field) == proposal_losses.get(field),
                f"proposal loss {field} differs at row {index}",
            )

        accepted = row.get("response_selection_feasible") is True
        _require(
            row.get("accepted") is accepted
            and row.get("rejected") is (not accepted),
            f"proposal acceptance differs at row {index}",
        )
        if accepted:
            anchor_epoch = index
            anchor_hash = str(proposal["trial_state_dict_sha256"])
            accepted_count += 1
        else:
            rejected_count += 1

        candidate_rank = _response_epoch_rank(row)
        best_updated = accepted and candidate_rank > best_rank
        if best_updated:
            best_epoch = index
            best_hash = str(proposal["trial_state_dict_sha256"])
            best_rank = candidate_rank
            stale = 0
        else:
            stale += 1
        patience_stop = stale >= patience
        _require(
            row.get("anchor_epoch_after_proposal") == anchor_epoch
            and row.get("anchor_state_dict_sha256_after_proposal") == anchor_hash
            and row.get("best_updated") is best_updated
            and row.get("best_epoch_after_proposal") == best_epoch
            and row.get("best_state_dict_sha256_after_proposal") == best_hash
            and row.get("patience_stale_after_proposal") == stale
            and row.get("patience_stop_after_proposal") is patience_stop
            and (not patience_stop or index == len(history) - 1),
            f"proposal transition differs at row {index}",
        )
    _require(
        len(history) == epochs + 1
        or history[-1].get("patience_stop_after_proposal") is True,
        "proposal history ended before the frozen epoch/patience stop condition",
    )
    return {
        "accepted_anchor": {
            "epoch": anchor_epoch,
            "state_dict_sha256": anchor_hash,
            "accepted_proposal_count": accepted_count,
            "rejected_proposal_count": rejected_count,
        },
        "best_epoch": best_epoch,
        "best_state_dict_sha256": best_hash,
        "history_hash_chain_sha256": _recompute_history_hash_chain(history),
    }


def recompute_response_epoch_selection(
    history: object, *, selection_contract: str = EPOCH_SELECTION
) -> tuple[int, float]:
    _require(
        selection_contract in {EPOCH_SELECTION, LEGACY_EPOCH_SELECTION},
        "response epoch-selection contract is not registered",
    )
    selected = _recompute_v3_response_epoch_selection(history)
    if selection_contract == EPOCH_SELECTION:
        assert isinstance(history, list)
        replay = _replay_v4_proposal_state_machine(
            history,
            epochs=int(TRAINING_CONTRACT["epochs"]),
            patience=int(TRAINING_CONTRACT["patience"]),
        )
        _require(
            replay["best_epoch"] == selected[0],
            "proposal replay and robust global-best selection differ",
        )
    return selected


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.is_floating_point():
        array = tensor.to(torch.float32).numpy().astype("<f4", copy=False)
        dtype = "float32-le"
    elif tensor.dtype == torch.bool:
        array = tensor.to(torch.uint8).numpy()
        dtype = "bool-u8"
    elif tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        array = tensor.to(torch.int64).numpy().astype("<i8", copy=False)
        dtype = "int64-le"
    else:
        raise ValueError(f"unsupported checkpoint tensor dtype: {tensor.dtype}")
    header = json.dumps(
        {"dtype": dtype, "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(state: object) -> str:
    _require(isinstance(state, Mapping) and bool(state), "checkpoint state_dict is invalid")
    records = []
    for name in sorted(state):
        tensor = state[name]
        _require(
            isinstance(name, str) and bool(name) and torch.is_tensor(tensor),
            "checkpoint state_dict fields differ",
        )
        contiguous = tensor.detach().cpu().contiguous()
        _require(
            not (contiguous.is_floating_point() or contiguous.is_complex())
            or bool(torch.isfinite(contiguous).all()),
            f"checkpoint state_dict tensor is non-finite: {name}",
        )
        records.append(
            {
                "name": name,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "tensor_sha256": _tensor_sha256(contiguous),
            }
        )
    return canonical_json_sha256(records)


def validate_v4_proposal_checkpoint(
    checkpoint: Mapping[str, Any], *, report: Mapping[str, Any] | None = None
) -> tuple[int, float]:
    _require(isinstance(checkpoint, Mapping), "response checkpoint is invalid")
    history = checkpoint.get("history")
    _require(isinstance(history, list), "response checkpoint history is invalid")
    selected = recompute_response_epoch_selection(
        history, selection_contract=EPOCH_SELECTION
    )
    training_config = checkpoint.get("training_config")
    provenance = checkpoint.get("provenance")
    _require(
        checkpoint.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE
        and isinstance(training_config, Mapping)
        and training_config.get("epochs") == TRAINING_CONTRACT["epochs"]
        and training_config.get("patience") == TRAINING_CONTRACT["patience"]
        and training_config.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE
        and isinstance(provenance, Mapping)
        and provenance.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE,
        "response checkpoint proposal contract differs",
    )
    replay = _replay_v4_proposal_state_machine(
        history,
        epochs=int(training_config["epochs"]),
        patience=int(training_config["patience"]),
    )
    best_state_hash = _state_dict_sha256(checkpoint.get("state_dict"))
    _require(
        checkpoint.get("accepted_anchor") == replay["accepted_anchor"]
        and checkpoint.get("best_epoch") == replay["best_epoch"] == selected[0]
        and checkpoint.get("best_state_dict_sha256")
        == replay["best_state_dict_sha256"]
        == best_state_hash
        and checkpoint.get("history_hash_chain_sha256")
        == replay["history_hash_chain_sha256"],
        "response checkpoint accepted-anchor/best/hash-chain fields differ",
    )
    if report is not None:
        _require(
            report.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE
            and report.get("accepted_anchor") == replay["accepted_anchor"]
            and report.get("best_epoch") == replay["best_epoch"]
            and report.get("best_state_dict_sha256") == best_state_hash
            and report.get("history_hash_chain_sha256")
            == replay["history_hash_chain_sha256"],
            "trainer report proposal/anchor/best/hash-chain fields differ",
        )
    return selected


def _expected_surface_control_binding(
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    control = _surface_controls_by_seed(manifest.get("surface_promotion", {}))[seed]
    record = dict(control["checkpoint"])
    control_payload, control_sha, control_path = (
        load_sha_bound_project_checkpoint_mapping(
            record["path"],
            expected_sha256=str(record["sha256"]),
            map_location="cpu",
            label=f"seed-{seed} frozen Surface control",
        )
    )
    architecture = checkpoint.get("architecture")
    control_config = control_payload.get("training_config")
    control_provenance = control_payload.get("provenance")
    _require(
        control_sha == record["sha256"]
        and control_payload.get("architecture") == architecture
        and isinstance(control_config, Mapping)
        and control_config.get("seed") == seed
        and control_config.get("context_pooling_mode") == CONTEXT_POOLING_MODE
        and isinstance(control_provenance, Mapping)
        and control_provenance.get("random_seed_contract", {}).get("seed") == seed
        and control_provenance.get("train", {}).get("cache_paths")
        == [row["path"] for row in manifest.get("train_caches", [])]
        and control_provenance.get("validation", {}).get("cache_paths")
        == [row["path"] for row in manifest.get("validation_caches", [])]
        and control_payload.get("best_epoch") == control.get("best_epoch")
        and math.isclose(
            _finite(
                control_payload.get("best_selection_score"),
                label=f"seed-{seed} Surface control best score",
            ),
            _finite(
                control.get("best_selection_score"),
                label=f"seed-{seed} indexed Surface control best score",
            ),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        f"seed-{seed} frozen Surface control binding differs",
    )
    return {
        "path": str(control_path),
        "sha256": control_sha,
        "seed": seed,
        "architecture": architecture,
        "train_caches": manifest["train_caches"],
        "validation_caches": manifest["validation_caches"],
        "source_best_epoch": int(control["best_epoch"]),
        "source_best_selection_score": float(control["best_selection_score"]),
    }


def _positive_epoch(value: object, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} is not a positive integer epoch",
    )
    return int(value)


def _kernel_journal_record(
    path: Path,
    *,
    seed: int,
    start_epoch: int,
    end_epoch: int,
) -> dict[str, Any]:
    def load(handle) -> str:
        try:
            return handle.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("kernel journal is not UTF-8") from exc

    text, digest, source = stable_descriptor_load(path, load, label="seed kernel journal")
    start = _positive_epoch(start_epoch, label=f"seed-{seed} journal start")
    end = _positive_epoch(end_epoch, label=f"seed-{seed} journal end")
    _require(end >= start, f"seed-{seed} kernel journal interval is reversed")
    lines = text.splitlines()
    _require(bool(lines), f"seed-{seed} kernel journal lacks its interval header")
    header = JOURNAL_HEADER_PATTERN.fullmatch(lines[0])
    _require(
        header is not None
        and int(header.group("seed")) == seed
        and int(header.group("start")) == start
        and int(header.group("end")) == end,
        f"seed-{seed} kernel journal interval header differs",
    )
    faults = [line for line in lines[1:] if KERNEL_FAULT_PATTERN.search(line)]
    _require(not faults, f"kernel journal contains Xid/PCIe faults: {faults[:3]}")
    return {
        "path": str(source),
        "sha256": digest,
        "seed": seed,
        "start_epoch": start,
        "end_epoch": end,
        "fault_count": 0,
    }


def _validate_gpu_check_record(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    payload, digest, source = load_json_object(path, label=f"GPU check {phase}")
    _require(
        payload.get("artifact_type") == GPU_CHECK_ARTIFACT_TYPE
        and payload.get("status") == "physical_gpu1_idle_and_pcie_responsive"
        and payload.get("phase") == phase
        and payload.get("gpu_identity") == manifest.get("gpu_identity")
        and payload.get("compute_owners") == [],
        f"GPU check differs: {phase}",
    )
    observed_epoch = _positive_epoch(
        payload.get("observed_epoch"), label=f"GPU check {phase} observed epoch"
    )
    return {"path": str(source), "sha256": digest, "observed_epoch": observed_epoch}


def _timestamp_epoch(value: object, *, label: str) -> float:
    _require(isinstance(value, str) and bool(value), f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        f"{label} lacks an explicit timezone",
    )
    epoch = parsed.timestamp()
    _require(math.isfinite(epoch) and epoch > 0.0, f"{label} epoch is invalid")
    return epoch


def _telemetry_interval_record(
    path: Path,
    *,
    seed: int,
    receipt_summary: Mapping[str, Any],
) -> dict[str, Any]:
    def load(handle) -> list[dict[str, str]]:
        try:
            text = handle.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("GPU telemetry is not UTF-8") from exc
        reader = csv.DictReader(io.StringIO(text))
        _require(
            tuple(reader.fieldnames or ()) == TELEMETRY_COLUMNS,
            "GPU telemetry header differs during interval validation",
        )
        return [dict(row) for row in reader]

    rows, digest, source = stable_descriptor_load(
        path,
        load,
        label=f"seed-{seed} GPU telemetry interval",
    )
    _require(bool(rows), f"seed-{seed} GPU telemetry interval is empty")
    epochs = [
        _timestamp_epoch(row.get("timestamp"), label=f"seed-{seed} telemetry row {index}")
        for index, row in enumerate(rows)
    ]
    _require(
        all(left <= right for left, right in zip(epochs, epochs[1:])),
        f"seed-{seed} GPU telemetry timestamps are not monotonic",
    )
    _require(
        receipt_summary.get("sample_count") == len(rows)
        and receipt_summary.get("first_timestamp") == rows[0]["timestamp"]
        and receipt_summary.get("last_timestamp") == rows[-1]["timestamp"],
        f"seed-{seed} GPU telemetry interval differs from its guard receipt",
    )
    return {
        "path": str(source),
        "sha256": digest,
        "seed": seed,
        "first_row": 0,
        "last_row": len(rows) - 1,
        "row_count": len(rows),
        "row_interval_sha256": canonical_json_sha256(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
    }


def validate_seed_execution_timeline(
    *,
    seed: int,
    gpu_preflight_epoch: int,
    command_prepared_epoch: int,
    journal_start_epoch: int,
    telemetry_first_epoch: float,
    telemetry_last_epoch: float,
    journal_end_epoch: int,
    gpu_postflight_epoch: int,
) -> dict[str, int | float]:
    pre = _positive_epoch(gpu_preflight_epoch, label=f"seed-{seed} preflight epoch")
    prepared = _positive_epoch(
        command_prepared_epoch, label=f"seed-{seed} command prepared epoch"
    )
    start = _positive_epoch(journal_start_epoch, label=f"seed-{seed} journal start")
    end = _positive_epoch(journal_end_epoch, label=f"seed-{seed} journal end")
    post = _positive_epoch(gpu_postflight_epoch, label=f"seed-{seed} postflight epoch")
    first = _finite(telemetry_first_epoch, label=f"seed-{seed} telemetry first epoch")
    last = _finite(telemetry_last_epoch, label=f"seed-{seed} telemetry last epoch")
    _require(
        pre <= prepared <= start <= first <= last <= end <= post,
        f"seed-{seed} GPU command/journal/telemetry timeline is not strictly bound",
    )
    return {
        "gpu_preflight_observed_epoch": pre,
        "command_prepared_epoch": prepared,
        "journal_start_epoch": start,
        "telemetry_first_epoch": first,
        "telemetry_last_epoch": last,
        "journal_end_epoch": end,
        "gpu_postflight_observed_epoch": post,
    }


def validate_response_distillation_provenance(
    value: object,
    *,
    manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fail closed on the fixed and run-bound response-loss provenance."""

    response_contract_keys = {
        "fit_split_only",
        "benchmark_vocabulary_opened",
        "fit_text_bank",
        "calibration_manifest",
        "calibration_manifest_sha256",
        "calibration_seed",
        "response_lambdas",
        "response_branch_gradient_target_ratio",
        "total_response_gradient_ratio_upper_bound",
        "losses",
        "scene_response_objective",
        "complete_scene_batching",
        "design_diagnostic",
    }
    fit_binding_keys = {
        "artifact_path",
        "artifact_sha256",
        "manifest_path",
        "manifest_sha256",
        "split",
        "query_count",
        "split_synset_tab_query_lf_sha256",
        "ordered_records_sha256",
        "vocabulary_sha256",
        "vocabulary_manifest_sha256",
        "embedding_semantic_sha256",
        "embedding_tensor_sha256",
        "text_encoder_snapshot_files_sha256",
    }
    response_contract = dict(value) if isinstance(value, Mapping) else {}
    fit_binding = response_contract.get("fit_text_bank")
    expected_fit_files = manifest.get("fit_text_bank")
    diagnostic_record = manifest.get("gradient_design_diagnostic")
    _require(
        isinstance(expected_fit_files, Mapping)
        and isinstance(diagnostic_record, Mapping),
        "distill manifest lacks response provenance inputs",
    )
    expected_design_diagnostic = {
        **dict(diagnostic_record),
        "role": "seed0_design_prior_only_per_seed_values_remeasured",
        "measured_seed": 0,
        "calibration_reuses_measured_values": False,
        "diagnostic_surface_control": _surface_controls_by_seed(
            manifest.get("surface_promotion", {})
        )[0]["checkpoint"],
    }
    _require(
        isinstance(value, Mapping)
        and set(response_contract) == response_contract_keys
        and response_contract.get("fit_split_only") is True
        and response_contract.get("benchmark_vocabulary_opened") is False
        and isinstance(fit_binding, Mapping)
        and set(fit_binding) == fit_binding_keys
        and fit_binding.get("artifact_path")
        == expected_fit_files.get("artifact", {}).get("path")
        and fit_binding.get("artifact_sha256")
        == expected_fit_files.get("artifact", {}).get("sha256")
        and fit_binding.get("manifest_path")
        == expected_fit_files.get("manifest", {}).get("path")
        and fit_binding.get("manifest_sha256")
        == expected_fit_files.get("manifest", {}).get("sha256")
        and fit_binding.get("split") == "fit"
        and isinstance(fit_binding.get("query_count"), int)
        and not isinstance(fit_binding.get("query_count"), bool)
        and fit_binding.get("query_count") > 0
        and all(
            SHA256_PATTERN.fullmatch(str(fit_binding.get(field, "")))
            is not None
            for field in (
                "split_synset_tab_query_lf_sha256",
                "ordered_records_sha256",
                "vocabulary_sha256",
                "vocabulary_manifest_sha256",
                "embedding_semantic_sha256",
                "embedding_tensor_sha256",
                "text_encoder_snapshot_files_sha256",
            )
        )
        and response_contract.get("calibration_seed") == seed
        and {
            "path": response_contract.get("calibration_manifest"),
            "sha256": response_contract.get("calibration_manifest_sha256"),
        }
        == calibration.get("manifest")
        and response_contract.get("response_lambdas")
        == calibration.get("response_lambdas")
        and response_contract.get("response_branch_gradient_target_ratio")
        == RESPONSE_BRANCH_GRADIENT_RATIO
        and response_contract.get("total_response_gradient_ratio_upper_bound")
        == TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND
        and response_contract.get("losses")
        == [INDEPENDENT_RESPONSE_LOSS, SCENE_RESPONSE_LOSS]
        and response_contract.get("scene_response_objective")
        == SCENE_RESPONSE_OBJECTIVE
        and response_contract.get("complete_scene_batching") is True
        and response_contract.get("design_diagnostic")
        == expected_design_diagnostic,
        f"seed-{seed} checkpoint response-loss provenance differs",
    )
    return response_contract


def _seed_evidence(
    manifest_path: Path,
    seed: int,
    *,
    journal_start_epoch: int,
    journal_end_epoch: int,
) -> dict[str, Any]:
    manifest, manifest_sha, manifest_source = _manifest(manifest_path)
    row = _seed_row(manifest, seed)
    report, report_sha, report_path = load_json_object(row["report"], label=f"seed-{seed} trainer report")
    checkpoint_sha = str(report.get("checkpoint_sha256", ""))
    _require(SHA256_PATTERN.fullmatch(checkpoint_sha) is not None, "trainer report lacks checkpoint SHA")
    checkpoint, observed_sha, checkpoint_path = load_sha_bound_project_checkpoint_mapping(
        row["checkpoint"],
        expected_sha256=checkpoint_sha,
        map_location="cpu",
        label=f"seed-{seed} response checkpoint",
    )
    selection_contract = _manifest_selection_contract(
        manifest, digest=manifest_sha, source=manifest_source
    )
    if selection_contract == EPOCH_SELECTION:
        best_epoch, best_score = validate_v4_proposal_checkpoint(
            checkpoint, report=report
        )
    else:
        best_epoch, best_score = recompute_response_epoch_selection(
            checkpoint.get("history"), selection_contract=selection_contract
        )
    calibration = _calibration_row(manifest, seed)
    validate_file_record(calibration["manifest"], label=f"seed-{seed} calibration")
    validate_file_record(calibration["audit"], label=f"seed-{seed} calibration audit")
    validate_response_distillation_provenance(
        checkpoint.get("provenance", {}).get("text_response_distillation"),
        manifest=manifest,
        calibration=calibration,
        seed=seed,
    )
    _require(
        report.get("calibration_manifest") == calibration["manifest"]["path"]
        and report.get("calibration_manifest_sha256")
        == calibration["manifest"]["sha256"]
        and report.get("response_lambdas") == calibration["response_lambdas"],
        f"seed-{seed} checkpoint/report calibration binding differs",
    )
    expected_control = _expected_surface_control_binding(
        manifest, checkpoint, seed=seed
    )
    control_validation = checkpoint.get("surface_control_validation")
    history = checkpoint.get("history")
    _require(
        checkpoint.get("surface_control_checkpoint") == expected_control
        and isinstance(control_validation, Mapping)
        and set(control_validation) == set(SURFACE_CONTROL_METRICS)
        and isinstance(history, list)
        and bool(history)
        and all(
            math.isclose(
                _finite(
                    control_validation[field],
                    label=f"seed-{seed} control validation {field}",
                ),
                _finite(
                    history[0].get(field),
                    label=f"seed-{seed} control history {field}",
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for field in SURFACE_CONTROL_METRICS
        ),
        f"seed-{seed} response checkpoint Surface control differs",
    )
    control_score = 0.5 * (
        float(control_validation["mean_descriptor_cosine"])
        + float(control_validation["all_view_descriptor_cosine"])
    )
    _require(
        checkpoint.get("best_epoch") == best_epoch
        and math.isclose(float(checkpoint.get("best_selection_score")), best_score, rel_tol=1e-9, abs_tol=1e-9)
        and report.get("best_epoch") == best_epoch
        and math.isclose(float(report.get("best_selection_score")), best_score, rel_tol=1e-9, abs_tol=1e-9)
        and report.get("distill_run_manifest_sha256") == manifest_sha
        and Path(str(report.get("distill_run_manifest", ""))).resolve() == manifest_source,
        f"seed-{seed} checkpoint/report selection or manifest binding drifted",
    )
    _require(
        report.get("surface_control_checkpoint") == expected_control
        and report.get("surface_control_validation") == dict(control_validation)
        and math.isclose(
            _finite(
                checkpoint.get("surface_control_score"),
                label=f"seed-{seed} checkpoint Surface control score",
            ),
            control_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            _finite(
                report.get("surface_control_score"),
                label=f"seed-{seed} report Surface control score",
            ),
            control_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            _finite(
                report.get("selection_score_delta"),
                label=f"seed-{seed} selection score delta",
            ),
            best_score - control_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        f"seed-{seed} checkpoint/report Surface control witness drifted",
    )
    audit, audit_sha, audit_path = load_json_object(row["audit_report"], label=f"seed-{seed} CPU audit")
    _require(
        audit.get("status") == "checkpoint_report_verified"
        and audit.get("seed") == seed
        and audit.get("checkpoint_sha256") == observed_sha
        and audit.get("report_sha256") == report_sha
        and audit.get("run_manifest_sha256") == manifest_sha
        and audit.get("device") == "cpu",
        f"seed-{seed} CPU audit differs",
    )
    receipt = validate_receipt(row["guard_receipt"])
    receipt_payload = receipt["payload"]
    guard_command_record = file_record(row["guard_command"])
    guard_telemetry_record = file_record(row["guard_telemetry"])
    prepared_epoch = validate_receipt_training_command(
        receipt["command"],
        manifest=manifest,
        manifest_path=manifest_source,
        manifest_sha256=manifest_sha,
        seed=seed,
    )
    thermal = manifest["thermal_safety_contract"]
    _require(
        receipt_payload.get("seed") == seed
        and receipt_payload.get("scene") == manifest.get("candidate")
        and receipt_payload.get("gpu_identity") == manifest.get("gpu_identity")
        and receipt_payload.get("command") == guard_command_record
        and receipt_payload.get("stage_output")
        == {"path": str(checkpoint_path), "sha256": observed_sha}
        and receipt_payload.get("guard") == thermal.get("guard")
        and receipt_payload.get("telemetry") == guard_telemetry_record
        and float(receipt_payload["telemetry_summary"]["maximum_temperature_c"])
        <= float(thermal["maximum_temperature_c"])
        and float(receipt_payload["telemetry_summary"]["maximum_reported_power_limit_w"])
        <= float(thermal["maximum_power_limit_w"]),
        f"seed-{seed} guard receipt violates the frozen thermal contract",
    )
    telemetry_interval = _telemetry_interval_record(
        Path(row["guard_telemetry"]),
        seed=seed,
        receipt_summary=receipt_payload["telemetry_summary"],
    )
    gpu_preflight = _validate_gpu_check_record(
        Path(row["gpu_preflight"]), manifest=manifest, phase=f"pre_seed{seed}"
    )
    gpu_postflight = _validate_gpu_check_record(
        Path(row["gpu_postflight"]), manifest=manifest, phase=f"post_seed{seed}"
    )
    kernel_journal = _kernel_journal_record(
        Path(row["kernel_journal"]),
        seed=seed,
        start_epoch=journal_start_epoch,
        end_epoch=journal_end_epoch,
    )
    execution_timeline = validate_seed_execution_timeline(
        seed=seed,
        gpu_preflight_epoch=gpu_preflight["observed_epoch"],
        command_prepared_epoch=prepared_epoch,
        journal_start_epoch=journal_start_epoch,
        telemetry_first_epoch=telemetry_interval["first_epoch"],
        telemetry_last_epoch=telemetry_interval["last_epoch"],
        journal_end_epoch=journal_end_epoch,
        gpu_postflight_epoch=gpu_postflight["observed_epoch"],
    )
    return {
        "seed": seed,
        "checkpoint": {"path": str(checkpoint_path), "sha256": observed_sha},
        "report": {"path": str(report_path), "sha256": report_sha},
        "training_log": file_record(row["training_log"]),
        "audit_report": {"path": str(audit_path), "sha256": audit_sha},
        "guard_command": guard_command_record,
        "guard_telemetry": guard_telemetry_record,
        "telemetry_interval": telemetry_interval,
        "guard_receipt": receipt["receipt"],
        "kernel_journal": kernel_journal,
        "gpu_preflight": gpu_preflight,
        "gpu_postflight": gpu_postflight,
        "execution_timeline": execution_timeline,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_contract": selection_contract,
        "surface_control": expected_control,
        "calibration": calibration,
    }


def _terminal_journal_bounds(terminal: Mapping[str, Any], *, seed: int) -> tuple[int, int]:
    evidence = terminal.get("evidence")
    journal = evidence.get("kernel_journal") if isinstance(evidence, Mapping) else None
    _require(
        isinstance(journal, Mapping) and journal.get("seed") == seed,
        f"seed-{seed} terminal lacks its bound kernel journal interval",
    )
    return (
        _positive_epoch(journal.get("start_epoch"), label=f"seed-{seed} terminal journal start"),
        _positive_epoch(journal.get("end_epoch"), label=f"seed-{seed} terminal journal end"),
    )


def finalize_seed(
    manifest_path: Path,
    seed: int,
    terminal_path: Path,
    *,
    journal_start_epoch: int,
    journal_end_epoch: int,
) -> dict[str, Any]:
    manifest, manifest_sha, manifest_source = _manifest(manifest_path)
    evidence = _seed_evidence(
        manifest_path,
        seed,
        journal_start_epoch=journal_start_epoch,
        journal_end_epoch=journal_end_epoch,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": SEED_TERMINAL_ARTIFACT_TYPE,
        "status": "complete_guarded_audited_no_xid_pcie_fault",
        "seed": seed,
        "candidate": manifest["candidate"],
        "run_manifest": {"path": str(manifest_source), "sha256": manifest_sha},
        "runtime_closure_digest": manifest["runtime_closure"]["digest"],
        "evidence": evidence,
    }
    write_frozen_json(terminal_path, payload)
    return {"terminal": file_record(terminal_path), "status": payload["status"]}


def validate_seed_terminal(manifest_path: Path, seed: int) -> dict[str, Any]:
    manifest, manifest_sha, manifest_source = _manifest(manifest_path)
    row = _seed_row(manifest, seed)
    terminal, digest, source = load_json_object(row["terminal"], label=f"seed-{seed} terminal")
    journal_start_epoch, journal_end_epoch = _terminal_journal_bounds(terminal, seed=seed)
    expected = {
        "schema_version": 1,
        "artifact_type": SEED_TERMINAL_ARTIFACT_TYPE,
        "status": "complete_guarded_audited_no_xid_pcie_fault",
        "seed": seed,
        "candidate": manifest["candidate"],
        "run_manifest": {"path": str(manifest_source), "sha256": manifest_sha},
        "runtime_closure_digest": manifest["runtime_closure"]["digest"],
        "evidence": _seed_evidence(
            manifest_path,
            seed,
            journal_start_epoch=journal_start_epoch,
            journal_end_epoch=journal_end_epoch,
        ),
    }
    _require(terminal == expected, f"seed-{seed} terminal differs from strict recomputation")
    return {"terminal": {"path": str(source), "sha256": digest}, "status": terminal["status"]}


def validate_cross_seed_replay(evidence_rows: Sequence[Mapping[str, Any]]) -> None:
    _require(
        len(evidence_rows) == len(REQUIRED_SEEDS),
        "cross-seed replay gate requires all three seed terminals",
    )
    for field, nested in (
        ("guard_command", "sha256"),
        ("telemetry_interval", "sha256"),
        ("telemetry_interval", "row_interval_sha256"),
        ("kernel_journal", "sha256"),
    ):
        values = [evidence[field][nested] for evidence in evidence_rows]
        _require(
            len(set(values)) == len(REQUIRED_SEEDS),
            f"cross-seed replay detected for {field}.{nested}",
        )
    for previous, current in zip(evidence_rows, evidence_rows[1:]):
        _require(
            previous["execution_timeline"]["journal_end_epoch"]
            <= current["execution_timeline"]["journal_start_epoch"],
            "seed execution intervals overlap or are out of order",
        )


def finalize_run(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest, manifest_sha, manifest_source = _manifest(manifest_path)
    selection_contract = _manifest_selection_contract(
        manifest, digest=manifest_sha, source=manifest_source
    )
    seeds = []
    seed_evidence: list[Mapping[str, Any]] = []
    for seed in REQUIRED_SEEDS:
        terminal = validate_seed_terminal(manifest_path, seed)
        row = _seed_row(manifest, seed)
        terminal_payload, _, _ = load_json_object(
            row["terminal"], label=f"seed-{seed} validated terminal"
        )
        evidence = terminal_payload["evidence"]
        _require(isinstance(evidence, Mapping), f"seed-{seed} validated evidence differs")
        seed_evidence.append(evidence)
        seeds.append(
            {
                "seed": seed,
                "checkpoint": evidence["checkpoint"],
                "report": evidence["report"],
                "guard_receipt": evidence["guard_receipt"],
                "terminal": terminal["terminal"],
                "best_epoch": evidence["best_epoch"],
                "best_selection_score": evidence["best_selection_score"],
                "surface_control": evidence["surface_control"],
                "calibration": evidence["calibration"],
            }
        )
    validate_cross_seed_replay(seed_evidence)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETION_ARTIFACT_TYPE,
        "status": "complete_three_seed_guarded_authority",
        "candidate": manifest["candidate"],
        "run_manifest": {"path": str(manifest_source), "sha256": manifest_sha},
        "calibrations": [dict(row) for row in manifest["calibrations"]],
        "gradient_design_diagnostic": dict(
            manifest["gradient_design_diagnostic"]
        ),
        "runtime_closure_digest": manifest["runtime_closure"]["digest"],
        "selection_contract": selection_contract,
        "seeds": seeds,
    }
    write_frozen_json(output, payload)
    return {"completion": file_record(output), "status": payload["status"]}


def _add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--lock-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--surface-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--fit-text-bank", required=True, type=Path)
    parser.add_argument("--fit-text-bank-manifest", required=True, type=Path)
    parser.add_argument("--radio-checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, action="append")
    parser.add_argument("--calibration-audit", required=True, action="append")
    parser.add_argument("--gradient-diagnostic", required=True, type=Path)
    parser.add_argument("--gradient-diagnostic-sha256", required=True)
    parser.add_argument("--initial-gpu-preflight", required=True, type=Path)
    parser.add_argument("--thermal-guard", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--gpu-max-temp-c", required=True, type=int)
    parser.add_argument("--gpu-start-max-temp-c", required=True, type=int)
    parser.add_argument("--gpu-max-power-limit-w", required=True, type=float)
    parser.add_argument("--gpu-poll-seconds", required=True, type=int)
    parser.add_argument("--gpu-soft-pause-temp-c", required=True, type=int)
    parser.add_argument("--gpu-soft-resume-temp-c", required=True, type=int)
    parser.add_argument("--gpu-peer-index", type=int)
    parser.add_argument("--gpu-peer-pause-temp-c", required=True, type=int)
    parser.add_argument("--gpu-peer-resume-temp-c", required=True, type=int)
    parser.add_argument("--gpu-peer-quiet-seconds", required=True, type=int)
    parser.add_argument("--gpu-peer-max-power-w", required=True, type=float)
    parser.add_argument("--gpu-peer-max-memory-mib", required=True, type=int)
    parser.add_argument("--gpu-peer-max-util-pct", required=True, type=int)
    parser.add_argument("--gpu-peer-activity-action", required=True)
    parser.add_argument("--gpu-owner-pid-namespace-mode", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight-lock-root")
    preflight.add_argument("--lock-root", required=True, type=Path)
    locked = subparsers.add_parser("run-locked")
    locked.add_argument("--repo-root", required=True, type=Path)
    locked.add_argument("--lock-root", required=True, type=Path)
    locked.add_argument("--output-root", required=True, type=Path)
    locked.add_argument("argv", nargs=argparse.REMAINDER)
    inherited = subparsers.add_parser("verify-lock-fds")
    inherited.add_argument("--lock-root", required=True, type=Path)
    inherited.add_argument("--output-root", required=True, type=Path)
    inherited.add_argument("--global-fd", required=True, type=int)
    inherited.add_argument("--run-fd", required=True, type=int)
    inherited.add_argument("--singleton-fd", required=True, type=int)
    gpu = subparsers.add_parser("record-gpu-check")
    gpu.add_argument("--output", required=True, type=Path)
    gpu.add_argument("--phase", required=True)
    gpu.add_argument("--gpu-uuid", required=True)
    gpu.add_argument("--gpu-bus-id", required=True)
    gpu.add_argument("--proc-bus-id", required=True)
    gpu.add_argument("--pci-prefix", required=True)
    gpu.add_argument("--observed-epoch", required=True, type=int)
    gpu.add_argument("--compute-owner", action="append", default=[])
    gpu.add_argument("--run-manifest", type=Path)
    for name in ("create-manifest", "verify-manifest"):
        _add_manifest_arguments(subparsers.add_parser(name))
    classify = subparsers.add_parser("classify-seed")
    classify.add_argument("--run-manifest", required=True, type=Path)
    classify.add_argument("--seed", required=True, type=int)
    finalize = subparsers.add_parser("finalize-seed")
    finalize.add_argument("--run-manifest", required=True, type=Path)
    finalize.add_argument("--seed", required=True, type=int)
    finalize.add_argument("--terminal", required=True, type=Path)
    finalize.add_argument("--journal-start-epoch", required=True, type=int)
    finalize.add_argument("--journal-end-epoch", required=True, type=int)
    verify = subparsers.add_parser("verify-seed")
    verify.add_argument("--run-manifest", required=True, type=Path)
    verify.add_argument("--seed", required=True, type=int)
    complete = subparsers.add_parser("finalize-run")
    complete.add_argument("--run-manifest", required=True, type=Path)
    complete.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight-lock-root":
        print(json.dumps(inspect_canonical_lock_root(args.lock_root), sort_keys=True))
        return
    if args.command == "run-locked":
        raise SystemExit(
            run_locked(
                repo_root=args.repo_root,
                lock_root=args.lock_root,
                output_root=args.output_root,
                command=args.argv,
            )
        )
    if args.command == "verify-lock-fds":
        result = verify_inherited_locks(
            lock_root=args.lock_root,
            output_root=args.output_root,
            global_descriptor=args.global_fd,
            run_descriptor=args.run_fd,
            singleton_descriptor=args.singleton_fd,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "record-gpu-check":
        result = record_gpu_check(args)
    elif args.command in {"create-manifest", "verify-manifest"}:
        result = create_or_verify_manifest(args)
        _require(
            args.command != "verify-manifest" or result["status"] == "verified",
            "verify-manifest cannot create a missing manifest",
        )
    elif args.command == "classify-seed":
        result = {"state": classify_seed(args.run_manifest, int(args.seed))}
    elif args.command == "finalize-seed":
        result = finalize_seed(
            args.run_manifest,
            int(args.seed),
            args.terminal,
            journal_start_epoch=args.journal_start_epoch,
            journal_end_epoch=args.journal_end_epoch,
        )
    elif args.command == "verify-seed":
        result = validate_seed_terminal(args.run_manifest, int(args.seed))
    else:
        result = finalize_run(args.run_manifest, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
