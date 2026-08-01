#!/usr/bin/env python3
"""Freeze and independently finalize the promoted Surface ScanNet text benchmark.

The benchmark may only be opened after the query-free Surface bundle and the
three-seed text-response audit have been strictly recomputed and accepted.
Exactly three promoted response checkpoints are evaluated on exactly three
frozen ScanNet scenes.  This module owns the immutable run manifest, per-stage
terminals, independent report validation, and the final seed/scene aggregate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts import eval_scannet_canonical_text_query as text_evaluator
from radio_gs.scripts.finalize_gpu_guard_receipt import (
    validate_receipt as validate_gpu_guard_receipt,
)
from radio_gs.scripts.eval_scannet_canonical_text_query import DEFAULT_PROMPTS
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
RUN_ARTIFACT_TYPE = "promoted_surface_scannet_text_benchmark_run"
SEMANTIC_TERMINAL_TYPE = "promoted_surface_scannet_semantic_stage_completion"
EVAL_TERMINAL_TYPE = "promoted_surface_scannet_text_eval_stage_completion"
AGGREGATE_ARTIFACT_TYPE = "promoted_surface_scannet_text_benchmark_aggregate"
COMPLETION_ARTIFACT_TYPE = "promoted_surface_scannet_text_benchmark_completion"

REQUIRED_SEEDS = (0, 1, 2)
REQUIRED_SCENES = ("scene0062_00", "scene0140_00", "scene0200_00")
REQUIRED_SPLITS = ("19", "15", "10")
PREPARED_ROOT = Path("/mnt/pool/sqy/3d_understanding/scannet_og")
RADIO_CHECKPOINT = Path("/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
TEXT_CACHE_BASE_NAME = "siglip2_scannet_og_text_embeddings_ens5.pt"
FIELD_ROOT_RELATIVE = Path("output/optimization_20260715/canonical_v2_validation")
FIELD_NAME = "canonical_mpr_v2_d256_l128_fusion.pth"
GRAPH_NAME = "v2_shared_support_graph_k16.pt"
BENCHMARK_CONTROL_POLICY = (
    "promoted_treatment_only_absolute_main_result;paired_control_was_"
    "consumed_by_dev_audit_selection;causal_benchmark_delta_requires_"
    "a_separate_preregistered_protocol"
)

PROTOCOL = {
    "evaluation_domain": "official_scannet_label_mesh_vertices",
    "primitive_to_mesh": "inverse_distance_knn",
    "projection_k": 8,
    "distance_epsilon": 1e-4,
    "evaluation_chunk_size": 2048,
    "evaluation_device": "cpu",
    "classification": "normalized_cosine_argmax",
    "semantic_scale_aggregation": "max_after_cosine",
    "scale_specificity_margin": 0.0,
    "text_encoder": "official_siglip2_g",
    "logit_calibration": "none",
    "spatial_postprocess": "none",
    "ground_truth_usage": "metrics_only",
    "prompt_templates": DEFAULT_PROMPTS.split("|"),
    "class_aliases": "none",
    "class_splits": list(REQUIRED_SPLITS),
    "knn_workers": 1,
    "torch_num_threads": 1,
}

IMPLEMENTATION_SOURCES = (
    "radio_gs/scripts/finalize_scannet_surface_text_authority_gate.py",
    "radio_gs/scripts/finalize_scannet_surface_text_benchmark.py",
    "radio_gs/scripts/run_scannet_surface_text_benchmark.sh",
    "radio_gs/scripts/finalize_gpu_guard_receipt.py",
    "radio_gs/scripts/build_surface_region_semantic_cache.py",
    "radio_gs/scripts/eval_scannet_canonical_text_query.py",
    "radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py",
    "radio_gs/scripts/build_primitive_text_score_cache.py",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
    "radio_gs/scripts/run_repo_python.sh",
    "radio_gs/interfaces/surface_region_contract.py",
    "radio_gs/interfaces/surface_region_summary.py",
    "radio_gs/models/siglip_projection.py",
    "radio_gs/querying/support_solver.py",
    "radio_gs/scannet_constants.py",
    "radio_gs/field/__init__.py",
    "radio_gs/field/canonical_gaussian_field.py",
    "radio_gs/field/primitive_fusion.py",
    "radio_gs/field/spatial_hash.py",
    "radio_gs/utils/immutable_artifacts.py",
)
AUTHORITY_RECEIPT_TYPE = "promoted_surface_text_benchmark_authority_receipt"
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


def _sha256(path: str | Path) -> str:
    return sha256_file(path)


def _file_record(path: str | Path) -> dict[str, str]:
    return file_record(path)


def _validate_file_record(record: object, label: str) -> Path:
    return validate_file_record(record, label=label)


def _json_object(path: str | Path) -> dict[str, Any]:
    payload, _, _ = load_json_object(path, label="benchmark JSON artifact")
    return payload


def _torch_mapping(path: str | Path, label: str) -> dict[str, Any]:
    payload, _, _ = load_torch_mapping(
        path,
        map_location="cpu",
        label=label,
    )
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_frozen_json(path, payload)


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_frozen_json(path, payload)


def validate_authority_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate the gate-only process receipt before opening benchmark data."""

    receipt, digest, receipt_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="benchmark authority receipt",
    )
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "required_seeds",
        "benchmark_data_opened",
        "forbidden_benchmark_modules_loaded",
        "promotion_authority",
        "authority_inputs",
        "implementation_sources",
    }
    _require(
        set(receipt) == expected_keys
        and receipt.get("schema_version") == 1
        and receipt.get("artifact_type") == AUTHORITY_RECEIPT_TYPE
        and receipt.get("status") == "accepted_before_benchmark_open"
        and receipt.get("required_seeds") == list(REQUIRED_SEEDS)
        and receipt.get("benchmark_data_opened") is False
        and receipt.get("forbidden_benchmark_modules_loaded") == [],
        "authority receipt schema/status differs",
    )
    repo_root = Path(__file__).resolve().parents[2]
    observed_sources = receipt.get("implementation_sources")
    _require(
        isinstance(observed_sources, list)
        and len(observed_sources) == len(AUTHORITY_IMPLEMENTATION_SOURCES),
        "authority receipt implementation binding is incomplete",
    )
    expected_sources = [
        {"relative_path": relative, **_file_record(repo_root / relative)}
        for relative in AUTHORITY_IMPLEMENTATION_SOURCES
    ]
    _require(
        observed_sources == expected_sources,
        "authority receipt implementation changed before benchmark opening",
    )
    authority = receipt.get("promotion_authority")
    _require(isinstance(authority, Mapping), "authority receipt lacks promotion")
    expected_authority_keys = {
        "selected_candidate",
        "method_id",
        "surface_manifest",
        "surface_completion",
        "text_audit_manifest",
        "text_audit_completion",
        "text_plan",
        "readouts",
    }
    _require(
        set(authority) == expected_authority_keys
        and isinstance(authority.get("selected_candidate"), str)
        and bool(authority["selected_candidate"])
        and isinstance(authority.get("method_id"), str)
        and bool(authority["method_id"]),
        "promotion authority receipt differs",
    )
    for role in (
        "surface_manifest",
        "surface_completion",
        "text_audit_manifest",
        "text_audit_completion",
        "text_plan",
    ):
        _validate_file_record(authority[role], f"authority {role}")
    authority_inputs = receipt.get("authority_inputs")
    _require(
        authority_inputs
        == {
            "surface_manifest": authority["surface_manifest"],
            "surface_completion": authority["surface_completion"],
            "text_audit_manifest": authority["text_audit_manifest"],
            "text_audit_completion": authority["text_audit_completion"],
        },
        "authority input receipt differs",
    )
    readouts = authority.get("readouts")
    _require(
        isinstance(readouts, list)
        and len(readouts) == len(REQUIRED_SEEDS)
        and all(isinstance(row, Mapping) for row in readouts)
        and {row.get("seed") for row in readouts} == set(REQUIRED_SEEDS),
        "authority receipt does not bind exact seeds 0/1/2",
    )
    for row in readouts:
        _require(
            set(row) == {"seed", "checkpoint", "sidecar"},
            "authority readout receipt fields differ",
        )
        _validate_file_record(row["checkpoint"], f"seed {row['seed']} readout")
        _validate_file_record(row["sidecar"], f"seed {row['seed']} readout sidecar")
    return {
        "receipt": {"path": str(receipt_path), "sha256": digest},
        "authority": dict(authority),
    }


def default_asset_paths(repo_root: Path | None = None) -> dict[str, Any]:
    repo = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    scenes: dict[str, dict[str, Path]] = {}
    for scene in REQUIRED_SCENES:
        short = scene.split("_", 1)[0]
        scene_root = (repo / FIELD_ROOT_RELATIVE / f"scannet_{short}").resolve()
        scenes[scene] = {
            "field": scene_root / FIELD_NAME,
            "graph": scene_root / GRAPH_NAME,
            "label": PREPARED_ROOT / scene / f"{scene}_vh_clean_2.labels.ply",
        }
    base = (repo / "checkpoints" / TEXT_CACHE_BASE_NAME).resolve()
    return {
        "prepared_root": PREPARED_ROOT,
        "radio_checkpoint": RADIO_CHECKPOINT,
        "text_cache_base": base,
        "text_split_caches": {
            split: base.with_name(f"{base.stem}_split{split}{base.suffix}")
            for split in REQUIRED_SPLITS
        },
        "scenes": scenes,
    }


def _validate_text_caches(
    split_records: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    for split in REQUIRED_SPLITS:
        path = _validate_file_record(split_records[split], f"split {split} text cache")
        payload = _torch_mapping(path, f"split {split} text cache")
        class_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        expected_queries = [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
        _require(
            payload.get("queries") == expected_queries,
            f"split {split} text queries differ",
        )
        _require(
            payload.get("prompt_templates") == PROTOCOL["prompt_templates"],
            f"split {split} prompt templates differ",
        )
        embeddings = torch.as_tensor(payload.get("embeddings"))
        _require(
            embeddings.shape == (len(expected_queries), 1536)
            and embeddings.dtype == torch.float32
            and bool(torch.isfinite(embeddings).all()),
            f"split {split} text embeddings are invalid",
        )
        _require(
            payload.get("text_encoder") in (None, "siglip2"),
            f"split {split} text encoder is not SigLIP2",
        )
        payloads[split] = payload
    authority = payloads["19"]
    _require(
        authority.get("text_encoder") == "siglip2"
        and authority.get("model_name") == "google/siglip2-giant-opt-patch16-384",
        "split 19 cache lacks explicit official SigLIP2 encoder provenance",
    )
    lookup = {query: index for index, query in enumerate(authority["queries"])}
    for split in ("15", "10"):
        expected = torch.stack(
            [
                authority["embeddings"][lookup[query]]
                for query in payloads[split]["queries"]
            ]
        )
        _require(
            torch.allclose(
                expected,
                torch.as_tensor(payloads[split]["embeddings"]),
                rtol=0.0,
                atol=2e-7,
            ),
            f"split {split} embeddings differ from the encoder-authoritative split 19 cache",
        )
    return {
        "encoder": "siglip2",
        "model_name": "google/siglip2-giant-opt-patch16-384",
        "encoder_authority_split": "19",
        "subset_embedding_max_abs_tolerance": 2e-7,
    }


def _materialize_input_registry(paths: Mapping[str, Any]) -> dict[str, Any]:
    prepared_root = Path(paths["prepared_root"]).resolve()
    _require(
        prepared_root == PREPARED_ROOT.resolve(),
        f"prepared root must be exactly {PREPARED_ROOT}",
    )
    radio = _file_record(paths["radio_checkpoint"])
    text_base = _file_record(paths["text_cache_base"])
    split_records = {
        split: _file_record(paths["text_split_caches"][split])
        for split in REQUIRED_SPLITS
    }
    text_provenance = _validate_text_caches(split_records)
    scenes = {}
    for scene in REQUIRED_SCENES:
        raw = paths["scenes"][scene]
        field_record = _file_record(raw["field"])
        graph_record = _file_record(raw["graph"])
        label_record = _file_record(raw["label"])
        field = _torch_mapping(field_record["path"], f"{scene} field")
        _require(
            field.get("benchmark_masks_opened") is False
            and field.get("text_queries_opened") is False,
            f"{scene} field is not query-free",
        )
        mpr_path = Path(str(field.get("mpr_cache", ""))).resolve()
        mpr_record = _file_record(mpr_path)
        graph = _torch_mapping(graph_record["path"], f"{scene} graph")
        metadata = graph.get("metadata")
        capability = (
            metadata.get("capability_metadata")
            if isinstance(metadata, Mapping)
            else None
        )
        _require(
            isinstance(metadata, Mapping)
            and metadata.get("source")
            == "canonical_official_dino_sam3_multichannel_support_graph"
            and isinstance(capability, Mapping)
            and Path(str(capability.get("field_checkpoint", ""))).resolve()
            == Path(field_record["path"])
            and capability.get("field_checkpoint_sha256") == field_record["sha256"]
            and Path(str(capability.get("mpr_cache", ""))).resolve() == mpr_path
            and capability.get("radio_checkpoint_sha256") == radio["sha256"],
            f"{scene} field/graph/RADIO provenance differs",
        )
        expected_label = prepared_root / scene / f"{scene}_vh_clean_2.labels.ply"
        _require(
            Path(label_record["path"]) == expected_label,
            f"{scene} label mesh is outside the frozen prepared root",
        )
        scenes[scene] = {
            "field": field_record,
            "mpr": mpr_record,
            "graph": graph_record,
            "label": label_record,
        }
        del field, graph
        gc.collect()
    return {
        "prepared_root": str(prepared_root),
        "radio_checkpoint": radio,
        "text_cache_base": text_base,
        "text_split_caches": split_records,
        "text_encoder_provenance": text_provenance,
        "scenes": scenes,
    }


def _implementation_records(repo_root: Path) -> list[dict[str, str]]:
    return [
        {"relative_path": relative, **_file_record(repo_root / relative)}
        for relative in IMPLEMENTATION_SOURCES
    ]


def _python_tree_binding(repo_root: Path) -> dict[str, Any]:
    python_root = (Path(repo_root) / "radio_gs").resolve()
    python_paths = sorted(
        python_root.rglob("*.py"),
        key=lambda value: value.as_posix(),
    )
    _require(
        all(not path.is_symlink() for path in python_paths),
        "radio_gs Python source tree contains a symlink",
    )
    entries = [
        {
            "relative_path": path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in python_paths
        if path.is_file()
    ]
    _require(entries, "radio_gs Python source tree is empty")
    return {
        "root_relative_path": "radio_gs",
        "ordering": "lexicographic_posix_relative_path",
        "file_count": len(entries),
        "ordered_entries": entries,
        "ordered_tree_sha256": canonical_json_sha256(entries),
    }


def _runtime_fingerprint() -> dict[str, Any]:
    import scipy

    payload = {
        "python_executable_invoked": str(Path(sys.executable).absolute()),
        "python_executable": _file_record(Path(sys.executable).resolve(strict=True)),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "torch_cudnn_build": str(torch.backends.cudnn.version()),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
    }
    return {**payload, "fingerprint_sha256": canonical_json_sha256(payload)}


def _stage_grid(output_root: Path) -> list[dict[str, Any]]:
    stages = []
    for seed in REQUIRED_SEEDS:
        for scene in REQUIRED_SCENES:
            root = output_root / f"seed{seed}" / scene
            stages.append(
                {
                    "seed": seed,
                    "scene": scene,
                    "semantic_cache": str((root / "semantic.pt").resolve()),
                    "semantic_terminal": str(
                        (root / "semantic.complete.json").resolve()
                    ),
                    "semantic_resume_dir": str((root / "semantic.resume").resolve()),
                    "semantic_guard_receipt": str(
                        (root / "semantic.guard.receipt.json").resolve()
                    ),
                    "evaluation_report": str((root / "evaluation.json").resolve()),
                    "evaluation_terminal": str(
                        (root / "evaluation.complete.json").resolve()
                    ),
                }
            )
    return stages


def _validate_thermal_contract(thermal_contract: Mapping[str, Any]) -> None:
    expected_keys = {
        "guarded",
        "guard",
        "physical_gpu",
        "maximum_temperature_c",
        "maximum_start_temperature_c",
        "maximum_power_limit_w",
        "poll_seconds",
        "soft_pause_temperature_c",
        "soft_resume_temperature_c",
        "peer_gpu",
        "peer_pause_temperature_c",
        "peer_resume_temperature_c",
        "peer_quiet_seconds_before_launch",
        "peer_maximum_power_w",
        "peer_maximum_memory_mib",
        "peer_maximum_utilization_percent",
        "peer_activity_gate",
        "semantic_builder_resume",
        "hard_guard_is_safety_authority",
        "soft_pause_safety_role",
    }
    _require(
        set(thermal_contract) == expected_keys,
        "thermal contract fields differ from the frozen GPU1 envelope",
    )
    _validate_file_record(thermal_contract["guard"], "GPU thermal guard")
    integer_fields = (
        "physical_gpu",
        "maximum_temperature_c",
        "maximum_start_temperature_c",
        "poll_seconds",
        "soft_pause_temperature_c",
        "soft_resume_temperature_c",
        "peer_gpu",
        "peer_pause_temperature_c",
        "peer_resume_temperature_c",
        "peer_quiet_seconds_before_launch",
        "peer_maximum_memory_mib",
        "peer_maximum_utilization_percent",
    )
    _require(
        all(
            isinstance(thermal_contract[field], int)
            and not isinstance(thermal_contract[field], bool)
            for field in integer_fields
        ),
        "thermal contract integer fields are malformed",
    )
    for field in ("maximum_power_limit_w", "peer_maximum_power_w"):
        value = thermal_contract[field]
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"thermal contract {field} is malformed",
        )
    _require(
        thermal_contract["physical_gpu"] == 1
        and thermal_contract["guarded"] is True
        and thermal_contract["semantic_builder_resume"]
        == "durable_per_batch_resume_with_strict_contract"
        and thermal_contract["hard_guard_is_safety_authority"] is True
        and thermal_contract["soft_pause_safety_role"]
        == "supplementary_only_not_a_safety_precondition",
        "semantic execution must rely on durable resume/pacing and the hard guard",
    )
    soft_pause = thermal_contract["soft_pause_temperature_c"]
    soft_resume = thermal_contract["soft_resume_temperature_c"]
    soft_contract_valid = (soft_pause == 0 and soft_resume == 0) or (
        0 < soft_resume < soft_pause < thermal_contract["maximum_temperature_c"]
    )
    _require(
        0 < thermal_contract["maximum_temperature_c"] <= 75
        and 0
        < thermal_contract["maximum_start_temperature_c"]
        < thermal_contract["maximum_temperature_c"]
        and 0.0 < float(thermal_contract["maximum_power_limit_w"]) <= 300.5
        and thermal_contract["poll_seconds"] == 1
        and soft_contract_valid
        and thermal_contract["peer_gpu"] == 0
        and 0
        < thermal_contract["peer_resume_temperature_c"]
        < thermal_contract["peer_pause_temperature_c"]
        and thermal_contract["peer_quiet_seconds_before_launch"] == 0
        and float(thermal_contract["peer_maximum_power_w"]) == 0.0
        and thermal_contract["peer_maximum_memory_mib"] == 0
        and thermal_contract["peer_maximum_utilization_percent"] == 100
        and thermal_contract["peer_activity_gate"]
        == "temperature_only_activity_limits_disabled",
        "thermal contract is weaker than the frozen GPU1 safety envelope",
    )


def _validate_gpu_identity(gpu_identity: Mapping[str, Any]) -> None:
    _require(
        set(gpu_identity) == {"physical_index", "uuid", "pci_bus_id"}
        and gpu_identity.get("physical_index") == 1
        and isinstance(gpu_identity.get("uuid"), str)
        and str(gpu_identity["uuid"]).startswith("GPU-")
        and len(str(gpu_identity["uuid"])) > 8
        and isinstance(gpu_identity.get("pci_bus_id"), str)
        and bool(str(gpu_identity["pci_bus_id"])),
        "physical GPU1 UUID/bus identity is invalid",
    )


def build_run_manifest(
    *,
    authority_receipt: Path,
    authority_receipt_sha256: str,
    output_root: Path,
    asset_paths: Mapping[str, Any] | None = None,
    semantic_radio_batch_size: int = 1024,
    semantic_batch_size: int = 64,
    semantic_pacing_seconds: float = 4.0,
    thermal_contract: Mapping[str, Any],
    gpu_identity: Mapping[str, Any],
) -> dict[str, Any]:
    # This receipt validation must remain first.  No benchmark scene, label,
    # vocabulary, field, or graph path may be opened before it succeeds.
    gate = validate_authority_receipt(
        authority_receipt,
        expected_sha256=authority_receipt_sha256,
    )
    authority = gate["authority"]
    _require(
        int(semantic_radio_batch_size) > 0, "semantic RADIO batch size must be positive"
    )
    _require(int(semantic_batch_size) > 0, "semantic batch size must be positive")
    _require(
        isinstance(semantic_pacing_seconds, (int, float))
        and not isinstance(semantic_pacing_seconds, bool)
        and math.isfinite(float(semantic_pacing_seconds))
        and float(semantic_pacing_seconds) > 0.0,
        "semantic pacing must be finite and positive",
    )
    _validate_thermal_contract(thermal_contract)
    _validate_gpu_identity(gpu_identity)
    output_root = Path(output_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    registry = _materialize_input_registry(
        asset_paths if asset_paths is not None else default_asset_paths(repo_root)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RUN_ARTIFACT_TYPE,
        "status": "accepted_authority_frozen_exact_nine_run_grid",
        "benchmark_gate": {
            "opened_only_after_surface_query_free_and_text_audit_acceptance": True,
            "text_audit_decision": "promote_confirmed",
            "main_result_eligible": True,
        },
        "method": {
            "selected_candidate": authority["selected_candidate"],
            "method_id": authority["method_id"],
            "seed_policy": "all_required_seeds_no_single_seed_selection",
            "benchmark_control_policy": BENCHMARK_CONTROL_POLICY,
        },
        "required_seeds": list(REQUIRED_SEEDS),
        "required_scenes": list(REQUIRED_SCENES),
        "benchmark_scope": {
            "semantic_cache_count": 9,
            "evaluation_report_count": 9,
            "control_evaluation_count": 0,
            "single_seed_or_scene_subset_is_main_result": False,
        },
        "authority_receipt": gate["receipt"],
        "promotion_authority": authority,
        "input_registry": registry,
        "protocol": {
            **PROTOCOL,
            "text_embedding_cache_base": registry["text_cache_base"]["path"],
        },
        "semantic_execution": {
            "radio_batch_size": int(semantic_radio_batch_size),
            "semantic_batch_size": int(semantic_batch_size),
            "device": "cuda:0_mapped_from_physical_gpu1",
            "full_multiscale_descriptor_cache": True,
            "stream_text_queries": False,
            "builder_output_publication": "same_filesystem_noclobber_link_fsync",
            "builder_internal_pacing": True,
            "builder_internal_resume": True,
            "resume_contract": "immutable_per_batch_tensor_plus_sha_terminal_v1",
            "pacing_seconds_after_each_committed_batch": float(
                semantic_pacing_seconds
            ),
        },
        "thermal_execution_contract": dict(thermal_contract),
        "gpu_identity": dict(gpu_identity),
        "output_root": str(output_root),
        "stage_grid": _stage_grid(output_root),
        "implementation_sources": _implementation_records(repo_root),
        "radio_gs_python_tree": _python_tree_binding(repo_root),
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def preflight(
    *,
    authority_receipt: Path,
    authority_receipt_sha256: str,
    output_root: Path,
    run_manifest: Path,
    semantic_radio_batch_size: int,
    semantic_batch_size: int,
    semantic_pacing_seconds: float,
    thermal_contract: Mapping[str, Any],
    gpu_identity: Mapping[str, Any],
    asset_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    run_manifest = Path(run_manifest).resolve()
    _require(
        run_manifest == output_root / "run_manifest.json",
        "run manifest must be OUTPUT_ROOT/run_manifest.json",
    )
    payload = build_run_manifest(
        authority_receipt=authority_receipt,
        authority_receipt_sha256=authority_receipt_sha256,
        output_root=output_root,
        asset_paths=asset_paths,
        semantic_radio_batch_size=semantic_radio_batch_size,
        semantic_batch_size=semantic_batch_size,
        semantic_pacing_seconds=semantic_pacing_seconds,
        thermal_contract=thermal_contract,
        gpu_identity=gpu_identity,
    )
    if not run_manifest.exists() and output_root.exists():
        allowed_preflight_files = {
            ".runner.lock",
            Path(authority_receipt).resolve().name,
        }
        existing = [
            path for path in output_root.iterdir() if path.name not in allowed_preflight_files
        ]
        _require(
            not existing, "benchmark output root contains artifacts but no run manifest"
        )
    if run_manifest.exists() or run_manifest.is_symlink():
        _require(
            _json_object(run_manifest) == payload,
            "existing run manifest differs from strict preflight",
        )
    else:
        _write_frozen_json(run_manifest, payload)
    return {
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": _sha256(run_manifest),
        "method_id": payload["method"]["method_id"],
        "required_seeds": list(REQUIRED_SEEDS),
        "required_scenes": list(REQUIRED_SCENES),
        "expected_semantic_caches": 9,
        "expected_evaluation_reports": 9,
        "device": "cpu_preflight_only",
    }


def _validate_run_manifest(path: Path, *, verify_registry: bool) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = _json_object(path)
    required_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "benchmark_gate",
        "method",
        "required_seeds",
        "required_scenes",
        "benchmark_scope",
        "authority_receipt",
        "promotion_authority",
        "input_registry",
        "protocol",
        "semantic_execution",
        "thermal_execution_contract",
        "gpu_identity",
        "output_root",
        "stage_grid",
        "implementation_sources",
        "radio_gs_python_tree",
        "runtime_fingerprint",
    }
    _require(
        set(manifest) == required_keys
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == RUN_ARTIFACT_TYPE
        and manifest.get("status") == "accepted_authority_frozen_exact_nine_run_grid",
        "invalid ScanNet text benchmark run manifest",
    )
    _require(
        manifest.get("required_seeds") == list(REQUIRED_SEEDS)
        and manifest.get("required_scenes") == list(REQUIRED_SCENES),
        "benchmark run manifest seed/scene grid differs",
    )
    receipt_record = manifest.get("authority_receipt")
    _require(
        isinstance(receipt_record, Mapping)
        and set(receipt_record) == {"path", "sha256"},
        "run manifest lacks the external authority receipt",
    )
    receipt = validate_authority_receipt(
        Path(str(receipt_record["path"])),
        expected_sha256=str(receipt_record["sha256"]),
    )
    _require(
        receipt["receipt"] == receipt_record
        and receipt["authority"] == manifest.get("promotion_authority"),
        "run manifest differs from the external authority receipt",
    )
    authority = manifest.get("promotion_authority")
    _require(
        isinstance(authority, Mapping)
        and isinstance(authority.get("selected_candidate"), str)
        and bool(authority["selected_candidate"])
        and isinstance(authority.get("method_id"), str)
        and bool(authority["method_id"])
        and isinstance(authority.get("readouts"), list)
        and len(authority["readouts"]) == len(REQUIRED_SEEDS)
        and {row.get("seed") for row in authority["readouts"]} == set(REQUIRED_SEEDS),
        "promotion authority seed/readout identity differs",
    )
    _require(
        manifest.get("benchmark_gate")
        == {
            "opened_only_after_surface_query_free_and_text_audit_acceptance": True,
            "text_audit_decision": "promote_confirmed",
            "main_result_eligible": True,
        }
        and manifest.get("method")
        == {
            "selected_candidate": authority["selected_candidate"],
            "method_id": authority["method_id"],
            "seed_policy": "all_required_seeds_no_single_seed_selection",
            "benchmark_control_policy": BENCHMARK_CONTROL_POLICY,
        },
        "benchmark gate/method differs from the accepted promotion authority",
    )
    scope = manifest.get("benchmark_scope")
    _require(
        scope
        == {
            "semantic_cache_count": 9,
            "evaluation_report_count": 9,
            "control_evaluation_count": 0,
            "single_seed_or_scene_subset_is_main_result": False,
        },
        "benchmark run scope differs",
    )
    output_root = Path(str(manifest.get("output_root", ""))).resolve()
    _require(
        path == output_root / "run_manifest.json",
        "run manifest path differs from its frozen output root",
    )
    _require(
        manifest.get("stage_grid") == _stage_grid(output_root),
        "benchmark stage grid differs from the exact 3x3 registry",
    )
    registry = manifest.get("input_registry")
    _require(
        isinstance(registry, Mapping)
        and Path(str(registry.get("prepared_root", ""))).resolve()
        == PREPARED_ROOT.resolve()
        and isinstance(registry.get("scenes"), Mapping)
        and set(registry["scenes"]) == set(REQUIRED_SCENES)
        and isinstance(registry.get("text_split_caches"), Mapping)
        and set(registry["text_split_caches"]) == set(REQUIRED_SPLITS),
        "benchmark input registry scene/text identity differs",
    )
    _require(
        manifest.get("protocol")
        == {
            **PROTOCOL,
            "text_embedding_cache_base": registry["text_cache_base"]["path"],
        },
        "benchmark evaluator protocol differs",
    )
    semantic_execution = manifest.get("semantic_execution")
    _require(
        isinstance(semantic_execution, Mapping)
        and set(semantic_execution)
        == {
            "radio_batch_size",
            "semantic_batch_size",
            "device",
            "full_multiscale_descriptor_cache",
            "stream_text_queries",
            "builder_output_publication",
            "builder_internal_pacing",
            "builder_internal_resume",
            "resume_contract",
            "pacing_seconds_after_each_committed_batch",
        }
        and isinstance(semantic_execution["radio_batch_size"], int)
        and not isinstance(semantic_execution["radio_batch_size"], bool)
        and semantic_execution["radio_batch_size"] > 0
        and isinstance(semantic_execution["semantic_batch_size"], int)
        and not isinstance(semantic_execution["semantic_batch_size"], bool)
        and semantic_execution["semantic_batch_size"] > 0
        and semantic_execution["device"] == "cuda:0_mapped_from_physical_gpu1"
        and semantic_execution["full_multiscale_descriptor_cache"] is True
        and semantic_execution["stream_text_queries"] is False
        and semantic_execution["builder_output_publication"]
        == "same_filesystem_noclobber_link_fsync"
        and semantic_execution["builder_internal_pacing"] is True
        and semantic_execution["builder_internal_resume"] is True
        and semantic_execution["resume_contract"]
        == "immutable_per_batch_tensor_plus_sha_terminal_v1"
        and isinstance(
            semantic_execution["pacing_seconds_after_each_committed_batch"],
            (int, float),
        )
        and not isinstance(
            semantic_execution["pacing_seconds_after_each_committed_batch"], bool
        )
        and math.isfinite(
            float(semantic_execution["pacing_seconds_after_each_committed_batch"])
        )
        and float(semantic_execution["pacing_seconds_after_each_committed_batch"])
        > 0.0,
        "semantic execution contract differs",
    )
    thermal_contract = manifest.get("thermal_execution_contract")
    _require(isinstance(thermal_contract, Mapping), "thermal contract is missing")
    _validate_thermal_contract(thermal_contract)
    gpu_identity = manifest.get("gpu_identity")
    _require(isinstance(gpu_identity, Mapping), "GPU identity is missing")
    _validate_gpu_identity(gpu_identity)
    observed_implementation = manifest.get("implementation_sources")
    _require(
        isinstance(observed_implementation, list), "implementation binding is invalid"
    )
    for record in observed_implementation:
        _require(isinstance(record, Mapping), "implementation source record is invalid")
        relative = str(record.get("relative_path", ""))
        _require(relative in IMPLEMENTATION_SOURCES, "unknown implementation source")
        expected = _file_record(Path(__file__).resolve().parents[2] / relative)
        _require(
            record == {"relative_path": relative, **expected},
            f"implementation source changed: {relative}",
        )
    _require(
        {str(record["relative_path"]) for record in observed_implementation}
        == set(IMPLEMENTATION_SOURCES)
        and len(observed_implementation) == len(IMPLEMENTATION_SOURCES),
        "implementation source binding is incomplete",
    )
    repo_root = Path(__file__).resolve().parents[2]
    _require(
        manifest.get("radio_gs_python_tree") == _python_tree_binding(repo_root),
        "ordered radio_gs Python tree digest changed",
    )
    _require(
        manifest.get("runtime_fingerprint") == _runtime_fingerprint(),
        "formal benchmark runtime fingerprint changed",
    )
    guard_record = next(
        record
        for record in observed_implementation
        if record["relative_path"] == "radio_gs/scripts/run_with_gpu_thermal_guard.sh"
    )
    _require(
        thermal_contract["guard"]
        == {"path": guard_record["path"], "sha256": guard_record["sha256"]},
        "thermal guard differs from the implementation binding",
    )
    if verify_registry:
        rematerialized = _materialize_input_registry(
            {
                "prepared_root": registry["prepared_root"],
                "radio_checkpoint": registry["radio_checkpoint"]["path"],
                "text_cache_base": registry["text_cache_base"]["path"],
                "text_split_caches": {
                    split: registry["text_split_caches"][split]["path"]
                    for split in REQUIRED_SPLITS
                },
                "scenes": {
                    scene: {
                        role: registry["scenes"][scene][role]["path"]
                        for role in ("field", "graph", "label")
                    }
                    for scene in REQUIRED_SCENES
                },
            }
        )
        _require(
            rematerialized == registry,
            "input registry differs after independent reopening",
        )
        for readout in manifest["promotion_authority"]["readouts"]:
            _validate_file_record(
                readout["checkpoint"], f"seed {readout['seed']} readout"
            )
            _validate_file_record(
                readout["sidecar"], f"seed {readout['seed']} readout sidecar"
            )
    return manifest


def _stage_record(manifest: Mapping[str, Any], seed: int, scene: str) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["stage_grid"]
        if row.get("seed") == seed and row.get("scene") == scene
    ]
    _require(len(matches) == 1, f"stage grid lacks seed {seed} scene {scene}")
    return dict(matches[0])


def _readout_record(manifest: Mapping[str, Any], seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["promotion_authority"]["readouts"]
        if row.get("seed") == seed
    ]
    _require(len(matches) == 1, f"promotion authority lacks readout seed {seed}")
    return dict(matches[0])


def _finite_tensor(value: torch.Tensor, label: str, chunk: int = 4_000_000) -> None:
    flat = torch.as_tensor(value).reshape(-1)
    for start in range(0, flat.numel(), int(chunk)):
        _require(
            bool(torch.isfinite(flat[start : start + int(chunk)]).all()),
            f"{label} contains NaN or infinity",
        )


def _float32_tensor_sha256(value: torch.Tensor) -> str:
    array = (
        torch.as_tensor(value)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require_unit_norms(
    value: torch.Tensor,
    label: str,
    *,
    row_chunk: int = 4096,
    tolerance: float = 2.5e-3,
) -> None:
    rows = torch.as_tensor(value)
    flat = rows.reshape(-1, rows.shape[-1])
    for start in range(0, flat.shape[0], int(row_chunk)):
        norms = torch.linalg.vector_norm(
            flat[start : start + int(row_chunk)].float(),
            dim=-1,
        )
        _require(
            bool((norms - 1.0).abs().le(float(tolerance)).all()),
            f"{label} rows are not unit-normalized",
        )


def _validate_semantic_cache(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    scene: str,
    semantic_cache: Path,
) -> dict[str, Any]:
    _require(seed in REQUIRED_SEEDS and scene in REQUIRED_SCENES, "unknown seed/scene")
    stage = _stage_record(manifest, seed, scene)
    semantic_cache = Path(semantic_cache).resolve()
    _require(
        semantic_cache == Path(stage["semantic_cache"]),
        "semantic cache path differs from the stage registry",
    )
    registry = manifest["input_registry"]
    scene_inputs = registry["scenes"][scene]
    readout = _readout_record(manifest, seed)
    for role in ("field", "mpr", "graph"):
        _validate_file_record(scene_inputs[role], f"{scene} {role}")
    _validate_file_record(readout["checkpoint"], f"seed {seed} readout")
    _validate_file_record(readout["sidecar"], f"seed {seed} readout sidecar")
    radio = _validate_file_record(registry["radio_checkpoint"], "RADIO checkpoint")
    field_payload = _torch_mapping(scene_inputs["field"]["path"], f"{scene} field")
    graph_payload = _torch_mapping(scene_inputs["graph"]["path"], f"{scene} graph")
    _, readout_payload, readout_digest, _ = (
        load_surface_region_summary_readout_v2(
            readout["checkpoint"]["path"],
            expected_sha256=readout["checkpoint"]["sha256"],
            map_location="cpu",
        )
    )
    _require(
        readout_digest == readout["checkpoint"]["sha256"],
        "readout checkpoint digest differs",
    )
    payload = _torch_mapping(semantic_cache, f"seed {seed} {scene} semantic cache")
    allowed_payload_keys = {
        "xyz",
        "features",
        "summary_features",
        "global_rows",
        "features_by_scale",
        "valid",
        "metadata",
        "primary_valid",
        "semantic_confidence",
    }
    _require(
        {
            "xyz",
            "features",
            "summary_features",
            "global_rows",
            "features_by_scale",
            "valid",
            "metadata",
        }.issubset(payload)
        and set(payload).issubset(allowed_payload_keys),
        "semantic cache payload keys differ from the frozen full-cache schema",
    )
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping), "semantic cache metadata is missing")
    contract_payload = metadata.get("region_contract")
    _require(
        isinstance(contract_payload, Mapping)
        and isinstance(contract_payload.get("radii_m"), list),
        "semantic cache lacks its frozen surface-region contract",
    )
    try:
        contract = SurfaceRegionContractV2(
            **{
                **dict(contract_payload),
                "radii_m": tuple(contract_payload["radii_m"]),
            }
        )
        contract.assert_compatible(dict(metadata))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("semantic cache surface-region contract differs") from exc
    _require(
        metadata.get("schema_version") == 5
        and metadata.get("feature_space")
        == "official_siglip2_summary_descriptor_multiscale"
        and metadata.get("source") == "canonical_radio_surface_region_readout"
        and metadata.get("construction")
        == "canonical_radio_surface_region_readout_then_official_summary_head"
        and metadata.get("canonical_radio_source") == "field_decode_only"
        and metadata.get("mpr_radio_features_opened") is False
        and metadata.get("official_summary_head") is True
        and metadata.get("custom_text_projection") is False
        and metadata.get("query_set_invariant") is True
        and metadata.get("benchmark_images_opened") is False
        and metadata.get("benchmark_masks_opened") is False
        and metadata.get("text_queries_opened") is False,
        "semantic cache violates the query-free method contract",
    )
    _require(
        Path(str(metadata.get("readout_checkpoint", ""))).resolve()
        == Path(readout["checkpoint"]["path"])
        and metadata.get("readout_checkpoint_sha256") == readout["checkpoint"]["sha256"]
        and metadata.get("bridge_checkpoint_sha256") == readout["checkpoint"]["sha256"]
        and metadata.get("bridge_training_scope") == "global_cross_scene"
        and str(metadata.get("bridge_training_scope_detail", "")).startswith(
            "global_cross_scene"
        )
        and Path(str(metadata.get("field_checkpoint", ""))).resolve()
        == Path(scene_inputs["field"]["path"])
        and metadata.get("field_checkpoint_sha256") == scene_inputs["field"]["sha256"]
        and Path(str(metadata.get("mpr_cache", ""))).resolve()
        == Path(scene_inputs["mpr"]["path"])
        and metadata.get("mpr_cache_sha256") == scene_inputs["mpr"]["sha256"]
        and Path(str(metadata.get("support_graph", ""))).resolve()
        == Path(scene_inputs["graph"]["path"])
        and metadata.get("support_graph_sha256") == scene_inputs["graph"]["sha256"]
        and metadata.get("radio_checkpoint_sha256")
        == registry["radio_checkpoint"]["sha256"]
        and metadata.get("official_radio_checkpoint_sha256")
        == registry["radio_checkpoint"]["sha256"]
        and metadata.get("readout_batch_size")
        == manifest["semantic_execution"]["semantic_batch_size"]
        and metadata.get("region_contract") == contract.to_dict()
        and metadata.get("region_topology") == contract.expansion
        and metadata.get("cache_role") == "disposable_derivative_not_scene_memory"
        and metadata.get("row_storage") == "sparse_valid_rows_with_global_row_index"
        and metadata.get("scale_storage")
        == "all_scales_preserved; mean_descriptor_legacy_only"
        and radio == Path(registry["radio_checkpoint"]["path"]),
        "semantic cache checkpoint/field/graph/RADIO provenance differs",
    )
    xyz = torch.as_tensor(payload.get("xyz"))
    valid = torch.as_tensor(payload.get("valid")).bool()
    rows = torch.as_tensor(payload.get("global_rows")).long()
    features = torch.as_tensor(payload.get("features"))
    summary = torch.as_tensor(payload.get("summary_features"))
    scales = torch.as_tensor(payload.get("features_by_scale"))
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    radii = metadata.get("region_radii_m")
    _require(
        xyz.ndim == 2
        and xyz.shape[1] == 3
        and xyz.dtype == torch.float32
        and valid.shape == (count,)
        and torch.as_tensor(payload.get("valid")).dtype == torch.bool
        and rows.ndim == 1
        and torch.as_tensor(payload.get("global_rows")).dtype == torch.int64
        and torch.equal(torch.where(valid)[0], rows)
        and rows.numel() > 0
        and features.shape == (rows.numel(), 1536)
        and features.dtype == torch.float16
        and summary.shape == features.shape
        and summary.dtype == torch.float16
        and torch.equal(features, summary)
        and isinstance(radii, list)
        and len(radii) > 0
        and all(
            isinstance(radius, (int, float))
            and not isinstance(radius, bool)
            and math.isfinite(float(radius))
            and float(radius) > 0.0
            for radius in radii
        )
        and [float(radius) for radius in radii]
        == sorted({float(radius) for radius in radii})
        and scales.shape == (rows.numel(), len(radii), 1536)
        and scales.dtype == torch.float16,
        "semantic cache tensor geometry differs",
    )
    field_architecture = field_payload.get("architecture")
    field_geometry = field_payload.get("geometry_fingerprint")
    graph_metadata = graph_payload.get("metadata")
    graph_capability = (
        graph_metadata.get("capability_metadata")
        if isinstance(graph_metadata, Mapping)
        else None
    )
    _require(
        isinstance(field_architecture, Mapping)
        and field_architecture.get("num_gaussians") == count
        and Path(str(field_payload.get("mpr_cache", ""))).resolve()
        == Path(scene_inputs["mpr"]["path"])
        and isinstance(field_geometry, Mapping)
        and field_geometry.get("num_gaussians") == count
        and field_geometry.get("xyz_sha256") == _float32_tensor_sha256(xyz)
        and graph_payload.get("schema_version") == 1
        and graph_payload.get("num_global_rows") == count
        and torch.equal(
            torch.as_tensor(graph_payload.get("global_rows")).long(),
            rows,
        )
        and torch.equal(
            torch.as_tensor(graph_payload.get("xyz")).float(),
            xyz[rows],
        )
        and isinstance(graph_capability, Mapping)
        and Path(str(graph_capability.get("field_checkpoint", ""))).resolve()
        == Path(scene_inputs["field"]["path"])
        and graph_capability.get("field_checkpoint_sha256")
        == scene_inputs["field"]["sha256"]
        and Path(str(graph_capability.get("mpr_cache", ""))).resolve()
        == Path(scene_inputs["mpr"]["path"]),
        "field/MPR/graph/semantic geometry closure differs",
    )
    readout_architecture = readout_payload.get("architecture")
    readout_provenance = readout_payload.get("provenance")
    _require(
        isinstance(readout_architecture, Mapping)
        and readout_architecture.get("feature_dim") == 1280
        and readout_architecture.get("geometry_dim") == 14
        and readout_architecture.get("contract_sha256") == contract.digest
        and isinstance(readout_provenance, Mapping)
        and readout_provenance.get("region_contract") == contract.to_dict()
        and readout_provenance.get("region_contract_sha256") == contract.digest
        and readout_provenance.get("scene_disjoint") is True
        and readout_provenance.get("uses_benchmark_scenes") is False
        and readout_provenance.get("uses_benchmark_test_vocabulary") is False,
        "semantic radii/contract differ from the frozen readout",
    )
    expected_features = F.normalize(
        scales.float().mean(dim=1),
        dim=-1,
        eps=1e-8,
    ).half()
    _require(
        torch.equal(features, expected_features),
        "semantic aggregate features are not derived from retained scales",
    )
    _require_unit_norms(scales, "semantic scale descriptors")
    _require_unit_norms(features, "semantic aggregate descriptors")
    if "primary_valid" in payload:
        primary_valid = torch.as_tensor(payload["primary_valid"])
        _require(
            primary_valid.dtype == torch.bool
            and primary_valid.shape == valid.shape
            and not bool((primary_valid & ~valid).any()),
            "semantic primary-valid partition differs",
        )
    if "semantic_confidence" in payload:
        semantic_confidence = torch.as_tensor(payload["semantic_confidence"])
        _require(
            semantic_confidence.shape == valid.shape
            and semantic_confidence.dtype == torch.float16,
            "semantic confidence tensor differs",
        )
        _finite_tensor(semantic_confidence, "semantic confidence")
    _finite_tensor(xyz, "semantic xyz")
    _finite_tensor(features, "semantic features")
    _finite_tensor(scales, "semantic multiscale features")
    result = {
        "semantic_cache": _file_record(semantic_cache),
        "num_primitives": count,
        "num_valid_primitives": int(valid.sum()),
        "num_semantic_scales": len(radii),
        "readout_checkpoint": dict(readout["checkpoint"]),
        "readout_sidecar": dict(readout["sidecar"]),
        "field": dict(scene_inputs["field"]),
        "mpr": dict(scene_inputs["mpr"]),
        "graph": dict(scene_inputs["graph"]),
        "radio_checkpoint": dict(registry["radio_checkpoint"]),
    }
    del (
        payload,
        field_payload,
        graph_payload,
        readout_payload,
        xyz,
        valid,
        rows,
        features,
        summary,
        scales,
        expected_features,
    )
    gc.collect()
    return result


def _command_option(argv: Sequence[str], option: str) -> str:
    matches = [index for index, value in enumerate(argv) if value == option]
    _require(len(matches) == 1, f"guarded semantic command lacks exact {option}")
    index = matches[0]
    _require(index + 1 < len(argv), f"guarded semantic command truncates {option}")
    return str(argv[index + 1])


def _validate_semantic_guard_receipt(
    manifest: Mapping[str, Any],
    *,
    run_manifest: Path,
    seed: int,
    scene: str,
    semantic_cache: Path,
    guard_receipt: Path,
) -> dict[str, str]:
    stage = _stage_record(manifest, seed, scene)
    guard_receipt = Path(guard_receipt).resolve()
    _require(
        guard_receipt == Path(stage["semantic_guard_receipt"]),
        "semantic guard receipt path differs from the stage registry",
    )
    validated = validate_gpu_guard_receipt(guard_receipt)
    payload = validated["payload"]
    command = validated["command"]
    _require(
        payload["seed"] == seed
        and payload["scene"] == scene
        and payload["gpu_identity"] == manifest["gpu_identity"]
        and payload["guard"] == manifest["thermal_execution_contract"]["guard"]
        and payload["stage_output"] == _file_record(semantic_cache)
        and command["run_manifest"] == _file_record(run_manifest),
        "semantic guard receipt identity/provenance differs",
    )
    telemetry_summary = payload.get("telemetry_summary")
    thermal_contract = manifest["thermal_execution_contract"]
    _require(
        isinstance(telemetry_summary, Mapping)
        and isinstance(telemetry_summary.get("sample_count"), int)
        and not isinstance(telemetry_summary.get("sample_count"), bool)
        and telemetry_summary["sample_count"] > 0
        and isinstance(
            telemetry_summary.get("maximum_temperature_c"), (int, float)
        )
        and not isinstance(
            telemetry_summary.get("maximum_temperature_c"), bool
        )
        and float(telemetry_summary["maximum_temperature_c"])
        < float(thermal_contract["maximum_temperature_c"])
        and isinstance(
            telemetry_summary.get("maximum_reported_power_limit_w"),
            (int, float),
        )
        and not isinstance(
            telemetry_summary.get("maximum_reported_power_limit_w"), bool
        )
        and 0.0
        < float(telemetry_summary["maximum_reported_power_limit_w"])
        <= float(thermal_contract["maximum_power_limit_w"]),
        "semantic guard telemetry exceeds the frozen safety envelope",
    )
    argv = command["argv"]
    repo_root = Path(__file__).resolve().parents[2]
    expected_builder = (
        repo_root / "radio_gs/scripts/build_surface_region_semantic_cache.py"
    ).resolve()
    builder_positions = [
        index
        for index, value in enumerate(argv)
        if Path(value).name == expected_builder.name
    ]
    _require(
        len(builder_positions) == 1
        and Path(argv[builder_positions[0]]).resolve() == expected_builder,
        "guarded command does not execute the frozen semantic builder",
    )
    scene_inputs = manifest["input_registry"]["scenes"][scene]
    readout = _readout_record(manifest, seed)
    expected_options = {
        "--field-checkpoint": scene_inputs["field"]["path"],
        "--field-checkpoint-sha256": scene_inputs["field"]["sha256"],
        "--support-graph": scene_inputs["graph"]["path"],
        "--support-graph-sha256": scene_inputs["graph"]["sha256"],
        "--readout-checkpoint": readout["checkpoint"]["path"],
        "--readout-checkpoint-sha256": readout["checkpoint"]["sha256"],
        "--mpr-cache-sha256": scene_inputs["mpr"]["sha256"],
        "--radio-checkpoint": manifest["input_registry"]["radio_checkpoint"]["path"],
        "--radio-checkpoint-sha256": manifest["input_registry"][
            "radio_checkpoint"
        ]["sha256"],
        "--output": str(Path(semantic_cache).resolve()),
        "--query-output": str(
            Path(semantic_cache).with_name("semantic_query.pt").resolve()
        ),
        "--resume-dir": stage["semantic_resume_dir"],
        "--radio-batch-size": str(
            manifest["semantic_execution"]["radio_batch_size"]
        ),
        "--semantic-batch-size": str(
            manifest["semantic_execution"]["semantic_batch_size"]
        ),
        "--thermal-pacing-seconds-per-batch": str(
            manifest["semantic_execution"][
                "pacing_seconds_after_each_committed_batch"
            ]
        ),
        "--device": "cuda:0",
    }
    for option, expected in expected_options.items():
        observed = _command_option(argv, option)
        if option.startswith("--") and option.endswith("checkpoint") or option in {
            "--support-graph",
            "--output",
            "--resume-dir",
        }:
            _require(
                Path(observed).resolve() == Path(str(expected)).resolve(),
                f"guarded semantic command {option} differs",
            )
        else:
            _require(observed == str(expected), f"guarded semantic command {option} differs")
    _require(
        "--stream-text-queries" not in argv,
        "formal semantic command must not open text queries",
    )
    return dict(validated["receipt"])


def finalize_semantic_stage(
    *,
    run_manifest: Path,
    seed: int,
    scene: str,
    semantic_cache: Path,
    guard_receipt: Path,
    terminal: Path,
    write: bool = True,
) -> dict[str, Any]:
    run_manifest = Path(run_manifest).resolve()
    manifest = _validate_run_manifest(run_manifest, verify_registry=False)
    stage = _stage_record(manifest, int(seed), str(scene))
    terminal = Path(terminal).resolve()
    _require(
        terminal == Path(stage["semantic_terminal"]),
        "semantic terminal path differs from the stage registry",
    )
    validation = _validate_semantic_cache(
        manifest,
        seed=int(seed),
        scene=str(scene),
        semantic_cache=Path(semantic_cache),
    )
    receipt_record = _validate_semantic_guard_receipt(
        manifest,
        run_manifest=run_manifest,
        seed=int(seed),
        scene=str(scene),
        semantic_cache=Path(semantic_cache),
        guard_receipt=Path(guard_receipt),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SEMANTIC_TERMINAL_TYPE,
        "status": "complete",
        "seed": int(seed),
        "scene": str(scene),
        "run_manifest": _file_record(run_manifest),
        "semantic": validation,
        "guard_receipt": receipt_record,
        "execution_safety": {
            "builder_internal_resume": True,
            "stage_output_noclobber_fsync": True,
            "thermal_guard_required": True,
            "physical_gpu": 1,
        },
    }
    if write:
        if terminal.exists() or terminal.is_symlink():
            _require(_json_object(terminal) == payload, "semantic terminal differs")
        else:
            _write_frozen_json(terminal, payload)
    elif terminal.exists():
        _require(_json_object(terminal) == payload, "semantic terminal differs")
    return payload


def _validate_semantic_terminal(
    manifest: Mapping[str, Any],
    *,
    run_manifest: Path,
    seed: int,
    scene: str,
    independently_validated: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen the frozen terminal and its cache hash without deserializing GBs.

    The semantic finalizer already performed the full safe tensor validation.
    Evaluation can therefore adopt that terminal by reopening its immutable
    JSON and rehashing the exact cache.  The final benchmark still performs
    one independent full tensor reopening for every one of the nine caches.
    """

    stage = _stage_record(manifest, seed, scene)
    terminal = Path(stage["semantic_terminal"]).resolve()
    _require(terminal.is_file(), "evaluation requires semantic completion")
    payload = _json_object(terminal)
    if independently_validated is not None:
        _require(
            payload == dict(independently_validated),
            "independent semantic validation differs from the frozen terminal",
        )
    _require(
        set(payload)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "seed",
            "scene",
            "run_manifest",
            "semantic",
            "guard_receipt",
            "execution_safety",
        }
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("artifact_type") == SEMANTIC_TERMINAL_TYPE
        and payload.get("status") == "complete"
        and payload.get("seed") == seed
        and payload.get("scene") == scene,
        "semantic terminal schema/identity differs",
    )
    run_record = payload.get("run_manifest")
    _require(
        isinstance(run_record, Mapping)
        and _validate_file_record(run_record, "semantic terminal run manifest")
        == Path(run_manifest).resolve(),
        "semantic terminal belongs to another run manifest",
    )
    semantic = payload.get("semantic")
    _require(
        isinstance(semantic, Mapping)
        and set(semantic)
        == {
            "semantic_cache",
            "num_primitives",
            "num_valid_primitives",
            "num_semantic_scales",
            "readout_checkpoint",
            "readout_sidecar",
            "field",
            "mpr",
            "graph",
            "radio_checkpoint",
        },
        "semantic terminal validation payload differs",
    )
    semantic_cache = (
        Path(str(semantic["semantic_cache"]["path"])).resolve()
        if independently_validated is not None
        else _validate_file_record(
            semantic["semantic_cache"], "semantic terminal cache"
        )
    )
    guard_receipt = _validate_file_record(
        payload.get("guard_receipt"),
        "semantic terminal guard receipt",
    )
    readout = _readout_record(manifest, seed)
    scene_inputs = manifest["input_registry"]["scenes"][scene]
    _require(
        semantic_cache == Path(stage["semantic_cache"])
        and guard_receipt == Path(stage["semantic_guard_receipt"])
        and semantic["readout_checkpoint"] == readout["checkpoint"]
        and semantic["readout_sidecar"] == readout["sidecar"]
        and semantic["field"] == scene_inputs["field"]
        and semantic["mpr"] == scene_inputs["mpr"]
        and semantic["graph"] == scene_inputs["graph"]
        and semantic["radio_checkpoint"]
        == manifest["input_registry"]["radio_checkpoint"]
        and isinstance(semantic["num_primitives"], int)
        and not isinstance(semantic["num_primitives"], bool)
        and isinstance(semantic["num_valid_primitives"], int)
        and not isinstance(semantic["num_valid_primitives"], bool)
        and 0 < semantic["num_valid_primitives"] <= semantic["num_primitives"]
        and isinstance(semantic["num_semantic_scales"], int)
        and not isinstance(semantic["num_semantic_scales"], bool)
        and semantic["num_semantic_scales"] > 0
        and payload["execution_safety"]
        == {
            "builder_internal_resume": True,
            "stage_output_noclobber_fsync": True,
            "thermal_guard_required": True,
            "physical_gpu": 1,
        },
        "semantic terminal cache/provenance differs",
    )
    return payload


def _close(first: object, second: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(first), float(second), rel_tol=0.0, abs_tol=float(tolerance)
        )
    except (TypeError, ValueError):
        return False


def _nonnegative_int(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _validate_split_report(
    split: str,
    payload: object,
    *,
    expected_num_valid: int,
) -> dict[str, float]:
    _require(isinstance(payload, Mapping), f"split {split} report is invalid")
    class_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
    class_names = [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
    _require(
        payload.get("class_ids") == class_ids
        and payload.get("class_names") == class_names
        and _nonnegative_int(payload.get("num_valid"), f"split {split} num_valid")
        == expected_num_valid,
        f"split {split} class/valid contract differs",
    )
    per_class = payload.get("per_class")
    _require(
        isinstance(per_class, Mapping)
        and set(per_class) == {str(class_id) for class_id in class_ids},
        f"split {split} per-class rows differ",
    )
    ious = []
    accuracies = []
    for class_id, class_name in zip(class_ids, class_names):
        row = per_class[str(class_id)]
        _require(isinstance(row, Mapping), f"split {split} class row is invalid")
        intersection = _nonnegative_int(
            row.get("intersection"), f"split {split} class {class_id} intersection"
        )
        union = _nonnegative_int(
            row.get("union"), f"split {split} class {class_id} union"
        )
        gt_count = _nonnegative_int(
            row.get("gt_count"), f"split {split} class {class_id} gt_count"
        )
        pred_count = _nonnegative_int(
            row.get("pred_count"), f"split {split} class {class_id} pred_count"
        )
        _require(
            row.get("name") == class_name
            and intersection <= gt_count
            and intersection <= pred_count
            and union == gt_count + pred_count - intersection,
            f"split {split} class {class_id} counts differ",
        )
        if gt_count > 0 and union > 0:
            iou = intersection / union
            _require(_close(row.get("iou"), iou), f"split {split} IoU differs")
            ious.append(iou)
        else:
            _require(row.get("iou") is None, f"split {split} empty IoU must be null")
        if gt_count > 0:
            accuracy = intersection / gt_count
            _require(
                _close(row.get("acc"), accuracy), f"split {split} accuracy differs"
            )
            accuracies.append(accuracy)
        else:
            _require(
                row.get("acc") is None, f"split {split} empty accuracy must be null"
            )
    _require(ious and accuracies, f"split {split} has no measurable classes")
    miou = statistics.fmean(ious)
    macc = statistics.fmean(accuracies)
    _require(
        _close(payload.get("miou"), miou) and _close(payload.get("macc"), macc),
        f"split {split} aggregate metrics differ from per-class counts",
    )
    return {"miou": miou, "macc": macc}


def _validate_evaluation_report(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    scene: str,
    report_path: Path,
    semantic_validation: Mapping[str, Any] | None = None,
    verify_text_caches: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = _stage_record(manifest, seed, scene)
    report_path = Path(report_path).resolve()
    _require(
        report_path == Path(stage["evaluation_report"]),
        "evaluation report path differs from the stage registry",
    )
    run_manifest = Path(manifest["output_root"]) / "run_manifest.json"
    semantic_payload = _validate_semantic_terminal(
        manifest,
        run_manifest=run_manifest,
        seed=seed,
        scene=scene,
        independently_validated=semantic_validation,
    )
    report = _json_object(report_path)
    expected_keys = {
        "schema_version",
        "scene",
        "label_ply",
        "label_ply_sha256",
        "semantic_cache",
        "semantic_cache_sha256",
        "semantic_source",
        "num_mesh_vertices",
        "num_primitives",
        "num_valid_primitives",
        "protocol",
        "splits",
    }
    _require(
        set(report) == expected_keys
        and report.get("schema_version") == 2
        and report.get("scene") == scene
        and report.get("semantic_source") == "canonical_radio_surface_region_readout",
        "evaluation report schema/source differs",
    )
    registry = manifest["input_registry"]
    label = _validate_file_record(registry["scenes"][scene]["label"], f"{scene} label")
    semantic_record = semantic_payload["semantic"]["semantic_cache"]
    report_num_mesh_vertices = _nonnegative_int(
        report.get("num_mesh_vertices"), "evaluation num_mesh_vertices"
    )
    report_num_primitives = _nonnegative_int(
        report.get("num_primitives"), "evaluation num_primitives"
    )
    report_num_valid_primitives = _nonnegative_int(
        report.get("num_valid_primitives"), "evaluation num_valid_primitives"
    )
    _require(
        Path(str(report.get("label_ply", ""))).resolve() == label
        and report.get("label_ply_sha256")
        == registry["scenes"][scene]["label"]["sha256"]
        and Path(str(report.get("semantic_cache", ""))).resolve()
        == Path(semantic_record["path"])
        and report.get("semantic_cache_sha256") == semantic_record["sha256"]
        and report_num_primitives == semantic_payload["semantic"]["num_primitives"]
        and report_num_valid_primitives
        == semantic_payload["semantic"]["num_valid_primitives"],
        "evaluation report label/semantic provenance differs",
    )
    protocol = report.get("protocol")
    expected_protocol = {
        key: value
        for key, value in manifest["protocol"].items()
        if key != "class_splits"
    }
    expected_protocol["num_semantic_scales"] = semantic_payload["semantic"][
        "num_semantic_scales"
    ]
    _require(protocol == expected_protocol, "evaluation protocol differs")
    if verify_text_caches:
        _validate_text_caches(registry["text_split_caches"])

    # Re-run the complete deterministic prediction and metric path from the
    # frozen semantic cache, label PLY, and text tensors.  Count identities
    # alone are insufficient because a forged intersection/GT/pred histogram
    # can remain internally self-consistent.
    cpu = torch.device("cpu")
    text_by_split: dict[str, torch.Tensor] = {}
    prompt_templates = list(manifest["protocol"]["prompt_templates"])
    for split in REQUIRED_SPLITS:
        class_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        class_names = [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
        text_path = _validate_file_record(
            registry["text_split_caches"][split],
            f"split {split} text cache for deterministic replay",
        )
        text_by_split[split] = text_evaluator.load_frozen_text_cache(
            text_path,
            class_names=class_names,
            prompt_templates=prompt_templates,
            device=cpu,
        )
    recomputed = text_evaluator.evaluate(
        scene=scene,
        label_ply=label,
        semantic_cache=Path(semantic_record["path"]),
        split_text_embeddings=text_by_split,
        split_names=list(REQUIRED_SPLITS),
        projection_k=int(manifest["protocol"]["projection_k"]),
        distance_epsilon=float(manifest["protocol"]["distance_epsilon"]),
        chunk_size=int(manifest["protocol"]["evaluation_chunk_size"]),
        device=cpu,
        allow_mpr_oracle=False,
        scale_aggregation="max",
        scale_specificity_margin=float(
            manifest["protocol"]["scale_specificity_margin"]
        ),
        tree_workers=int(manifest["protocol"]["knn_workers"]),
        torch_threads=int(manifest["protocol"]["torch_num_threads"]),
    )
    recomputed["protocol"]["prompt_templates"] = prompt_templates
    recomputed["protocol"]["class_aliases"] = "none"
    recomputed["protocol"]["text_embedding_cache_base"] = registry[
        "text_cache_base"
    ]["path"]
    _require(
        report == recomputed,
        "evaluation report differs from full deterministic prediction replay",
    )

    mesh_xyz, gt_labels, label_digest = text_evaluator.read_label_ply_secure(label)
    _require(
        report_num_mesh_vertices == int(mesh_xyz.shape[0])
        and label_digest == registry["scenes"][scene]["label"]["sha256"],
        "evaluation mesh vertex count differs from the label PLY",
    )
    splits = report.get("splits")
    _require(
        isinstance(splits, Mapping) and set(splits) == set(REQUIRED_SPLITS),
        "evaluation split set differs",
    )
    metrics = {}
    for split in REQUIRED_SPLITS:
        expected_valid = int(
            np.isin(
                gt_labels,
                np.asarray(OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]),
            ).sum()
        )
        metrics[split] = _validate_split_report(
            split,
            splits[split],
            expected_num_valid=expected_valid,
        )
    return report, metrics


def finalize_evaluation_stage(
    *,
    run_manifest: Path,
    seed: int,
    scene: str,
    report: Path,
    terminal: Path,
    write: bool = True,
    semantic_validation: Mapping[str, Any] | None = None,
    verify_text_caches: bool = True,
) -> dict[str, Any]:
    run_manifest = Path(run_manifest).resolve()
    manifest = _validate_run_manifest(run_manifest, verify_registry=False)
    stage = _stage_record(manifest, int(seed), str(scene))
    terminal = Path(terminal).resolve()
    _require(
        terminal == Path(stage["evaluation_terminal"]),
        "evaluation terminal path differs from the stage registry",
    )
    report_payload, metrics = _validate_evaluation_report(
        manifest,
        seed=int(seed),
        scene=str(scene),
        report_path=Path(report),
        semantic_validation=semantic_validation,
        verify_text_caches=verify_text_caches,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": EVAL_TERMINAL_TYPE,
        "status": "complete",
        "seed": int(seed),
        "scene": str(scene),
        "run_manifest": _file_record(run_manifest),
        "semantic_terminal": _file_record(stage["semantic_terminal"]),
        "evaluation_report": _file_record(report),
        "validated_metrics": metrics,
        "evaluator_provenance": next(
            record
            for record in manifest["implementation_sources"]
            if record["relative_path"]
            == "radio_gs/scripts/eval_scannet_canonical_text_query.py"
        ),
        "report_schema_version": report_payload["schema_version"],
    }
    if write:
        if terminal.exists() or terminal.is_symlink():
            _require(_json_object(terminal) == payload, "evaluation terminal differs")
        else:
            _write_frozen_json(terminal, payload)
    elif terminal.exists():
        _require(_json_object(terminal) == payload, "evaluation terminal differs")
    return payload


def _distribution(values: Sequence[float]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    _require(
        len(numeric) >= 2 and all(math.isfinite(value) for value in numeric),
        "metric distribution is invalid",
    )
    return {
        "mean": statistics.fmean(numeric),
        "population_variance": statistics.pvariance(numeric),
        "sample_variance": statistics.variance(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def aggregate_validated_metrics(
    metrics: Mapping[tuple[int, str], Mapping[str, Mapping[str, float]]],
    *,
    method_id: str,
) -> dict[str, Any]:
    expected = {(seed, scene) for seed in REQUIRED_SEEDS for scene in REQUIRED_SCENES}
    _require(
        set(metrics) == expected,
        "aggregate requires exactly all nine seed/scene reports",
    )
    splits: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        split_result: dict[str, Any] = {}
        for metric in ("miou", "macc"):
            per_seed = {
                str(seed): _distribution(
                    [metrics[(seed, scene)][split][metric] for scene in REQUIRED_SCENES]
                )
                for seed in REQUIRED_SEEDS
            }
            per_scene = {
                scene: _distribution(
                    [metrics[(seed, scene)][split][metric] for seed in REQUIRED_SEEDS]
                )
                for scene in REQUIRED_SCENES
            }
            seed_scene_macros = [per_seed[str(seed)]["mean"] for seed in REQUIRED_SEEDS]
            scene_seed_macros = [per_scene[scene]["mean"] for scene in REQUIRED_SCENES]
            all_nine = [
                metrics[(seed, scene)][split][metric]
                for seed in REQUIRED_SEEDS
                for scene in REQUIRED_SCENES
            ]
            split_result[metric] = {
                "per_seed_scene_distribution": per_seed,
                "per_scene_seed_distribution": per_scene,
                "seed_macro_distribution_of_scene_macros": _distribution(
                    seed_scene_macros
                ),
                "scene_macro_distribution_of_seed_macros": _distribution(
                    scene_seed_macros
                ),
                "all_nine_seed_scene_distribution": _distribution(all_nine),
                "grand_macro_mean": statistics.fmean(all_nine),
            }
            _require(
                _close(
                    statistics.fmean(seed_scene_macros),
                    statistics.fmean(all_nine),
                ),
                "seed and scene macro means disagree",
            )
            _require(
                _close(
                    statistics.fmean(scene_seed_macros),
                    statistics.fmean(all_nine),
                ),
                "scene and seed macro means disagree",
            )
        splits[split] = split_result
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": AGGREGATE_ARTIFACT_TYPE,
        "status": "complete_exact_three_seed_three_scene_macro",
        "method_id": str(method_id),
        "required_seeds": list(REQUIRED_SEEDS),
        "required_scenes": list(REQUIRED_SCENES),
        "report_count": 9,
        "variance_definition": {
            "population_variance": "sum((x-mean)^2)/N",
            "sample_variance": "sum((x-mean)^2)/(N-1)",
            "scene_macro": "equal_weight_over_three_required_scenes",
            "seed_macro": "equal_weight_over_three_required_seeds",
        },
        "splits": splits,
    }


def finalize_benchmark(
    *,
    run_manifest: Path,
    aggregate_output: Path,
    completion: Path,
) -> dict[str, Any]:
    run_manifest = Path(run_manifest).resolve()
    manifest = _validate_run_manifest(run_manifest, verify_registry=True)
    authority = manifest["promotion_authority"]
    receipt_record = manifest["authority_receipt"]
    recomputed_gate = validate_authority_receipt(
        Path(receipt_record["path"]),
        expected_sha256=receipt_record["sha256"],
    )
    _require(
        recomputed_gate["receipt"] == receipt_record
        and recomputed_gate["authority"] == authority,
        "promotion authority changed after benchmark opening",
    )
    metrics: dict[tuple[int, str], Mapping[str, Mapping[str, float]]] = {}
    semantic_records = []
    report_records = []
    for seed in REQUIRED_SEEDS:
        for scene in REQUIRED_SCENES:
            stage = _stage_record(manifest, seed, scene)
            semantic = finalize_semantic_stage(
                run_manifest=run_manifest,
                seed=seed,
                scene=scene,
                semantic_cache=Path(stage["semantic_cache"]),
                guard_receipt=Path(stage["semantic_guard_receipt"]),
                terminal=Path(stage["semantic_terminal"]),
                write=False,
            )
            _require(
                Path(stage["semantic_terminal"]).is_file(),
                "semantic terminal is missing",
            )
            evaluation = finalize_evaluation_stage(
                run_manifest=run_manifest,
                seed=seed,
                scene=scene,
                report=Path(stage["evaluation_report"]),
                terminal=Path(stage["evaluation_terminal"]),
                write=False,
                semantic_validation=semantic,
                verify_text_caches=False,
            )
            _require(
                Path(stage["evaluation_terminal"]).is_file(),
                "evaluation terminal is missing",
            )
            metrics[(seed, scene)] = evaluation["validated_metrics"]
            semantic_records.append(
                {
                    "seed": seed,
                    "scene": scene,
                    **semantic["semantic"]["semantic_cache"],
                    "terminal": _file_record(stage["semantic_terminal"]),
                }
            )
            report_records.append(
                {
                    "seed": seed,
                    "scene": scene,
                    **evaluation["evaluation_report"],
                    "terminal": _file_record(stage["evaluation_terminal"]),
                }
            )
    aggregate = aggregate_validated_metrics(
        metrics,
        method_id=manifest["method"]["method_id"],
    )
    aggregate["run_manifest"] = _file_record(run_manifest)
    aggregate["semantic_caches"] = semantic_records
    aggregate["evaluation_reports"] = report_records
    aggregate_output = Path(aggregate_output).resolve()
    completion = Path(completion).resolve()
    expected_root = Path(manifest["output_root"])
    _require(
        aggregate_output == expected_root / "aggregate.json"
        and completion == expected_root / "benchmark.complete.json",
        "aggregate/completion paths differ from the run root",
    )
    if aggregate_output.exists() or aggregate_output.is_symlink():
        _require(_json_object(aggregate_output) == aggregate, "aggregate differs")
    else:
        _write_frozen_json(aggregate_output, aggregate)
    completion_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETION_ARTIFACT_TYPE,
        "status": "complete_all_three_seeds_all_three_scenes",
        "main_result_eligible": True,
        "run_manifest": _file_record(run_manifest),
        "aggregate": _file_record(aggregate_output),
        "method_id": manifest["method"]["method_id"],
        "required_seeds": list(REQUIRED_SEEDS),
        "required_scenes": list(REQUIRED_SCENES),
        "semantic_cache_count": len(semantic_records),
        "evaluation_report_count": len(report_records),
        "single_seed_or_subset_result_forbidden": True,
    }
    if completion.exists() or completion.is_symlink():
        _require(
            _json_object(completion) == completion_payload,
            "benchmark completion differs",
        )
    else:
        _write_frozen_json(completion, completion_payload)
    return {
        "aggregate": str(aggregate_output),
        "aggregate_sha256": _sha256(aggregate_output),
        "completion": str(completion),
        "completion_sha256": _sha256(completion),
        "method_id": manifest["method"]["method_id"],
        "semantic_cache_count": 9,
        "evaluation_report_count": 9,
        "main_result_eligible": True,
        "device": "cpu_finalizer",
    }


def stage_info(*, run_manifest: Path, seed: int, scene: str) -> dict[str, Any]:
    manifest = _validate_run_manifest(Path(run_manifest), verify_registry=False)
    stage = _stage_record(manifest, int(seed), str(scene))
    return {
        "seed": int(seed),
        "scene": str(scene),
        "stage": stage,
        "readout": _readout_record(manifest, int(seed)),
        "scene_inputs": manifest["input_registry"][str(scene)],
        "radio_checkpoint": manifest["input_registry"]["radio_checkpoint"],
        "gpu_identity": manifest["gpu_identity"],
        "semantic_execution": manifest["semantic_execution"],
    }


def _thermal_contract_from_args(args: argparse.Namespace) -> dict[str, Any]:
    guard = Path(args.thermal_guard).resolve()
    return {
        "guarded": True,
        "guard": _file_record(guard),
        "physical_gpu": int(args.gpu),
        "maximum_temperature_c": int(args.gpu_max_temp_c),
        "maximum_start_temperature_c": int(args.gpu_start_max_temp_c),
        "maximum_power_limit_w": float(args.gpu_max_power_limit_w),
        "poll_seconds": int(args.gpu_poll_seconds),
        "soft_pause_temperature_c": int(args.gpu_soft_pause_temp_c),
        "soft_resume_temperature_c": int(args.gpu_soft_resume_temp_c),
        "peer_gpu": int(args.gpu_peer_index),
        "peer_pause_temperature_c": int(args.gpu_peer_pause_temp_c),
        "peer_resume_temperature_c": int(args.gpu_peer_resume_temp_c),
        "peer_quiet_seconds_before_launch": int(args.gpu_peer_quiet_seconds),
        "peer_maximum_power_w": float(args.gpu_peer_max_power_w),
        "peer_maximum_memory_mib": int(args.gpu_peer_max_memory_mib),
        "peer_maximum_utilization_percent": int(args.gpu_peer_max_util_pct),
        "peer_activity_gate": (
            "temperature_only_activity_limits_disabled"
            if float(args.gpu_peer_max_power_w) == 0.0
            and int(args.gpu_peer_max_memory_mib) == 0
            and int(args.gpu_peer_max_util_pct) == 100
            and int(args.gpu_peer_quiet_seconds) == 0
            else "custom_peer_activity_gate"
        ),
        "semantic_builder_resume": "durable_per_batch_resume_with_strict_contract",
        "hard_guard_is_safety_authority": True,
        "soft_pause_safety_role": "supplementary_only_not_a_safety_precondition",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--authority-receipt", required=True, type=Path)
    pre.add_argument("--authority-receipt-sha256", required=True)
    pre.add_argument("--output-root", required=True, type=Path)
    pre.add_argument("--run-manifest", required=True, type=Path)
    pre.add_argument("--semantic-radio-batch-size", type=int, default=1024)
    pre.add_argument("--semantic-batch-size", type=int, default=64)
    pre.add_argument("--semantic-pacing-seconds", type=float, default=4.0)
    pre.add_argument("--thermal-guard", required=True, type=Path)
    pre.add_argument("--gpu", type=int, required=True)
    pre.add_argument("--gpu-uuid", required=True)
    pre.add_argument("--gpu-bus-id", required=True)
    pre.add_argument("--gpu-max-temp-c", type=int, required=True)
    pre.add_argument("--gpu-start-max-temp-c", type=int, required=True)
    pre.add_argument("--gpu-max-power-limit-w", type=float, required=True)
    pre.add_argument("--gpu-poll-seconds", type=int, required=True)
    pre.add_argument("--gpu-soft-pause-temp-c", type=int, required=True)
    pre.add_argument("--gpu-soft-resume-temp-c", type=int, required=True)
    pre.add_argument("--gpu-peer-index", type=int, required=True)
    pre.add_argument("--gpu-peer-pause-temp-c", type=int, required=True)
    pre.add_argument("--gpu-peer-resume-temp-c", type=int, required=True)
    pre.add_argument("--gpu-peer-quiet-seconds", type=int, required=True)
    pre.add_argument("--gpu-peer-max-power-w", type=float, required=True)
    pre.add_argument("--gpu-peer-max-memory-mib", type=int, required=True)
    pre.add_argument("--gpu-peer-max-util-pct", type=int, required=True)

    semantic = subparsers.add_parser("finalize-semantic")
    semantic.add_argument("--run-manifest", required=True, type=Path)
    semantic.add_argument("--seed", required=True, type=int)
    semantic.add_argument("--scene", required=True)
    semantic.add_argument("--semantic-cache", required=True, type=Path)
    semantic.add_argument("--guard-receipt", required=True, type=Path)
    semantic.add_argument("--terminal", required=True, type=Path)

    evaluation = subparsers.add_parser("finalize-eval")
    evaluation.add_argument("--run-manifest", required=True, type=Path)
    evaluation.add_argument("--seed", required=True, type=int)
    evaluation.add_argument("--scene", required=True)
    evaluation.add_argument("--report", required=True, type=Path)
    evaluation.add_argument("--terminal", required=True, type=Path)

    info = subparsers.add_parser("stage-info")
    info.add_argument("--run-manifest", required=True, type=Path)
    info.add_argument("--seed", required=True, type=int)
    info.add_argument("--scene", required=True)

    final = subparsers.add_parser("finalize-benchmark")
    final.add_argument("--run-manifest", required=True, type=Path)
    final.add_argument("--aggregate-output", required=True, type=Path)
    final.add_argument("--completion", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        result = preflight(
            authority_receipt=args.authority_receipt,
            authority_receipt_sha256=args.authority_receipt_sha256,
            output_root=args.output_root,
            run_manifest=args.run_manifest,
            semantic_radio_batch_size=args.semantic_radio_batch_size,
            semantic_batch_size=args.semantic_batch_size,
            semantic_pacing_seconds=args.semantic_pacing_seconds,
            thermal_contract=_thermal_contract_from_args(args),
            gpu_identity={
                "physical_index": args.gpu,
                "uuid": args.gpu_uuid,
                "pci_bus_id": args.gpu_bus_id,
            },
        )
    elif args.command == "finalize-semantic":
        result = finalize_semantic_stage(
            run_manifest=args.run_manifest,
            seed=args.seed,
            scene=args.scene,
            semantic_cache=args.semantic_cache,
            guard_receipt=args.guard_receipt,
            terminal=args.terminal,
        )
    elif args.command == "finalize-eval":
        result = finalize_evaluation_stage(
            run_manifest=args.run_manifest,
            seed=args.seed,
            scene=args.scene,
            report=args.report,
            terminal=args.terminal,
        )
    elif args.command == "stage-info":
        result = stage_info(
            run_manifest=args.run_manifest,
            seed=args.seed,
            scene=args.scene,
        )
    else:
        result = finalize_benchmark(
            run_manifest=args.run_manifest,
            aggregate_output=args.aggregate_output,
            completion=args.completion,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
