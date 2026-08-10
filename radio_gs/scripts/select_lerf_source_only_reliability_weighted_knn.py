#!/usr/bin/env python3
"""Select one global weighted-kNN policy using query-free source descriptors.

For every source scene, a deterministic subset of primitives is predicted
from its other valid spatial neighbours.  The target is the durable source
teacher mean; the centre itself is removed, so the audit cannot win by copying
its target.  Candidate weights are exactly the deployed spatial/reliability
kernels.  Selection requires pooled mean improvement and mean/p05
non-regression in every source scene.  No text bank, benchmark image, mask,
label, renderer, or target metric is opened.

Large descriptor payloads are PyTorch ZIP archives with uncompressed aligned
storages.  This script memory-maps only the declared tensors and never loads
the multi-GiB compact field or full teacher matrix into host memory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.querying import (
    reliability_weighted_valid_domain_knn_readout as weighted,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


PREREGISTRATION_SCHEMA = (
    "radio_gs.lerf_reliability_weighted_valid_domain_knn_preregistration.v1"
)
RESULT_SCHEMA = (
    "radio_gs.lerf_source_only_reliability_weighted_valid_domain_knn_result.v1"
)
SIDECAR_SCHEMA = "radio_gs.lerf_source_view_reliability_sidecar.v1"
SCHEMA_VERSION = 1
DESCRIPTOR_DIMENSION = 1536
MAXIMUM_AUDIT_ROWS_PER_SCENE = 8192
AUDIT_CHUNK_ROWS = 256
NONREGRESSION_TOLERANCE = 1e-7


def method_contract() -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "weighted_readout_contract": weighted.CONTRACT,
        "weighted_readout_implementation": file_record(
            Path(weighted.__file__).resolve()
        ),
        "candidate_grid": [policy.policy_id for policy in weighted.POLICIES],
        "candidate_count": len(weighted.POLICIES),
        "knn_k": weighted.KNN_K,
        "source_target": "durable_normalized_equal_view_teacher_mean",
        "source_prediction": (
            "normalized_weighted_mean_of_other_valid_spatial_neighbors"
        ),
        "center_primitive_removed_from_prediction": True,
        "sampling": (
            "sorted_teacher_valid_global_rows_evenly_spaced_max8192_per_scene"
        ),
        "statistics": ["mean_cosine", "exact_linear_p05_cosine"],
        "selection": {
            "baseline": "uniform",
            "eligible": (
                "strict_pooled_mean_improvement_and_every_scene_mean_and_p05_"
                "nonregression"
            ),
            "winner": "maximum_pooled_mean_cosine_then_candidate_grid_order",
            "fallback": "uniform",
        },
        "one_global_policy": True,
        "scene_or_query_specific_parameters": False,
        "query_embeddings_or_scores_opened": False,
        "benchmark_images_masks_labels_metrics_opened": False,
        "gpu_used": False,
        "target_metric_execution_authorized": False,
    }


def access_audit() -> dict[str, bool]:
    return {
        "source_base_geometry_opened": True,
        "source_teacher_mean_opened": True,
        "source_view_count_and_agreement_opened": True,
        "query_embeddings_or_scores_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_or_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def validate_preregistration(path: str | Path, digest: str) -> dict[str, Any]:
    payload, _, _ = load_json_object(
        path,
        expected_sha256=digest,
        label="weighted-kNN source-only preregistration",
    )
    expected = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "method_contract",
        "candidate_results_opened_at_seal",
        "target_data_or_metrics_opened_at_seal",
        "source_scene_count_minimum",
        "metric_execution_authorized",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema") != PREREGISTRATION_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status")
        != "sealed_before_source_gate_or_target_result_inspection"
        or payload.get("implementation")
        != file_record(Path(__file__).resolve())
        or payload.get("method_contract") != method_contract()
        or payload.get("candidate_results_opened_at_seal") is not False
        or payload.get("target_data_or_metrics_opened_at_seal") is not False
        or payload.get("source_scene_count_minimum") != 2
        or payload.get("metric_execution_authorized") is not False
    ):
        raise ValueError("weighted-kNN preregistration differs")
    return dict(payload)


def _zip_storage_memmap(
    path: str | Path,
    *,
    storage_index: int,
    dtype: str,
    shape: tuple[int, ...],
) -> np.memmap:
    """Map one ZIP_STORED PyTorch tensor storage without allocating siblings."""

    source = Path(path).expanduser().resolve()
    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        pickle_names = [name for name in names if name.endswith("/data.pkl")]
        if len(pickle_names) != 1:
            raise ValueError("PyTorch ZIP must contain one data.pkl")
        prefix = pickle_names[0][: -len("data.pkl")]
        info = archive.getinfo(f"{prefix}data/{storage_index}")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("selective tensor storage must be ZIP_STORED")
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if info.file_size != expected_bytes:
        raise ValueError("selective tensor storage size differs")
    with source.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
    if len(header) != 30:
        raise ValueError("truncated ZIP local header")
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50 or fields[3] != zipfile.ZIP_STORED:
        raise ValueError("ZIP local tensor header differs")
    offset = info.header_offset + 30 + fields[-2] + fields[-1]
    return np.memmap(source, mode="r", dtype=dtype, offset=offset, shape=shape)


def load_base_geometry_selective(
    record: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    """Load only xyz/global_rows/valid from a frozen native-v2 descriptor."""

    source = _record(record, label="source base descriptor")
    if sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("source base descriptor SHA differs")
    with zipfile.ZipFile(source["path"], "r") as archive:
        pickle_name = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")),
            None,
        )
        if pickle_name is None:
            raise ValueError("source base descriptor data.pkl missing")
        pickle_data = archive.read(pickle_name)
        prefix = pickle_name[: -len("data.pkl")]
        xyz_bytes = archive.getinfo(f"{prefix}data/0").file_size
        row_bytes = archive.getinfo(f"{prefix}data/2").file_size
        valid_bytes = archive.getinfo(f"{prefix}data/4").file_size
    required_strings = (
        b"xyz",
        b"features_by_scale",
        b"global_rows",
        b"valid",
        b"official_siglip2_summary_descriptor_multiscale",
    )
    if any(value not in pickle_data for value in required_strings):
        raise ValueError("source base descriptor schema differs")
    if xyz_bytes % 12 != 0:
        raise ValueError("source xyz storage size differs")
    total = xyz_bytes // 12
    if valid_bytes != total or row_bytes % 8 != 0:
        raise ValueError("source geometry axes differ")
    accepted = row_bytes // 8
    xyz = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=0, dtype="<f4", shape=(total, 3)
            ),
            copy=True,
        )
    )
    rows = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=2, dtype="<i8", shape=(accepted,)
            ),
            copy=True,
        )
    ).long()
    valid = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=4, dtype="u1", shape=(total,)
            ),
            copy=True,
        )
    ).bool()
    if (
        not bool(torch.isfinite(xyz).all())
        or rows.shape != (accepted,)
        or not torch.equal(rows, torch.where(valid)[0])
    ):
        raise ValueError("source base geometry content differs")
    return {"xyz": xyz.contiguous(), "global_rows": rows, "valid": valid}


def load_teacher_selective(
    record: Mapping[str, str],
) -> dict[str, Any]:
    """Map teacher descriptors and load only its four small aligned tensors."""

    source = _record(record, label="source teacher payload")
    if sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("source teacher payload SHA differs")
    with zipfile.ZipFile(source["path"], "r") as archive:
        pickle_name = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")),
            None,
        )
        if pickle_name is None:
            raise ValueError("source teacher data.pkl missing")
        pickle_data = archive.read(pickle_name)
        prefix = pickle_name[: -len("data.pkl")]
        sizes = [
            archive.getinfo(f"{prefix}data/{index}").file_size
            for index in range(5)
        ]
    required_strings = (
        b"radio_gs.lerf_source_teacher_mean_siglip.v2",
        b"global_rows",
        b"teacher_mean",
        b"teacher_valid",
        b"retained_view_count",
        b"teacher_view_directional_resultant",
    )
    if any(value not in pickle_data for value in required_strings):
        raise ValueError("source teacher payload schema differs")
    if sizes[0] % 8 != 0:
        raise ValueError("teacher global-row storage size differs")
    rows = sizes[0] // 8
    if sizes != [rows * 8, rows * DESCRIPTOR_DIMENSION * 2, rows, rows, rows * 4]:
        raise ValueError("teacher selective storage layout differs")
    global_rows = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=0, dtype="<i8", shape=(rows,)
            ),
            copy=True,
        )
    ).long()
    valid = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=2, dtype="u1", shape=(rows,)
            ),
            copy=True,
        )
    ).bool()
    count = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=3, dtype="u1", shape=(rows,)
            ),
            copy=True,
        )
    ).to(torch.uint8)
    agreement = torch.from_numpy(
        np.array(
            _zip_storage_memmap(
                source["path"], storage_index=4, dtype="<f4", shape=(rows,)
            ),
            copy=True,
        )
    ).float()
    descriptor = _zip_storage_memmap(
        source["path"],
        storage_index=1,
        dtype="<f2",
        shape=(rows, DESCRIPTOR_DIMENSION),
    )
    reliability = weighted.source_view_reliability(
        count, agreement, valid_mask=valid
    )
    if (
        not torch.equal(valid, count > 0)
        or bool((agreement[~valid] != 0).any())
        or not bool(torch.isfinite(agreement).all())
    ):
        raise ValueError("teacher reliability tensors differ")
    return {
        "global_rows": global_rows,
        "teacher_valid": valid,
        "retained_view_count": count,
        "directional_resultant": agreement,
        "reliability": reliability,
        "teacher_mean_memmap": descriptor,
        "source": source,
    }


def deterministic_audit_rows(rows: torch.Tensor, *, maximum: int) -> torch.Tensor:
    values = torch.as_tensor(rows).detach().long().cpu().reshape(-1)
    if values.numel() == 0 or not torch.equal(values, values.sort().values):
        raise ValueError("audit rows must be non-empty and sorted")
    if maximum <= 0:
        raise ValueError("maximum audit rows must be positive")
    if values.numel() <= maximum:
        return values
    indices = torch.linspace(0, values.numel() - 1, maximum).round().long()
    return values[indices].contiguous()


def _candidate_weights_for_loo(
    distances: torch.Tensor,
    neighbor_reliability: torch.Tensor,
    usable: torch.Tensor,
    neighbor_is_self: torch.Tensor,
    *,
    policy_id: str,
) -> torch.Tensor:
    weights = weighted.normalized_neighbor_weights(
        distances,
        neighbor_reliability,
        policy_id=policy_id,
        neighbor_is_self=neighbor_is_self,
    )
    usable_mask = torch.as_tensor(usable).bool()
    masked = weights * usable_mask
    mass = masked.sum(dim=1, keepdim=True)
    # If precision masking removes all mass, reliability is uninformative;
    # use the same policy's spatial component over usable neighbours.
    spatial_policy = "gaussian" if weighted.POLICY_BY_ID[
        policy_id
    ].gaussian_distance else "uniform"
    fallback = weighted.normalized_neighbor_weights(
        distances, torch.zeros_like(neighbor_reliability), policy_id=spatial_policy
    ) * usable_mask
    fallback_mass = fallback.sum(dim=1, keepdim=True)
    if bool((fallback_mass <= 0).any()):
        raise ValueError("LOO row has no usable non-centre teacher neighbour")
    return torch.where(
        mass > 0,
        masked / mass.clamp_min(torch.finfo(torch.float32).tiny),
        fallback / fallback_mass,
    )


def evaluate_scene(
    *,
    scene_id: str,
    geometry: Mapping[str, torch.Tensor],
    teacher: Mapping[str, Any],
    maximum_audit_rows: int = MAXIMUM_AUDIT_ROWS_PER_SCENE,
) -> dict[str, Any]:
    """Evaluate all policies on deterministic leave-one-primitive-out source data."""

    xyz = torch.as_tensor(geometry["xyz"]).float().cpu().contiguous()
    valid = torch.as_tensor(geometry["valid"]).bool().cpu().reshape(-1)
    base_rows = torch.as_tensor(geometry["global_rows"]).long().cpu().reshape(-1)
    teacher_rows = torch.as_tensor(teacher["global_rows"]).long().cpu().reshape(-1)
    teacher_valid = torch.as_tensor(teacher["teacher_valid"]).bool().cpu()
    local_reliability = torch.as_tensor(teacher["reliability"]).float().cpu()
    if not torch.equal(base_rows, teacher_rows):
        raise ValueError(f"{scene_id}: base/teacher global rows differ")
    if xyz.shape != (valid.numel(), 3) or not torch.equal(base_rows, torch.where(valid)[0]):
        raise ValueError(f"{scene_id}: geometry validity domain differs")
    global_to_local = torch.full((valid.numel(),), -1, dtype=torch.long)
    global_to_local[teacher_rows] = torch.arange(teacher_rows.numel())
    global_reliability = torch.zeros(valid.numel(), dtype=torch.float32)
    global_reliability[teacher_rows] = local_reliability

    eligible_centres = teacher_rows[teacher_valid]
    centres = deterministic_audit_rows(eligible_centres, maximum=maximum_audit_rows)
    from sklearn.neighbors import NearestNeighbors

    valid_rows = torch.where(valid)[0]
    neighbor_count = min(weighted.KNN_K + 1, int(valid_rows.numel()))
    estimator = NearestNeighbors(n_neighbors=neighbor_count).fit(
        xyz[valid_rows].numpy()
    )
    distance_np, local_np = estimator.kneighbors(
        xyz[centres].numpy(), return_distance=True
    )
    distances = torch.from_numpy(distance_np).float()
    neighbors = valid_rows[torch.from_numpy(local_np).long()]
    neighbor_local = global_to_local[neighbors]
    neighbor_has_teacher = neighbor_local >= 0
    safe_neighbor_local = neighbor_local.clamp_min(0)
    neighbor_teacher_valid = torch.zeros_like(neighbor_has_teacher)
    neighbor_teacher_valid[neighbor_has_teacher] = teacher_valid[
        safe_neighbor_local[neighbor_has_teacher]
    ]
    usable = neighbor_teacher_valid & (neighbors != centres[:, None])
    keep = usable[:, : min(weighted.KNN_K, neighbor_count)].any(dim=1)
    if not bool(keep.any()):
        raise ValueError(f"{scene_id}: no source LOO audit rows remain")
    centres = centres[keep]
    distances = distances[keep]
    neighbors = neighbors[keep]
    safe_neighbor_local = safe_neighbor_local[keep]
    usable = usable[keep]
    center_local = global_to_local[centres]
    neighbor_reliability = global_reliability[neighbors]
    policies = [policy.policy_id for policy in weighted.POLICIES]
    cosine_parts: dict[str, list[torch.Tensor]] = {name: [] for name in policies}
    descriptor_memmap = teacher["teacher_mean_memmap"]

    for start in range(0, centres.numel(), AUDIT_CHUNK_ROWS):
        stop = min(start + AUDIT_CHUNK_ROWS, centres.numel())
        target = torch.from_numpy(
            np.array(descriptor_memmap[center_local[start:stop].numpy()], copy=True)
        ).float()
        target = F.normalize(target, dim=-1)
        neighbor_descriptor = torch.from_numpy(
            np.array(
                descriptor_memmap[safe_neighbor_local[start:stop].numpy()],
                copy=True,
            )
        ).float()
        neighbor_descriptor = F.normalize(neighbor_descriptor, dim=-1)
        for policy_id in policies:
            policy = weighted.POLICY_BY_ID[policy_id]
            width = (
                min(weighted.KNN_K + 1, neighbor_count)
                if policy.exclude_self
                else min(weighted.KNN_K, neighbor_count)
            )
            weights = _candidate_weights_for_loo(
                distances[start:stop, :width],
                neighbor_reliability[start:stop, :width],
                usable[start:stop, :width],
                neighbors[start:stop, :width] == centres[start:stop, None],
                policy_id=policy_id,
            )
            prediction = F.normalize(
                (neighbor_descriptor[:, :width] * weights[..., None]).sum(dim=1),
                dim=-1,
            )
            cosine_parts[policy_id].append((prediction * target).sum(dim=-1))

    rows = []
    for policy_id in policies:
        cosine = torch.cat(cosine_parts[policy_id]).double()
        rows.append(
            {
                "policy_id": policy_id,
                "observations": int(cosine.numel()),
                "cosine_sum": float(cosine.sum()),
                "mean_cosine": float(cosine.mean()),
                "p05_cosine": float(torch.quantile(cosine, 0.05)),
            }
        )
    return {
        "scene_id": scene_id,
        "valid_primitives": int(valid.sum()),
        "teacher_valid_primitives": int(teacher_valid.sum()),
        "audit_rows_requested": int(
            min(int(eligible_centres.numel()), maximum_audit_rows)
        ),
        "audit_rows_evaluated": int(centres.numel()),
        "candidate_statistics": rows,
        "global_reliability": global_reliability,
    }


def select_policy(scene_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(scene_results) < 2:
        raise ValueError("weighted-kNN gate requires at least two source scenes")
    policy_ids = [policy.policy_id for policy in weighted.POLICIES]
    per_policy: dict[str, list[Mapping[str, Any]]] = {name: [] for name in policy_ids}
    for scene in scene_results:
        rows = scene.get("candidate_statistics")
        if not isinstance(rows, list) or [row.get("policy_id") for row in rows] != policy_ids:
            raise ValueError("source scene candidate grid differs")
        for row in rows:
            per_policy[str(row["policy_id"])].append(row)
    baseline_rows = per_policy["uniform"]
    candidates = []
    for policy_id in policy_ids:
        rows = per_policy[policy_id]
        observations = sum(int(row["observations"]) for row in rows)
        pooled_mean = sum(float(row["cosine_sum"]) for row in rows) / observations
        baseline_observations = sum(int(row["observations"]) for row in baseline_rows)
        baseline_mean = (
            sum(float(row["cosine_sum"]) for row in baseline_rows)
            / baseline_observations
        )
        every_mean = all(
            float(row["mean_cosine"])
            >= float(base["mean_cosine"]) - NONREGRESSION_TOLERANCE
            for row, base in zip(rows, baseline_rows)
        )
        every_p05 = all(
            float(row["p05_cosine"])
            >= float(base["p05_cosine"]) - NONREGRESSION_TOLERANCE
            for row, base in zip(rows, baseline_rows)
        )
        strict = pooled_mean > baseline_mean + NONREGRESSION_TOLERANCE
        candidates.append(
            {
                "policy_id": policy_id,
                "pooled_observations": observations,
                "pooled_mean_cosine": pooled_mean,
                "pooled_delta_mean_cosine_vs_uniform": pooled_mean - baseline_mean,
                "every_source_mean_nonregression": every_mean,
                "every_source_p05_nonregression": every_p05,
                "strict_pooled_mean_improvement": strict,
                "eligible": strict and every_mean and every_p05,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["pooled_mean_cosine"]),
                -policy_ids.index(str(row["policy_id"])),
            ),
        )["policy_id"]
        if eligible
        else "uniform"
    )
    return {
        "candidate_grid": candidates,
        "selected_policy_id": selected,
        "fallback_uniform_used": not bool(eligible),
        "target_metric_execution_authorized": selected != "uniform",
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    validate_preregistration(args.preregistration, args.preregistration_sha256)
    manifest, _, _ = load_json_object(
        args.source_manifest,
        expected_sha256=args.source_manifest_sha256,
        label="weighted-kNN source manifest",
    )
    scenes = manifest.get("scenes") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(scenes, list)
        or len(scenes) < 2
        or [item.get("scene_id") for item in scenes]
        != sorted(item.get("scene_id") for item in scenes)
        or len({item.get("scene_id") for item in scenes}) != len(scenes)
    ):
        raise ValueError("weighted-kNN source scene manifest differs")
    output = Path(args.output_report).expanduser().resolve()
    sidecar_dir = Path(args.sidecar_dir).expanduser().resolve()
    if str(output) != args.output_report or str(sidecar_dir) != args.sidecar_dir:
        raise ValueError("output paths must be canonical absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"weighted-kNN output exists: {output}")
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    inputs = []
    sidecars = []
    for item in scenes:
        if not isinstance(item, Mapping) or set(item) != {
            "scene_id", "base_descriptor", "teacher_payload"
        }:
            raise ValueError("weighted-kNN source scene entry differs")
        scene_id = str(item["scene_id"])
        base_record = _record(item["base_descriptor"], label=f"{scene_id} base")
        teacher_record = _record(item["teacher_payload"], label=f"{scene_id} teacher")
        geometry = load_base_geometry_selective(base_record)
        teacher = load_teacher_selective(teacher_record)
        evaluated = evaluate_scene(
            scene_id=scene_id,
            geometry=geometry,
            teacher=teacher,
            maximum_audit_rows=args.maximum_audit_rows,
        )
        sidecar_path = (sidecar_dir / f"{scene_id}_source_reliability.pt").resolve()
        if sidecar_path.exists() or sidecar_path.is_symlink():
            raise FileExistsError(f"weighted-kNN sidecar exists: {sidecar_path}")
        sidecar_payload = {
            "schema": SIDECAR_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene_id,
            "global_rows": geometry["global_rows"].long().contiguous(),
            "valid": geometry["valid"].bool().contiguous(),
            "reliability": evaluated["global_reliability"].float().contiguous(),
            "source": {
                "base_descriptor": base_record,
                "teacher_payload": teacher_record,
            },
            "method": {
                "formula": (
                    "sqrt(clamp((retained_view_count-1)/3,0,1)*"
                    "teacher_view_directional_resultant)"
                ),
                "query_independent": True,
            },
        }
        write_torch_noclobber(sidecar_path, sidecar_payload)
        sidecar_record = file_record(sidecar_path)
        sidecars.append({"scene_id": scene_id, "sidecar": sidecar_record})
        summary = {key: value for key, value in evaluated.items() if key != "global_reliability"}
        summaries.append(summary)
        inputs.append(
            {
                "scene_id": scene_id,
                "base_descriptor": base_record,
                "teacher_payload": teacher_record,
            }
        )
        del geometry, teacher, evaluated

    selection = select_policy(summaries)
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_only_weighted_knn_gate",
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(),
        "preregistration": file_record(args.preregistration),
        "source_manifest": file_record(args.source_manifest),
        "source_inputs": inputs,
        "source_scene_results": summaries,
        "selection": selection,
        "reliability_sidecars": sidecars,
        "access_audit": access_audit(),
        "metric_executed": False,
        "metric_execution_authorized": bool(
            selection["target_metric_execution_authorized"]
        ),
        "next_gate": (
            "materialize_one_global_selected_policy_then_one_shot_frozen_metric"
            if selection["target_metric_execution_authorized"]
            else "reject_weighted_knn_and_retain_uniform_valid_domain_v1"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_frozen_json(output, result)
    return {**result, "output_report": file_record(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--maximum-audit-rows", type=int, default=MAXIMUM_AUDIT_ROWS_PER_SCENE
    )
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "MAXIMUM_AUDIT_ROWS_PER_SCENE",
    "PREREGISTRATION_SCHEMA",
    "RESULT_SCHEMA",
    "SIDECAR_SCHEMA",
    "access_audit",
    "deterministic_audit_rows",
    "evaluate_scene",
    "load_base_geometry_selective",
    "load_teacher_selective",
    "materialize",
    "method_contract",
    "select_policy",
    "validate_preregistration",
]
