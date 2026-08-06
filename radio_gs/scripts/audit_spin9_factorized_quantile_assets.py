#!/usr/bin/env python3
"""Target-blind CPU inventory for the SPIn9 factorized quantile expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCENE_ORDER = (
    "orchids",
    "leaves",
    "fern",
    "room",
    "horns",
    "fortress",
    "pinecone",
    "truck",
    "lego",
)

# These three carriers were recovered under the frozen exact-local9 protocol rather
# than the older queue's incomplete geometry directories.
EXTERNAL_CARRIER_CONFIGS = {
    scene: Path(
        f"radio_gs/configs/generated/frozen_eval_20260802/"
        f"spin_{scene}_local9_external_geometry_field.yaml"
    )
    for scene in ("room", "horns", "truck")
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def ply_vertex_count(path: Path) -> int:
    """Read only the ASCII PLY header, never the (potentially huge) payload."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        for _ in range(256):
            line = handle.readline()
            if not line:
                break
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded.startswith("element vertex "):
                return int(decoded.rsplit(" ", 1)[1])
            if decoded == "end_header":
                break
    raise ValueError(f"PLY vertex count absent from header: {path}")


def yaml_scalar(path: Path, key: str) -> str:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("'\"")
    raise ValueError(f"{key!r} absent from {path}")


def asset(path: Path, *, hash_small: bool = False) -> dict[str, Any]:
    present = path.is_file()
    record: dict[str, Any] = {"path": str(path.resolve()), "present": present}
    if present:
        record["bytes"] = path.stat().st_size
        if hash_small:
            record["sha256"] = sha256_file(path)
    return record


def sidecar_rows(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    report = json_object(path)
    global_rows = report.get("num_global_rows", report.get("num_gaussians"))
    active_rows = report.get(
        "num_nodes", report.get("valid_gaussians", report.get("valid_count"))
    )
    return (
        int(global_rows) if global_rows is not None else None,
        int(active_rows) if active_rows is not None else None,
    )


def resolve_carrier(
    scene: str, queue_scene: Mapping[str, Any], repo_root: Path
) -> tuple[Path, str]:
    queue_ply = Path(str(queue_scene["artifacts"]["geometry_ply"]["path"]))
    if queue_ply.is_file():
        return queue_ply.resolve(), "frozen_spin9_queue"
    config = repo_root / EXTERNAL_CARRIER_CONFIGS[scene]
    if config.is_file():
        recovered = Path(yaml_scalar(config, "ply_path")).expanduser().resolve()
        if recovered.is_file():
            return recovered, "frozen_exact_local9_external_carrier"
    return queue_ply.resolve(), "missing"


def estimate_resources(global_rows: int, active_rows: int, frames: int) -> dict[str, Any]:
    """Conservative planning estimates; these are not runtime measurements."""

    lego_rows = 636_148
    lego_active = 128_780
    active_scale = active_rows / lego_active
    # The Lego builder reported 5.032 GB CPU peak.  Most terms scale with all
    # carrier rows; retain a 256 MiB fixed allowance.
    cpu_peak = int(256 * 1024**2 + global_rows * ((5_032_039_188 - 256 * 1024**2) / lego_rows))
    # Field + materialized capabilities + surface graph + K201 relation/knn.
    # Exact-W storage is scene/image dependent and is therefore stated separately.
    deterministic_disk = int(
        global_rows * 2560
        + active_rows * (10240 + 2544 + 390 + 1828)
    )
    return {
        "estimated_cpu_peak_gib_factorized_builder": round(cpu_peak / 1024**3, 2),
        "estimated_new_disk_gib_before_exact_w_and_predictions": round(
            deterministic_disk / 1024**3, 2
        ),
        "estimated_gpu_field_capability_graph_hours": [
            round(max(0.45, 0.75 * active_scale), 2),
            round(max(0.9, 1.35 * active_scale), 2),
        ],
        "estimated_gpu_vram_gib": [6, min(22, round(8 + 2.0 * active_scale))],
        "estimated_fullfit_and_target_minutes": [
            max(2, round(3 * active_scale + frames / 20)),
            max(6, round(8 * active_scale + frames / 8)),
        ],
        "basis": (
            "scaled conservatively from the completed Lego factorized build; "
            "exact-W size and render time remain resolution/sparsity dependent"
        ),
    }


def audit(
    *,
    manifest_path: Path,
    queue_path: Path,
    factorized_root: Path,
    exact_w_root: Path,
    historical_root: Path,
    compact_primary_root: Path,
    compact_recovered_root: Path,
    output: Path,
    repo_root: Path,
) -> dict[str, Any]:
    manifest = json_object(manifest_path)
    queue = json_object(queue_path)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("manifest protocol must be an object")
    if tuple(protocol.get("cohort", ())) != SCENE_ORDER:
        raise ValueError("SPIn9 cohort/order differs from the frozen authority")
    queue_scenes = {
        str(entry["scene_id"]): entry for entry in queue.get("scenes", [])
    }
    manifest_scenes = {
        str(entry["scene_id"]): entry for entry in manifest.get("scenes", [])
    }
    if set(queue_scenes) != set(SCENE_ORDER) or set(manifest_scenes) != set(SCENE_ORDER):
        raise ValueError("manifest/queue scene coverage differs from SPIn9")

    scenes: list[dict[str, Any]] = []
    for scene in SCENE_ORDER:
        carrier, carrier_source = resolve_carrier(scene, queue_scenes[scene], repo_root)
        carrier_present = carrier.is_file()
        carrier_rows = ply_vertex_count(carrier) if carrier_present else None
        frames = len(manifest_scenes[scene].get("evaluation_frame_ids", []))
        train_views = len(manifest_scenes[scene].get("training_frames", []))

        current = factorized_root / scene
        field = current / "factorized_field_d256_l128.pth"
        field_sidecar = field.with_suffix(field.suffix + ".json")
        capability = current / "official_dino_sam3_views.pt"
        capability_sidecar = capability.with_suffix(capability.suffix + ".json")
        graph = current / "shared_support_graph_surface_safe_k16.pt"
        graph_sidecar = graph.with_suffix(graph.suffix + ".json")
        matched = current / "query_diffusion_matched_interface_v1"
        knn = matched / f"{scene}_euclidean_k200_self.pt"
        knn_sidecar = knn.with_suffix(knn.suffix + ".json")
        relation = matched / f"{scene}_dino_signed_hash256.pt"
        relation_sidecar = relation.with_suffix(relation.suffix + ".json")
        exact_w = exact_w_root / scene / "reference_exact_w.pt"
        exact_w_report = exact_w_root / scene / "reference_exact_w_report.json"

        field_rows, field_active = sidecar_rows(field_sidecar)
        capability_rows, capability_active = sidecar_rows(capability_sidecar)
        graph_rows, graph_active = sidecar_rows(graph_sidecar)
        knn_rows, knn_active = sidecar_rows(knn_sidecar)
        relation_rows, relation_active = sidecar_rows(relation_sidecar)
        row_candidates = [
            value
            for value in (
                field_rows,
                capability_rows,
                graph_rows,
                knn_rows,
                relation_rows,
            )
            if value is not None
        ]
        current_rows_match = bool(
            carrier_rows is not None
            and row_candidates
            and all(value == carrier_rows for value in row_candidates)
        )
        current_active_candidates = [
            value
            for value in (
                field_active,
                capability_active,
                graph_active,
                knn_active,
                relation_active,
            )
            if value is not None
        ]
        current_active_match = bool(
            current_active_candidates
            and len(set(current_active_candidates)) == 1
        )

        historical_knn = historical_root / "knn" / f"{scene}_euclidean_k200_self.pt"
        historical_knn_sidecar = historical_knn.with_suffix(
            historical_knn.suffix + ".json"
        )
        historical_relation = historical_root / "features" / f"{scene}_dino_signed_hash256.pt"
        historical_relation_sidecar = historical_relation.with_suffix(
            historical_relation.suffix + ".json"
        )
        historical_rows, historical_active = sidecar_rows(historical_knn_sidecar)

        compact_root = compact_primary_root / scene
        if not (compact_root / "official_dino_sam3_views.pt").is_file():
            compact_root = compact_recovered_root / scene
        compact_capability = compact_root / "official_dino_sam3_views.pt"
        compact_field = compact_root / "canonical_d256_l128_capability_first.pth"

        all_required_present = all(
            path.is_file()
            for path in (
                field,
                field_sidecar,
                capability,
                capability_sidecar,
                graph,
                graph_sidecar,
                knn,
                knn_sidecar,
                relation,
                relation_sidecar,
                exact_w,
                exact_w_report,
            )
        )
        source_authority_ready = bool(
            carrier_present
            and all_required_present
            and current_rows_match
            and current_active_match
        )
        blockers: list[str] = []
        if not carrier_present:
            blockers.append("frozen_carrier_missing")
        if not field.is_file():
            blockers.append("current_factorized_field_missing")
        if not capability.is_file():
            blockers.append("current_factorized_capability_missing")
        if not graph.is_file():
            blockers.append("current_surface_safe_graph_missing")
        if not knn.is_file() or not relation.is_file():
            blockers.append("current_matched_k201_interface_missing")
        if not exact_w.is_file() or not exact_w_report.is_file():
            blockers.append("exact_source_responsibility_authority_missing")
        if row_candidates and not current_rows_match:
            blockers.append("current_asset_carrier_row_mismatch")
        if current_active_candidates and not current_active_match:
            blockers.append("current_asset_active_row_mismatch")

        active_for_estimate = (
            current_active_candidates[0]
            if current_active_candidates and current_active_match
            else historical_active
        )
        scene_record: dict[str, Any] = {
            "scene": scene,
            "evaluation_frames": frames,
            "training_views": train_views,
            "carrier": {
                **asset(carrier),
                "source": carrier_source,
                "num_primitives": carrier_rows,
            },
            "current_factorized_v1": {
                "field": asset(field),
                "capability": asset(capability),
                "surface_safe_graph": asset(graph),
                "matched_k201_knn": asset(knn),
                "matched_k201_relation": asset(relation),
                "declared_global_rows": row_candidates,
                "declared_active_rows": current_active_candidates,
                "carrier_rows_match": current_rows_match,
                "active_rows_match": current_active_match,
            },
            "source_calibration_authority": {
                "exact_w": asset(exact_w),
                "exact_w_report": asset(exact_w_report, hash_small=True),
                "ready": source_authority_ready,
                "decision": "eligible" if source_authority_ready else "fail_closed",
                "blockers": blockers,
            },
            "historical_compact_context_only": {
                "capability": asset(compact_capability),
                "field": asset(compact_field),
                "k201_knn": asset(historical_knn),
                "k201_relation": asset(historical_relation),
                "declared_global_rows": historical_rows,
                "declared_active_rows": historical_active,
                "carrier_rows_match": historical_rows == carrier_rows,
                "permitted_use": "rebuild input and historical context only",
                "forbidden_use": "silent substitution into factorized-v1 cohort",
            },
        }
        if carrier_rows is not None and active_for_estimate is not None:
            scene_record["planning_estimate"] = estimate_resources(
                carrier_rows, active_for_estimate, frames
            )
        scenes.append(scene_record)

    eligible = [entry["scene"] for entry in scenes if entry["source_calibration_authority"]["ready"]]
    blocked = [entry["scene"] for entry in scenes if not entry["source_calibration_authority"]["ready"]]
    report = {
        "schema_version": 1,
        "kind": "spin9_factorized_source_quantile_full9_asset_inventory",
        "status": "fail_closed_pending_source_authority" if blocked else "ready",
        "audit_mode": "cpu_metadata_and_ply_headers_only",
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "protocol_hash": manifest.get("protocol_hash"),
        },
        "queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
        "representation_contract": {
            "required": "canonical_factorized_radio_v1 with matched surface-safe K201 interface",
            "historical_compact_mixing": "forbidden",
            "scene_specific_method_overrides": "forbidden",
        },
        "scenes": scenes,
        "summary": {
            "scenes": len(scenes),
            "eligible_source_authority": eligible,
            "fail_closed": blocked,
            "eligible_count": len(eligible),
            "fail_closed_count": len(blocked),
            "target_masks_opened_by_audit": False,
            "target_rgb_opened_by_audit": False,
            "gpu_used": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/mnt/pool/sqy/results/RADIO-GS/output/unified_query/manifests/"
        "spin_nerf_full_reference_mask_9scene_diagnostic_v1.json",
    )
    parser.add_argument(
        "--queue",
        default="/mnt/pool/sqy/results/RADIO-GS/output/unified_query/"
        "spin9_gaussfm_queue_20260712/queue_plan.json",
    )
    parser.add_argument(
        "--factorized-root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/"
        "canonical_factorized_radio_v1",
    )
    parser.add_argument(
        "--exact-w-root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/"
        "spin_source_exact_w_v1",
    )
    parser.add_argument(
        "--historical-root",
        default="/root/RADIO-GS/output/optimization_20260803/"
        "spin9_query_conditioned_diffusion_v1",
    )
    parser.add_argument(
        "--compact-primary-root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/evaluation_closeout_20260716/"
        "canonical_mpr_v3_spin9",
    )
    parser.add_argument(
        "--compact-recovered-root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/"
        "spin9_exact_local9/fields",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(
        manifest_path=Path(args.manifest).expanduser().resolve(),
        queue_path=Path(args.queue).expanduser().resolve(),
        factorized_root=Path(args.factorized_root).expanduser().resolve(),
        exact_w_root=Path(args.exact_w_root).expanduser().resolve(),
        historical_root=Path(args.historical_root).expanduser().resolve(),
        compact_primary_root=Path(args.compact_primary_root).expanduser().resolve(),
        compact_recovered_root=Path(args.compact_recovered_root).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        repo_root=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
