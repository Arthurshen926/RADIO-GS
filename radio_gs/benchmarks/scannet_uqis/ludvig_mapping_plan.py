"""Freeze per-scene CLIP and DINO mapping jobs for the LUDVIG comparator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .construction_authority import audit_construction_authority
from .method_fields import ludvig_modality_field_plan
from .protocol import BENCHMARK_VERSION, canonical_json_sha256, sha256_file


LUDVIG_MAPPING_PLAN_SCHEMA = "scannet_uqis_ludvig_mapping_plan_v1"
FROZEN_MAPPING_VIEW_COUNT = 120


def select_mapping_frame_ids(
    legal_field_frame_ids: Sequence[str], *, maximum_views: int = FROZEN_MAPPING_VIEW_COUNT
) -> tuple[str, ...]:
    """Select an endpoint-preserving, evenly spaced subset without randomness."""

    frames = tuple(map(str, legal_field_frame_ids))
    if not frames or len(set(frames)) != len(frames) or list(frames) != sorted(frames):
        raise ValueError("legal field-frame inventory must be non-empty, unique, and sorted")
    if maximum_views <= 0:
        raise ValueError("maximum_views must be positive")
    if len(frames) <= maximum_views:
        return frames
    indices = [
        (index * (len(frames) - 1)) // (maximum_views - 1)
        for index in range(maximum_views)
    ]
    if len(set(indices)) != maximum_views:
        raise RuntimeError("mapping frame sampler produced duplicate indices")
    return tuple(frames[index] for index in indices)


def _binding(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _source_tree_binding(root: Path) -> dict[str, Any]:
    included_roots = (
        root / "predictors",
        root / "clip_utils",
        root / "dinov2",
        root / "diffusion",
    )
    rows = []
    for directory in included_roots:
        for path in sorted(directory.rglob("*.py")):
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise ValueError("LUDVIG source tree contains no Python sources")
    return {
        "root": str(root),
        "python_file_count": len(rows),
        "python_source_tree_sha256": canonical_json_sha256(rows),
    }


def build_ludvig_mapping_plan(
    authority_path: str | Path,
    *,
    ludvig_upstream: str | Path,
    dino_checkpoint: str | Path,
    openclip_checkpoint: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Emit 18 authority-bound mapping jobs (two fields for each scene)."""

    authority_file = Path(authority_path).resolve()
    audit = audit_construction_authority(authority_file, check_files=True)
    if not audit["valid"]:
        raise ValueError("construction authority audit failed: " + "; ".join(audit["errors"]))
    authority = json.loads(authority_file.read_text(encoding="utf-8"))
    scene_records_binding = authority["construction_inputs"]["scene_records.json"]
    scene_records_path = Path(scene_records_binding["path"])
    if sha256_file(scene_records_path) != scene_records_binding["sha256"]:
        raise ValueError("authority-bound scene records changed")
    scene_records = json.loads(scene_records_path.read_text(encoding="utf-8"))["scenes"]

    upstream = Path(ludvig_upstream).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("unable to identify LUDVIG upstream commit") from error
    models = {
        "dinov2_vitg14_reg4": _binding(dino_checkpoint),
        "openclip_vit_b_16_laion2b_s34b_b88k": _binding(openclip_checkpoint),
    }
    implementation = {
        "upstream_git_commit": commit,
        "source_tree": _source_tree_binding(upstream),
        "benchmark_local_adapter": True,
        "official_ludvig_reproduction": False,
        "paper_metric_comparable": False,
    }
    field_plan = list(ludvig_modality_field_plan())
    identity_body = {
        "benchmark_version": BENCHMARK_VERSION,
        "construction_authority_sha256": authority["authority_sha256"],
        "mapping_view_selection": {
            "rule": "endpoint_preserving_integer_even_spacing",
            "maximum_views": FROZEN_MAPPING_VIEW_COUNT,
        },
        "field_plan": field_plan,
        "models": models,
        "implementation": implementation,
    }
    method_identity_sha256 = canonical_json_sha256(identity_body)
    receipt_hashes = authority["scene_derivation_receipt_sha256"]
    source_by_scene = authority["verified_scene_sources"]
    jobs = []
    for scene in scene_records:
        scene_id = scene["scene_id"]
        chosen = select_mapping_frame_ids(scene["field_frame_ids"])
        common = {
            "scene_id": scene_id,
            "construction_scene_receipt_sha256": receipt_hashes[scene_id],
            "sens": source_by_scene[scene_id]["sens"],
            "legal_field_frame_count": len(scene["field_frame_ids"]),
            "legal_field_frame_ids_sha256": canonical_json_sha256(scene["field_frame_ids"]),
            "selected_mapping_frame_count": len(chosen),
            "selected_mapping_frame_ids": list(chosen),
            "selected_mapping_frame_ids_sha256": canonical_json_sha256(list(chosen)),
            "withheld_frame_ids_sha256": canonical_json_sha256(scene["withheld_frame_ids"]),
        }
        for field in field_plan:
            job_body = {
                **common,
                "field_id": field["field_id"],
                "field_family": field["field_family"],
                "modalities": field["modalities"],
                "method_identity_sha256": method_identity_sha256,
                "status": "planned_not_run",
            }
            jobs.append(
                {
                    **job_body,
                    "job_sha256": canonical_json_sha256(job_body),
                }
            )
    body = {
        "schema_version": LUDVIG_MAPPING_PLAN_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "mapping_planned_not_run",
        "construction_authority": {
            "path": str(authority_file),
            "sha256": authority["authority_sha256"],
            "fresh_audit": audit,
        },
        "method_system_id": "ludvig_uqis9_clip_dino_system_v1",
        "method_identity_sha256": method_identity_sha256,
        "representation_scope": "modality_specific_multi_field",
        "mapping_view_selection": identity_body["mapping_view_selection"],
        "models": models,
        "implementation": implementation,
        "scene_count": len(scene_records),
        "field_count": len(jobs),
        "jobs": jobs,
    }
    plan = {**body, "plan_sha256": canonical_json_sha256(body)}
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan

