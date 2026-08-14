"""Bridge authority-bound UQIS observations into audited LUDVIG DINO phases."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import (
    LUDVIG_AUDITED_COMMIT,
    OFFICIAL_DINO_CHECKPOINT_SHA256,
    OFFICIAL_DINO_CHECKPOINT_SIZE,
    UPSTREAM_AUDIT_FILES,
    audit_checkpoint,
    audit_ludvig_upstream,
    _orthonormalized_c2w,
    stage_colmap_text,
)

from .ludvig_mapping_stage import STAGE_SCHEMA
from .protocol import BENCHMARK_VERSION, canonical_json_sha256, sha256_file


DINO_BRIDGE_SCHEMA = "scannet_uqis_ludvig_dino_phase_a_bridge_v1"
DINO_WIDTH = 640
DINO_HEIGHT = 480


def _gaussian_count(path: Path) -> int:
    with path.open("rb") as handle:
        for _ in range(512):
            line = handle.readline().decode("ascii", errors="strict").strip()
            match = re.fullmatch(r"element vertex ([0-9]+)", line)
            if match:
                return int(match.group(1))
            if line == "end_header":
                break
    raise ValueError(f"invalid Gaussian PLY: {path}")


def _relative_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: path escapes observation root")
    path = root / relative
    if (
        not path.is_file()
        or path.stat().st_size != int(binding.get("bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise ValueError(f"{label}: file binding changed")
    return path


def stage_uqis_ludvig_dino_bridge(
    observation_receipt_path: str | Path,
    *,
    expected_observation_receipt_sha256: str,
    construction_authority_path: str | Path,
    expected_construction_authority_sha256: str,
    geometry_run_receipt_path: str | Path,
    expected_geometry_run_receipt_sha256: str,
    dino_checkpoint: str | Path,
    ludvig_upstream: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize the exact 640x480 CPU seam consumed by legacy Phase B/C."""

    observation_path = Path(observation_receipt_path).resolve()
    observation_root = observation_path.parent
    observations = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_body = {
        key: value for key, value in observations.items() if key != "receipt_sha256"
    }
    if (
        observations.get("schema_version") != STAGE_SCHEMA
        or observations.get("receipt_sha256") != expected_observation_receipt_sha256
        or observations.get("receipt_sha256") != canonical_json_sha256(observation_body)
    ):
        raise ValueError("mapping-observation receipt identity/digest changed")
    authority_path = Path(construction_authority_path).resolve()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_body = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    if (
        authority.get("authority_sha256") != expected_construction_authority_sha256
        or authority.get("authority_sha256") != canonical_json_sha256(authority_body)
    ):
        raise ValueError("construction authority identity/digest changed")
    geometry_receipt_path = Path(geometry_run_receipt_path).resolve()
    geometry = json.loads(geometry_receipt_path.read_text(encoding="utf-8"))
    geometry_body = {
        key: value for key, value in geometry.items() if key != "receipt_sha256"
    }
    if (
        geometry.get("receipt_sha256") != expected_geometry_run_receipt_sha256
        or geometry.get("receipt_sha256") != canonical_json_sha256(geometry_body)
        or geometry.get("status") != "geometry_complete"
        or geometry.get("formal_field_eligible") is not True
        or geometry.get("iterations") != 30000
    ):
        raise ValueError("formal Gaussian geometry receipt changed or is ineligible")
    scene_id = str(geometry["scene_id"])
    scene_rows = [row for row in observations["scenes"] if row["scene_id"] == scene_id]
    if len(scene_rows) != 1:
        raise ValueError(f"observations contain no unique scene {scene_id}")
    scene = scene_rows[0]
    if (
        scene["construction_scene_receipt_sha256"]
        != authority["scene_derivation_receipt_sha256"].get(scene_id)
    ):
        raise ValueError("scene derivation authority changed")
    geometry_ply = Path(geometry["point_cloud"]["path"]).resolve()
    if (
        not geometry_ply.is_file()
        or geometry_ply.stat().st_size != geometry["point_cloud"]["bytes"]
        or sha256_file(geometry_ply) != geometry["point_cloud"]["sha256"]
    ):
        raise ValueError("formal Gaussian geometry artifact changed")

    checkpoint = audit_checkpoint(
        Path(dino_checkpoint).resolve(),
        expected_size=OFFICIAL_DINO_CHECKPOINT_SIZE,
        expected_sha256=OFFICIAL_DINO_CHECKPOINT_SHA256,
    )
    upstream = audit_ludvig_upstream(
        Path(ludvig_upstream).resolve(),
        expected_commit=LUDVIG_AUDITED_COMMIT,
        audited_files=UPSTREAM_AUDIT_FILES,
    )
    matrices = {
        name: np.load(
            _relative_file(observation_root, binding, f"{scene_id}/{name}"),
            allow_pickle=False,
        )
        for name, binding in scene["camera_matrices"].items()
    }
    source_k = np.asarray(matrices["intrinsic_color"], dtype=np.float64)
    first_color = _relative_file(
        observation_root, scene["frames"][0]["color"], f"{scene_id}/first_color"
    )
    with Image.open(first_color) as image:
        source_width, source_height = image.size
    scale_x = DINO_WIDTH / source_width
    scale_y = DINO_HEIGHT / source_height
    resized_k = source_k.copy()
    resized_k[0] *= scale_x
    resized_k[1] *= scale_y
    intrinsics = {
        "selected_role": "UQIS_authorized_color_intrinsics_scaled_for_LUDVIG_DINO",
        "image_dimensions": [DINO_WIDTH, DINO_HEIGHT],
        "fx": float(resized_k[0, 0]),
        "fy": float(resized_k[1, 1]),
        "cx": float(resized_k[0, 2]),
        "cy": float(resized_k[1, 2]),
        "matrix": resized_k.tolist(),
        "source_dimensions": [source_width, source_height],
        "scale_xy": [scale_x, scale_y],
    }
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent))
    try:
        resized_root = temporary / "staging" / "resized_rgb"
        resized_root.mkdir(parents=True)
        frame_ids: list[int] = []
        colors: dict[int, Path] = {}
        poses: dict[int, np.ndarray] = {}
        ordered_inventory = []
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        for frame in scene["frames"]:
            frame_id = int(frame["frame_id"])
            color = _relative_file(
                observation_root, frame["color"], f"{scene_id}/{frame_id}/color"
            )
            pose_path = _relative_file(
                observation_root,
                frame["camera_to_world"],
                f"{scene_id}/{frame_id}/pose",
            )
            destination = resized_root / f"{frame_id:06d}.png"
            with Image.open(color) as image:
                image.convert("RGB").resize(
                    (DINO_WIDTH, DINO_HEIGHT), resampling
                ).save(destination, format="PNG", compress_level=1)
            pose = np.load(pose_path, allow_pickle=False).astype(np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"{scene_id}/{frame_id}: pose is invalid")
            pose, _correction = _orthonormalized_c2w(pose, frame_id)
            binding = {
                "path": str(output / "staging" / "resized_rgb" / destination.name),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
            frame_ids.append(frame_id)
            # The COLMAP staging helper creates absolute symlinks. Point them
            # at the final immutable root that will exist after atomic rename,
            # rather than at the temporary directory name.
            colors[frame_id] = output / "staging" / "resized_rgb" / destination.name
            poses[frame_id] = pose
            ordered_inventory.append({"frame_id": frame_id, "color": binding})
        colmap = stage_colmap_text(
            temporary / "staging" / "colmap", frame_ids, poses, colors, intrinsics
        )
        ledger_body = {
            "schema_version": DINO_BRIDGE_SCHEMA,
            "benchmark_version": BENCHMARK_VERSION,
            "scene_id": scene_id,
            "construction_authority_sha256": expected_construction_authority_sha256,
            "mapping_observation_receipt_sha256": expected_observation_receipt_sha256,
            "geometry_run_receipt_sha256": expected_geometry_run_receipt_sha256,
            "ordered_frame_ids": frame_ids,
            "ordered_frame_ids_sha256": canonical_json_sha256(frame_ids),
            "resize_contract": {
                "source_dimensions": [source_width, source_height],
                "output_dimensions": [DINO_WIDTH, DINO_HEIGHT],
                "implementation": "PIL_RGB_bilinear_lossless_PNG_compress_level_1",
                "intrinsics_rule": "independent_x_y_row_scaling",
            },
            "query_frames_opened": False,
            "evaluator_labels_opened": False,
        }
        ledger = {**ledger_body, "ledger_sha256": canonical_json_sha256(ledger_body)}
        ledger_path = temporary / "source_adapter_ledger.json"
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        ledger_file_sha = sha256_file(ledger_path)
        gaussian_count = _gaussian_count(geometry_ply)
        manifest = {
            "schema_version": "ludvig_pfpr_phase_a_v1",
            "status": "phase_a_complete_phase_b_available_not_run",
            "result_eligible": False,
            "formal_benchmark_row_eligible": False,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "benchmark_local_adapter": True,
            "scene_id": scene_id,
            "attempt_dir": str(output),
            "gpu_work_started": False,
            "torch_imported_by_phase_a": False,
            "evaluator_private_manifest_opened": False,
            "checkpoint": checkpoint,
            "ludvig_upstream": upstream,
            "construction_authority": {
                "path": str(authority_path),
                "sha256": expected_construction_authority_sha256,
            },
            "mapping_observation_receipt": {
                "path": str(observation_path),
                "sha256": expected_observation_receipt_sha256,
            },
            "geometry_run_receipt": {
                "path": str(geometry_receipt_path),
                "sha256": expected_geometry_run_receipt_sha256,
            },
            "source_adapter_ledger": {
                "path": str(output / "source_adapter_ledger.json"),
                "sha256": ledger_file_sha,
                "coverage_prefix_sha256": canonical_json_sha256(frame_ids),
            },
            "geometry": {
                "path": str(geometry_ply),
                "bytes": geometry_ply.stat().st_size,
                "sha256": geometry["point_cloud"]["sha256"],
                "gaussians": gaussian_count,
            },
            "view_selection": {
                "policy": "UQIS_frozen_even_spacing_over_legal_field_frames",
                "count": len(frame_ids),
                "ordered_frame_ids": frame_ids,
                "ordered_frame_ids_sha256": canonical_json_sha256(frame_ids),
                "query_source_frames_excluded_by_authority": True,
            },
            "source_inventory": {
                "ordered_inventory": ordered_inventory,
                "ordered_inventory_sha256": canonical_json_sha256(ordered_inventory),
            },
            "camera_intrinsics": intrinsics,
            "colmap_staging": colmap,
            "bridge_contract": ledger_body["resize_contract"],
            "phase_status": {
                "phase_a_cpu_staging": "complete",
                "phase_b_dino_scene_features_and_pca": "available_separate_not_run",
                "phase_c_inverse_render_uplift": "not_run",
            },
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        manifest_sha = sha256_file(manifest_path)
        (temporary / "run_manifest.sha256").write_text(manifest_sha + "\n")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": manifest_sha}
