#!/usr/bin/env python3
"""Seal a carrier-native NVOS all-view candidate plan before SAM inference.

The frozen signed field prompt supplies calibrated positive/negative object
identity, while the official scribbles clamp its observed source evidence.
Both are lifted through the exact accepted-hit 3DGS compositor and reprojected
into the complete registered source/target view cohort.  The native target-view
prompt is kept before any lossy W.T/W round trip.  Ten exchangeable candidate
records bind ten deterministic signed-point trials in the downstream official
SAM3 executor.  No candidate/view is selected and no target mask or metric is
opened.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.data.lerf_dataset import _parse_colmap_sparse
from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.build_nvos_two_round_exact_consensus import (
    exact_adjoint_probability,
    exact_forward_probability,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_EVALUATION_CONTRACT,
    DEFAULT_METHOD_AUTHORITY,
    SAM_HEIGHT,
    SAM_WIDTH,
    load_signed_field_prompt,
)
from radio_gs.scripts.render_promptable_nvs_features import (
    resolve_protocol_views,
    validate_locked_camera_mapping,
)
from radio_gs.scripts.run_nvos_method_v1_scene import (
    DATASET_MANIFEST,
    NVOS_AUTHORITY,
    resolve_scene_assets,
)
from radio_gs.utils.immutable_artifacts import sha256_file


PLAN_TYPE = "nvos_synchronous_multiview_candidate_plan_v1"
COMPOSITOR_MODE = (
    "gsplat_exact_accepted_hits_front_to_back_alpha_times_exclusive_transmittance"
)
DEFAULT_FIELD_ROOT = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/"
    "core_method_v1/nvos"
)


def expand_to_all_registered_views(
    protocol_views: Sequence[Mapping[str, Any]],
    mapping_records: Sequence[Mapping[str, Any]],
    colmap_by_stem: Mapping[str, tuple[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Expand prompt/evaluation views to every queue-locked RGB camera.

    The locked camera map, rather than directory enumeration, owns the view
    cohort.  Protocol roles are preserved exactly; every other registered RGB
    is a mapping-time observation and cannot become a new prompt/evaluation
    view by ordering or filename convention.
    """

    protocol_by_camera: dict[str, dict[str, Any]] = {}
    for raw in protocol_views:
        view = dict(raw)
        camera = str(view.get("camera_name", ""))
        if not camera or camera in protocol_by_camera:
            raise ValueError("protocol camera identity is empty or repeated")
        protocol_by_camera[camera] = view
    output: list[dict[str, Any]] = []
    used_frames: set[str] = set()
    used_colmap: set[str] = set()
    for raw in mapping_records:
        record = dict(raw)
        camera = str(record.get("rgb_camera_name", ""))
        colmap_camera = str(record.get("colmap_camera_name", ""))
        rgb_path = Path(str(record.get("rgb_path", ""))).expanduser().resolve()
        if (
            not camera
            or not colmap_camera
            or colmap_camera not in colmap_by_stem
            or not rgb_path.is_file()
            or rgb_path.is_symlink()
            or colmap_camera in used_colmap
        ):
            raise ValueError("registered RGB/camera authority differs")
        colmap_path, c2w = colmap_by_stem[colmap_camera]
        locked_colmap_path = Path(str(record.get("colmap_file_path", "")))
        if Path(colmap_path) != locked_colmap_path:
            raise ValueError("locked registered COLMAP path changed")
        protocol = protocol_by_camera.get(camera)
        frame_id = str(protocol["frame_id"]) if protocol is not None else camera
        if frame_id in used_frames:
            raise ValueError("registered frame identity is repeated")
        used_frames.add(frame_id)
        used_colmap.add(colmap_camera)
        output.append(
            {
                "frame_id": frame_id,
                "camera_name": camera,
                "colmap_camera_name": colmap_camera,
                "camera_match_rule": str(record.get("match_rule", "")),
                "role": (
                    str(protocol["role"])
                    if protocol is not None
                    else "registered_mapping"
                ),
                "colmap_file_path": str(colmap_path),
                "rgb_path": str(rgb_path),
                "w2c": np.linalg.inv(np.asarray(c2w, dtype=np.float32)).astype(
                    np.float32
                ),
            }
        )
    if set(protocol_by_camera) - {str(row["camera_name"]) for row in output}:
        raise ValueError("protocol camera is absent from complete registered cohort")
    return output


def _load_assignment(
    record: Mapping[str, Any],
    *,
    num_gaussians: int,
    geometry_sha256: str,
) -> dict[str, torch.Tensor]:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError("exact assignment SHA-256 differs")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("exact assignment is not one mapping")
    if (
        int(value.get("num_gaussians", -1)) != int(num_gaussians)
        or str(value.get("geometry_xyz_sha256", "")) != str(geometry_sha256)
        or value.get("compositor_mode") != COMPOSITOR_MODE
    ):
        raise ValueError("exact assignment carrier identity differs")
    tensors = {
        key: torch.as_tensor(value.get(key)).cpu()
        for key in ("gaussian_ids", "pixel_ids", "weights")
    }
    if (
        tensors["gaussian_ids"].ndim != 1
        or tensors["pixel_ids"].shape != tensors["gaussian_ids"].shape
        or tensors["weights"].shape != tensors["gaussian_ids"].shape
    ):
        raise ValueError("exact assignment triplet axes differ")
    return tensors


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _resize_probability(value: np.ndarray, shape: tuple[int, int]) -> torch.Tensor:
    source = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
    return F.interpolate(
        source, size=shape, mode="bilinear", align_corners=False
    )[0, 0]


def _resize_mask(value: np.ndarray, shape: tuple[int, int]) -> torch.Tensor:
    source = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
    return F.interpolate(source, size=shape, mode="nearest")[0, 0] > 0.5


def exclusive_projected_authority(
    positive: torch.Tensor,
    negative: torch.Tensor,
    visibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn projected signed mass into disjoint explicit point authority.

    Common support is unknown.  A negative point can therefore only arise from
    transported negative scribble mass that strictly dominates transported
    positive mass; low object posterior never creates negative evidence.
    """

    pos = torch.as_tensor(positive).float()
    neg = torch.as_tensor(negative).float()
    visible = torch.as_tensor(visibility).bool()
    if pos.shape != neg.shape or pos.shape != visible.shape or pos.ndim != 2:
        raise ValueError("projected signed authority axes differ")
    if not bool(torch.isfinite(pos).all()) or not bool(torch.isfinite(neg).all()):
        raise ValueError("projected signed authority is nonfinite")
    if bool((pos < 0).any()) or bool((neg < 0).any()):
        raise ValueError("projected signed authority must be nonnegative")
    positive_only = visible & (pos > neg) & (pos > 0)
    negative_only = visible & (neg > pos) & (neg > 0)
    if bool((positive_only & negative_only).any()):
        raise RuntimeError("exclusive signed authority overlaps")
    return positive_only, negative_only


def exchangeable_candidates(
    *,
    scene_id: str,
    plan_identity: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Create K equal-likelihood trial identities over one sealed view cohort."""

    if int(count) <= 0 or not views:
        raise ValueError("candidate count and view cohort must be nonempty")
    output: list[dict[str, Any]] = []
    for rank in range(int(count)):
        digest = _canonical_digest(
            {
                "schema": PLAN_TYPE,
                "scene_id": str(scene_id),
                "plan_identity": dict(plan_identity),
                "trial_rank": rank,
            }
        )
        output.append(
            {
                "candidate_digest": digest,
                "candidate_logit": 0.0,
                "trial_rank": rank,
                # Bind the systematic point-trial rank into every Cartesian
                # candidate/view cell.  The downstream executor therefore
                # cannot silently recover the old digest-hashed point sampler
                # or make the trial depend on candidate traversal order.
                "views": [
                    {**dict(view), "candidate_trial_rank": rank}
                    for view in views
                ],
            }
        )
    if len({row["candidate_digest"] for row in output}) != int(count):
        raise RuntimeError("exchangeable candidate identities collided")
    return output


def _manifest_scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    values = [
        row
        for row in manifest.get("scenes", [])
        if isinstance(row, Mapping) and str(row.get("scene_id")) == scene_id
    ]
    if len(values) != 1:
        raise ValueError("NVOS dataset scene authority differs")
    return values[0]


def _frame_rgb(scene: Mapping[str, Any], frame_id: str) -> Path:
    frames = scene.get("frames", [])
    values = frames.values() if isinstance(frames, Mapping) else frames
    matches = [
        row
        for row in values
        if isinstance(row, Mapping) and str(row.get("frame_id")) == str(frame_id)
    ]
    if len(matches) != 1:
        raise ValueError("NVOS registered RGB authority differs")
    return Path(str(matches[0].get("rgb_path", ""))).expanduser().resolve(strict=True)


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, Any]:
    if (SAM_HEIGHT, SAM_WIDTH) != (756, 1008):
        raise RuntimeError("frozen Method-v1 SAM raster authority changed")
    scene_id = str(args.scene_id)
    output = Path(args.output_dir).expanduser().resolve()
    plan_path = output / "candidate_plan.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)

    source = load_signed_field_prompt(
        dataset_manifest_path=args.manifest,
        prompt_manifest_path=args.signed_field_prompt_manifest,
        method_authority_path=args.method_authority,
        evaluation_contract_path=args.evaluation_contract,
        scene_id=scene_id,
    )
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_scene = _manifest_scene(manifest, scene_id)
    prompt = raw_scene.get("prompt", {})
    if not isinstance(prompt, Mapping) or prompt.get("type") != "positive_negative_scribbles":
        raise ValueError("NVOS native plan requires official signed scribbles")
    positive_path = Path(str(prompt.get("positive_path", ""))).resolve(strict=True)
    negative_path = Path(str(prompt.get("negative_path", ""))).resolve(strict=True)
    positive_native = load_ground_truth_mask(positive_path)
    negative_native = load_ground_truth_mask(negative_path)
    if positive_native.shape != negative_native.shape or bool(
        (positive_native & negative_native).any()
    ):
        raise ValueError("official NVOS signed scribbles are malformed")

    assets = resolve_scene_assets(scene_id)
    field_root = Path(args.field_root).expanduser().resolve(strict=True)
    config_path = field_root / scene_id / "method_v1.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    exact_authority = json.loads(Path(NVOS_AUTHORITY).read_text(encoding="utf-8"))
    rows = {
        str(row["scene_id"]): row for row in exact_authority.get("scenes", [])
    }
    camera_record = rows[scene_id]["camera_map"]
    camera_path = Path(str(camera_record["path"])).resolve(strict=True)
    if sha256_file(camera_path) != str(camera_record["sha256"]):
        raise ValueError("NVOS camera mapping SHA-256 differs")
    camera_mapping = json.loads(camera_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))
    views = resolve_protocol_views(
        manifest,
        scene_id=scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    if len(views) != 2 or {str(view["role"]) for view in views} != {
        "prompt",
        "evaluation",
    }:
        raise ValueError("NVOS protocol cohort must contain source and target")
    if bool(args.all_registered_views):
        colmap = _parse_colmap_sparse(Path(str(config.scene_root)).resolve())
        colmap_by_stem: dict[str, tuple[str, np.ndarray]] = {}
        for file_path, c2w in zip(colmap["file_paths"], colmap["c2w_list"]):
            stem = Path(str(file_path)).stem
            if stem in colmap_by_stem:
                raise ValueError("registered COLMAP camera identity is repeated")
            colmap_by_stem[stem] = (
                str(file_path),
                np.asarray(c2w, dtype=np.float32),
            )
        views = expand_to_all_registered_views(
            views,
            validate_locked_camera_mapping(camera_mapping, scene_id=scene_id),
            colmap_by_stem,
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact NVOS native plan requires CUDA")
    model, _codec, renderer, _sharpener, refiner, _cfg, _hybrid = load_render_pipeline(
        str(config_path),
        str(assets.geometry),
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    if refiner is not None:
        raise ValueError("native candidate plan forbids an RGB screen refiner")
    num_gaussians = int(model.get_xyz().shape[0])
    geometry_sha = _float32_rows_sha256(model.get_xyz())

    assignment_records: dict[str, dict[str, str]] = {}
    static_records: dict[str, dict[str, Any]] = {}
    for view in views:
        frame_id = str(view["frame_id"])
        pose = torch.from_numpy(np.asarray(view["w2c"], dtype=np.float32)).to(device)
        hits = rasterize_single_view_contributions(
            model, renderer, pose, height=SAM_HEIGHT, width=SAM_WIDTH
        )
        keep = hits["weights"] > 0
        assignment = {
            "gaussian_ids": hits["gaussian_ids"][keep].long().cpu(),
            "pixel_ids": hits["pixel_ids"][keep].long().cpu(),
            "weights": hits["weights"][keep].float().cpu(),
        }
        if not assignment["gaussian_ids"].numel():
            raise ValueError(f"{scene_id}/{frame_id} exact assignment is empty")
        assignment_path = output / "assignments" / f"{frame_id}.pt"
        assignment_sha = _atomic_torch(
            assignment_path,
            {
                **assignment,
                "height": SAM_HEIGHT,
                "width": SAM_WIDTH,
                "num_gaussians": num_gaussians,
                "geometry_xyz_sha256": geometry_sha,
                "compositor_mode": COMPOSITOR_MODE,
            },
        )
        pixels = SAM_HEIGHT * SAM_WIDTH
        pixel_mass = torch.zeros(pixels, dtype=torch.float32)
        pixel_mass.index_add_(
            0, assignment["pixel_ids"], assignment["weights"]
        )
        rgb_path = (
            Path(str(view["rgb_path"])).expanduser().resolve(strict=True)
            if view.get("rgb_path")
            else _frame_rgb(raw_scene, frame_id)
        )
        rgb_sha = sha256_file(rgb_path)
        pose_sha = hashlib.sha256(
            np.asarray(view["w2c"], dtype="<f4").tobytes()
        ).hexdigest()
        view_digest = _canonical_digest(
            {
                "scene_id": scene_id,
                "frame_id": frame_id,
                "pose_sha256": pose_sha,
                "rgb_sha256": rgb_sha,
                "assignment_sha256": assignment_sha,
            }
        )
        precision = math.log(max(float(pixel_mass.mean()), 1e-8))
        assignment_records[frame_id] = {
            "path": str(assignment_path),
            "sha256": assignment_sha,
        }
        static_records[frame_id] = {
            "frame_id": frame_id,
            "role": str(view["role"]),
            "view_digest": view_digest,
            "rgb": {"path": str(rgb_path), "sha256": rgb_sha},
            "assignment": dict(assignment_records[frame_id]),
            "log_precision": precision,
            "query_independent_precision": "log_mean_exact_pixel_visible_mass",
        }
        del assignment, hits, keep, pixel_mass

    prompt_view = next(view for view in views if view["role"] == "prompt")
    target_view = next(view for view in views if view["role"] == "evaluation")
    prompt_assignment = _load_assignment(
        assignment_records[str(prompt_view["frame_id"])],
        num_gaussians=num_gaussians,
        geometry_sha256=geometry_sha,
    )
    target_assignment = _load_assignment(
        assignment_records[str(target_view["frame_id"])],
        num_gaussians=num_gaussians,
        geometry_sha256=geometry_sha,
    )
    target_margin = _resize_probability(
        np.asarray(source["signed_margin"], dtype=np.float32),
        (SAM_HEIGHT, SAM_WIDTH),
    )
    target_probability = torch.sigmoid(target_margin).reshape(-1)
    primitive_probability, _target_visible_mass = exact_adjoint_probability(
        target_assignment["gaussian_ids"],
        target_assignment["pixel_ids"],
        target_assignment["weights"],
        target_probability,
        num_gaussians=num_gaussians,
    )
    positive_source = _resize_mask(
        positive_native, (SAM_HEIGHT, SAM_WIDTH)
    ).float().reshape(-1)
    negative_source = _resize_mask(
        negative_native, (SAM_HEIGHT, SAM_WIDTH)
    ).float().reshape(-1)
    primitive_positive, _ = exact_adjoint_probability(
        prompt_assignment["gaussian_ids"],
        prompt_assignment["pixel_ids"],
        prompt_assignment["weights"],
        positive_source,
        num_gaussians=num_gaussians,
    )
    primitive_negative, _ = exact_adjoint_probability(
        prompt_assignment["gaussian_ids"],
        prompt_assignment["pixel_ids"],
        prompt_assignment["weights"],
        negative_source,
        num_gaussians=num_gaussians,
    )
    # exact_adjoint_probability uses 0.5 for invisible rows.  Scribble
    # authority is mass, not a Bernoulli posterior, so remove that neutral fill.
    prompt_visible = torch.zeros(num_gaussians, dtype=torch.float32)
    prompt_visible.index_add_(
        0, prompt_assignment["gaussian_ids"], prompt_assignment["weights"]
    )
    primitive_positive[prompt_visible <= 0] = 0
    primitive_negative[prompt_visible <= 0] = 0
    primitive_probability[primitive_positive > primitive_negative] = torch.maximum(
        primitive_probability[primitive_positive > primitive_negative],
        torch.tensor(0.95),
    )
    primitive_probability[primitive_negative > primitive_positive] = torch.minimum(
        primitive_probability[primitive_negative > primitive_positive],
        torch.tensor(0.05),
    )
    del prompt_assignment, target_assignment, prompt_visible

    view_records: list[dict[str, Any]] = []
    skipped_registered_views: list[dict[str, Any]] = []
    zeros = torch.zeros(SAM_HEIGHT * SAM_WIDTH, dtype=torch.float32)
    for view in views:
        frame_id = str(view["frame_id"])
        assignment = _load_assignment(
            assignment_records[frame_id],
            num_gaussians=num_gaussians,
            geometry_sha256=geometry_sha,
        )
        if frame_id == str(target_view["frame_id"]):
            # The sealed signed field prompt was rendered in this registered
            # view.  Preserve that observation exactly instead of applying a
            # lossy W.T/W round trip before the official SAM call.
            projected = target_probability.reshape(SAM_HEIGHT, SAM_WIDTH)
            pixel_mass = torch.zeros(SAM_HEIGHT * SAM_WIDTH, dtype=torch.float32)
            pixel_mass.index_add_(
                0, assignment["pixel_ids"], assignment["weights"]
            )
            pixel_mass = pixel_mass.reshape(SAM_HEIGHT, SAM_WIDTH)
            projection_semantics = "sealed_native_signed_field_prompt"
        else:
            projected, pixel_mass = exact_forward_probability(
                assignment["gaussian_ids"],
                assignment["pixel_ids"],
                assignment["weights"],
                primitive_probability,
                height=SAM_HEIGHT,
                width=SAM_WIDTH,
                unsupported_fallback=zeros,
            )
            projection_semantics = "exact_adjoint_forward_transport"
        projected_positive, _ = exact_forward_probability(
            assignment["gaussian_ids"],
            assignment["pixel_ids"],
            assignment["weights"],
            primitive_positive,
            height=SAM_HEIGHT,
            width=SAM_WIDTH,
            unsupported_fallback=zeros,
        )
        projected_negative, _ = exact_forward_probability(
            assignment["gaussian_ids"],
            assignment["pixel_ids"],
            assignment["weights"],
            primitive_negative,
            height=SAM_HEIGHT,
            width=SAM_WIDTH,
            unsupported_fallback=zeros,
        )
        visibility = pixel_mass > 0
        positive_authority, negative_authority = exclusive_projected_authority(
            projected_positive, projected_negative, visibility
        )
        if min(int(positive_authority.sum()), int(negative_authority.sum())) < int(
            args.points_per_sign
        ):
            if str(view["role"]) != "registered_mapping":
                raise ValueError(
                    f"{scene_id}/{frame_id} lacks projected explicit signed support"
                )
            skipped_registered_views.append(
                {
                    "frame_id": frame_id,
                    "reason": "insufficient_projected_explicit_signed_support",
                    "positive_pixels": int(positive_authority.sum()),
                    "negative_pixels": int(negative_authority.sum()),
                }
            )
            del assignment, projected, pixel_mass
            del projected_positive, projected_negative, visibility
            del positive_authority, negative_authority
            continue
        records: dict[str, dict[str, str]] = {}
        for label, value in (
            ("projected_probability", projected.numpy().astype(np.float32)),
            ("visibility", visibility.numpy()),
            ("positive_authority", positive_authority.numpy()),
            ("negative_authority", negative_authority.numpy()),
        ):
            path = output / "projected" / frame_id / f"{label}.npy"
            records[label] = {"path": str(path), "sha256": _atomic_numpy(path, value)}
        view_records.append(
            {
                **static_records[frame_id],
                **records,
                "projected_probability_semantics": projection_semantics,
            }
        )
        del assignment, projected, pixel_mass
        del projected_positive, projected_negative, visibility
        del positive_authority, negative_authority

    plan_identity = {
        "scene_id": scene_id,
        "geometry_xyz_sha256": geometry_sha,
        "signed_field_prompt_sha256": str(source["signed_margin_sha256"]),
        "positive_scribble_sha256": sha256_file(positive_path),
        "negative_scribble_sha256": sha256_file(negative_path),
        "view_digests": sorted(row["view_digest"] for row in view_records),
    }
    candidates = exchangeable_candidates(
        scene_id=scene_id,
        plan_identity=plan_identity,
        views=view_records,
        count=int(args.candidates),
    )
    plan = {
        "schema_version": 1,
        "artifact_type": PLAN_TYPE,
        "scene_id": scene_id,
        "num_gaussians": num_gaussians,
        "geometry_xyz_sha256": geometry_sha,
        "candidate_count": len(candidates),
        "view_count": len(view_records),
        "registered_camera_count": len(views),
        "skipped_registered_views": skipped_registered_views,
        "candidates": candidates,
        "candidate_semantics": (
            "exchangeable_deterministic_signed_point_trials_equal_logit"
        ),
        "signed_authority": (
            "exact_transport_of_official_positive_and_negative_scribbles;"
            "common_mass_unknown;low_posterior_never_negative"
        ),
        "compositor_mode": COMPOSITOR_MODE,
        "inputs": {
            "dataset_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "method_authority": {
                "path": str(Path(args.method_authority).resolve()),
                "sha256": sha256_file(Path(args.method_authority)),
            },
            "evaluation_contract": {
                "path": str(Path(args.evaluation_contract).resolve()),
                "sha256": sha256_file(Path(args.evaluation_contract)),
            },
            "signed_field_prompt": {
                "path": str(source["signed_margin_path"]),
                "sha256": str(source["signed_margin_sha256"]),
            },
            "positive_scribble": {
                "path": str(positive_path),
                "sha256": sha256_file(positive_path),
            },
            "negative_scribble": {
                "path": str(negative_path),
                "sha256": sha256_file(negative_path),
            },
            "geometry_checkpoint": {
                "path": str(assets.geometry),
                "sha256": str(assets.geometry_sha256),
            },
        },
        "all_candidate_view_inputs_sealed": True,
        "candidate_selection": False,
        "view_selection": (
            "explicit_signed_support_eligibility_before_sam"
            if skipped_registered_views
            else False
        ),
        "registered_view_contract": (
            "complete_queue_locked_rgb_camera_map"
            if bool(args.all_registered_views)
            else "protocol_prompt_and_evaluation_only"
        ),
        "registered_rgb_decoded_by_plan_producer": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    plan_sha = _atomic_json(plan_path, plan)
    del model, renderer, assignment_records
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "plan": str(plan_path),
        "plan_sha256": plan_sha,
        "scene_id": scene_id,
        "candidate_count": len(candidates),
        "view_count": len(view_records),
        "target_mask_opened": False,
        "target_metric_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--manifest", default=str(DATASET_MANIFEST))
    parser.add_argument("--signed-field-prompt-manifest", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument(
        "--evaluation-contract", default=str(DEFAULT_EVALUATION_CONTRACT)
    )
    parser.add_argument("--field-root", default=str(DEFAULT_FIELD_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--points-per-sign", type=int, default=3)
    parser.add_argument("--all-registered-views", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
