#!/usr/bin/env python3
"""Export an authority-bound exact prompt-view compositor matrix.

Official prompt files are opened only for native dimensions and source hashes.
A SPIn reference mask is a source-file authority only: its pixels are never
decoded or interpreted.  Evaluation RGB and target masks are never opened.
Historical top-1 ``responsibility.pt`` files are neither read nor accepted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile

from PIL import Image
import torch

from radio_gs.config import load_config
from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    build_prompt_responsibility_cache,
    save_prompt_responsibility_cache,
    sha256_file,
    tensor_sha256,
)
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
import radio_gs.rendering.contribution_compositor as compositor_module
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


def _scene_record(manifest: dict, scene_id: str) -> dict:
    matches = [scene for scene in manifest.get("scenes", []) if scene.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest scene {scene_id!r}")
    return matches[0]


def _view_by_frame(views: list[dict], frame_id: str) -> dict:
    matches = [view for view in views if str(view.get("frame_id")) == str(frame_id)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one protocol view {frame_id!r}")
    return matches[0]


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = (
        value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _native_prompt_shape(scene: dict) -> tuple[int, int, dict[str, Path]]:
    prompt = scene.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("type") != "positive_negative_scribbles":
        raise ValueError("exporter requires official positive/negative prompt scribbles")
    paths = {
        "positive_scribble": Path(str(prompt.get("positive_path"))).resolve(),
        "negative_scribble": Path(str(prompt.get("negative_path"))).resolve(),
    }
    sizes: dict[str, tuple[int, int]] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        # Header-only use is sufficient; no semantic labels enter W.
        with Image.open(path) as image:
            sizes[role] = tuple(map(int, image.size))
    if len(set(sizes.values())) != 1:
        raise ValueError(f"positive/negative prompt dimensions differ: {sizes}")
    width, height = next(iter(sizes.values()))
    return height, width, paths


@dataclass(frozen=True)
class NativePromptHeaderAuthority:
    """Prompt-file identity and raster geometry, never prompt pixels."""

    prompt_type: str
    reference_frame_id: str
    height: int
    width: int
    paths: dict[str, Path]
    source_sha256: dict[str, str] | None = None


def _require_lowercase_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _native_reference_mask_header_authority(
    scene: dict,
    *,
    expected_prompt_type: str | None,
    expected_reference_frame_id: str | None,
    expected_reference_mask_sha256: str | None,
    expected_native_height: int | None,
    expected_native_width: int | None,
) -> NativePromptHeaderAuthority:
    """Seal a full-mask source file without decoding or interpreting its pixels."""

    prompt = scene.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("type") != "reference_binary_mask":
        raise ValueError("reference-mask authority requires prompt type reference_binary_mask")
    if expected_prompt_type != "reference_binary_mask":
        raise ValueError(
            "reference-mask authority requires --expected-prompt-type "
            "reference_binary_mask"
        )
    reference_frame_id = str(prompt.get("frame_id", ""))
    if not reference_frame_id or str(expected_reference_frame_id or "") != reference_frame_id:
        raise ValueError("reference-mask authority frame id differs from the expected frame")
    declared_frames = scene.get("prompt_frame_ids")
    if not isinstance(declared_frames, list) or [str(value) for value in declared_frames] != [
        reference_frame_id
    ]:
        raise ValueError("reference-mask authority requires exactly one matching prompt frame")
    expected_sha256 = _require_lowercase_sha256(
        expected_reference_mask_sha256,
        label="expected reference mask SHA-256",
    )
    if expected_native_height is None or int(expected_native_height) <= 0:
        raise ValueError("expected native height must be positive")
    if expected_native_width is None or int(expected_native_width) <= 0:
        raise ValueError("expected native width must be positive")

    mask_path = Path(str(prompt.get("mask_path", ""))).resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    # PIL is intentionally used only as a lazy header parser.  Do not call
    # load(), convert(), getdata(), __array__(), or otherwise access pixels.
    with Image.open(mask_path) as image:
        width, height = tuple(map(int, image.size))
    if height <= 0 or width <= 0:
        raise ValueError("reference mask header reported a non-positive native shape")
    if (height, width) != (int(expected_native_height), int(expected_native_width)):
        raise ValueError("reference mask native shape differs from the expected shape")
    mask_sha256 = sha256_file(mask_path)
    if mask_sha256 != expected_sha256:
        raise ValueError("reference mask file SHA-256 differs from the expected authority")
    return NativePromptHeaderAuthority(
        prompt_type="reference_binary_mask",
        reference_frame_id=reference_frame_id,
        height=height,
        width=width,
        paths={"reference_binary_mask": mask_path},
        source_sha256={"reference_binary_mask": mask_sha256},
    )


def _native_prompt_header_authority(
    scene: dict,
    *,
    expected_prompt_type: str | None = None,
    expected_reference_frame_id: str | None = None,
    expected_reference_mask_sha256: str | None = None,
    expected_native_height: int | None = None,
    expected_native_width: int | None = None,
) -> NativePromptHeaderAuthority:
    prompt = scene.get("prompt")
    prompt_type = prompt.get("type") if isinstance(prompt, dict) else None
    if expected_prompt_type is not None and expected_prompt_type != prompt_type:
        raise ValueError("manifest prompt type differs from --expected-prompt-type")
    if prompt_type == "positive_negative_scribbles":
        if any(
            value is not None
            for value in (
                expected_reference_frame_id,
                expected_reference_mask_sha256,
                expected_native_height,
                expected_native_width,
            )
        ):
            raise ValueError("reference-mask expectations are invalid for scribble prompts")
        # Preserve the legacy helper and its exact positive/negative path order.
        height, width, paths = _native_prompt_shape(scene)
        prompt_frames = scene.get("prompt_frame_ids", [prompt.get("frame_id")])
        frame_id = str(prompt_frames[0])
        return NativePromptHeaderAuthority(
            prompt_type="positive_negative_scribbles",
            reference_frame_id=frame_id,
            height=height,
            width=width,
            paths=paths,
        )
    if prompt_type == "reference_binary_mask":
        return _native_reference_mask_header_authority(
            scene,
            expected_prompt_type=expected_prompt_type,
            expected_reference_frame_id=expected_reference_frame_id,
            expected_reference_mask_sha256=expected_reference_mask_sha256,
            expected_native_height=expected_native_height,
            expected_native_width=expected_native_width,
        )
    raise ValueError(f"unsupported prompt type for exact-W export: {prompt_type!r}")


def _reference_mask_report_metadata(
    prompt: NativePromptHeaderAuthority,
    source_sha256: dict[str, str],
    authority: PromptResponsibilityAuthority,
) -> dict[str, object]:
    """Build and revalidate the no-pixel/no-target full-mask report binding."""

    if prompt.prompt_type != "reference_binary_mask":
        raise ValueError("reference-mask report requires reference_binary_mask authority")
    prompt_sources = dict(prompt.source_sha256 or {})
    if set(prompt.paths) != {"reference_binary_mask"} or set(prompt_sources) != {
        "reference_binary_mask"
    }:
        raise ValueError("reference-mask authority requires one clear source SHA key")
    if source_sha256.get("reference_binary_mask") != prompt_sources[
        "reference_binary_mask"
    ]:
        raise ValueError("reference-mask source SHA binding changed before report")
    if (
        authority.frame_id != prompt.reference_frame_id
        or authority.height != prompt.height
        or authority.width != prompt.width
    ):
        raise ValueError("reference-mask raster/frame authority changed before report")
    required_bindings = {
        "camera_mapping",
        "contribution_compositor_source",
        "exporter_source",
        "gaussfm_config",
        "geometry_checkpoint",
    }
    if not required_bindings.issubset(source_sha256):
        raise ValueError("reference-mask report is missing geometry/camera/implementation SHA bindings")
    return {
        "prompt_type": prompt.prompt_type,
        "reference_frame_id": prompt.reference_frame_id,
        "native_height": prompt.height,
        "native_width": prompt.width,
        "source_sha256_key": "reference_binary_mask",
        "reference_binary_mask_sha256": prompt_sources["reference_binary_mask"],
        "source_mask_pixels_decoded": False,
        "source_mask_pixels_interpreted": False,
        "query_or_evidence_constructed": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "geometry_camera_implementation_bindings": {
            "camera_mapping_sha256": source_sha256["camera_mapping"],
            "contribution_compositor_source_sha256": source_sha256[
                "contribution_compositor_source"
            ],
            "exporter_source_sha256": source_sha256["exporter_source"],
            "gaussfm_config_sha256": source_sha256["gaussfm_config"],
            "geometry_checkpoint_sha256": authority.geometry_checkpoint_sha256,
            "geometry_xyz_sha256": authority.geometry_xyz_sha256,
            "intrinsics_sha256": authority.intrinsics_sha256,
            "pose_sha256": authority.pose_sha256,
        },
    }


@contextmanager
def _exclusive_cpu_staging_lock(path: str | None):
    """Serialize native triplet CPU residency across parallel GPU exporters."""

    if not path:
        yield
        return
    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _telemetry_peak_temperature(path: str | None) -> int | None:
    if not path:
        return None
    source = Path(path).resolve()
    if not source.is_file():
        return None
    temperatures: list[int] = []
    for line in source.read_text(encoding="utf-8").splitlines()[1:]:
        columns = line.split(",")
        if len(columns) >= 4:
            try:
                temperatures.append(int(columns[3]))
            except ValueError:
                continue
    return max(temperatures) if temperatures else None


def _commit_local_artifact(
    local_path: Path, remote_path: Path, *, overwrite: bool
) -> None:
    """Sequentially commit a locally hashed artifact without re-reading NFS."""

    remote_path.parent.mkdir(parents=True, exist_ok=True)
    if remote_path.exists() and not overwrite:
        raise FileExistsError(remote_path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{remote_path.name}.", suffix=".tmp", dir=remote_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with local_path.open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != local_path.stat().st_size:
            raise IOError("remote responsibility copy size differs from local artifact")
        if overwrite:
            os.replace(temporary, remote_path)
        else:
            os.link(temporary, remote_path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def export(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact gsplat responsibility export requires an available CUDA device")
    manifest_path = Path(args.manifest).resolve()
    queue_root = Path(args.queue_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = _scene_record(manifest, args.scene_id)
    base_scene_id = str(scene.get("base_scene_id") or args.scene_id)
    queue_scene = queue_root / "scenes" / args.scene_id
    if not queue_scene.is_dir():
        queue_scene = queue_root / "scenes" / base_scene_id
    if not queue_scene.is_dir():
        raise FileNotFoundError(queue_scene)
    config_path = queue_scene / "gaussfm_main_track.yaml"
    checkpoint_path = queue_scene / "feature_field" / "checkpoints" / "best.pth"
    camera_map_path = queue_scene / "rgb_to_colmap_camera_mapping.json"
    for path in (config_path, checkpoint_path, camera_map_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_load_path = checkpoint_path
    if args.geometry_checkpoint_local_copy:
        checkpoint_load_path = Path(args.geometry_checkpoint_local_copy).resolve()
        if not checkpoint_load_path.is_file():
            raise FileNotFoundError(checkpoint_load_path)
        expected_checkpoint_sha256 = str(
            args.expected_geometry_checkpoint_sha256 or ""
        )
        if (
            len(expected_checkpoint_sha256) != 64
            or sha256_file(checkpoint_load_path) != expected_checkpoint_sha256
            or checkpoint_load_path.stat().st_size != checkpoint_path.stat().st_size
        ):
            raise ValueError("local geometry checkpoint copy differs from authority")

    prompt_authority = _native_prompt_header_authority(
        scene,
        expected_prompt_type=getattr(args, "expected_prompt_type", None),
        expected_reference_frame_id=getattr(args, "expected_reference_frame_id", None),
        expected_reference_mask_sha256=getattr(
            args, "expected_reference_mask_sha256", None
        ),
        expected_native_height=getattr(args, "expected_native_height", None),
        expected_native_width=getattr(args, "expected_native_width", None),
    )
    height, width, prompt_paths = (
        prompt_authority.height,
        prompt_authority.width,
        prompt_authority.paths,
    )
    config = load_config(str(config_path))
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    prompt_frame = prompt_authority.reference_frame_id
    view = _view_by_frame(views, prompt_frame)

    model, _codec, renderer, _sharpener, refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(config_path),
            str(checkpoint_load_path),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    if refiner is not None:
        raise ValueError("prompt responsibility carrier must not use an RGB screen refiner")
    xyz = model.get_xyz()
    num_gaussians = int(xyz.shape[0])
    pose = torch.from_numpy(view["w2c"]).to(device=device, dtype=torch.float32)
    intrinsics = renderer.scaled_intrinsics(width, height).detach().float().cpu().contiguous()
    pose_cpu = pose.detach().cpu().contiguous()
    geometry_xyz_sha256 = _float32_rows_sha256(xyz)

    hits = rasterize_single_view_contributions(
        model, renderer, pose, height=height, width=width
    )
    gids = hits["gaussian_ids"]
    pids = hits["pixel_ids"]
    weights = hits["weights"]
    if gids.numel() == 0:
        raise ValueError("exact prompt compositor returned no accepted hits")
    keep = weights > 0
    zero_weight_hits_dropped = int((~keep).sum().item())
    gids, pids, weights = gids[keep], pids[keep], weights[keep]
    del hits, keep
    if bool((pids[1:] < pids[:-1]).any()):
        raise ValueError("exact compositor did not return grouped pixel ids")
    # Pair uniqueness is checked fail-closed in bounded complete-pixel chunks
    # while building/loading the CPU cache.  Avoid a redundant full GPU sort,
    # whose temporary memory can exceed the native triplet payload.

    prompt_source_sha256 = prompt_authority.source_sha256 or {
        role: sha256_file(path) for role, path in prompt_paths.items()
    }
    source_sha256 = {
        "benchmark_manifest": sha256_file(manifest_path),
        "camera_mapping": sha256_file(camera_map_path),
        "contribution_compositor_source": sha256_file(Path(compositor_module.__file__).resolve()),
        "exporter_source": sha256_file(Path(__file__).resolve()),
        "gaussfm_config": sha256_file(config_path),
        "geometry_checkpoint": sha256_file(checkpoint_path),
        **prompt_source_sha256,
    }
    if (
        args.geometry_checkpoint_local_copy
        and source_sha256["geometry_checkpoint"]
        != str(args.expected_geometry_checkpoint_sha256)
    ):
        raise ValueError("geometry checkpoint authority changed after local staging")
    authority = PromptResponsibilityAuthority(
        scene_id=str(args.scene_id),
        frame_id=prompt_frame,
        camera_name=str(view["camera_name"]),
        colmap_camera_name=str(view["colmap_camera_name"]),
        geometry_checkpoint_sha256=source_sha256["geometry_checkpoint"],
        geometry_xyz_sha256=geometry_xyz_sha256,
        pose_sha256=tensor_sha256(pose_cpu),
        intrinsics_sha256=tensor_sha256(intrinsics),
        height=height,
        width=width,
        num_gaussians=num_gaussians,
        alpha_threshold=0.0,
        compositor_contract=COMPOSITOR_CONTRACT,
        target_rgb_opened=False,
        target_mask_opened=False,
        source_sha256=source_sha256,
    )

    # Copy only the three sparse columns retained by the artifact.  Scenes are
    # exported in separate processes so two native triplet sets never coexist.
    with _exclusive_cpu_staging_lock(args.cpu_staging_lock):
        cache = build_prompt_responsibility_cache(
            authority=authority,
            gaussian_ids=gids.detach().cpu(),
            pixel_ids=pids.detach().cpu(),
            weights=weights.detach().cpu(),
        )
        del gids, pids, weights, model
        torch.cuda.empty_cache()
        if args.local_artifact_staging_dir:
            staging_root = Path(args.local_artifact_staging_dir).resolve(strict=True)
            if not staging_root.is_dir():
                raise NotADirectoryError(staging_root)
            with tempfile.TemporaryDirectory(
                prefix=f"nvos_exact_w_{args.scene_id}_", dir=staging_root
            ) as staging_name:
                local_artifact = Path(staging_name) / "responsibility.pt"
                artifact = save_prompt_responsibility_cache(cache, local_artifact)
                remote_artifact = Path(args.output).expanduser().absolute()
                _commit_local_artifact(
                    local_artifact, remote_artifact, overwrite=bool(args.overwrite)
                )
                artifact_path = str(remote_artifact)
                artifact_file_sha256 = artifact.file_sha256
        else:
            artifact = save_prompt_responsibility_cache(
                cache, args.output, overwrite=bool(args.overwrite)
            )
            artifact_path = artifact.path
            artifact_file_sha256 = artifact.file_sha256
        tensor_digests = dict(cache.tensor_sha256)
        triplet_count = int(cache.weights.numel())
        visible_gaussians = int((cache.visible_mass > 0).sum())
        visible_mass_sum = float(cache.visible_mass.sum())
        del cache
        gc.collect()
    report = {
        "artifact_path": artifact_path,
        "file_sha256": artifact_file_sha256,
        "authority": authority.to_dict(),
        "authority_sha256": artifact.authority_sha256,
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": artifact.tensor_bundle_sha256,
        "triplet_count": triplet_count,
        "zero_weight_hits_dropped": zero_weight_hits_dropped,
        "visible_gaussians": visible_gaussians,
        "visible_mass_sum": visible_mass_sum,
        "prompt_assets_opened_for_header_and_hash_only": True,
        "historical_top1_responsibility_opened": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "execution_log": str(Path(args.execution_log).resolve()) if args.execution_log else None,
        "gpu_telemetry_log": str(Path(args.telemetry_log).resolve()) if args.telemetry_log else None,
        "gpu_temperature_peak_c_observed_before_report": _telemetry_peak_temperature(
            args.telemetry_log
        ),
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "geometry_checkpoint_loaded_from_verified_local_copy": bool(
            args.geometry_checkpoint_local_copy
        ),
        "artifact_hashed_before_remote_commit": bool(args.local_artifact_staging_dir),
    }
    if prompt_authority.prompt_type == "reference_binary_mask":
        report["reference_mask_header_authority"] = _reference_mask_report_metadata(
            prompt_authority, source_sha256, authority
        )
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geometry-checkpoint-local-copy")
    parser.add_argument("--expected-geometry-checkpoint-sha256")
    parser.add_argument("--local-artifact-staging-dir")
    parser.add_argument(
        "--cpu-staging-lock",
        help="optional flock path serializing native CPU triplets across exporters",
    )
    parser.add_argument("--telemetry-log", help="thermal-guard CSV to bind and summarize")
    parser.add_argument("--execution-log", help="captured stdout/stderr path to record")
    parser.add_argument(
        "--expected-prompt-type",
        choices=("positive_negative_scribbles", "reference_binary_mask"),
        help=(
            "optional legacy scribble assertion; required as reference_binary_mask "
            "for a full-mask header authority"
        ),
    )
    parser.add_argument(
        "--expected-reference-frame-id",
        help="required exact source reference frame for a reference_binary_mask",
    )
    parser.add_argument(
        "--expected-reference-mask-sha256",
        help="required lowercase complete-file SHA-256 for a reference_binary_mask",
    )
    parser.add_argument(
        "--expected-native-height",
        type=int,
        help="required source raster height for a reference_binary_mask",
    )
    parser.add_argument(
        "--expected-native-width",
        type=int,
        help="required source raster width for a reference_binary_mask",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
