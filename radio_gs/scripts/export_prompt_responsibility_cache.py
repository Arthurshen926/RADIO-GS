#!/usr/bin/env python3
"""Export an authority-bound exact prompt-view compositor matrix.

Only the two official prompt scribble files are opened (for native dimensions
and source hashes).  Evaluation RGB and ground-truth masks are never opened.
Historical top-1 ``responsibility.pt`` files are neither read nor accepted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import resource

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

    height, width, prompt_paths = _native_prompt_shape(scene)
    config = load_config(str(config_path))
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    prompt_frame = str(scene.get("prompt_frame_ids", [scene["prompt"]["frame_id"]])[0])
    view = _view_by_frame(views, prompt_frame)

    model, _codec, renderer, _sharpener, refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(config_path),
            str(checkpoint_path),
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

    source_sha256 = {
        "benchmark_manifest": sha256_file(manifest_path),
        "camera_mapping": sha256_file(camera_map_path),
        "contribution_compositor_source": sha256_file(Path(compositor_module.__file__).resolve()),
        "exporter_source": sha256_file(Path(__file__).resolve()),
        "gaussfm_config": sha256_file(config_path),
        "geometry_checkpoint": sha256_file(checkpoint_path),
        **{role: sha256_file(path) for role, path in prompt_paths.items()},
    }
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
        artifact = save_prompt_responsibility_cache(
            cache, args.output, overwrite=bool(args.overwrite)
        )
        tensor_digests = dict(cache.tensor_sha256)
        triplet_count = int(cache.weights.numel())
        visible_gaussians = int((cache.visible_mass > 0).sum())
        visible_mass_sum = float(cache.visible_mass.sum())
        del cache
        gc.collect()
    report = {
        "artifact_path": artifact.path,
        "file_sha256": artifact.file_sha256,
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
    }
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
    parser.add_argument(
        "--cpu-staging-lock",
        help="optional flock path serializing native CPU triplets across exporters",
    )
    parser.add_argument("--telemetry-log", help="thermal-guard CSV to bind and summarize")
    parser.add_argument("--execution-log", help="captured stdout/stderr path to record")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
