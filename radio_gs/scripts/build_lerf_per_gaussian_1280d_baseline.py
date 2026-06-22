"""Evaluate a per-Gaussian 1280-D explicit RADIO-memory LERF baseline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    OPEN_GAUSSIAN_LERF_FRAMES,
    resolve_registration_split_frame_ids,
    sample_multiview_radio_targets,
    select_registration_frame_ids,
)
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    LERF_OVS_SCENES,
    SigLIP2SummaryHead,
    evaluate_scene,
    load_lerf_ovs_labels,
    load_or_generate_prompt_ensemble_embeddings,
    load_render_pipeline,
    parse_prompt_templates,
    resolve_lerf_label_dir,
    resolve_lerf_scene_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
DEFAULT_COMPONENT_JSON = REPORT_DIR / "lerf_component_ablation.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "radio_gs" / "lerf_per_gaussian_1280d_baseline_20260517"
DEFAULT_MARKDOWN = REPORT_DIR / "lerf_per_gaussian_1280d_baseline.md"
DEFAULT_JSON = REPORT_DIR / "lerf_per_gaussian_1280d_baseline.json"
DEFAULT_LATEX = REPO_ROOT / "paper" / "lerf_per_gaussian_1280d_baseline_table.tex"
DEFAULT_SUMMARY_HEAD = REPO_ROOT / "checkpoints" / "siglip2_summary_head.pth"
DEFAULT_NEAREST_VIEW_TEXT_CACHE_ROOT = (
    REPO_ROOT / "output" / "radio_gs" / "lerf_nearest_view_cache_baseline_20260517" / "text_cache"
)
DEFAULT_PROMPT_ENSEMBLE_TEXT_CACHES = {
    "figurines": REPO_ROOT / "checkpoints" / "siglip2_lerf_text_embeddings_promptens_20260515_figurines.pt",
    "ramen": REPO_ROOT / "checkpoints" / "siglip2_lerf_text_embeddings_promptens_20260515_ramen.pt",
    "teatime": REPO_ROOT / "checkpoints" / "siglip2_lerf_text_embeddings_promptens_20260515_teatime.pt",
    "waldo_kitchen": REPO_ROOT / "checkpoints" / "siglip2_lerf_text_embeddings_promptens_20260515_waldo_kitchen.pt",
}
CACHE_VERSION = 1


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    config: Path
    checkpoint: Path


@dataclass(frozen=True)
class SceneResult:
    scene: str
    loc_acc: float
    miou: float
    n: int
    total_gaussians: int
    registered_gaussians: int
    storage_mib: float


class IdentityCodec(nn.Module):
    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value


class PerGaussianFeatureMemory(nn.Module):
    """Geometry proxy with explicit 1280-D features stored on every Gaussian."""

    def __init__(self, base_model: nn.Module, features: torch.Tensor) -> None:
        super().__init__()
        if features.ndim != 2:
            raise ValueError(f"Expected features [N,D], got {tuple(features.shape)}")
        if int(features.shape[0]) != int(base_model.get_xyz().shape[0]):
            raise ValueError(
                f"Feature row count {features.shape[0]} does not match "
                f"Gaussian count {base_model.get_xyz().shape[0]}"
            )
        self.base_model = base_model
        self.features = features

    def get_xyz(self) -> torch.Tensor:
        return self.base_model.get_xyz()

    def get_rotation(self) -> torch.Tensor:
        return self.base_model.get_rotation()

    def get_scaling(self) -> torch.Tensor:
        return self.base_model.get_scaling()

    def get_opacity(self) -> torch.Tensor:
        return self.base_model.get_opacity()

    def get_features(self) -> torch.Tensor:
        return self.features


def _round4(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _display_scene(scene: str) -> str:
    return {
        "figurines": "Figurines",
        "ramen": "Ramen",
        "teatime": "Teatime",
        "waldo_kitchen": "Waldo Kitchen",
    }.get(scene, scene)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _storage_mib(num_gaussians: int, feature_dim: int, bytes_per_value: int = 2) -> float:
    return round(float(num_gaussians) * float(feature_dim) * float(bytes_per_value) / (1024.0**2), 1)


def summarize_rows(rows: list[SceneResult], *, protocol: dict[str, Any]) -> dict[str, Any]:
    total_n = sum(int(row.n) for row in rows)
    weighted_loc = sum(float(row.loc_acc) * int(row.n) for row in rows) / total_n if total_n else 0.0
    weighted_miou = sum(float(row.miou) * int(row.n) for row in rows) / total_n if total_n else 0.0
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol,
        "rows": [
            {
                "scene": row.scene,
                "loc_acc": _round4(row.loc_acc),
                "miou": _round4(row.miou),
                "n": int(row.n),
                "total_gaussians": int(row.total_gaussians),
                "registered_gaussians": int(row.registered_gaussians),
                "registered_fraction": _round4(row.registered_gaussians / max(row.total_gaussians, 1)),
                "storage_mib": _round4(row.storage_mib),
            }
            for row in rows
        ],
        "macro": {
            "loc_acc": _round4(_mean(row.loc_acc for row in rows)),
            "miou": _round4(_mean(row.miou for row in rows)),
        },
        "weighted": {
            "loc_acc": _round4(weighted_loc),
            "miou": _round4(weighted_miou),
        },
        "mean_registered_fraction": _round4(
            _mean(row.registered_gaussians / max(row.total_gaussians, 1) for row in rows)
        ),
        "mean_storage_mib": _round4(_mean(row.storage_mib for row in rows)),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LERF Per-Gaussian 1280-D Explicit Baseline",
        "",
        "Protocol: Per-Gaussian 1280-D explicit RADIO memory. Cached frame-wise RADIO feature maps are registered to visible Gaussian centers, stored as fp16 1280-D vectors, rendered back to LERF views, and evaluated with the same frozen SigLIP2 text scorer.",
        "",
        "| Scene | LocAcc | mIoU | N | Registered | Fraction | Storage MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {scene} | {loc:.4f} | {miou:.4f} | {n} | {reg}/{total} | {frac:.4f} | {storage:.1f} |".format(
                scene=row["scene"],
                loc=float(row.get("loc_acc", 0.0)),
                miou=float(row.get("miou", 0.0)),
                n=int(row.get("n", 0)),
                reg=int(row.get("registered_gaussians", 0)),
                total=int(row.get("total_gaussians", 0)),
                frac=float(row.get("registered_fraction", 0.0)),
                storage=float(row.get("storage_mib", 0.0)),
            )
        )
    macro = summary.get("macro", {})
    weighted = summary.get("weighted", {})
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Aggregate | LocAcc | mIoU | Registered fraction | Storage MiB |",
            "|---|---:|---:|---:|---:|",
            "| Macro | {loc:.4f} | {miou:.4f} | {frac:.4f} | {storage:.1f} |".format(
                loc=float(macro.get("loc_acc", 0.0)),
                miou=float(macro.get("miou", 0.0)),
                frac=float(summary.get("mean_registered_fraction", 0.0)),
                storage=float(summary.get("mean_storage_mib", 0.0)),
            ),
            "| Query-weighted | {loc:.4f} | {miou:.4f} | -- | -- |".format(
                loc=float(weighted.get("loc_acc", 0.0)),
                miou=float(weighted.get("miou", 0.0)),
            ),
            "",
            "## Interpretation",
            "",
            "- This row is not compact: it stores fp16 1280-D RADIO features per Gaussian.",
            "- It is a 3D scene-memory baseline because features are attached to Gaussian primitives and rendered to novel views.",
            "- Invalid or never-visible Gaussians receive zero features; registered fraction is reported per scene.",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-Gaussian 1280-D explicit RADIO-memory baseline on LERF-OVS. Cached RADIO reference features are registered to Gaussian centers, stored as fp16 vectors, rendered to annotated views, and evaluated with the same frozen SigLIP2 scorer.}",
        "\\label{tab:lerf_per_gaussian_1280d_baseline}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scene & LocAcc & mIoU & Reg. frac. & Storage MiB \\\\",
        "\\midrule",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "{scene} & {loc:.4f} & {miou:.4f} & {frac:.4f} & {storage:.1f} \\\\".format(
                scene=_latex_escape(str(row.get("scene", ""))),
                loc=float(row.get("loc_acc", 0.0)),
                miou=float(row.get("miou", 0.0)),
                frac=float(row.get("registered_fraction", 0.0)),
                storage=float(row.get("storage_mib", 0.0)),
            )
        )
    macro = summary.get("macro", {})
    lines.extend(
        [
            "\\midrule",
            "Macro & {loc:.4f} & {miou:.4f} & {frac:.4f} & {storage:.1f} \\\\".format(
                loc=float(macro.get("loc_acc", 0.0)),
                miou=float(macro.get("miou", 0.0)),
                frac=float(summary.get("mean_registered_fraction", 0.0)),
                storage=float(summary.get("mean_storage_mib", 0.0)),
            ),
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    markdown_path: str | Path = DEFAULT_MARKDOWN,
    json_path: str | Path = DEFAULT_JSON,
    latex_path: str | Path = DEFAULT_LATEX,
) -> dict[str, Path]:
    markdown_out = Path(markdown_path)
    json_out = Path(json_path)
    latex_out = Path(latex_path)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    latex_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(build_markdown(summary), encoding="utf-8")
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latex_out.write_text(build_latex_table(summary), encoding="utf-8")
    return {"markdown": markdown_out, "json": json_out, "latex": latex_out}


def _load_projection(device: torch.device, summary_head_path: str | Path) -> nn.Module:
    proj = SigLIP2SummaryHead.from_extracted_weights(str(summary_head_path))
    proj = proj.to(device)
    return (proj.half() if device.type == "cuda" else proj.float()).eval()


def _scene_specs_from_component_json(path: str | Path, scenes: Iterable[str]) -> dict[str, SceneSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("results", {}).get("full", {})
    selected = {str(scene) for scene in scenes}
    specs: dict[str, SceneSpec] = {}
    for scene, row in rows.items():
        if scene not in selected:
            continue
        specs[scene] = SceneSpec(
            scene=scene,
            config=Path(str(row["config_path"] if row.get("config_path") else row["config"])),
            checkpoint=Path(str(row["checkpoint"])),
        )
    missing = sorted(selected - set(specs))
    if missing:
        raise KeyError(f"Missing full component provenance for scenes: {missing}")
    return specs


def _parse_scenes(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return LERF_OVS_SCENES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_temperature_map(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    values: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        scene, value = part.split(":", 1)
        values[scene.strip()] = float(value)
    return values


def _parse_text_embedding_cache_map(raw: str) -> dict[str, Path]:
    if not raw:
        return {}
    values: dict[str, Path] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "Text embedding cache map entries must use scene=/path/to/cache.pt; "
                f"got {part!r}"
            )
        scene, path = part.split("=", 1)
        scene = scene.strip()
        path = path.strip()
        if not scene or not path:
            raise ValueError(f"Invalid text embedding cache map entry: {part!r}")
        values[scene] = Path(path)
    return values


def _is_query_only_prompt_templates(prompt_templates: list[str]) -> bool:
    return len(prompt_templates) == 1 and prompt_templates[0] == "{query}"


def _cache_matches_prompt_templates(path: Path, prompt_templates: list[str]) -> bool:
    if _is_query_only_prompt_templates(prompt_templates):
        return path.exists()
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    cached_templates = [str(value) for value in payload.get("prompt_templates", ["{query}"])]
    return cached_templates == list(prompt_templates)


def resolve_text_embedding_cache_path(
    scene: str,
    *,
    text_cache_root: str | Path,
    text_embedding_cache_map: dict[str, Path] | None = None,
    fallback_text_cache_roots: Iterable[str | Path] = (DEFAULT_NEAREST_VIEW_TEXT_CACHE_ROOT,),
    prompt_templates: list[str] | None = None,
) -> Path:
    templates = prompt_templates or ["{query}"]
    scene_cache_map = text_embedding_cache_map or {}
    if scene in scene_cache_map:
        return Path(scene_cache_map[scene])

    cache_name = f"{scene}_siglip2_text_embeddings.pt"
    requested = Path(text_cache_root) / cache_name
    if _cache_matches_prompt_templates(requested, templates):
        return requested

    for root in fallback_text_cache_roots:
        candidate = Path(root) / cache_name
        if _cache_matches_prompt_templates(candidate, templates):
            return candidate

    if not _is_query_only_prompt_templates(templates):
        candidate = DEFAULT_PROMPT_ENSEMBLE_TEXT_CACHES.get(scene)
        if candidate is not None and _cache_matches_prompt_templates(candidate, templates):
            return candidate

    return requested


def _load_teacher_feature(path: Path, device: torch.device) -> torch.Tensor:
    feature = torch.load(path, map_location=device)
    if feature.dim() == 3:
        feature = feature.unsqueeze(0)
    if feature.ndim != 4 or int(feature.shape[1]) != 1280:
        raise ValueError(f"Expected RADIO feature [1,1280,H,W] at {path}, got {tuple(feature.shape)}")
    return feature.float()


def _feature_path_by_frame(dataset: LERFDataset) -> dict[int, Path]:
    return {
        int(frame_id): Path(path)
        for frame_id, path in zip(dataset.frame_indices, dataset.feature_paths)
    }


def build_registration_dataset(scene: str, config: object, label_dir: str, *, height: int, width: int) -> LERFDataset:
    return LERFDataset(
        scene_root=str(resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))),
        feature_dir=str(Path(getattr(config, "feature_dir", "") or DEFAULT_GT_FEATURE_ROOT) / ""),
        annotation_dir=str(REPO_ROOT / "__no_lerf_annotations__"),
        feature_height=height,
        feature_width=width,
    )


@torch.no_grad()
def register_teacher_features(
    *,
    scene: str,
    model: nn.Module,
    renderer: nn.Module,
    config: object,
    label_dir: str,
    feature_root: str | Path,
    registration_frame_mode: str,
    registration_max_frames: int,
    registration_chunk_size: int,
    depth_tolerance: float,
    relative_depth_tolerance: float,
    alpha_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    _, _, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)
    registration_dataset = LERFDataset(
        scene_root=str(resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))),
        feature_dir=str(Path(feature_root) / scene),
        annotation_dir=str(REPO_ROOT / "__no_lerf_annotations__"),
        feature_height=img_h,
        feature_width=img_w,
    )
    frame_paths = _feature_path_by_frame(registration_dataset)
    frame_ids = select_registration_frame_ids(
        available_pose_ids=registration_dataset.pose_by_frame_idx.keys(),
        annotated_frame_ids=OPEN_GAUSSIAN_LERF_FRAMES.get(scene, []),
        official_frame_ids=OPEN_GAUSSIAN_LERF_FRAMES.get(scene, []),
        train_frame_ids=resolve_registration_split_frame_ids(config, "train"),
        val_frame_ids=resolve_registration_split_frame_ids(config, "val"),
        mode=registration_frame_mode,
        max_frames=registration_max_frames,
    )
    if not frame_ids:
        raise RuntimeError(f"No registration frames selected for {scene}")

    xyz_cpu = model.get_xyz().detach().cpu().float()
    n_gaussians = int(xyz_cpu.shape[0])
    feature_sum = torch.zeros(n_gaussians, 1280, dtype=torch.float32)
    feature_counts = torch.zeros(n_gaussians, dtype=torch.float32)
    chunk_size = max(int(registration_chunk_size), 1)

    for frame_id in tqdm(frame_ids, desc=f"  register teacher {scene}", leave=False):
        feature_path = frame_paths.get(int(frame_id))
        pose_w2c = registration_dataset.pose_by_frame_idx.get(int(frame_id))
        if feature_path is None or pose_w2c is None:
            continue
        teacher = _load_teacher_feature(feature_path, device)
        viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device).unsqueeze(0)
        aux = renderer.render_features(model, viewmat.squeeze(0))
        depth_map = aux["depth_map"].detach().float().unsqueeze(0)
        alpha_map = aux["alpha_map"].detach().float().unsqueeze(0)
        for start in range(0, n_gaussians, chunk_size):
            end = min(start + chunk_size, n_gaussians)
            points = xyz_cpu[start:end].to(device=device, dtype=torch.float32)
            targets, valid, counts = sample_multiview_radio_targets(
                points,
                teacher,
                viewmat,
                renderer.K,
                depth_map=depth_map,
                alpha_map=alpha_map,
                depth_tolerance=depth_tolerance,
                relative_depth_tolerance=relative_depth_tolerance,
                alpha_threshold=alpha_threshold,
                normalize_sampled_features=False,
            )
            valid_cpu = valid.detach().cpu()
            if valid_cpu.any():
                weights = counts[valid].detach().float().cpu().clamp_min(1.0)
                feature_sum[start:end][valid_cpu] += targets[valid].detach().float().cpu() * weights.unsqueeze(1)
                feature_counts[start:end][valid_cpu] += weights
        del teacher, aux, depth_map, alpha_map
        if device.type == "cuda":
            torch.cuda.empty_cache()

    valid = feature_counts > 0
    features = torch.zeros_like(feature_sum, dtype=torch.float16)
    if bool(valid.any()):
        averaged = feature_sum[valid] / feature_counts[valid].clamp_min(1.0).unsqueeze(1)
        features[valid] = averaged.half()
    return features, feature_counts, frame_ids


def save_feature_cache(
    path: str | Path,
    *,
    scene: str,
    features: torch.Tensor,
    view_counts: torch.Tensor,
    metadata: dict[str, Any],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CACHE_VERSION,
            "scene": scene,
            "feature_dim": int(features.shape[1]),
            "storage_dtype": "float16",
            "metadata": metadata,
            "features": features.cpu().half(),
            "valid": (view_counts > 0).cpu().bool(),
            "view_counts": view_counts.cpu().float(),
        },
        out,
    )
    return out


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if int(payload.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported cache version in {path}")
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or int(features.shape[1]) != 1280:
        raise ValueError(f"Cache {path} must contain features [N,1280]")
    return payload


def _features_for_render(features_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    return features_cpu.float().to(device=device)


def evaluate_scene_baseline(
    *,
    spec: SceneSpec,
    label_dir: str,
    feature_root: str | Path,
    output_root: str | Path,
    projection: nn.Module,
    text_cache_root: str | Path,
    text_embedding_cache_map: dict[str, Path],
    prompt_templates: list[str],
    temperature: float,
    scoring: str,
    iou_threshold: float,
    heatmap_upsample: int,
    registration_frame_mode: str,
    registration_max_frames: int,
    registration_chunk_size: int,
    depth_tolerance: float,
    relative_depth_tolerance: float,
    alpha_threshold: float,
    device: torch.device,
) -> tuple[SceneResult, dict[str, Any]]:
    model, _codec, renderer, _sharpener, _refiner, config, _is_hybrid = load_render_pipeline(
        str(spec.config),
        str(spec.checkpoint),
        device,
    )
    cache_path = Path(output_root) / "per_gaussian_1280d" / spec.scene / "features.pt"
    metadata = {
        "scene": spec.scene,
        "config": str(spec.config),
        "checkpoint": str(spec.checkpoint),
        "feature_root": str(feature_root),
        "registration_frame_mode": registration_frame_mode,
        "registration_max_frames": int(registration_max_frames),
        "depth_tolerance": float(depth_tolerance),
        "relative_depth_tolerance": float(relative_depth_tolerance),
        "alpha_threshold": float(alpha_threshold),
    }
    if cache_path.exists():
        payload = load_feature_cache(cache_path)
        features_cpu = payload["features"]
        view_counts = payload["view_counts"]
        frame_ids = list(payload.get("metadata", {}).get("frame_ids", []))
    else:
        features_cpu, view_counts, frame_ids = register_teacher_features(
            scene=spec.scene,
            model=model,
            renderer=renderer,
            config=config,
            label_dir=label_dir,
            feature_root=feature_root,
            registration_frame_mode=registration_frame_mode,
            registration_max_frames=registration_max_frames,
            registration_chunk_size=registration_chunk_size,
            depth_tolerance=depth_tolerance,
            relative_depth_tolerance=relative_depth_tolerance,
            alpha_threshold=alpha_threshold,
            device=device,
        )
        metadata["frame_ids"] = [int(frame_id) for frame_id in frame_ids]
        save_feature_cache(
            cache_path,
            scene=spec.scene,
            features=features_cpu,
            view_counts=view_counts,
            metadata=metadata,
        )

    _, categories, _, _ = load_lerf_ovs_labels(label_dir, spec.scene)
    text_cache = resolve_text_embedding_cache_path(
        spec.scene,
        text_cache_root=text_cache_root,
        text_embedding_cache_map=text_embedding_cache_map,
        prompt_templates=prompt_templates,
    )
    text_embeddings = load_or_generate_prompt_ensemble_embeddings(
        categories,
        device,
        cache_path=str(text_cache),
        prompt_templates=prompt_templates,
    )
    text_embeddings = text_embeddings.half() if device.type == "cuda" else text_embeddings.float()

    features = _features_for_render(features_cpu, device)
    proxy = PerGaussianFeatureMemory(model, features)
    raw_pipeline = (
        proxy,
        IdentityCodec().to(device),
        renderer,
        nn.Identity().to(device),
        None,
        config,
        False,
    )
    eval_dataset = LERFDataset(
        scene_root=str(resolve_lerf_scene_root(spec.scene, getattr(config, "scene_root", ""))),
        feature_dir=str(Path(feature_root) / spec.scene),
        annotation_dir=str(Path(label_dir) / spec.scene),
        feature_height=getattr(config, "feature_height", 30),
        feature_width=getattr(config, "feature_width", 40),
    )
    result = evaluate_scene(
        scene=spec.scene,
        label_dir=label_dir,
        proj=projection,
        text_embeddings=text_embeddings,
        categories=categories,
        device=device,
        render_pipeline=raw_pipeline,
        lerf_dataset=eval_dataset,
        iou_threshold=iou_threshold,
        scoring=scoring,
        heatmap_upsample=heatmap_upsample,
        temperature=temperature,
    )
    rendered = result.get("rendered", {})
    valid = view_counts > 0
    scene_result = SceneResult(
        scene=_display_scene(spec.scene),
        loc_acc=float(rendered.get("loc_acc", 0.0)),
        miou=float(rendered.get("miou", 0.0)),
        n=int(rendered.get("loc_total", 0)),
        total_gaussians=int(features_cpu.shape[0]),
        registered_gaussians=int(valid.sum().item()),
        storage_mib=_storage_mib(int(features_cpu.shape[0]), int(features_cpu.shape[1])),
    )
    scene_payload = {
        "cache_path": str(cache_path),
        "registration_frame_ids": [int(frame_id) for frame_id in frame_ids],
        "metrics": rendered,
        "registered_gaussians": scene_result.registered_gaussians,
        "total_gaussians": scene_result.total_gaussians,
        "registered_fraction": _round4(scene_result.registered_gaussians / max(scene_result.total_gaussians, 1)),
        "storage_mib": _round4(scene_result.storage_mib),
        "temperature": float(temperature),
        "text_embedding_cache": str(text_cache),
    }
    del features, proxy, raw_pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scene_result, scene_payload


def evaluate_baseline(
    *,
    scenes: Iterable[str],
    component_json: str | Path,
    label_dir: str,
    feature_root: str | Path,
    output_root: str | Path,
    summary_head_path: str | Path,
    text_cache_root: str | Path,
    text_embedding_cache_map: dict[str, Path],
    prompt_templates: list[str],
    temperatures: dict[str, float],
    scoring: str,
    iou_threshold: float,
    heatmap_upsample: int,
    registration_frame_mode: str,
    registration_max_frames: int,
    registration_chunk_size: int,
    depth_tolerance: float,
    relative_depth_tolerance: float,
    alpha_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    scene_names = tuple(scenes)
    specs = _scene_specs_from_component_json(component_json, scene_names)
    projection = _load_projection(device, summary_head_path)
    rows: list[SceneResult] = []
    scene_payloads: dict[str, Any] = {}
    for scene in scene_names:
        row, payload = evaluate_scene_baseline(
            spec=specs[scene],
            label_dir=label_dir,
            feature_root=feature_root,
            output_root=output_root,
            projection=projection,
            text_cache_root=text_cache_root,
            text_embedding_cache_map=text_embedding_cache_map,
            prompt_templates=prompt_templates,
            temperature=float(temperatures.get(scene, 25.0)),
            scoring=scoring,
            iou_threshold=iou_threshold,
            heatmap_upsample=heatmap_upsample,
            registration_frame_mode=registration_frame_mode,
            registration_max_frames=registration_max_frames,
            registration_chunk_size=registration_chunk_size,
            depth_tolerance=depth_tolerance,
            relative_depth_tolerance=relative_depth_tolerance,
            alpha_threshold=alpha_threshold,
            device=device,
        )
        rows.append(row)
        scene_payloads[scene] = payload

    protocol = {
        "feature_source": "registered RADIO 1280-D RADIO reference features",
        "feature_dim": 1280,
        "storage_dtype": "fp16",
        "compact": False,
        "3d_memory": True,
        "novel_view_feature": True,
        "direct_3d_query": "partial",
        "registration_frame_mode": registration_frame_mode,
        "registration_max_frames": int(registration_max_frames),
        "depth_tolerance": float(depth_tolerance),
        "relative_depth_tolerance": float(relative_depth_tolerance),
        "alpha_threshold": float(alpha_threshold),
        "iou_threshold": float(iou_threshold),
        "scoring": scoring,
        "heatmap_upsample": int(heatmap_upsample),
        "prompt_templates": prompt_templates,
        "text_cache_root": str(text_cache_root),
        "text_embedding_cache_map": {scene: str(path) for scene, path in text_embedding_cache_map.items()},
    }
    summary = summarize_rows(rows, protocol=protocol)
    summary["scene_payloads"] = scene_payloads
    return summary


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--component_json", default=str(DEFAULT_COMPONENT_JSON))
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature_root", default=DEFAULT_GT_FEATURE_ROOT)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary_head_weights", default=str(DEFAULT_SUMMARY_HEAD))
    parser.add_argument("--text_cache_root", default=str(DEFAULT_OUTPUT_ROOT / "text_cache"))
    parser.add_argument(
        "--text_embedding_cache_map",
        default="",
        help="Optional comma-separated scene=/path/to/cache.pt overrides for SigLIP2 text caches.",
    )
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--temperatures", default="figurines:50,ramen:40,teatime:25,waldo_kitchen:25")
    parser.add_argument("--scoring", default="softmax_scene")
    parser.add_argument("--iou_threshold", type=float, default=0.60)
    parser.add_argument("--heatmap_upsample", type=int, default=4)
    parser.add_argument("--registration_frame_mode", choices=["official", "annotated", "all_poses", "train", "val"], default="train")
    parser.add_argument("--registration_max_frames", type=int, default=64)
    parser.add_argument("--registration_chunk_size", type=int, default=32768)
    parser.add_argument("--registration_depth_tolerance", type=float, default=0.08)
    parser.add_argument("--registration_relative_depth_tolerance", type=float, default=0.02)
    parser.add_argument("--registration_alpha_threshold", type=float, default=0.02)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_LATEX))
    args = parser.parse_args(argv)

    label_dir = resolve_lerf_label_dir(args.label_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    summary = evaluate_baseline(
        scenes=_parse_scenes(args.scenes),
        component_json=args.component_json,
        label_dir=label_dir,
        feature_root=args.feature_root,
        output_root=args.output_root,
        summary_head_path=args.summary_head_weights,
        text_cache_root=args.text_cache_root,
        text_embedding_cache_map=_parse_text_embedding_cache_map(args.text_embedding_cache_map),
        prompt_templates=parse_prompt_templates(args.prompt_templates),
        temperatures=_parse_temperature_map(args.temperatures),
        scoring=args.scoring,
        iou_threshold=args.iou_threshold,
        heatmap_upsample=args.heatmap_upsample,
        registration_frame_mode=args.registration_frame_mode,
        registration_max_frames=args.registration_max_frames,
        registration_chunk_size=args.registration_chunk_size,
        depth_tolerance=args.registration_depth_tolerance,
        relative_depth_tolerance=args.registration_relative_depth_tolerance,
        alpha_threshold=args.registration_alpha_threshold,
        device=device,
    )
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
