#!/usr/bin/env python3
"""Build paper-facing storage footprint evidence for RADIO-GS/CTF-GS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


TEACHER_DIM = 1280
SIGLIP_DIM = 1536
FP16_BYTES = 2
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SceneFootprintInput:
    scene: str
    ply_path: Path
    checkpoint_path: Path
    query_count: int = 0


@dataclass(frozen=True)
class StorageRow:
    scene: str
    gaussians: int
    direct_fp16_bytes: int
    latent_payload_bytes: int
    fixed_feature_package_bytes: int
    feature_memory_package_bytes: int
    model_bytes: int
    codec_bytes: int
    refiner_bytes: int
    total_compact_bytes: int
    vpr_embedding_cache_bytes: int
    voxel_score_cache_bytes: int
    compact_plus_persistent_vpr_bytes: int
    compact_plus_optional_vpr_cache_bytes: int
    saving_ratio: float
    saving_ratio_with_optional_vpr_cache: float
    checkpoint_path: Path


def parse_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"PLY header does not contain an element vertex line: {path}")


def _state_dict_bytes(state_dict: object) -> int:
    if not isinstance(state_dict, dict):
        return 0
    total = 0
    for value in state_dict.values():
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += int(value.numel()) * int(value.element_size())
    return total


def _selected_state_bytes(state_dict: object, prefixes: tuple[str, ...]) -> int:
    if not isinstance(state_dict, dict):
        return 0
    total = 0
    for key, value in state_dict.items():
        if not any(str(key).startswith(prefix) for prefix in prefixes):
            continue
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += int(value.numel()) * int(value.element_size())
    return total


def checkpoint_tensor_bytes(path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict", {})
    codec = _state_dict_bytes(checkpoint.get("codec_state_dict", {}))
    refiner = _state_dict_bytes(checkpoint.get("refiner_state_dict", {}))
    latent = _selected_state_bytes(model_state, ("_latent",))
    field_heads = _selected_state_bytes(
        model_state,
        (
            "hash_field",
            "fusion_head",
            "coarse_decoder",
            "fine_decoder",
            "quality_head",
            "visibility_head",
            "point_summary_adapter",
        ),
    )
    return {
        "model": _state_dict_bytes(model_state),
        "codec": codec,
        "refiner": refiner,
        "latent": latent,
        "field_heads": field_heads,
        "fixed_feature_package": field_heads + codec + refiner,
    }


def build_storage_row(item: SceneFootprintInput) -> StorageRow:
    gaussians = parse_ply_vertex_count(item.ply_path)
    direct = int(gaussians) * TEACHER_DIM * FP16_BYTES
    parts = checkpoint_tensor_bytes(item.checkpoint_path)
    total = int(parts["model"] + parts["codec"] + parts["refiner"])
    feature_memory_package = int(
        parts["latent"] + parts["fixed_feature_package"]
    )
    vpr_embedding_cache = int(gaussians) * SIGLIP_DIM * FP16_BYTES
    voxel_score_cache = int(gaussians) * max(int(item.query_count), 0) * FP16_BYTES
    compact_plus_persistent_vpr = total
    compact_plus_optional_vpr = total + vpr_embedding_cache + voxel_score_cache
    ratio = float(direct) / float(total) if total else 0.0
    ratio_with_optional_vpr = (
        float(direct) / float(compact_plus_optional_vpr)
        if compact_plus_optional_vpr
        else 0.0
    )
    return StorageRow(
        scene=item.scene,
        gaussians=gaussians,
        direct_fp16_bytes=direct,
        latent_payload_bytes=int(parts["latent"]),
        fixed_feature_package_bytes=int(parts["fixed_feature_package"]),
        feature_memory_package_bytes=feature_memory_package,
        model_bytes=int(parts["model"]),
        codec_bytes=int(parts["codec"]),
        refiner_bytes=int(parts["refiner"]),
        total_compact_bytes=total,
        vpr_embedding_cache_bytes=vpr_embedding_cache,
        voxel_score_cache_bytes=voxel_score_cache,
        compact_plus_persistent_vpr_bytes=compact_plus_persistent_vpr,
        compact_plus_optional_vpr_cache_bytes=compact_plus_optional_vpr,
        saving_ratio=ratio,
        saving_ratio_with_optional_vpr_cache=ratio_with_optional_vpr,
        checkpoint_path=item.checkpoint_path,
    )


def default_inputs() -> list[SceneFootprintInput]:
    return [
        SceneFootprintInput(
            scene="Figurines",
            ply_path=REPO_ROOT
            / "output/3dgs_models/figurines/point_cloud/iteration_30000/point_cloud.ply",
            checkpoint_path=REPO_ROOT
            / "output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth",
            query_count=21,
        ),
        SceneFootprintInput(
            scene="Ramen",
            ply_path=REPO_ROOT
            / "output/3dgs_models/ramen/point_cloud/iteration_30000/point_cloud.ply",
            checkpoint_path=REPO_ROOT
            / "output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth",
            query_count=14,
        ),
        SceneFootprintInput(
            scene="Teatime",
            ply_path=REPO_ROOT
            / "output/3dgs_models/teatime/point_cloud/iteration_30000/point_cloud.ply",
            checkpoint_path=REPO_ROOT
            / "output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth",
            query_count=14,
        ),
        SceneFootprintInput(
            scene="Waldo Kitchen",
            ply_path=REPO_ROOT
            / "output/3dgs_models/waldo_kitchen/point_cloud/iteration_30000/point_cloud.ply",
            checkpoint_path=REPO_ROOT
            / "output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth",
            query_count=18,
        ),
    ]


def _mib(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_markdown(rows: Iterable[StorageRow], path: Path) -> None:
    rows = list(rows)
    lines = [
        "# Storage Footprint Report",
        "",
        "This report separates three storage accounting levels so that the compact "
        "feature-memory claim is not hidden by carried 3DGS geometry/RGB tensors. "
        "Direct storage assumes one 1280-D fp16 teacher feature per Gaussian. "
        "Latent payload counts only the stored compact per-Gaussian semantic code. "
        "Feature-memory package counts the latent payload plus global field heads, "
        "CTR/HCD codec, and VFA/refiner tensors. Full checkpoint is a conservative "
        "deployable accounting that also includes ordinary 3DGS geometry/RGB and "
        "appearance tensors carried inside the model state dict. VPR caches are "
        "reported separately because they are optional inference artifacts, not "
        "persistent trained state.",
        "",
        "| Scene | #Gaussians | Direct 1280-D fp16 | Latent payload | Latent saving | Feature-memory package | Package saving | Full checkpoint | Full saving | Optional VPR emb. cache | Voxel score cache |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {n:,} | {direct:.1f} MiB | {latent:.1f} MiB | "
            "{latent_ratio:.2f}x | {package:.1f} MiB | {package_ratio:.2f}x | "
            "{total:.1f} MiB | {ratio:.2f}x | {vpr:.1f} MiB | {voxel:.1f} MiB |".format(
                scene=row.scene,
                n=row.gaussians,
                direct=_mib(row.direct_fp16_bytes),
                latent=_mib(row.latent_payload_bytes),
                latent_ratio=float(row.direct_fp16_bytes) / float(row.latent_payload_bytes)
                if row.latent_payload_bytes
                else 0.0,
                package=_mib(row.feature_memory_package_bytes),
                package_ratio=float(row.direct_fp16_bytes) / float(row.feature_memory_package_bytes)
                if row.feature_memory_package_bytes
                else 0.0,
                total=_mib(row.total_compact_bytes),
                ratio=row.saving_ratio,
                vpr=_mib(row.vpr_embedding_cache_bytes),
                voxel=_mib(row.voxel_score_cache_bytes),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Direct storage assumes storing only the 1280-D teacher feature as fp16 "
            "per Gaussian; it excludes ordinary 3DGS RGB/geometry attributes.",
            "- The latent payload is the clean per-Gaussian semantic storage number. "
            "It is approximately the 64-D compact code and gives the expected "
            "20x reduction relative to 1280-D fp16 teacher features.",
            "- The feature-memory package adds scene-global heads and decoders. "
            "Those fixed tensors do not scale with the number of Gaussians, so "
            "their overhead is amortized more strongly on larger indoor/outdoor "
            "scenes; the per-scene growing term remains the compact latent payload.",
            "- Full checkpoint is intentionally conservative because it counts the "
            "whole feature-field model state dict, including existing 3DGS geometry/"
            "RGB tensors carried inside the checkpoint, plus decoder and refiner "
            "tensors. Use it for deployable footprint, not for pure feature-memory "
            "compression.",
            "- VPR does not add persistent trained parameters. If one persists the "
            "registered 1536-D SigLIP2 primitive embeddings and scene-query voxel "
            "scores instead of streaming them during evaluation, that optional cache "
            "is larger than direct 1280-D feature storage on these scenes; the paper "
            "therefore separates stored compact checkpoint footprint from optional "
            "VPR inference cache footprint.",
            "- The paper should report latent/package/full-checkpoint columns rather "
            "than only the full checkpoint; otherwise the compact feature-memory "
            "advantage is underestimated.",
            "",
            "## Sources",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row.scene}: `{_rel(row.checkpoint_path)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: Iterable[StorageRow], path: Path) -> None:
    rows = list(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Storage footprint accounting. Direct stores one 1280-D fp16 teacher feature per Gaussian. Latent is the compact per-Gaussian semantic payload. Feature package adds scene-global heads, CTR/HCD, and VFA/refiner tensors; these fixed tensors are amortized as scene scale grows. Full checkpoint additionally counts carried 3DGS geometry/RGB state. Lower is better for storage; higher saving ratios are better.}",
        r"\label{tab:storage_footprint}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Scene & \#G & Direct & Latent & Save & Feature pkg. & Save & Full ckpt. & Save \\",
        r" &  & MiB & MiB &  & MiB &  & MiB &  \\",
        r"\midrule",
    ]
    for row in rows:
        latent_ratio = (
            float(row.direct_fp16_bytes) / float(row.latent_payload_bytes)
            if row.latent_payload_bytes
            else 0.0
        )
        package_ratio = (
            float(row.direct_fp16_bytes) / float(row.feature_memory_package_bytes)
            if row.feature_memory_package_bytes
            else 0.0
        )
        lines.append(
            f"{row.scene} & {row.gaussians:,} & "
            f"{_mib(row.direct_fp16_bytes):.1f} & "
            f"{_mib(row.latent_payload_bytes):.1f} & "
            f"{latent_ratio:.1f}$\\times$ & "
            f"{_mib(row.feature_memory_package_bytes):.1f} & "
            f"{package_ratio:.2f}$\\times$ & "
            f"{_mib(row.total_compact_bytes):.1f} & "
            f"{row.saving_ratio:.2f}$\\times$ \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_rows(inputs: Iterable[SceneFootprintInput] | None = None) -> list[StorageRow]:
    return [build_storage_row(item) for item in (inputs or default_inputs())]


def main() -> None:
    rows = build_rows()
    write_markdown(rows, REPO_ROOT / "output/radio_gs/reports/storage_footprint_report.md")
    write_markdown(rows, REPO_ROOT / "paper/artifacts/storage_footprint_report.md")
    write_latex(rows, REPO_ROOT / "paper/storage_footprint_table.tex")
    print("Wrote output/radio_gs/reports/storage_footprint_report.md")
    print("Wrote paper/artifacts/storage_footprint_report.md")
    print("Wrote paper/storage_footprint_table.tex")


if __name__ == "__main__":
    main()
