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


def checkpoint_tensor_bytes(path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "model": _state_dict_bytes(checkpoint.get("model_state_dict", {})),
        "codec": _state_dict_bytes(checkpoint.get("codec_state_dict", {})),
        "refiner": _state_dict_bytes(checkpoint.get("refiner_state_dict", {})),
    }


def build_storage_row(item: SceneFootprintInput) -> StorageRow:
    gaussians = parse_ply_vertex_count(item.ply_path)
    direct = int(gaussians) * TEACHER_DIM * FP16_BYTES
    parts = checkpoint_tensor_bytes(item.checkpoint_path)
    total = int(parts["model"] + parts["codec"] + parts["refiner"])
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
        "This report compares direct per-Gaussian 1280-D fp16 teacher-feature storage "
        "against the stored compact CTF-GS checkpoint footprint. The compact "
        "footprint includes the Gaussian feature-field state dict, CTR/HCD codec, "
        "and VFA/screen refiner tensors. VPR is an inference-time readout; the "
        "registered SigLIP2 primitive cache and per-query voxel-score cache are "
        "reported separately because they are not persistent checkpoint state.",
        "",
        "| Scene | #Gaussians | Direct 1280-D fp16 | Compact ckpt | Saving | Optional VPR emb. cache | Voxel score cache | Compact + optional VPR | Saving w/ optional VPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {n:,} | {direct:.1f} MiB | {total:.1f} MiB | {ratio:.2f}x | "
            "{vpr:.1f} MiB | {voxel:.1f} MiB | {optional:.1f} MiB | {ratio_optional:.2f}x |".format(
                scene=row.scene,
                n=row.gaussians,
                direct=_mib(row.direct_fp16_bytes),
                total=_mib(row.total_compact_bytes),
                ratio=row.saving_ratio,
                vpr=_mib(row.vpr_embedding_cache_bytes),
                voxel=_mib(row.voxel_score_cache_bytes),
                optional=_mib(row.compact_plus_optional_vpr_cache_bytes),
                ratio_optional=row.saving_ratio_with_optional_vpr_cache,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Direct storage assumes storing only the 1280-D teacher feature as fp16 "
            "per Gaussian; it excludes ordinary 3DGS RGB/geometry attributes.",
            "- Compact total is conservative because it counts the whole feature-field "
            "model state dict, including existing 3DGS geometry/RGB tensors carried "
            "inside the checkpoint, plus decoder and refiner tensors.",
            "- VPR does not add persistent trained parameters. If one persists the "
            "registered 1536-D SigLIP2 primitive embeddings and scene-query voxel "
            "scores instead of streaming them during evaluation, that optional cache "
            "is larger than direct 1280-D feature storage on these scenes; the paper "
            "therefore separates stored compact checkpoint footprint from optional "
            "VPR inference cache footprint.",
            "- The paper should describe this as a footprint accounting table rather "
            "than a pure per-primitive compression ratio.",
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
        r"\caption{Storage footprint accounting for compact teacher-feature fields and optional VPR caches. Direct storage assumes one 1280-D fp16 teacher feature per Gaussian. Compact ckpt counts the stored feature-field model, CTR/HCD codec, and VFA/refiner tensors. VPR caches are inference-time artifacts, not persistent trained state.}",
        r"\label{tab:storage_footprint}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Scene & \#G & Direct & Compact & Saving & Opt. VPR cache & Saving w/ cache \\",
        r" &  & MiB & MiB &  & MiB &  \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.scene} & {row.gaussians:,} & "
            f"{_mib(row.direct_fp16_bytes):.1f} & "
            f"{_mib(row.total_compact_bytes):.1f} & "
            f"{row.saving_ratio:.2f}$\\times$ & "
            f"{_mib(row.vpr_embedding_cache_bytes + row.voxel_score_cache_bytes):.1f} & "
            f"{row.saving_ratio_with_optional_vpr_cache:.2f}$\\times$ \\\\"
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
    write_latex(rows, REPO_ROOT / "paper/storage_footprint_table.tex")
    print("Wrote output/radio_gs/reports/storage_footprint_report.md")
    print("Wrote paper/storage_footprint_table.tex")


if __name__ == "__main__":
    main()
