#!/usr/bin/env python3
"""Draw the paper framework figure for CTF-GS."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "input": "#DCEBFF",
    "field": "#E7F6EC",
    "loss": "#FFF3D8",
    "readout": "#F0E8FF",
    "eval": "#F4F4F4",
    "edge": "#2F3A45",
    "text": "#17212B",
}


def box(ax, xy, wh, text, *, color, fontsize=10, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=lw,
        edgecolor=COLORS["edge"],
        facecolor=color,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, *, text="", rad=0.0, color="#2F3A45"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.035,
            text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
        )


def main() -> None:
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.955,
        "CTF-GS: Dual-Readout Compact Teacher Feature Field",
        fontsize=17,
        fontweight="bold",
        color=COLORS["text"],
        va="top",
    )
    ax.text(
        0.03,
        0.908,
        "One compact Gaussian feature memory supports rendered-view grounding and direct primitive/point-level querying.",
        fontsize=10.5,
        color="#4B5563",
        va="top",
    )

    # Inputs.
    box(
        ax,
        (0.035, 0.70),
        (0.17, 0.13),
        "Posed RGB views\n+ 3DGS geometry",
        color=COLORS["input"],
    )
    box(
        ax,
        (0.035, 0.52),
        (0.17, 0.13),
        "Frozen RADIO teacher\n1280d dense features",
        color=COLORS["input"],
    )
    box(
        ax,
        (0.035, 0.34),
        (0.17, 0.13),
        "Frozen heads\nSigLIP2 / DINOv3 /\nSAM3 adaptor",
        color=COLORS["input"],
        fontsize=9.2,
    )

    # Field and decoder.
    box(
        ax,
        (0.27, 0.63),
        (0.18, 0.15),
        "Hybrid Gaussian\nCode Field\nper-Gaussian z_i",
        color=COLORS["field"],
    )
    box(
        ax,
        (0.27, 0.41),
        (0.18, 0.15),
        "Voxel / spatial\ncontext branch\nh(x)",
        color=COLORS["field"],
    )
    box(
        ax,
        (0.50, 0.55),
        (0.16, 0.16),
        "CTR / HCD\nteacher decoder\ncompact -> 1280d",
        color=COLORS["field"],
    )
    box(
        ax,
        (0.50, 0.31),
        (0.16, 0.14),
        "VFA\nview-space feature\nalignment",
        color=COLORS["field"],
    )

    # Losses.
    box(
        ax,
        (0.27, 0.16),
        (0.18, 0.14),
        "Training constraints\nteacher feature\n+ FGC geometry",
        color=COLORS["loss"],
        fontsize=9.3,
    )
    box(
        ax,
        (0.50, 0.10),
        (0.18, 0.14),
        "Adaptor supervision\nDINO relation\nSAM3 mask logits",
        color=COLORS["loss"],
        fontsize=9.3,
    )

    # Readouts.
    box(
        ax,
        (0.73, 0.68),
        (0.19, 0.13),
        "Rendered-view readout\nsplat compact codes\n-> dense feature map",
        color=COLORS["readout"],
        fontsize=9.2,
    )
    box(
        ax,
        (0.73, 0.47),
        (0.19, 0.14),
        "Raster VPR readout\nGaussian-pixel hits\n-> primitive scores",
        color=COLORS["readout"],
        fontsize=9.2,
    )
    box(
        ax,
        (0.73, 0.25),
        (0.19, 0.14),
        "Direct point readout\nquery 3D points\n-> teacher-space features",
        color=COLORS["readout"],
        fontsize=9.2,
    )
    box(
        ax,
        (0.73, 0.06),
        (0.19, 0.12),
        "Proposal / OPR\nconnected 3D support\nobject-aware masks",
        color=COLORS["readout"],
        fontsize=9.2,
    )

    # Outputs.
    box(
        ax,
        (0.955, 0.68),
        (0.03, 0.13),
        "2D\nOVG",
        color=COLORS["eval"],
        fontsize=8,
    )
    box(
        ax,
        (0.955, 0.47),
        (0.03, 0.14),
        "3D\nOVS",
        color=COLORS["eval"],
        fontsize=8,
    )
    box(
        ax,
        (0.955, 0.25),
        (0.03, 0.14),
        "ScanNet\npoint",
        color=COLORS["eval"],
        fontsize=7.5,
    )
    box(
        ax,
        (0.955, 0.06),
        (0.03, 0.12),
        "SAM\nDINO",
        color=COLORS["eval"],
        fontsize=7.5,
    )

    # Arrows from inputs to field/losses.
    arrow(ax, (0.205, 0.765), (0.27, 0.705), text="initialize / render")
    arrow(ax, (0.205, 0.585), (0.27, 0.705), text="distill")
    arrow(ax, (0.205, 0.405), (0.27, 0.23), text="frozen probes", rad=-0.12)
    arrow(ax, (0.36, 0.63), (0.50, 0.63), text="fuse")
    arrow(ax, (0.45, 0.485), (0.50, 0.61), rad=-0.08)
    arrow(ax, (0.58, 0.55), (0.58, 0.45), text="screen refine")
    arrow(ax, (0.36, 0.30), (0.36, 0.41), rad=-0.08)
    arrow(ax, (0.59, 0.24), (0.58, 0.31), rad=0.05)

    # Arrows from decoder to readouts.
    arrow(ax, (0.66, 0.64), (0.73, 0.745), text="rasterize")
    arrow(ax, (0.66, 0.61), (0.73, 0.54), text="register")
    arrow(ax, (0.66, 0.58), (0.73, 0.32), text="query")
    arrow(ax, (0.66, 0.39), (0.73, 0.12), text="group")

    # Readout outputs.
    arrow(ax, (0.92, 0.745), (0.955, 0.745))
    arrow(ax, (0.92, 0.54), (0.955, 0.54))
    arrow(ax, (0.92, 0.32), (0.955, 0.32))
    arrow(ax, (0.92, 0.12), (0.955, 0.12))

    ax.text(
        0.742,
        0.90,
        "Evaluation interfaces",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
        va="center",
    )
    ax.text(
        0.275,
        0.86,
        "Compact feature memory",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
        va="center",
    )
    ax.text(
        0.275,
        0.08,
        "GT-free method additions: raster contribution registration + proposal-level object readout.",
        fontsize=9.2,
        color="#4B5563",
        va="center",
    )

    fig.savefig(out_dir / "radio_gs_framework.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / "radio_gs_framework.pdf", bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()
