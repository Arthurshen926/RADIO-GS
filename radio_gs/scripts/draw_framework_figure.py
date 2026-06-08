#!/usr/bin/env python3
"""Draw the paper framework figure for CTF-GS.

The figure is intentionally vector-first and compact: it separates training-only
supervision from inference-time readouts and makes the "one compact map" claim
visually explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "bg": "#FFFFFF",
    "ink": "#16212C",
    "muted": "#5B6673",
    "edge": "#2E3A45",
    "input": "#DCEBFF",
    "teacher": "#EDF4FF",
    "field": "#E3F6EA",
    "field_deep": "#BFE8D0",
    "head": "#EEE8FF",
    "train": "#FFF1D4",
    "eval": "#F7F8FA",
    "blue": "#316BE6",
    "green": "#20A386",
    "purple": "#7556D9",
    "orange": "#E66B3D",
    "yellow": "#E7B52E",
    "gray": "#D7DBDF",
}


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str = "",
    *,
    color: str,
    fontsize: float = 8.5,
    weight: str = "normal",
    lw: float = 1.05,
    radius: float = 0.018,
    align: str = "center",
    alpha: float = 1.0,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=lw,
        edgecolor=COLORS["edge"],
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    if text:
        tx = x + w / 2 if align == "center" else x + 0.014
        ha = "center" if align == "center" else "left"
        ax.text(
            tx,
            y + h / 2,
            text,
            ha=ha,
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=COLORS["ink"],
            linespacing=1.10,
        )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    style: str = "solid",
    color: str = COLORS["edge"],
    rad: float = 0.0,
    lw: float = 1.35,
    scale: float = 12.5,
    label_offset: float = 0.015,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + label_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=color,
        )


def add_elbow_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    label: str = "",
    color: str = COLORS["edge"],
    style: str = "solid",
    lw: float = 1.25,
) -> None:
    """Draw an orthogonal route with an arrowhead on the last segment."""
    if len(points) < 2:
        return
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, linestyle=style, linewidth=lw, solid_capstyle="round")
    arrow = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=1,
        shrinkB=4,
    )
    ax.add_patch(arrow)
    if label:
        mid = points[len(points) // 2]
        ax.text(mid[0], mid[1] + 0.014, label, ha="center", va="bottom", fontsize=7.0, color=color)


def add_panel_label(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=COLORS["ink"],
    )


def add_tag(ax, x: float, y: float, text: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), 0.006, 0.046, facecolor=color, edgecolor="none"))
    ax.text(x + 0.012, y + 0.023, text, ha="left", va="center", fontsize=7.2, color=COLORS["muted"])


def add_gaussian_cloud(ax, x: float, y: float, w: float, h: float) -> None:
    rng = np.random.default_rng(7)
    palette = [COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["orange"], COLORS["yellow"]]
    for i in range(28):
        px = x + 0.06 * w + rng.random() * 0.88 * w
        py = y + 0.08 * h + rng.random() * 0.76 * h
        r = (0.010 + 0.012 * rng.random()) * min(w / 0.22, h / 0.16)
        ax.add_patch(
            Circle(
                (px, py),
                r,
                facecolor=palette[i % len(palette)],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.74,
            )
        )
    ax.text(
        x + w / 2,
        y + 0.055 * h,
        "compact codes + spatial context",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=COLORS["muted"],
    )


def add_heatmap_icon(ax, x: float, y: float, w: float, h: float) -> None:
    nx, ny = 18, 12
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    z = np.exp(-((xx - 0.25) ** 2 / 0.18 + (yy + 0.15) ** 2 / 0.24))
    z += 0.45 * np.exp(-((xx + 0.55) ** 2 / 0.05 + (yy - 0.45) ** 2 / 0.08))
    for iy in range(ny):
        for ix in range(nx):
            c = plt.cm.turbo(float(z[iy, ix] / z.max()))
            ax.add_patch(
                Rectangle(
                    (x + ix * w / nx, y + iy * h / ny),
                    w / nx,
                    h / ny,
                    facecolor=c,
                    edgecolor="none",
                    alpha=0.92,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["edge"], linewidth=0.8))


def add_selection_icon(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor=COLORS["edge"], linewidth=0.8))
    rng = np.random.default_rng(12)
    for _ in range(20):
        ax.add_patch(
            Circle(
                (x + rng.random() * w, y + rng.random() * h),
                0.004,
                facecolor="#DDE2E6",
                edgecolor="none",
                alpha=0.8,
            )
        )
    for px, py, rw, rh in [(0.42, 0.50, 0.070, 0.045), (0.60, 0.60, 0.050, 0.035), (0.72, 0.38, 0.035, 0.030)]:
        ax.add_patch(
            FancyBboxPatch(
                (x + px * w, y + py * h),
                rw,
                rh,
                boxstyle="round,pad=0.002,rounding_size=0.010",
                facecolor=COLORS["orange"],
                edgecolor=COLORS["orange"],
                alpha=0.72,
            )
        )


def add_point_icon(ax, x: float, y: float, w: float, h: float) -> None:
    rng = np.random.default_rng(21)
    pts = rng.random((90, 2))
    pts[:, 0] = x + pts[:, 0] * w
    pts[:, 1] = y + pts[:, 1] * h
    ax.scatter(pts[:, 0], pts[:, 1], s=2.1, c="#CCD2D8", linewidths=0)
    sel = pts[(pts[:, 0] > x + 0.56 * w) & (pts[:, 1] > y + 0.30 * h)]
    ax.scatter(sel[:, 0], sel[:, 1], s=3.2, c=COLORS["green"], linewidths=0)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["edge"], linewidth=0.8))


def label_list(ax, x: float, y: float, lines: Iterable[str], *, color: str = COLORS["muted"]) -> None:
    for i, line in enumerate(lines):
        ax.text(x, y - 0.026 * i, line, ha="left", va="top", fontsize=7.6, color=color)


def main() -> None:
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(15.8, 6.9), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(
        0.035,
        0.955,
        "CTF-GS: one compact foundation-feature Gaussian map",
        ha="left",
        va="top",
        fontsize=17.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.035,
        0.917,
        "Frozen foundation heads supervise training; the deployed scene reads the same compact map as 2D features, 3D primitives, and 3D points.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=COLORS["muted"],
    )

    add_panel_label(ax, 0.045, 0.840, "Offline supervision")
    add_panel_label(ax, 0.300, 0.840, "Stored compact map")
    add_panel_label(ax, 0.600, 0.840, "Global readout heads")
    add_panel_label(ax, 0.800, 0.840, "Protocol evidence")

    # Left column.
    add_box(ax, (0.045, 0.690), (0.195, 0.092), "posed RGB views\n+ 3DGS geometry", color=COLORS["input"], fontsize=8.8, weight="bold")
    add_box(ax, (0.045, 0.560), (0.195, 0.092), "dense RADIO teacher\n1280-D features", color=COLORS["teacher"], fontsize=8.4)
    add_box(ax, (0.045, 0.430), (0.195, 0.092), "frozen task heads\nSigLIP2 / DINO / SAM3", color=COLORS["teacher"], fontsize=8.0)
    add_box(ax, (0.045, 0.300), (0.195, 0.092), "registered multiview\nsupport teacher (VPR)", color=COLORS["teacher"], fontsize=8.1)

    # Central compact map.
    field = add_box(ax, (0.300, 0.365), (0.245, 0.390), "", color=COLORS["field"], lw=1.35, radius=0.026)
    ax.text(0.4225, 0.722, "Hybrid Gaussian Code Field", ha="center", va="center", fontsize=11.2, fontweight="bold", color=COLORS["ink"])
    ax.text(0.4225, 0.694, "stored once per scene", ha="center", va="center", fontsize=7.8, color=COLORS["muted"])
    add_gaussian_cloud(ax, 0.335, 0.505, 0.175, 0.150)
    add_box(ax, (0.325, 0.440), (0.195, 0.038), r"$z_i$: compact per-Gaussian code", color="#F8FCFA", fontsize=7.7, radius=0.012, lw=0.75)
    add_box(ax, (0.325, 0.392), (0.195, 0.038), r"$h(x)$: voxel / spatial context", color="#F8FCFA", fontsize=7.7, radius=0.012, lw=0.75)

    # Shared heads.
    add_box(ax, (0.600, 0.665), (0.145, 0.078), "Compact-to-Teacher\nDecoder", color=COLORS["field"], fontsize=8.4, weight="bold")
    add_box(ax, (0.600, 0.535), (0.145, 0.078), "View-Space\nAligner", color=COLORS["field"], fontsize=8.3)
    add_box(ax, (0.600, 0.405), (0.145, 0.078), "Support-Aware\nPrimitive Policy", color=COLORS["field_deep"], fontsize=8.2)
    add_box(ax, (0.600, 0.275), (0.145, 0.078), "Frozen-Head\nAdaptors", color=COLORS["field"], fontsize=8.2)

    # Right column with visual output icons.
    readouts = [
        (0.805, 0.670, COLORS["blue"], "LERF rendered-view OVS", "2D heatmap / mask"),
        (0.805, 0.515, COLORS["green"], "LERF direct 3D OVS", "primitive support"),
        (0.805, 0.360, COLORS["purple"], "ScanNet point query", "open-vocabulary points"),
        (0.805, 0.205, COLORS["orange"], "Frozen-head probes", "SigLIP2 / DINO / SAM3"),
    ]
    for x, y, c, title, subtitle in readouts:
        add_box(ax, (x, y), (0.155, 0.100), "", color=COLORS["head"], lw=1.05)
        ax.add_patch(Rectangle((x + 0.010, y + 0.017), 0.006, 0.066, facecolor=c, edgecolor="none"))
        ax.text(x + 0.024, y + 0.065, title, ha="left", va="center", fontsize=7.9, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.024, y + 0.037, subtitle, ha="left", va="center", fontsize=7.1, color=COLORS["muted"])

    add_heatmap_icon(ax, 0.966, 0.685, 0.030, 0.064)
    add_selection_icon(ax, 0.966, 0.530, 0.030, 0.064)
    add_point_icon(ax, 0.966, 0.375, 0.030, 0.064)
    add_heatmap_icon(ax, 0.966, 0.220, 0.030, 0.064)

    # Training-only band.
    add_box(ax, (0.300, 0.095), (0.445, 0.135), "", color=COLORS["train"], lw=1.0, radius=0.020)
    ax.text(0.320, 0.195, "Training-only constraints", ha="left", va="center", fontsize=9.0, fontweight="bold", color=COLORS["ink"])
    label_list(
        ax,
        0.320,
        0.166,
        [
            "feature reconstruction + frozen-head consistency",
            "visibility-aware support distillation from registered VPR",
            "geometry regularization + SAM3 mask-logit/boundary supervision",
        ],
        color=COLORS["ink"],
    )

    # Inference arrows.
    add_arrow(ax, (0.240, 0.735), (0.300, 0.615), label="initialize", lw=1.25)
    add_arrow(ax, (0.545, 0.650), (0.600, 0.705), label="decode", lw=1.25)
    add_arrow(ax, (0.672, 0.665), (0.672, 0.613), label="align", lw=1.15, label_offset=0.006)
    add_arrow(ax, (0.745, 0.704), (0.805, 0.720), lw=1.35)
    add_arrow(ax, (0.745, 0.444), (0.805, 0.565), lw=1.35)
    add_arrow(ax, (0.545, 0.500), (0.600, 0.445), label="score", lw=1.15, label_offset=0.006)
    add_elbow_arrow(
        ax,
        [(0.545, 0.392), (0.575, 0.372), (0.780, 0.372), (0.805, 0.410)],
        label="point query",
        lw=1.15,
    )
    add_arrow(ax, (0.745, 0.314), (0.805, 0.255), lw=1.25)

    # Training arrows.
    dashed = dict(style="dashed", color=COLORS["muted"], lw=1.10, scale=11.0)
    add_arrow(ax, (0.240, 0.606), (0.300, 0.170), label="teacher loss", rad=-0.18, **dashed)
    add_arrow(ax, (0.240, 0.476), (0.300, 0.150), label="head loss", rad=-0.10, **dashed)
    add_arrow(ax, (0.240, 0.346), (0.300, 0.128), label="support loss", rad=-0.02, **dashed)
    add_arrow(ax, (0.522, 0.230), (0.430, 0.365), label="", rad=0.10, **dashed)
    add_arrow(ax, (0.708, 0.230), (0.674, 0.405), label="", rad=-0.05, **dashed)

    # Protocol note kept short; details live in the figure caption.
    ax.text(
        0.800,
        0.122,
        "Dashed paths supervise training only; deployed readouts share the stored compact map.",
        ha="left",
        va="center",
        fontsize=7.4,
        color=COLORS["muted"],
    )

    handles = [
        Line2D([0], [0], color=COLORS["edge"], lw=1.5, linestyle="solid", label="inference"),
        Line2D([0], [0], color=COLORS["muted"], lw=1.2, linestyle="dashed", label="training only"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.045, 0.095), frameon=False, fontsize=8.0, handlelength=2.6)

    field.set_linewidth(1.55)
    fig.savefig(out_dir / "radio_gs_framework.png", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_dir / "radio_gs_framework.pdf", bbox_inches="tight", pad_inches=0.025)


if __name__ == "__main__":
    main()
