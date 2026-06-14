#!/usr/bin/env python3
"""Draw paper-ready method figures for CTF-GS.

The script intentionally mixes vector structure with real qualitative
thumbnails. This avoids a box-only framework figure while keeping labels,
arrows, and module boundaries editable in PDF/SVG outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"

COLORS = {
    "ink": "#17212b",
    "muted": "#5f6b7a",
    "line": "#95a1ad",
    "panel": "#fbfbf7",
    "blue": "#2f6fb3",
    "teal": "#159a8c",
    "green": "#3aa66b",
    "orange": "#d97835",
    "purple": "#7665c9",
    "yellow": "#f0c35a",
    "red": "#c94f4f",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def read_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("L")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def read_wide_column(path: Path, *, index: int = 5, columns: int = 6) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    if image.width < columns:
        return np.asarray(image)
    panel_w = image.width // columns
    x0 = min(index, columns - 1) * panel_w
    x1 = image.width if index == columns - 1 else x0 + panel_w
    return np.asarray(image.crop((x0, 0, x1, image.height)))


def crop_to_aspect(image: np.ndarray, aspect: float) -> np.ndarray:
    height, width = image.shape[:2]
    current = width / max(height, 1)
    if current > aspect:
        new_w = int(round(height * aspect))
        x0 = max(0, (width - new_w) // 2)
        return image[:, x0 : x0 + new_w]
    new_h = int(round(width / aspect))
    y0 = max(0, (height - new_h) // 2)
    return image[y0 : y0 + new_h, :]


def selected_support(rgb: np.ndarray | None, mask: np.ndarray | None, color: str) -> np.ndarray:
    if rgb is None:
        return synthetic_support(color)
    output = np.full_like(rgb, 255)
    if mask is None or not np.any(mask):
        return output
    tint = np.array(hex_to_rgb(color), dtype=np.float32).reshape(1, 1, 3)
    pixels = rgb.astype(np.float32)
    output[mask] = (0.72 * pixels[mask] + 0.28 * tint).astype(np.uint8)
    return output


def synthetic_support(color: str, size: tuple[int, int] = (200, 280)) -> np.ndarray:
    height, width = size
    yy, xx = np.ogrid[:height, :width]
    cx, cy = width * 0.55, height * 0.52
    mask = ((xx - cx) ** 2 / (width * 0.18) ** 2 + (yy - cy) ** 2 / (height * 0.24) ** 2) < 1
    output = np.full((height, width, 3), 255, dtype=np.uint8)
    output[mask] = np.array(hex_to_rgb(color), dtype=np.uint8)
    return output


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def draw_image(ax, image: np.ndarray | None, x: float, y: float, w: float, h: float, *, border: str = "#d5dbe3", lw: float = 0.8) -> None:
    if image is None:
        image = synthetic_support("#d8dde6")
    image = crop_to_aspect(image, w / h)
    ax.imshow(image, extent=(x, x + w, y, y + h), zorder=2)
    ax.set_aspect("auto")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=border, linewidth=lw, zorder=3))


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str = "#33424f",
    lw: float = 1.0,
    radius: float = 0.018,
    z: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x: float, y: float, text: str, *, size: float = 9, weight: str = "normal", color: str = COLORS["ink"], ha: str = "left", va: str = "center") -> None:
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=5)


def text_box(ax, x: float, y: float, w: float, h: float, title: str, body: str = "", *, fc: str = "#ffffff", stripe: str | None = None, size: float = 8.5) -> None:
    rounded_box(ax, x, y, w, h, fc=fc, ec="#9aa6b2", lw=0.9, radius=0.012)
    if stripe is not None:
        ax.add_patch(Rectangle((x, y), 0.008, h, facecolor=stripe, edgecolor="none", zorder=4))
    if body and h >= 0.052:
        label(ax, x + 0.018, y + h * 0.63, title, size=size, weight="bold")
        label(ax, x + 0.018, y + h * 0.28, body, size=max(5.8, size - 1.8), color=COLORS["muted"])
    else:
        label(ax, x + 0.018, y + h * 0.50, title, size=size, weight="bold")


def arrow(ax, xy1: tuple[float, float], xy2: tuple[float, float], *, color: str = "#33424f", lw: float = 1.4, dashed: bool = False, text: str | None = None, tpos: float = 0.5) -> None:
    patch = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        zorder=4,
    )
    ax.add_patch(patch)
    if text:
        tx = xy1[0] * (1 - tpos) + xy2[0] * tpos
        ty = xy1[1] * (1 - tpos) + xy2[1] * tpos
        label(ax, tx, ty + 0.018, text, size=7.5, color=color, ha="center")


def draw_memory(ax, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, fc="#eaf6ef", ec="#24473d", lw=1.2, radius=0.018)
    compact = h < 0.36
    label(
        ax,
        x + 0.018,
        y + h - 0.045,
        "Compact Gaussian Memory" if compact else "Compact Foundation-Feature Memory",
        size=8.7 if compact else 10.5,
        weight="bold",
    )
    if not compact:
        label(ax, x + 0.018, y + h - 0.078, "stored once per scene; reconstruct features on demand", size=7.5, color=COLORS["muted"])
    rng = np.random.default_rng(9)
    bottom_margin = 0.115 if compact else 0.20
    top_margin = 0.075 if compact else 0.13
    low = np.array([x + 0.045, y + min(bottom_margin, h * 0.45)])
    high = np.array([x + w - 0.045, y + h - min(top_margin, h * 0.35)])
    if np.any(high <= low):
        low = np.array([x + 0.045, y + h * 0.33])
        high = np.array([x + w - 0.045, y + h * 0.66])
    count = 20 if compact else 34
    centers = rng.uniform(low, high, size=(count, 2))
    palette = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["yellow"]]
    for idx, (cx, cy) in enumerate(centers):
        ell = Ellipse(
            (cx, cy),
            rng.uniform(0.014, 0.026) if compact else rng.uniform(0.018, 0.035),
            rng.uniform(0.008, 0.018) if compact else rng.uniform(0.010, 0.024),
            angle=rng.uniform(-25, 25),
            facecolor=palette[idx % len(palette)],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.82,
            zorder=3,
        )
        ax.add_patch(ell)
    if compact:
        text_box(ax, x + 0.030, y + 0.035, w - 0.060, 0.045, "z_i + h(x)", "compact", fc="#f8fffb", stripe=COLORS["teal"], size=7.6)
    else:
        text_box(ax, x + 0.035, y + 0.050, w - 0.070, 0.055, "compact code z_i + spatial context h(x)", "not explicit 1280-D feature storage", fc="#f8fffb", stripe=COLORS["teal"], size=8.5)


def load_assets() -> dict[str, np.ndarray | None]:
    rgb = read_rgb(Path("/mnt/pool/sqy/3d_understanding/lerf_ovs/label/figurines/frame_00105.jpg"))
    rgb_t = read_rgb(Path("/mnt/pool/sqy/3d_understanding/lerf_ovs/label/teatime/frame_00140.jpg"))
    rgb_w = read_rgb(Path("/mnt/pool/sqy/3d_understanding/lerf_ovs/label/waldo_kitchen/frame_00140.jpg"))
    rendered = read_wide_column(
        REPO_ROOT
        / "output/radio_gs/freeze_eval/lerf_figurines_overlay_calibrated_thr0p60_vis_20260514/visualisations/figurines/lerf_grounding_frame_00105_rendered_pumpkin.png"
    )
    mask = read_mask(
        REPO_ROOT
        / "output/radio_gs/lerf_direct3d_prompt_ensemble_policy_masks_20260528/pred_masks/thr0p65/figurines/frame_00105_pumpkin.png",
        rgb.shape[:2] if rgb is not None else (728, 986),
    )
    direct = selected_support(rgb, mask, COLORS["orange"])
    scannet = read_rgb(REPO_ROOT / "paper/figures/scannet_openvocab_3d_query_qualitative.png")
    downstream = read_rgb(REPO_ROOT / "paper/figures/lerf_adaptor_downstream_qualitative.png")
    if downstream is None:
        downstream = read_rgb(REPO_ROOT / "paper/figures/lerf_sam_dino_tasks_qualitative.png")
    return {
        "rgb": rgb,
        "rgb_t": rgb_t,
        "rgb_w": rgb_w,
        "rendered": rendered,
        "direct": direct,
        "scannet": scannet,
        "downstream": downstream,
    }


def draw_figure1(assets: dict[str, np.ndarray | None]) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfbf7")
    ax.set_facecolor("#fbfbf7")

    label(ax, 0.03, 0.955, "CTF-GS: Compact Foundation-Feature Gaussian Memory", size=14.5, weight="bold")
    label(ax, 0.03, 0.918, "Multiview evidence is compressed into a compact 3D memory, then query-conditioned feature/support reconstruction serves 2D and 3D open-vocabulary tasks.", size=8.5, color=COLORS["muted"])

    label(ax, 0.045, 0.860, "A. Multiview Evidence", size=10.5, weight="bold")
    rounded_box(ax, 0.025, 0.105, 0.270, 0.720, fc="#f4f8fd", ec="#9fb4cc", lw=1.0, radius=0.014)
    draw_image(ax, assets["rgb_w"], 0.050, 0.615, 0.105, 0.110, border="#ffffff", lw=1.0)
    draw_image(ax, assets["rgb_t"], 0.095, 0.570, 0.105, 0.110, border="#ffffff", lw=1.0)
    draw_image(ax, assets["rgb"], 0.140, 0.525, 0.105, 0.110, border="#ffffff", lw=1.0)
    label(ax, 0.052, 0.492, "posed RGB views + 3DGS geometry", size=8.5, weight="bold")
    text_box(ax, 0.050, 0.400, 0.210, 0.070, "Frame-wise foundation features", "RADIO / SigLIP2 / DINO / SAM", fc="#ffffff", stripe=COLORS["blue"])
    text_box(ax, 0.050, 0.302, 0.210, 0.070, "Multiview primitive registration", "visibility, contribution, confidence", fc="#ffffff", stripe=COLORS["teal"])
    text_box(ax, 0.050, 0.205, 0.210, 0.070, "Training constraints", "feature, support, boundary, topology", fc="#fff8e8", stripe=COLORS["orange"])

    draw_memory(ax, 0.365, 0.195, 0.285, 0.630)
    label(ax, 0.405, 0.860, "B. Compact Scene Memory", size=10.5, weight="bold")
    arrow(ax, (0.295, 0.520), (0.365, 0.520), color=COLORS["teal"], text="compress")
    arrow(ax, (0.280, 0.255), (0.420, 0.205), color=COLORS["muted"], dashed=True, text="training only", tpos=0.45)

    text_box(ax, 0.685, 0.690, 0.280, 0.120, "LERF rendered-view OVS", "dense score map -> 2D mask", fc="#f5f1ff", stripe=COLORS["purple"])
    draw_image(ax, assets["rendered"], 0.865, 0.705, 0.085, 0.075)
    text_box(ax, 0.685, 0.515, 0.280, 0.120, "LERF direct 3D OVS", "primitive scores -> selected mask", fc="#eefaf6", stripe=COLORS["teal"])
    draw_image(ax, assets["direct"], 0.865, 0.530, 0.085, 0.075)
    text_box(ax, 0.685, 0.340, 0.280, 0.120, "ScanNet point query", "open-vocabulary points", fc="#f4f1ff", stripe=COLORS["blue"])
    draw_image(ax, assets["scannet"], 0.865, 0.355, 0.085, 0.075)
    text_box(ax, 0.685, 0.165, 0.280, 0.120, "Frozen-head probes", "SigLIP2 / SAM / DINO probes", fc="#fff4ed", stripe=COLORS["orange"])
    draw_image(ax, assets["downstream"], 0.865, 0.180, 0.085, 0.075)
    label(ax, 0.705, 0.860, "C. Query-Conditioned Tasks", size=10.5, weight="bold")

    arrow(ax, (0.650, 0.575), (0.685, 0.750), color=COLORS["purple"], text="reconstruct")
    arrow(ax, (0.650, 0.515), (0.685, 0.575), color=COLORS["teal"], text="calibrate")
    arrow(ax, (0.650, 0.455), (0.685, 0.400), color=COLORS["blue"], text="query")
    arrow(ax, (0.650, 0.395), (0.685, 0.225), color=COLORS["orange"], text="probe")

    ax.plot([0.065, 0.105], [0.138, 0.138], color=COLORS["ink"], linewidth=1.5)
    arrow(ax, (0.065, 0.138), (0.105, 0.138), color=COLORS["ink"], lw=1.2)
    label(ax, 0.118, 0.138, "inference", size=7.5, color=COLORS["muted"])
    ax.plot([0.190, 0.230], [0.138, 0.138], color=COLORS["muted"], linewidth=1.3, linestyle=(0, (4, 3)))
    label(ax, 0.242, 0.138, "training only", size=7.5, color=COLORS["muted"])

    save_all(fig, "figure1_overall_framework")


def draw_panel_frame(ax, x: float, y: float, w: float, h: float, title: str) -> None:
    rounded_box(ax, x, y, w, h, fc="#ffffff", ec="#c9d1da", lw=0.9, radius=0.012)
    label(ax, x + 0.018, y + h - 0.035, title, size=10.2, weight="bold")


def draw_figure2(assets: dict[str, np.ndarray | None]) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfbf7")
    ax.set_facecolor("#fbfbf7")
    label(ax, 0.03, 0.955, "Method Details", size=15, weight="bold")
    label(ax, 0.03, 0.920, "The same compact memory is trained by multiview primitive evidence, reconstructed into foundation-feature spaces, and calibrated into stable object support.", size=8.5, color=COLORS["muted"])

    draw_panel_frame(ax, 0.030, 0.090, 0.435, 0.790, "(a) Multiview Primitive Registration")
    draw_image(ax, assets["rgb_w"], 0.060, 0.685, 0.100, 0.100)
    draw_image(ax, assets["rgb_t"], 0.185, 0.705, 0.100, 0.100)
    draw_image(ax, assets["rgb"], 0.310, 0.685, 0.100, 0.100)
    for cx in [0.110, 0.235, 0.360]:
        ax.add_patch(Circle((cx, 0.635), 0.018, facecolor="#ffffff", edgecolor=COLORS["ink"], linewidth=1.0, zorder=3))
        ax.plot([cx - 0.030, cx, cx + 0.030], [0.590, 0.635, 0.590], color=COLORS["ink"], linewidth=0.9, zorder=3)
        arrow(ax, (cx, 0.590), (0.248, 0.455), color=COLORS["teal"], lw=1.0, dashed=True)
    draw_memory(ax, 0.125, 0.295, 0.245, 0.225)
    label(ax, 0.105, 0.560, "visible primitive evidence", size=8.5, color=COLORS["teal"], weight="bold")
    text_box(ax, 0.060, 0.185, 0.160, 0.070, "visibility weights", "view_count, alpha, depth", fc="#f4fbfa", stripe=COLORS["teal"])
    text_box(ax, 0.245, 0.185, 0.170, 0.070, "registered target", "primitive feature + confidence", fc="#fff8e8", stripe=COLORS["orange"])
    arrow(ax, (0.210, 0.300), (0.145, 0.255), color=COLORS["teal"], dashed=True)
    arrow(ax, (0.285, 0.300), (0.330, 0.255), color=COLORS["orange"], dashed=True)
    label(ax, 0.060, 0.130, "Training only: multiview evidence is distilled into compact codes; no VPR cache is read at inference.", size=7.5, color=COLORS["muted"])

    draw_panel_frame(ax, 0.515, 0.540, 0.455, 0.340, "(b) Query-Conditioned Feature Reconstruction")
    rounded_box(ax, 0.550, 0.655, 0.135, 0.105, fc="#eaf6ef", ec="#24473d", lw=1.0)
    label(ax, 0.570, 0.725, "compact z_i", size=8.5, weight="bold")
    label(ax, 0.570, 0.690, "+ context h(x)", size=7.7, color=COLORS["muted"])
    text_box(ax, 0.735, 0.690, 0.180, 0.060, "feature decoder", "reconstruct task space", fc="#f6fbff", stripe=COLORS["blue"])
    arrow(ax, (0.685, 0.707), (0.735, 0.720), color=COLORS["blue"], text="decode")
    for idx, (name, col) in enumerate([("SigLIP2", COLORS["purple"]), ("SAM", COLORS["green"]), ("DINO", COLORS["orange"])]):
        y = 0.620 - idx * 0.048
        text_box(ax, 0.735, y, 0.180, 0.038, name + " consistency", "", fc="#ffffff", stripe=col, size=7.6)
    text_box(ax, 0.550, 0.565, 0.135, 0.055, "text/query q", "open vocabulary", fc="#fff4ed", stripe=COLORS["orange"], size=7.8)
    arrow(ax, (0.685, 0.593), (0.735, 0.640), color=COLORS["orange"], lw=1.1)
    label(ax, 0.550, 0.815, "A low-dimensional memory reconstructs features only when a query/task needs them.", size=7.7, color=COLORS["muted"])

    draw_panel_frame(ax, 0.515, 0.090, 0.455, 0.395, "(c) Support-Calibrated Selection")
    text_box(ax, 0.545, 0.365, 0.135, 0.055, "score primitives", "cos(f_i, t_q)", fc="#ffffff", stripe=COLORS["purple"], size=7.8)
    text_box(ax, 0.545, 0.280, 0.135, 0.055, "support policy", "visibility + confidence", fc="#f4fbfa", stripe=COLORS["teal"], size=7.8)
    text_box(ax, 0.545, 0.195, 0.135, 0.055, "component guard", "complete object support", fc="#fff8e8", stripe=COLORS["orange"], size=7.8)
    arrow(ax, (0.612, 0.365), (0.612, 0.335), color=COLORS["ink"])
    arrow(ax, (0.612, 0.280), (0.612, 0.250), color=COLORS["ink"])

    xs = np.linspace(0.720, 0.860, 16)
    heights = np.array([0.02, 0.028, 0.035, 0.050, 0.070, 0.095, 0.130, 0.115, 0.080, 0.060, 0.048, 0.035, 0.026, 0.022, 0.018, 0.014])
    for x, h in zip(xs, heights):
        ax.add_patch(Rectangle((x, 0.315), 0.006, h, facecolor=COLORS["purple"], edgecolor="none", alpha=0.75, zorder=3))
    label(ax, 0.720, 0.470, "query score distribution", size=7.7, color=COLORS["muted"])
    draw_image(ax, assets["rendered"], 0.715, 0.150, 0.100, 0.080)
    label(ax, 0.715, 0.245, "2D score map", size=7.5, weight="bold")
    draw_image(ax, assets["direct"], 0.840, 0.150, 0.100, 0.080)
    label(ax, 0.840, 0.245, "3D selected support", size=7.5, weight="bold")
    arrow(ax, (0.680, 0.308), (0.735, 0.228), color=COLORS["purple"], text="render")
    arrow(ax, (0.680, 0.225), (0.855, 0.228), color=COLORS["teal"], text="select")
    label(ax, 0.720, 0.115, "The calibration acts on support, not just feature cosine, reducing fragmented or low-visibility selections.", size=7.4, color=COLORS["muted"])

    save_all(fig, "figure2_method_details")


def save_all(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    assets = load_assets()
    draw_figure1(assets)
    draw_figure2(assets)
    print(f"Wrote {FIG_DIR / 'figure1_overall_framework.pdf'}")
    print(f"Wrote {FIG_DIR / 'figure1_overall_framework.svg'}")
    print(f"Wrote {FIG_DIR / 'figure1_overall_framework.png'}")
    print(f"Wrote {FIG_DIR / 'figure2_method_details.pdf'}")
    print(f"Wrote {FIG_DIR / 'figure2_method_details.svg'}")
    print(f"Wrote {FIG_DIR / 'figure2_method_details.png'}")


if __name__ == "__main__":
    main()
