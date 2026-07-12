#!/usr/bin/env python3
"""
Extract and save RADIO features for all training images in a scene.

Usage:
    python radio_gs/scripts/extract_radio_features.py \
        --scene room_0 \
        --image_dir dataset/room_0/Sequence_1/rgb/ \
        --output_dir output/radio_features/room_0/ \
        --radio_repo /root/RADIO \
        --radio_version c-radio_v4-h \
        --batch_size 4 \
        --extract_adaptors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.data.benchmark_paths import extract_feature_frame_index
from radio_gs.data.view_split import (
    load_excluded_image_stems,
    select_image_indices,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRAME_ID_MODES = ("auto", "source_rank")


# ---- image loading helpers ------------------------------------------------

def _collect_image_paths(image_dir: str) -> tuple[list[Path], str]:
    """Return image paths with numeric frame-order when indices are parseable."""
    paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    indexed: list[tuple[int, Path]] = []
    for path in paths:
        try:
            indexed.append((extract_feature_frame_index(path), path))
        except ValueError:
            indexed = []
            break

    if indexed:
        # Numeric suffixes are not unique in several NVOS captures (for
        # example multiple Horns filenames end in ``_427``).  A filename
        # tiebreaker makes source_rank stable across processes/filesystems.
        indexed.sort(key=lambda item: (item[0], item[1].name))
        return [path for _, path in indexed], "numeric_then_exact_filename"

    return sorted(paths, key=lambda path: path.name), "lexicographic_filename"


def _apply_subsampling(
    image_paths: list[Path],
    frame_stride: int,
    max_frames: int | None,
) -> list[Path]:
    sampled = image_paths[:: max(1, frame_stride)]
    if max_frames is not None:
        sampled = sampled[:max_frames]
    return sampled


def _saved_frame_indices(
    image_paths: list[Path],
    *,
    mode: str,
) -> list[int]:
    """Resolve output ids once and reject any would-be cache overwrite."""
    if mode not in FRAME_ID_MODES:
        raise ValueError(f"frame_id_mode must be one of {FRAME_ID_MODES}")
    frame_indices: list[int] = []
    by_index: dict[int, Path] = {}
    for source_rank, source_path in enumerate(image_paths):
        if mode == "source_rank":
            frame_idx = source_rank
        else:
            try:
                frame_idx = extract_feature_frame_index(source_path)
            except ValueError:
                frame_idx = source_rank
        if frame_idx in by_index:
            raise ValueError(
                f"Feature output collision at rgb_{frame_idx}.pt: "
                f"{by_index[frame_idx].name} and {source_path.name}. "
                "Use --frame-id-mode source_rank for a unique dense index."
            )
        by_index[frame_idx] = source_path
        frame_indices.append(frame_idx)
    return frame_indices


def _nearest_radio_resolution(h: int, w: int, patch_size: int = 16) -> tuple[int, int]:
    """Round h, w to nearest multiples of *patch_size*."""
    return (
        max(patch_size, round(h / patch_size) * patch_size),
        max(patch_size, round(w / patch_size) * patch_size),
    )


def _compute_scaled_radio_resolution(
    h: int,
    w: int,
    resolution_scale: float,
    patch_size: int = 16,
) -> tuple[int, int]:
    """Scale an image size and snap it to RADIO patch multiples."""
    if resolution_scale <= 0:
        raise ValueError("--resolution_scale must be positive")
    return _nearest_radio_resolution(
        h * resolution_scale,
        w * resolution_scale,
        patch_size=patch_size,
    )


def _load_and_preprocess(
    paths: list[Path],
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Load images, resize, normalize to [0, 1] and stack → (B, 3, H, W)."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0)
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _sliding_starts(length: int, tile_size: int, tile_overlap: int) -> list[int]:
    """Return start indices that cover *length* with overlapping tiles."""
    if tile_size <= 0:
        raise ValueError("--tile_size must be positive")
    if tile_overlap < 0:
        raise ValueError("--tile_overlap must be non-negative")
    if tile_overlap >= tile_size:
        raise ValueError("--tile_overlap must be smaller than --tile_size")
    if length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _stitch_sliding_window_features(
    output_shape: tuple[int, int, int, int] | torch.Size,
    tiles: list[tuple[int, int, torch.Tensor]],
) -> torch.Tensor:
    """Average overlapping feature tiles into one feature map.

    Tile coordinates are in feature-grid units, not input-pixel units.
    """
    if not tiles:
        raise ValueError("No sliding-window tiles to stitch")
    first = tiles[0][2]
    accum = first.new_zeros(tuple(output_shape), dtype=torch.float32)
    weight = first.new_zeros(tuple(output_shape), dtype=torch.float32)

    for top, left, feat in tiles:
        _, _, h, w = feat.shape
        accum[:, :, top : top + h, left : left + w] += feat.float()
        weight[:, :, top : top + h, left : left + w] += 1.0

    return accum / weight.clamp_min(1.0)


def _parse_adaptor_names(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names


def _adaptor_output_subdir(name: str) -> str:
    if name == "siglip2-g":
        return "siglip2"
    return name.replace("-", "_")


def _split_radio_output_pair(value) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(value):
        return None, value
    if isinstance(value, dict):
        summary = value.get("summary")
        features = value.get("features")
        if features is None:
            features = value.get("spatial")
        if features is None:
            raise ValueError("RADIO output dict must contain features/spatial")
        return summary, features
    if hasattr(value, "summary") and hasattr(value, "features"):
        return value.summary, value.features
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    raise TypeError(f"Unsupported RADIO output value type: {type(value)!r}")


def _spatial_to_feature_grid(
    spatial: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    if spatial.ndim == 4:
        return spatial
    if spatial.ndim != 3:
        raise ValueError(f"Expected RADIO spatial features as [B,N,D] or [B,D,H,W], got {tuple(spatial.shape)}")
    B, _, D = spatial.shape
    return spatial.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)


def _unpack_radio_output(
    output,
    patch_h: int,
    patch_w: int,
    adaptor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Convert RADIO output into summary, backbone grid, adaptor grids."""
    if isinstance(output, dict):
        if "backbone" not in output:
            raise ValueError("RADIO dict output is missing the 'backbone' entry")
        summary, spatial = _split_radio_output_pair(output["backbone"])
        adaptor_outputs = {
            name: value for name, value in output.items() if name != "backbone"
        }
    elif isinstance(output, tuple) and len(output) == 2:
        summary, spatial = _split_radio_output_pair(output)
        adaptor_outputs = {}
    else:
        summary, spatial, adaptor_outputs = output

    spatial_2d = _spatial_to_feature_grid(spatial, patch_h, patch_w)

    adaptor_2d: dict[str, torch.Tensor] = {}
    for name in adaptor_names or []:
        if not adaptor_outputs or name not in adaptor_outputs:
            continue
        ad_out = adaptor_outputs[name]
        _, ad_spatial = _split_radio_output_pair(ad_out)
        adaptor_2d[name] = _spatial_to_feature_grid(ad_spatial, patch_h, patch_w)
    return summary, spatial_2d, adaptor_2d


def _run_radio_batch(
    model,
    conditioner,
    imgs: torch.Tensor,
    amp: bool,
    patch_h: int,
    patch_w: int,
    adaptor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    imgs = conditioner(imgs)
    with torch.cuda.amp.autocast(enabled=amp):
        output = model(imgs)
    return _unpack_radio_output(output, patch_h, patch_w, adaptor_names=adaptor_names)


def _extract_sliding_window_single(
    model,
    conditioner,
    img: torch.Tensor,
    amp: bool,
    tile_size: int,
    tile_overlap: int,
    patch_size: int = 16,
    adaptor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Extract and stitch features for a single preprocessed image tensor."""
    if img.shape[0] != 1:
        raise ValueError("Sliding-window extraction expects single-image batches")

    _, _, target_h, target_w = img.shape
    tile_size = _nearest_radio_resolution(tile_size, tile_size, patch_size)[0]
    tile_overlap = max(0, round(tile_overlap / patch_size) * patch_size)

    if target_h <= tile_size and target_w <= tile_size:
        return _run_radio_batch(
            model,
            conditioner,
            img,
            amp,
            target_h // patch_size,
            target_w // patch_size,
            adaptor_names=adaptor_names,
        )

    row_tile = min(tile_size, target_h)
    col_tile = min(tile_size, target_w)
    row_overlap = min(tile_overlap, max(0, row_tile - patch_size))
    col_overlap = min(tile_overlap, max(0, col_tile - patch_size))
    row_starts = _sliding_starts(target_h, row_tile, row_overlap)
    col_starts = _sliding_starts(target_w, col_tile, col_overlap)
    summaries: list[torch.Tensor] = []
    backbone_tiles: list[tuple[int, int, torch.Tensor]] = []
    adaptor_tiles: dict[str, list[tuple[int, int, torch.Tensor]]] = {}

    for top in row_starts:
        for left in col_starts:
            bottom = min(top + row_tile, target_h)
            right = min(left + col_tile, target_w)
            top = bottom - row_tile
            left = right - col_tile
            tile = img[:, :, top:bottom, left:right]
            patch_h = tile.shape[-2] // patch_size
            patch_w = tile.shape[-1] // patch_size
            summary, spatial_2d, adaptors = _run_radio_batch(
                model,
                conditioner,
                tile,
                amp,
                patch_h,
                patch_w,
                adaptor_names=adaptor_names,
            )
            summaries.append(summary)
            grid_top = top // patch_size
            grid_left = left // patch_size
            backbone_tiles.append((grid_top, grid_left, spatial_2d))
            for name, value in adaptors.items():
                adaptor_tiles.setdefault(name, []).append((grid_top, grid_left, value))

    full_shape = (
        1,
        backbone_tiles[0][2].shape[1],
        target_h // patch_size,
        target_w // patch_size,
    )
    spatial_full = _stitch_sliding_window_features(full_shape, backbone_tiles)
    adaptor_full = {
        name: _stitch_sliding_window_features(
            (1, tiles[0][2].shape[1], target_h // patch_size, target_w // patch_size),
            tiles,
        )
        for name, tiles in adaptor_tiles.items()
    }
    return torch.stack(summaries, dim=0).mean(dim=0), spatial_full, adaptor_full


# ---- model loading --------------------------------------------------------

def _load_radio_model(
    radio_repo: str,
    radio_version: str,
    adaptor_names: list[str] | None,
    device: torch.device,
):
    """Load RADIO via torch.hub and return (model, input_conditioner)."""
    kwargs: dict = {"source": "local"}
    if adaptor_names:
        kwargs["adaptor_names"] = adaptor_names

    model = torch.hub.load(radio_repo, "radio_model", version=radio_version, **kwargs)
    model = model.to(device).eval()

    input_conditioner = model.make_preprocessor_external()

    return model, input_conditioner


# ---- PCA statistics -------------------------------------------------------

def _compute_pca_stats(
    all_features: list[torch.Tensor],
    n_components: int = 64,
) -> dict[str, torch.Tensor]:
    """Compute mean, std, and top-*n_components* PCA components from features.

    Args:
        all_features: list of (C, Hp, Wp) tensors (float32).
        n_components: number of PCA components to keep.

    Returns:
        dict with 'mean' (C,), 'std' (C,), 'components_64' (n_components, C).
    """
    # Stack all spatial tokens: (N_total, C)
    pixels = torch.cat([f.reshape(f.shape[0], -1).T for f in all_features], dim=0)
    mean = pixels.mean(dim=0)
    std = pixels.std(dim=0).clamp(min=1e-6)
    centered = pixels - mean
    # Economy SVD — only need top components
    k = min(n_components, centered.shape[0], centered.shape[1])
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:k]  # (k, C)
    return {"mean": mean, "std": std, "components_64": components}


# ---- main extraction loop -------------------------------------------------

@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine adaptor setup
    adaptor_names: list[str] | None = None
    if args.extract_adaptors:
        adaptor_names = _parse_adaptor_names(args.adaptor_names)
        print(f"[RADIO] Extracting with adaptors: {adaptor_names}")

    # Load model
    print(f"[RADIO] Loading {args.radio_version} from {args.radio_repo} …")
    model, conditioner = _load_radio_model(
        args.radio_repo, args.radio_version, adaptor_names, device
    )

    # Collect images
    image_paths, image_sort_mode = _collect_image_paths(args.image_dir)
    source_image_count = len(image_paths)
    excluded_image_stems = load_excluded_image_stems(
        args.exclude_image_stem,
        args.exclude_image_stems_file,
    )
    retained_indices, excluded_image_names = select_image_indices(
        image_paths,
        excluded_image_stems,
        min_remaining=1,
    )
    image_paths = [image_paths[index] for index in retained_indices]
    image_paths = _apply_subsampling(
        image_paths,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    frame_id_mode = getattr(args, "frame_id_mode", "auto")
    saved_frame_indices = _saved_frame_indices(
        image_paths,
        mode=frame_id_mode,
    )
    print(f"[RADIO] Found {len(image_paths)} images in {args.image_dir}")
    print(f"[RADIO] Image ordering: {image_sort_mode}")
    if excluded_image_names:
        print(
            "[RADIO] Excluded RGB feature views: "
            + ", ".join(excluded_image_names)
        )

    # Probe resolution from first image
    probe_img = Image.open(image_paths[0])
    orig_w, orig_h = probe_img.size
    target_h, target_w = _compute_scaled_radio_resolution(
        orig_h,
        orig_w,
        args.resolution_scale,
        patch_size=16,
    )
    patch_h, patch_w = target_h // 16, target_w // 16
    print(
        f"[RADIO] Input resolution: {orig_h}×{orig_w} "
        f"→ scale {args.resolution_scale:g} → {target_h}×{target_w}"
    )
    print(f"[RADIO] Feature grid: {patch_h}×{patch_w}")
    if args.sliding_window:
        if args.batch_size != 1:
            print("[RADIO] Sliding-window mode uses single-image batches; overriding batch_size=1")
            args.batch_size = 1
        print(
            f"[RADIO] Sliding-window extraction: tile={args.tile_size}px, "
            f"overlap={args.tile_overlap}px"
        )

    # Prepare output dirs
    subdirs = ["backbone", "summary"]
    if adaptor_names:
        subdirs += [_adaptor_output_subdir(name) for name in adaptor_names]
    for sd in subdirs:
        os.makedirs(os.path.join(args.output_dir, sd), exist_ok=True)

    pca_accumulator: list[torch.Tensor] = []
    frame_manifest: list[dict[str, object]] = []
    total_bytes: int = 0
    t0 = time.time()

    # Process in batches
    n = len(image_paths)
    for start in tqdm(range(0, n, args.batch_size), desc="Extracting RADIO features"):
        batch_paths = image_paths[start : start + args.batch_size]
        imgs = _load_and_preprocess(batch_paths, target_h, target_w, device)

        if args.sliding_window:
            summary, spatial_2d, adaptor_2d = _extract_sliding_window_single(
                model,
                conditioner,
                imgs,
                args.amp,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                adaptor_names=adaptor_names,
            )
        else:
            summary, spatial_2d, adaptor_2d = _run_radio_batch(
                model,
                conditioner,
                imgs,
                args.amp,
                patch_h,
                patch_w,
                adaptor_names=adaptor_names,
            )

        B, D, _, _ = spatial_2d.shape

        # Save per-frame
        for i in range(B):
            source_path = batch_paths[i]
            source_rank = start + i
            frame_idx = saved_frame_indices[source_rank]
            stem = f"rgb_{frame_idx}"
            frame_manifest.append(
                {
                    "source_rank": source_rank,
                    "frame_idx": frame_idx,
                    "source_file": source_path.name,
                    "saved_stem": stem,
                }
            )

            # Backbone: float16
            bb = spatial_2d[i].cpu().half()
            bb_path = os.path.join(args.output_dir, "backbone", f"{stem}.pt")
            torch.save(bb, bb_path)
            total_bytes += bb.nelement() * bb.element_size()

            # Summary: float32
            sm = summary[i].cpu().float()
            sm_path = os.path.join(args.output_dir, "summary", f"{stem}.pt")
            torch.save(sm, sm_path)
            total_bytes += sm.nelement() * sm.element_size()

            # Accumulate for PCA (convert back to float32)
            pca_accumulator.append(bb.float())

            # Adaptor features
            if adaptor_names and adaptor_2d:
                for name in adaptor_names:
                    if name not in adaptor_2d:
                        continue
                    short_name = _adaptor_output_subdir(name)
                    ad_frame = adaptor_2d[name][i].cpu().half()
                    ad_path = os.path.join(args.output_dir, short_name, f"{stem}.pt")
                    torch.save(ad_frame, ad_path)
                    total_bytes += ad_frame.nelement() * ad_frame.element_size()

    # PCA statistics
    print("[RADIO] Computing PCA statistics …")
    pca_stats = _compute_pca_stats(pca_accumulator, n_components=64)
    pca_path = os.path.join(args.output_dir, "pca_stats.pt")
    torch.save(pca_stats, pca_path)

    manifest_path = Path(args.output_dir) / "frame_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scene": args.scene,
                "radio": {
                    "version": args.radio_version,
                    "repo": str(Path(args.radio_repo).expanduser().resolve()),
                    "requested_adaptors": list(adaptor_names or []),
                },
                "image_dir": str(Path(args.image_dir).resolve()),
                "image_sort_mode": image_sort_mode,
                "frame_id_mode": frame_id_mode,
                "resolution_scale": float(args.resolution_scale),
                "radio_input_resolution_hw": [int(target_h), int(target_w)],
                "source_image_count_before_exclusion": source_image_count,
                "excluded_image_stems": list(excluded_image_stems),
                "excluded_image_names": excluded_image_names,
                "num_frames": len(frame_manifest),
                "features": {
                    "backbone": {
                        "subdir": "backbone",
                        "dim": int(D),
                        "grid": [int(patch_h), int(patch_w)],
                        "dtype": "float16",
                    },
                    "summary": {
                        "subdir": "summary",
                        "dim": int(summary.shape[-1]),
                        "dtype": "float32",
                    },
                    "adaptors": [
                        {
                            "name": name,
                            "subdir": _adaptor_output_subdir(name),
                            "dim": int(adaptor_2d[name].shape[1]) if name in adaptor_2d else None,
                            "grid": (
                                [
                                    int(adaptor_2d[name].shape[2]),
                                    int(adaptor_2d[name].shape[3]),
                                ]
                                if name in adaptor_2d and adaptor_2d[name].ndim == 4
                                else None
                            ),
                            "dtype": "float16",
                        }
                        for name in (adaptor_names or [])
                    ],
                },
                "frames": frame_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Summary
    elapsed = time.time() - t0
    disk_mb = total_bytes / (1024 * 1024)
    print("=" * 60)
    print(f"  Scene       : {args.scene}")
    print(f"  Frames      : {n}")
    print(f"  Backbone dim: {D} × {patch_h}×{patch_w}")
    print(f"  Summary dim : {summary.shape[-1]}")
    print(f"  Disk usage  : {disk_mb:.1f} MB  (float16 spatial + float32 summary)")
    print(f"  PCA saved   : {pca_path}")
    print(f"  Manifest    : {manifest_path}")
    print(f"  Time        : {elapsed:.1f}s  ({elapsed / n:.2f}s/frame)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RADIO features for all images in a scene."
    )
    parser.add_argument("--scene", type=str, default="room_0", help="Scene name")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory of RGB images (.png/.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save extracted features",
    )
    parser.add_argument(
        "--radio_repo",
        type=str,
        default="/root/RADIO",
        help="Local path to RADIO repo for torch.hub",
    )
    parser.add_argument(
        "--radio_version",
        type=str,
        default="c-radio_v4-h",
        help="RADIO model version string",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Images per batch")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every Nth image from image_dir",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on the number of images to extract",
    )
    parser.add_argument(
        "--frame-id-mode",
        choices=FRAME_ID_MODES,
        default="auto",
        help=(
            "Output feature id policy. 'auto' preserves legacy numeric suffixes; "
            "'source_rank' assigns a unique dense id after exclusion/subsampling."
        ),
    )
    parser.add_argument(
        "--exclude-image-stem",
        action="append",
        default=[],
        help=(
            "Exact case-sensitive image basename stem to exclude before feature "
            "extraction; repeat for multiple held-out views"
        ),
    )
    parser.add_argument(
        "--exclude-image-stems-file",
        default="",
        help="Optional JSON/text file of exact image stems to exclude",
    )
    parser.add_argument(
        "--extract_adaptors",
        action="store_true",
        help="Also extract adaptor features listed by --adaptor_names",
    )
    parser.add_argument(
        "--adaptor_names",
        type=str,
        default="siglip2-g,sam3",
        help=(
            "Comma-separated RADIO adaptor names to extract when --extract_adaptors "
            "is set, e.g. siglip2-g,dino_v3,sam3"
        ),
    )
    parser.add_argument(
        "--resolution_scale",
        type=float,
        default=1.0,
        help="Scale input images before RADIO extraction (default: 1.0)",
    )
    parser.add_argument(
        "--sliding_window",
        action="store_true",
        help="Extract high-resolution features by stitching overlapping single-image tiles",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Sliding-window tile size in input pixels (default: 1024)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=128,
        help="Sliding-window tile overlap in input pixels (default: 128)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device (default: cuda)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)",
    )

    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
