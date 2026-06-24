#!/usr/bin/env python3
"""Convert LangSplat language feature caches to RADIO-GS tensor caches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


_LERF_FRAME_RE = re.compile(r"^frame_(\d+)$")


def feature_stem_to_frame_id(stem: str) -> int:
    """Return the integer frame id from a LERF or ScanNet language-feature stem."""
    match = _LERF_FRAME_RE.match(stem)
    if match:
        return int(match.group(1))
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(f"Cannot parse language-feature frame id from stem: {stem!r}") from exc


def output_feature_name(stem: str) -> str:
    """Return the RADIO-GS tensor-cache filename for a language-feature stem."""
    return f"rgb_{feature_stem_to_frame_id(stem)}.pt"


def _normalise_feature_map(dense: torch.Tensor) -> torch.Tensor:
    norms = dense.float().norm(dim=0, keepdim=True)
    valid = norms > 1e-8
    out = torch.zeros_like(dense.float())
    out = torch.where(valid, dense.float() / norms.clamp_min(1e-8), out)
    return out


def materialize_dense_feature(
    feature_map: np.ndarray,
    seg_map: np.ndarray,
    *,
    level: int,
    output_size: tuple[int, int] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert one segment-prototype map into a dense [C,H,W] tensor."""
    if feature_map.ndim != 2:
        raise ValueError(f"Expected feature_map [N,C], got {feature_map.shape}")
    if seg_map.ndim != 3:
        raise ValueError(f"Expected seg_map [L,H,W], got {seg_map.shape}")
    if level < 0 or level >= int(seg_map.shape[0]):
        raise ValueError(f"Level {level} is outside seg_map levels [0,{int(seg_map.shape[0])})")

    seg = np.asarray(seg_map[level], dtype=np.int64)
    valid = seg >= 0
    if np.any(valid):
        max_id = int(seg[valid].max())
        min_id = int(seg[valid].min())
        if min_id < 0 or max_id >= int(feature_map.shape[0]):
            raise ValueError(
                f"Segment ids out of range for level {level}: "
                f"valid min={min_id} max={max_id}, features={int(feature_map.shape[0])}"
            )

    height, width = int(seg.shape[0]), int(seg.shape[1])
    channels = int(feature_map.shape[1])
    dense_hw_c = torch.zeros((height * width, channels), dtype=torch.float32)
    if np.any(valid):
        feature_tensor = torch.as_tensor(feature_map, dtype=torch.float32)
        feature_tensor = F.normalize(feature_tensor, dim=-1, eps=1e-8)
        valid_flat = torch.as_tensor(valid.reshape(-1), dtype=torch.bool)
        ids_flat = torch.as_tensor(seg.reshape(-1)[valid.reshape(-1)], dtype=torch.long)
        dense_hw_c[valid_flat] = feature_tensor[ids_flat]

    dense = dense_hw_c.view(height, width, channels).permute(2, 0, 1).contiguous()
    if output_size is not None:
        out_h, out_w = int(output_size[0]), int(output_size[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if (out_h, out_w) != (height, width):
            dense = F.interpolate(
                dense.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            dense = _normalise_feature_map(dense)
    return dense.to(dtype=dtype).contiguous()


def discover_feature_pairs(language_feature_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return sorted (stem, f_path, s_path) tuples with both cache files present."""
    language_feature_dir = Path(language_feature_dir)
    pairs: list[tuple[str, Path, Path]] = []
    for f_path in sorted(language_feature_dir.glob("*_f.npy")):
        stem = f_path.name[: -len("_f.npy")]
        s_path = language_feature_dir / f"{stem}_s.npy"
        if s_path.exists():
            pairs.append((stem, f_path, s_path))
    pairs.sort(key=lambda item: feature_stem_to_frame_id(item[0]))
    return pairs


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def convert_scene(
    language_feature_dir: Path,
    output_root: Path,
    *,
    levels: Sequence[int],
    output_size: tuple[int, int] | None,
    dtype: torch.dtype,
    dry_run: bool,
) -> dict[str, object]:
    """Convert all cache pairs in a scene and return the manifest payload."""
    language_feature_dir = Path(language_feature_dir)
    output_root = Path(output_root)
    pairs = discover_feature_pairs(language_feature_dir)
    if not pairs:
        raise FileNotFoundError(f"No matching *_f.npy/*_s.npy pairs found in {language_feature_dir}")

    levels = [int(level) for level in levels]
    if not levels:
        raise ValueError("At least one level is required")

    first_feature = np.load(pairs[0][1], mmap_mode="r")
    first_seg = np.load(pairs[0][2], mmap_mode="r")
    feature_dim = int(first_feature.shape[1])
    available_levels = int(first_seg.shape[0])
    for level in levels:
        if level < 0 or level >= available_levels:
            raise ValueError(f"Level {level} is outside available levels [0,{available_levels})")

    aggregate_outputs: list[dict[str, object]] = []
    for level in levels:
        level_root = output_root / f"l{level}"
        backbone_root = level_root / "backbone"
        outputs: list[dict[str, object]] = []
        for stem, f_path, s_path in pairs:
            frame_id = feature_stem_to_frame_id(stem)
            tensor_rel = Path("backbone") / output_feature_name(stem)
            if not dry_run:
                feature_map = np.load(f_path)
                seg_map = np.load(s_path)
                tensor = materialize_dense_feature(
                    feature_map,
                    seg_map,
                    level=level,
                    output_size=output_size,
                    dtype=dtype,
                )
                backbone_root.mkdir(parents=True, exist_ok=True)
                torch.save(tensor.cpu(), level_root / tensor_rel)
            entry = {
                "frame_id": frame_id,
                "stem": stem,
                "feature": str(f_path),
                "segments": str(s_path),
                "tensor": tensor_rel.as_posix(),
            }
            outputs.append(entry)
            aggregate_outputs.append({"level": level, **entry})

        manifest = {
            "source_dir": str(language_feature_dir),
            "output_dir": str(level_root),
            "levels": [level],
            "level": level,
            "frames_converted": len(outputs),
            "feature_dim": feature_dim,
            "source_feature_shape": list(first_feature.shape),
            "source_segment_shape": list(first_seg.shape),
            "output_size": list(output_size) if output_size is not None else None,
            "dtype": str(dtype).replace("torch.", ""),
            "dry_run": bool(dry_run),
            "outputs": outputs,
        }
        if not dry_run:
            _write_manifest(level_root / "samclip_manifest.json", manifest)

    return {
        "source_dir": str(language_feature_dir),
        "output_root": str(output_root),
        "levels": levels,
        "frames_converted": len(pairs),
        "feature_dim": feature_dim,
        "source_feature_shape": list(first_feature.shape),
        "source_segment_shape": list(first_seg.shape),
        "output_size": list(output_size) if output_size is not None else None,
        "dtype": str(dtype).replace("torch.", ""),
        "dry_run": bool(dry_run),
        "outputs": aggregate_outputs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language-feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1])
    parser.add_argument("--output-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = convert_scene(
        args.language_feature_dir,
        args.output_root,
        levels=args.levels,
        output_size=tuple(args.output_size) if args.output_size is not None else None,
        dtype=_dtype_from_name(args.dtype),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({key: manifest[key] for key in ("source_dir", "output_root", "levels", "frames_converted", "feature_dim", "output_size", "dtype", "dry_run")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
