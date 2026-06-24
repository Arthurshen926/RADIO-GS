#!/usr/bin/env python3
"""Convert rendered ``rgb_*.pt`` feature maps to LERF ``frame_*.npy`` files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def output_frame_name(path: Path, *, prefix: str = "frame_", digits: int = 5) -> str:
    stem = path.stem
    if not stem.startswith("rgb_"):
        raise ValueError(f"Expected rgb_<id>.pt filename, got {path.name}")
    frame_id = int(stem.removeprefix("rgb_"))
    return f"{prefix}{frame_id:0{digits}d}.npy"


def _load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        for key in ("feature", "features", "feat", "tensor"):
            value = tensor.get(key)
            if isinstance(value, torch.Tensor):
                tensor = value
                break
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path}: expected tensor or tensor dict, got {type(tensor).__name__}")
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{path}: expected [C,H,W], got {tuple(tensor.shape)}")
    return tensor.detach().cpu()


def convert_tensor(
    tensor: torch.Tensor,
    *,
    normalize: bool,
    dtype: str,
) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(tensor.shape)}")
    tensor = tensor.float()
    if normalize:
        tensor = F.normalize(tensor, dim=0, eps=1e-6)
    array = tensor.permute(1, 2, 0).contiguous().numpy()
    if dtype == "fp16":
        return array.astype(np.float16)
    if dtype == "fp32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    normalize: bool = True,
    dtype: str = "fp16",
    prefix: str = "frame_",
    digits: int = 5,
) -> list[Path]:
    feature_dir = input_dir / "backbone" if (input_dir / "backbone").is_dir() else input_dir
    paths = sorted(feature_dir.glob("rgb_*.pt"), key=lambda path: int(path.stem.removeprefix("rgb_")))
    if not paths:
        raise FileNotFoundError(f"No rgb_*.pt files found under {feature_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in paths:
        tensor = _load_tensor(path)
        output = output_dir / output_frame_name(path, prefix=prefix, digits=digits)
        np.save(output, convert_tensor(tensor, normalize=normalize, dtype=dtype))
        written.append(output)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--feature-mode", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--frame-prefix", default="frame_")
    parser.add_argument("--digits", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    written = convert_directory(
        args.input_dir,
        args.output_dir,
        normalize=args.feature_mode == "normalized",
        dtype=args.dtype,
        prefix=args.frame_prefix,
        digits=args.digits,
    )
    print(f"Converted {len(written)} feature maps to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
