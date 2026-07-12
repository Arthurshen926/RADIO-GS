#!/usr/bin/env python3
"""Extract dense OpenCLIP ViT patch-token features for RADIO-GS ablations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from radio_gs.config import load_config
from radio_gs.data.benchmark_paths import (
    extract_feature_frame_index,
    list_feature_paths,
    resolve_dataset_type,
    resolve_rgb_path,
    resolve_scene_root,
    resolve_split_data_dir,
    resolve_split_feature_dir,
    resolve_split_frame_ids,
)


LOGGER = logging.getLogger("extract_openclip_dense_features")
TOKEN_MODES = ("patch", "maskclip_value")


def project_patch_tokens(
    tokens: torch.Tensor,
    projection: torch.Tensor | None,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Project OpenCLIP patch tokens from visual width to text-embedding width."""
    if tokens.ndim != 4:
        raise ValueError(f"Expected patch tokens [B,C,H,W], got {tuple(tokens.shape)}")

    if projection is None:
        projected = tokens
    else:
        proj = projection.detach().to(device=tokens.device, dtype=tokens.dtype)
        if proj.ndim != 2 or proj.shape[0] != tokens.shape[1]:
            raise ValueError(
                "Projection must have shape [token_dim, output_dim], got "
                f"{tuple(proj.shape)} for token_dim={tokens.shape[1]}"
            )
        projected = torch.einsum("bdhw,de->behw", tokens, proj)

    if normalize:
        projected = F.normalize(projected.float(), dim=1, eps=1e-8).to(dtype=projected.dtype)
    return projected.contiguous()


def _unwrap_intermediates(output: Any) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict):
        for key in ("image_intermediates", "intermediates", "features"):
            value = output.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)
        raise TypeError(f"Could not find intermediates in keys: {sorted(output)}")
    if isinstance(output, (list, tuple)):
        if output and all(isinstance(item, torch.Tensor) for item in output):
            return list(output)
        for item in output:
            if isinstance(item, (list, tuple)) and item and all(isinstance(t, torch.Tensor) for t in item):
                return list(item)
    raise TypeError(f"Unsupported forward_intermediates output type: {type(output)!r}")


def extract_dense_feature_from_visual(
    visual: Any,
    image: torch.Tensor,
    *,
    intermediate_index: int,
    token_mode: str = "patch",
    normalize: bool,
) -> torch.Tensor:
    """Return dense projected OpenCLIP features as ``[B,512,H,W]``."""
    if token_mode == "maskclip_value":
        return extract_maskclip_value_feature_from_visual(visual, image, normalize=normalize)
    if token_mode != "patch":
        raise ValueError(f"Unsupported OpenCLIP token mode: {token_mode}")

    output = visual.forward_intermediates(
        image,
        indices=[int(intermediate_index)],
        normalize_intermediates=True,
        intermediates_only=True,
        output_fmt="NCHW",
    )
    intermediates = _unwrap_intermediates(output)
    if not intermediates:
        raise ValueError("OpenCLIP visual.forward_intermediates returned no intermediates")
    tokens = intermediates[-1]
    if tokens.ndim != 4:
        raise ValueError(
            "Expected OpenCLIP intermediates in NCHW patch-grid format, got "
            f"{tuple(tokens.shape)}"
        )
    projection = getattr(visual, "proj", None)
    return project_patch_tokens(tokens, projection, normalize=normalize)


def _infer_patch_grid(visual: Any, patch_count: int) -> tuple[int, int]:
    grid_size = getattr(visual, "grid_size", None)
    if isinstance(grid_size, Sequence) and len(grid_size) == 2:
        height, width = int(grid_size[0]), int(grid_size[1])
        if height * width == patch_count:
            return height, width
    side = int(round(float(patch_count) ** 0.5))
    if side * side == patch_count:
        return side, side
    raise ValueError(f"Could not infer OpenCLIP patch grid for {patch_count} tokens")


def extract_maskclip_value_feature_from_visual(
    visual: Any,
    image: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Return MaskCLIP-style final-block value tokens projected to CLIP text space.

    Plain CLIP ViT patch tokens are not trained as dense language descriptors.
    This mode follows the common MaskCLIP trick of using the final attention
    block's value projection and output projection, without the query/key
    attention mixing that collapses local evidence into the class token.
    """
    if not hasattr(visual, "_embeds"):
        raise TypeError("maskclip_value mode requires an OpenCLIP VisionTransformer visual._embeds method")
    transformer = getattr(visual, "transformer", None)
    blocks = list(getattr(transformer, "resblocks", []))
    if not blocks:
        raise TypeError("maskclip_value mode requires visual.transformer.resblocks")

    x = visual._embeds(image)
    for block in blocks[:-1]:
        x = block(x)

    block = blocks[-1]
    if not hasattr(block, "ln_1") or not hasattr(block, "attn"):
        raise TypeError("maskclip_value mode requires a final block with ln_1 and attn")
    normalized = block.ln_1(x)
    attn = block.attn
    embed_dim = int(getattr(attn, "embed_dim", normalized.shape[-1]))
    in_proj_weight = getattr(attn, "in_proj_weight", None)
    if in_proj_weight is None or in_proj_weight.shape[0] < 3 * embed_dim:
        raise TypeError("maskclip_value mode requires packed qkv in_proj_weight")
    in_proj_bias = getattr(attn, "in_proj_bias", None)
    value_weight = in_proj_weight[2 * embed_dim : 3 * embed_dim]
    value_bias = in_proj_bias[2 * embed_dim : 3 * embed_dim] if in_proj_bias is not None else None
    value_tokens = F.linear(normalized, value_weight, value_bias)
    value_tokens = attn.out_proj(value_tokens)

    ln_post = getattr(visual, "ln_post", None)
    if ln_post is not None:
        value_tokens = ln_post(value_tokens)
    patch_tokens = value_tokens[:, 1:, :]
    height, width = _infer_patch_grid(visual, int(patch_tokens.shape[1]))
    patch_grid = patch_tokens.reshape(patch_tokens.shape[0], height, width, patch_tokens.shape[-1])
    patch_grid = patch_grid.permute(0, 3, 1, 2).contiguous()
    projection = getattr(visual, "proj", None)
    return project_patch_tokens(patch_grid, projection, normalize=normalize)


def discover_source_frames(
    feature_dir: str | Path,
    frame_ids: Sequence[int] | None = None,
) -> list[int]:
    """Discover sorted frame ids from a RADIO-GS-style feature directory."""
    return [extract_feature_frame_index(path) for path in list_feature_paths(feature_dir, frame_ids)]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _resolve_rgb_root(config: Any, split: str, override: str | Path | None) -> Path:
    if override:
        root = Path(override)
        if not root.exists():
            raise FileNotFoundError(f"RGB directory does not exist: {root}")
        return root

    dataset_type = resolve_dataset_type(config)
    scene_root = resolve_scene_root(config)
    candidates: list[Path] = []
    resolved = resolve_split_data_dir(config, split, "rgb")
    if resolved is not None:
        candidates.append(Path(resolved))
    if dataset_type == "lerf":
        candidates.append(scene_root / "images")
    elif dataset_type == "scannet":
        candidates.append(scene_root / "color")
    else:
        split_name = getattr(config, "train_split", "Sequence_1") if split == "train" else getattr(config, "val_split", "Sequence_2")
        candidates.append(scene_root / split_name / "rgb")

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No RGB directory found. Checked: "
        + ", ".join(str(candidate) for candidate in unique_candidates)
    )


def _load_image_batch(
    paths: Sequence[Path],
    preprocess: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(preprocess(handle.convert("RGB")))
    batch = torch.stack(images, dim=0)
    return batch.to(device=device, dtype=dtype, non_blocking=True)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_scene(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_feature_dir: str | Path | None = None,
    rgb_dir: str | Path | None = None,
    split: str = "train",
    frame_ids: Sequence[int] | None = None,
    limit: int | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    openclip_model: str = "ViT-B-16",
    openclip_pretrained: str = "laion2b_s34b_b88k",
    intermediate_index: int = -1,
    token_mode: str = "patch",
    dtype: torch.dtype = torch.float16,
    normalize: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract dense OpenCLIP patch-token tensors for frames in a config."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_config(str(config_path))
    dataset_type = resolve_dataset_type(config)
    feature_root = Path(source_feature_dir) if source_feature_dir is not None else resolve_split_feature_dir(config, split)
    selected_frames = list(frame_ids) if frame_ids is not None else resolve_split_frame_ids(config, split)
    frames = discover_source_frames(feature_root, selected_frames)
    if limit is not None:
        frames = frames[: int(limit)]
    if not frames:
        raise FileNotFoundError(f"No source feature frames found in {feature_root}")

    rgb_root = _resolve_rgb_root(config, split, rgb_dir)
    records: list[dict[str, Any]] = []
    missing: list[Path] = []
    for frame_id in frames:
        rgb_path = resolve_rgb_path(rgb_root, frame_id, dataset_type)
        if rgb_path is None or not rgb_path.exists():
            missing.append(Path(rgb_path) if rgb_path is not None else rgb_root / f"{frame_id}")
            continue
        records.append({"frame_id": int(frame_id), "rgb": Path(rgb_path)})
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} RGB files, first entries: {preview}")
    if not records:
        raise FileNotFoundError(f"No RGB files resolved under {rgb_root}")

    backbone_root = output_dir / "backbone"
    if not dry_run:
        backbone_root.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    precision = "fp16" if device_obj.type == "cuda" and dtype == torch.float16 else "fp32"

    feature_dim: int | None = None
    output_size: list[int] | None = None
    image_size: Any = None
    patch_grid: Any = None
    outputs: list[dict[str, Any]] = []

    if dry_run:
        outputs = [
            {
                "frame_id": entry["frame_id"],
                "rgb": str(entry["rgb"]),
                "tensor": f"backbone/rgb_{entry['frame_id']}.pt",
            }
            for entry in records
        ]
    else:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            openclip_model,
            pretrained=openclip_pretrained,
            precision=precision,
            device=device_obj,
        )
        model.eval()
        visual = model.visual
        image_size = getattr(visual, "image_size", None)
        patch_grid = getattr(visual, "grid_size", None)
        conv1 = getattr(visual, "conv1", None)
        model_dtype = conv1.weight.dtype if conv1 is not None else next(visual.parameters()).dtype
        save_dtype = dtype

        with torch.inference_mode():
            for batch_records in _batched(records, batch_size):
                paths = [entry["rgb"] for entry in batch_records]
                images = _load_image_batch(paths, preprocess, device=device_obj, dtype=model_dtype)
                dense = extract_dense_feature_from_visual(
                    visual,
                    images,
                    intermediate_index=intermediate_index,
                    token_mode=token_mode,
                    normalize=normalize,
                )
                if feature_dim is None:
                    feature_dim = int(dense.shape[1])
                    output_size = [int(dense.shape[2]), int(dense.shape[3])]
                if len(batch_records) != int(dense.shape[0]):
                    raise RuntimeError(
                        f"Batch size mismatch: {len(batch_records)} records vs {int(dense.shape[0])} features"
                    )
                for entry, feature in zip(batch_records, dense):
                    tensor_name = f"rgb_{entry['frame_id']}.pt"
                    tensor_rel = Path("backbone") / tensor_name
                    torch.save(feature.detach().cpu().to(dtype=save_dtype).contiguous(), output_dir / tensor_rel)
                    outputs.append(
                        {
                            "frame_id": entry["frame_id"],
                            "rgb": str(entry["rgb"]),
                            "tensor": tensor_rel.as_posix(),
                        }
                    )

    manifest = {
        "source_config": str(config_path),
        "source_feature_dir": str(feature_root),
        "rgb_dir": str(rgb_root),
        "output_dir": str(output_dir),
        "dataset_type": dataset_type,
        "split": split,
        "openclip_model": openclip_model,
        "openclip_pretrained": openclip_pretrained,
        "token_mode": token_mode,
        "intermediate_index": int(intermediate_index),
        "normalize": bool(normalize),
        "dtype": str(dtype).replace("torch.", ""),
        "image_size": list(image_size) if isinstance(image_size, tuple) else image_size,
        "patch_grid": list(patch_grid) if isinstance(patch_grid, tuple) else patch_grid,
        "feature_dim": feature_dim,
        "output_size": output_size,
        "frames_converted": len(outputs),
        "dry_run": bool(dry_run),
        "outputs": outputs,
    }
    if not dry_run:
        _write_manifest(output_dir / "openclip_dense_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-feature-dir", type=Path, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--frame-ids", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--openclip-model", default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--token-mode", choices=TOKEN_MODES, default="patch")
    parser.add_argument("--intermediate-index", type=int, default=-1)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    manifest = extract_scene(
        args.config,
        args.output_dir,
        source_feature_dir=args.source_feature_dir,
        rgb_dir=args.rgb_dir,
        split=args.split,
        frame_ids=args.frame_ids,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
        openclip_model=args.openclip_model,
        openclip_pretrained=args.openclip_pretrained,
        token_mode=args.token_mode,
        intermediate_index=args.intermediate_index,
        dtype=_dtype_from_name(args.dtype),
        normalize=not args.no_normalize,
        dry_run=bool(args.dry_run),
    )
    LOGGER.info(
        "Prepared %d OpenCLIP dense features in %s",
        manifest["frames_converted"],
        manifest["output_dir"],
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "source_config",
                    "source_feature_dir",
                    "rgb_dir",
                    "output_dir",
                    "dataset_type",
                    "openclip_model",
                    "openclip_pretrained",
                    "token_mode",
                    "feature_dim",
                    "output_size",
                    "frames_converted",
                    "dry_run",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
