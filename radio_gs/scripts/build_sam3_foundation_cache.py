#!/usr/bin/env python3
"""Build official SAM3 mask-logit foundation caches for RADIO-GS training."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SAM3_DTYPE_CHOICES = ("auto", "float32", "bfloat16")
SAM3_AMP_DTYPE_CHOICES = ("auto", "off", "bfloat16")


def parse_queries(raw: str, query_file: str = "") -> list[str]:
    queries: list[str] = []
    if query_file:
        payload = json.loads(Path(query_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("queries", [])
        if not isinstance(payload, list):
            raise ValueError("query file must be a list or {'queries': [...]}")
        queries.extend(str(item).strip() for item in payload)
    if raw:
        queries.extend(part.strip() for part in raw.replace("\n", ",").split(","))
    queries = [query for query in queries if query]
    if not queries:
        raise ValueError("at least one SAM3 text query is required")
    return queries


def iter_images(image_root: str, pattern: str) -> list[Path]:
    root = Path(image_root).expanduser()
    if root.is_file():
        return [root]
    images = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no images found under {root} with pattern {pattern!r}")
    return images


def frame_id_from_path(path: Path) -> str:
    return path.stem


def _frame_id_aliases(path: Path) -> set[str]:
    stem = frame_id_from_path(path)
    aliases = {stem}
    if stem.startswith("frame_"):
        suffix = stem[len("frame_") :]
        aliases.add(suffix)
        try:
            aliases.add(str(int(suffix)))
        except ValueError:
            pass
    return aliases


def filter_images_by_frame_ids(images: list[Path], raw_frame_ids: str) -> list[Path]:
    raw = str(raw_frame_ids or "").strip()
    if not raw:
        return images
    wanted = {part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()}
    selected = [path for path in images if _frame_id_aliases(path) & wanted]
    if not selected:
        raise ValueError(f"No images matched --frame_ids={raw!r}")
    return selected


def output_path_for_image(output_root: Path, image_path: Path) -> Path:
    return output_root / f"{frame_id_from_path(image_path)}.pt"


def normalize_sam3_device(device: str) -> str:
    device_name = str(device)
    if device_name.startswith("cuda"):
        return "cuda"
    if device_name == "cpu":
        return "cpu"
    raise ValueError("SAM3 device must be 'cuda', 'cuda:<index>', or 'cpu'")


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_sam3_cache_payload(
    *,
    frame_id: str,
    queries: list[str],
    masks: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    image_path: str,
    backend: str = "facebookresearch/sam3",
    decoder: str = "Sam3Image+Sam3Processor",
    checkpoint_path: str = "",
    checkpoint_source: str = "",
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    if masks.dim() != 3:
        raise ValueError("SAM3 masks must be [M,H,W]")
    return {
        "version": 1,
        "frame_id": frame_id,
        "heads": {
            "sam3": {
                "mask_logits": masks.float().contiguous(),
                "producer": {
                    "official": True,
                    "backend": backend,
                    "decoder": decoder,
                    "source": image_path,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_source": checkpoint_source,
                    "checkpoint_sha256": checkpoint_sha256,
                },
                "queries": list(queries),
                "scores": scores.float().contiguous(),
                "boxes_xyxy": boxes.float().contiguous(),
            }
        },
    }


def resolve_sam3_dtype(device: str, dtype: str) -> torch.dtype:
    dtype_name = str(dtype).lower()
    if dtype_name not in SAM3_DTYPE_CHOICES:
        raise ValueError(f"dtype must be one of: {', '.join(SAM3_DTYPE_CHOICES)}")
    if dtype_name in {"auto", "float32"}:
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported SAM3 model dtype: {dtype}")


def resolve_sam3_amp_dtype(device: str, dtype: str) -> torch.dtype | None:
    dtype_name = str(dtype).lower()
    if dtype_name not in SAM3_AMP_DTYPE_CHOICES:
        raise ValueError(f"amp_dtype must be one of: {', '.join(SAM3_AMP_DTYPE_CHOICES)}")
    if dtype_name == "off":
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if str(device).startswith("cuda") else None


def cast_sam3_model_for_inference(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    if dtype == torch.float32:
        return model.float()
    if dtype == torch.bfloat16:
        return model.to(dtype=torch.bfloat16)
    raise ValueError(f"Unsupported SAM3 inference dtype: {dtype}")


def make_sam3_processor(
    processor_cls: type,
    model: torch.nn.Module,
    *,
    device: str,
    confidence_threshold: float,
    resolution: int,
):
    return processor_cls(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
        resolution=int(resolution),
    )


def validate_sam3_resolution(resolution: int, *, allow_unsafe: bool) -> int:
    value = int(resolution)
    if value != 1008 and not allow_unsafe:
        raise ValueError(
            "The official SAM3 image model expects 1008 resolution with the "
            "current RoPE setup. Use --allow_unsafe_resolution only for explicit "
            "diagnostics."
        )
    return value


def sam3_autocast_context(device: str, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return contextlib.nullcontext()
    device_name = str(device)
    if device_name.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    if device_name == "cpu":
        return torch.autocast(device_type="cpu", dtype=amp_dtype)
    return contextlib.nullcontext()


def _load_sam3_model(
    *,
    checkpoint_path: str,
    device: str,
    confidence_threshold: float,
    dtype: str,
    resolution: int,
):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = normalize_sam3_device(device)
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path or None,
        load_from_HF=not checkpoint_path,
        device=device,
        eval_mode=True,
    )
    model = cast_sam3_model_for_inference(model, resolve_sam3_dtype(device, dtype))
    return make_sam3_processor(
        Sam3Processor,
        model,
        device=device,
        confidence_threshold=confidence_threshold,
        resolution=resolution,
    )


def run_sam3_on_image(
    processor: Any,
    image_path: Path,
    queries: Iterable[str],
    *,
    max_masks_per_query: int,
    amp_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    with sam3_autocast_context(str(processor.device), amp_dtype):
        state = processor.set_image(image)
    masks: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    for query in queries:
        query_state = dict(state)
        with sam3_autocast_context(str(processor.device), amp_dtype):
            output = processor.set_text_prompt(str(query), query_state)
        query_masks = _as_tensor(output.get("masks_logits", torch.empty(0, image.height, image.width)))
        query_scores = _as_tensor(output.get("scores", torch.empty(0)))
        query_boxes = _as_tensor(output.get("boxes", torch.empty(0, 4)))
        if query_masks.dim() == 4 and query_masks.shape[1] == 1:
            query_masks = query_masks[:, 0]
        if max_masks_per_query > 0:
            query_masks = query_masks[:max_masks_per_query]
            query_scores = query_scores[:max_masks_per_query]
            query_boxes = query_boxes[:max_masks_per_query]
        masks.append(query_masks)
        scores.append(query_scores)
        boxes.append(query_boxes)
    height, width = image.height, image.width
    all_masks = torch.cat(masks, dim=0) if masks else torch.empty(0, height, width)
    all_scores = torch.cat(scores, dim=0) if scores else torch.empty(0)
    all_boxes = torch.cat(boxes, dim=0) if boxes else torch.empty(0, 4)
    return all_masks, all_scores, all_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--image_glob", default="*")
    parser.add_argument("--frame_ids", default="")
    parser.add_argument("--queries", default="")
    parser.add_argument("--query_file", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--checkpoint_source", default="")
    parser.add_argument("--checkpoint_sha256", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=SAM3_DTYPE_CHOICES, default="auto")
    parser.add_argument("--amp_dtype", choices=SAM3_AMP_DTYPE_CHOICES, default="auto")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--allow_unsafe_resolution", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--max_masks_per_query", type=int, default=8)
    args = parser.parse_args()

    queries = parse_queries(args.queries, args.query_file)
    images = filter_images_by_frame_ids(
        iter_images(args.image_root, args.image_glob),
        args.frame_ids,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolution = validate_sam3_resolution(
        args.resolution,
        allow_unsafe=args.allow_unsafe_resolution,
    )
    device = normalize_sam3_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        confidence_threshold=args.confidence_threshold,
        dtype=args.dtype,
        resolution=resolution,
    )
    amp_dtype = resolve_sam3_amp_dtype(device, args.amp_dtype)
    checkpoint_sha256 = ""
    if args.checkpoint_path:
        if str(args.checkpoint_sha256).lower() == "auto":
            checkpoint_sha256 = sha256_file(args.checkpoint_path)
        else:
            checkpoint_sha256 = str(args.checkpoint_sha256)
    for image_path in images:
        output_path = output_path_for_image(output_root, image_path)
        if args.skip_existing and output_path.exists():
            print(f"skip existing {output_path}")
            continue
        masks, scores, boxes = run_sam3_on_image(
            processor,
            image_path,
            queries,
            max_masks_per_query=args.max_masks_per_query,
            amp_dtype=amp_dtype,
        )
        payload = make_sam3_cache_payload(
            frame_id=frame_id_from_path(image_path),
            queries=queries,
            masks=masks,
            scores=scores,
            boxes=boxes,
            image_path=str(image_path),
            checkpoint_path=str(args.checkpoint_path),
            checkpoint_source=str(args.checkpoint_source),
            checkpoint_sha256=checkpoint_sha256,
        )
        torch.save(payload, output_path)
        print(f"wrote {output_path} masks={tuple(masks.shape)}")


if __name__ == "__main__":
    main()
