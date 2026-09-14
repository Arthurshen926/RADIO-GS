"""Build query-free region-level language keys for LERF SAM fragments.

Every source SAM proposal remains an observed fragment.  Its masked crop and
an expanded context crop are re-encoded by the complete frozen C-RADIO model;
only the genuine SigLIP2 summary token is projected into text space.  This is
the legal replacement for applying ``SigLIP2SummaryHead`` independently to
spatial pixels.

The output does not merge fragments, open text queries, or read benchmark
labels.  Proposal order is byte-bound to the source SAM manifests so a later
global fragment-to-object stage can retain multiple view prototypes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.scripts.build_sam_mask_aligned_language_teacher import (
    build_crop_pairs,
    encode_in_batches,
)
from radio_gs.scripts.extract_official_crop_summary_teacher import (
    _atomic_json_write,
    _atomic_torch_save,
)
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import (
    validate_source_authority as _validate_source_authority,
)
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records


FRAME_SCHEMA = "radio_gs.surface_object_memory_v4.fragment_language_frame.v1"
MANIFEST_SCHEMA = "radio_gs.surface_object_memory_v4.fragment_language_manifest.v1"


def _frame_index(image_id: str) -> int:
    digits = "".join(character for character in Path(image_id).stem if character.isdigit())
    if not digits:
        raise ValueError(f"source image id has no frame index: {image_id!r}")
    return int(digits)


def _validate_frame_payload(
    payload: dict[str, Any],
    *,
    image_path: Path,
) -> tuple[np.ndarray, torch.Tensor]:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("fragment language memory requires multiscale SAM schema v1")
    shape = tuple(int(value) for value in payload.get("mask_shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("SAM fragment payload has an invalid mask shape")
    with Image.open(image_path) as image:
        if (image.height, image.width) != shape:
            raise ValueError("source RGB and SAM fragment raster differ")
    packed = torch.as_tensor(payload.get("packed_masks"), dtype=torch.uint8)
    if packed.ndim != 3:
        raise ValueError("SAM fragment payload has malformed packed masks")
    masks = unpack_masks(packed, width=shape[1])
    if masks.ndim != 3 or tuple(masks.shape[1:]) != shape or masks.shape[0] == 0:
        raise ValueError("SAM fragment payload has empty or misaligned masks")
    boxes = torch.as_tensor(payload.get("boxes_xyxy"), dtype=torch.int32)
    count = int(masks.shape[0])
    if boxes.shape != (count, 4):
        raise ValueError("SAM fragment boxes do not align with masks")
    for key in ("quality", "stability", "parent_index"):
        value = torch.as_tensor(payload.get(key))
        if value.shape != (count,):
            raise ValueError(f"SAM fragment {key} does not align with masks")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"SAM fragment {key} contains non-finite values")
    for index, mask in enumerate(masks):
        y, x = np.where(mask)
        if not x.size:
            raise ValueError("SAM fragment mask is empty")
        expected = torch.tensor(
            [x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=torch.int32
        )
        if not torch.equal(boxes[index], expected):
            raise ValueError("SAM fragment box differs from its packed support")
    return masks, boxes


def preflight_inputs(
    *,
    scene_label: str,
    source_rgb_authority: Path,
    sam_manifests: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    authority_path = source_rgb_authority.resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    source_frames = _validate_source_authority(authority, scene_label)
    source_records = {
        _frame_index(str(record["image_id"])): record for record in authority["images"]
    }
    if set(source_records) != set(source_frames):
        raise ValueError("source RGB authority frame inventory differs")
    sam_records = _load_sam_records([path.resolve(strict=True) for path in sam_manifests])
    if set(sam_records) != set(source_frames):
        raise ValueError("SAM fragment frames differ from source RGB authority")

    records: list[dict[str, Any]] = []
    for frame_id in source_frames:
        source = source_records[frame_id]
        image_path = Path(source["path"]).resolve(strict=True)
        if sha256_file(image_path) != source["sha256"]:
            raise ValueError(f"source RGB digest differs for frame {frame_id}")
        sam_path = Path(sam_records[frame_id]["output"]).resolve(strict=True)
        payload = torch.load(sam_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("SAM fragment payload is not a dictionary")
        masks, boxes = _validate_frame_payload(payload, image_path=image_path)
        if int(sam_records[frame_id].get("proposal_count", -1)) != int(masks.shape[0]):
            raise ValueError("SAM manifest proposal count differs from payload")
        records.append({
            "frame_id": frame_id,
            "image_path": image_path,
            "image_sha256": source["sha256"],
            "sam_path": sam_path,
            "sam_sha256": sam_records[frame_id]["output_sha256"],
            "payload": payload,
            "masks": masks,
            "boxes": boxes,
        })
    return authority, records


def build_frame_memory(
    record: dict[str, Any],
    runtime: Any,
    *,
    device: torch.device,
    crop_resolution: int,
    batch_size: int,
    context_expansion: float,
    masked_background_rgb: tuple[float, float, float],
) -> dict[str, Any]:
    image = pil_to_tensor(Image.open(record["image_path"]).convert("RGB")).float().div_(255)
    masked, context = build_crop_pairs(
        image,
        record["masks"],
        record["boxes"],
        context_expansion=context_expansion,
        crop_resolution=crop_resolution,
        masked_background_rgb=masked_background_rgb,
    )
    masked_descriptor = encode_in_batches(
        runtime, masked, batch_size=batch_size, device=device
    )
    context_descriptor = encode_in_batches(
        runtime, context, batch_size=batch_size, device=device
    )
    payload = record["payload"]
    return {
        "schema": FRAME_SCHEMA,
        "frame_id": int(record["frame_id"]),
        "masked_crop_descriptor": masked_descriptor,
        "context_crop_descriptor": context_descriptor,
        "quality": torch.as_tensor(payload["quality"]).float(),
        "stability": torch.as_tensor(payload["stability"]).float(),
        "parent_index": torch.as_tensor(payload["parent_index"]).long(),
        "boxes_xyxy": record["boxes"].int(),
        "proposal_area_fraction": torch.as_tensor(payload["proposal_area_fraction"]).float(),
        "metadata": {
            "teacher_space": "official_siglip2_crop_summary",
            "text_compatibility": "official_siglip2_g_text_space",
            "source_image": str(record["image_path"]),
            "source_image_sha256": record["image_sha256"],
            "sam_fragment_payload": str(record["sam_path"]),
            "sam_fragment_payload_sha256": record["sam_sha256"],
            "query_free": True,
            "source_only": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.crop_resolution <= 0 or args.batch_size <= 0:
        raise ValueError("crop resolution and batch size must be positive")
    if args.context_expansion < 1.0:
        raise ValueError("context expansion must be at least one")
    fill = tuple(float(value) for value in args.masked_background_rgb)
    if len(fill) != 3 or any(value < 0 or value > 1 for value in fill):
        raise ValueError("masked background RGB must contain three values in [0,1]")
    authority_path = Path(args.source_rgb_authority)
    sam_paths = [Path(value) for value in args.sam_manifest]
    _authority, records = preflight_inputs(
        scene_label=args.scene_label,
        source_rgb_authority=authority_path,
        sam_manifests=sam_paths,
    )
    checkpoint = Path(args.radio_checkpoint).resolve(strict=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    if args.expected_radio_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("RADIO checkpoint digest differs from the sealed expectation")
    output_root = Path(args.output_root).resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("schema") != MANIFEST_SCHEMA
            or existing.get("scene_label") != args.scene_label
            or existing.get("radio_checkpoint_sha256") != checkpoint_sha256
        ):
            raise FileExistsError(
                f"incompatible fragment language manifest already exists: {manifest_path}"
            )
        for record in existing.get("outputs", []):
            output = Path(record["output"]).resolve(strict=True)
            if sha256_file(output) != record["output_sha256"]:
                raise ValueError("existing fragment language frame digest differs")
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runtime = None

    outputs = []
    for index, record in enumerate(records, start=1):
        print(
            f"encoding fragment language memory: {index}/{len(records)} "
            f"frame={record['frame_id']} proposals={len(record['masks'])}",
            flush=True,
        )
        output = output_root / f"frame_{record['frame_id']:05d}.pt"
        if output.exists():
            frame = torch.load(output, map_location="cpu", weights_only=False)
            if (
                frame.get("schema") != FRAME_SCHEMA
                or int(frame.get("frame_id", -1)) != int(record["frame_id"])
                or frame.get("metadata", {}).get("source_image_sha256")
                != record["image_sha256"]
                or frame.get("metadata", {}).get("sam_fragment_payload_sha256")
                != record["sam_sha256"]
                or torch.as_tensor(frame.get("masked_crop_descriptor")).shape
                != (len(record["masks"]), 1536)
                or torch.as_tensor(frame.get("context_crop_descriptor")).shape
                != (len(record["masks"]), 1536)
            ):
                raise ValueError("partial fragment language frame cannot be resumed safely")
            print(f"reusing verified fragment language frame={record['frame_id']}", flush=True)
        else:
            if runtime is None:
                runtime = OfficialCropSummaryRuntime.load(
                    checkpoint_path=checkpoint,
                    radio_repo=args.radio_repo,
                    version=args.radio_version,
                    device=device,
                    parameter_dtype=(
                        torch.float16
                        if args.runtime_dtype == "float16" else torch.float32
                    ),
                )
                if runtime.radio_checkpoint_sha256 != checkpoint_sha256:
                    raise ValueError("loaded RADIO runtime differs from the sealed checkpoint")
            frame = build_frame_memory(
                record,
                runtime,
                device=device,
                crop_resolution=args.crop_resolution,
                batch_size=args.batch_size,
                context_expansion=args.context_expansion,
                masked_background_rgb=fill,
            )
            _atomic_torch_save(frame, output)
        outputs.append({
            "frame_id": int(record["frame_id"]),
            "source_image": str(record["image_path"]),
            "source_image_sha256": record["image_sha256"],
            "sam_fragment_payload": str(record["sam_path"]),
            "sam_fragment_payload_sha256": record["sam_sha256"],
            "output": str(output),
            "output_sha256": sha256_file(output),
            "proposal_count": int(frame["masked_crop_descriptor"].shape[0]),
        })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scene_label": args.scene_label,
        "status": "query_free_fragment_region_keys_materialized",
        "source_rgb_authority": str(authority_path.resolve()),
        "source_rgb_authority_sha256": sha256_file(authority_path.resolve()),
        "sam_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in sam_paths
        ],
        "radio_checkpoint": str(checkpoint),
        "radio_checkpoint_sha256": checkpoint_sha256,
        "radio_version": args.radio_version,
        "descriptor_contract": {
            "teacher_space": "official_siglip2_crop_summary",
            "masked_background_rgb": list(fill),
            "context_expansion": float(args.context_expansion),
            "crop_resolution": int(args.crop_resolution),
            "batch_size": int(args.batch_size),
            "output_dtype": "float16",
            "runtime_parameter_dtype": args.runtime_dtype,
            "per_token_summary_head_forbidden": True,
            "multiple_fragment_prototypes_preserved": True,
        },
        "information_policy": {
            "query_free": True,
            "source_only": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_rgb_opened": False,
        },
        "outputs": outputs,
    }
    _atomic_json_write(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--expected-radio-checkpoint-sha256", required=True)
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-resolution", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--runtime-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--context-expansion", type=float, default=1.5)
    parser.add_argument(
        "--masked-background-rgb", nargs=3, type=float, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "scene_label": report["scene_label"],
        "frame_count": len(report["outputs"]),
        "proposal_count": sum(item["proposal_count"] for item in report["outputs"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
