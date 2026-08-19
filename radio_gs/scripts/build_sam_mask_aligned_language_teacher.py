#!/usr/bin/env python3
"""Build a query-free SAM-mask-aligned official SigLIP2 language teacher.

The producer consumes only pre-registered source RGBs and a byte-bound official
SAM proposal cache.  It emits two frozen descriptors per proposal: a tight
masked crop for region identity and an expanded unmasked crop for contextual
disambiguation.  Geometric containment and shared-parent sibling candidates
are preserved as metadata; they are deliberately not treated as semantic
labels.

This reuses the official crop-summary runtime and atomic writers from
``extract_official_crop_summary_teacher.py``.  It never opens benchmark masks,
labels, vocabulary, evaluation RGBs, or text queries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.scripts.build_sam3_automatic_mask_cache import (
    unpack_masks,
    validate_automatic_mask_source_binding,
)
from radio_gs.scripts.extract_official_crop_summary_teacher import (
    _atomic_json_write,
    _atomic_torch_save,
)
from radio_gs.utils.immutable_artifacts import sha256_file


PREREGISTRATION_SCHEMA = (
    "radio_gs.sam_mask_aligned_language_teacher_preregistration.v1"
)
TEACHER_SCHEMA = "radio_gs.sam_mask_aligned_language_teacher.v1"
MANIFEST_SCHEMA = "radio_gs.sam_mask_aligned_language_teacher_manifest.v1"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_exact_keys(mapping: dict, keys: Iterable[str], *, label: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ValueError(f"{label} lacks required keys: {missing}")


def validate_preregistration(preregistration: dict) -> dict:
    """Validate the immutable, query-free source-view execution contract."""

    _require_exact_keys(
        preregistration,
        (
            "schema",
            "candidate_id",
            "scene",
            "source_contract",
            "inputs",
            "descriptor_contract",
            "topology_contract",
            "output_root",
        ),
        label="teacher preregistration",
    )
    if preregistration["schema"] != PREREGISTRATION_SCHEMA:
        raise ValueError("teacher preregistration schema differs")
    source = dict(preregistration["source_contract"])
    required_false = (
        "benchmark_labels_opened",
        "benchmark_masks_opened",
        "benchmark_vocabulary_opened",
        "evaluation_rgb_opened",
        "text_queries_opened",
    )
    if source.get("query_free") is not True or source.get("source_only") is not True:
        raise ValueError("teacher must be query-free and source-only")
    if any(source.get(key) is not False for key in required_false):
        raise ValueError("teacher source contract opens a forbidden channel")

    inputs = dict(preregistration["inputs"])
    _require_exact_keys(
        inputs,
        (
            "source_image_root",
            "source_image_stems",
            "mask_root",
            "mask_manifest",
            "required_mask_generation_contract",
            "radio_checkpoint",
            "radio_checkpoint_sha256",
            "radio_repo",
            "radio_version",
        ),
        label="teacher inputs",
    )
    stems = list(inputs["source_image_stems"])
    if not stems or any(not isinstance(stem, str) or not stem for stem in stems):
        raise ValueError("source_image_stems must be a non-empty string list")
    if len(stems) != len(set(stems)) or stems != sorted(stems):
        raise ValueError("source_image_stems must be unique and sorted")
    generation = dict(inputs["required_mask_generation_contract"])
    if (
        generation.get("query_free") is not True
        or generation.get("official_decoder") is not True
        or generation.get("schema_version") != 2
    ):
        raise ValueError("official query-free SAM generation contract is required")

    descriptor = dict(preregistration["descriptor_contract"])
    _require_exact_keys(
        descriptor,
        (
            "teacher_space",
            "masked_background_rgb",
            "context_expansion",
            "crop_resolution",
            "batch_size",
            "output_dtype",
        ),
        label="descriptor contract",
    )
    if descriptor["teacher_space"] != "official_siglip2_crop_summary":
        raise ValueError("only the official SigLIP2 crop-summary space is allowed")
    fill = list(descriptor["masked_background_rgb"])
    if len(fill) != 3 or any(not 0.0 <= float(value) <= 1.0 for value in fill):
        raise ValueError("masked_background_rgb must contain three values in [0,1]")
    if float(descriptor["context_expansion"]) < 1.0:
        raise ValueError("context_expansion must be at least one")
    if int(descriptor["crop_resolution"]) <= 0 or int(descriptor["batch_size"]) <= 0:
        raise ValueError("crop resolution and batch size must be positive")
    if descriptor["output_dtype"] != "float16":
        raise ValueError("v1 descriptor output must be float16")

    topology = dict(preregistration["topology_contract"])
    _require_exact_keys(
        topology,
        (
            "containment_threshold",
            "maximum_child_parent_area_ratio",
            "sibling_maximum_iou",
            "semantic_assertion",
        ),
        label="topology contract",
    )
    if not 0.0 < float(topology["containment_threshold"]) <= 1.0:
        raise ValueError("containment_threshold must lie in (0,1]")
    if not 0.0 < float(topology["maximum_child_parent_area_ratio"]) < 1.0:
        raise ValueError("maximum_child_parent_area_ratio must lie in (0,1)")
    if not 0.0 <= float(topology["sibling_maximum_iou"]) <= 1.0:
        raise ValueError("sibling_maximum_iou must lie in [0,1]")
    if topology["semantic_assertion"] != "geometric_candidates_only":
        raise ValueError("mask topology cannot be promoted to semantic truth")
    return preregistration


def _manifest_records(manifest: dict) -> dict[str, dict]:
    records = manifest.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("automatic-mask manifest has no image records")
    result: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("automatic-mask image record is not an object")
        image = _resolved(record.get("image", ""))
        stem = image.stem
        if not stem or stem in result:
            raise ValueError("automatic-mask image stems are empty or duplicated")
        result[stem] = record
    return result


def _validate_mask_payload(
    payload: dict,
    *,
    image_path: Path,
    expected_generation_contract: dict,
) -> np.ndarray:
    validate_automatic_mask_source_binding(
        payload,
        image_path,
        expected_generation_contract=expected_generation_contract,
    )
    required = (
        "packed_masks",
        "mask_shape",
        "scores",
        "stability",
        "seed_xy",
        "prompt_index",
        "candidate_index",
        "boxes_xyxy",
        "proposal_area_fraction",
    )
    _require_exact_keys(payload, required, label=f"automatic-mask payload {image_path}")
    shape = tuple(int(value) for value in payload["mask_shape"])
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("automatic-mask shape must be positive H,W")
    with Image.open(image_path) as image:
        if (image.height, image.width) != shape:
            raise ValueError("automatic masks and source RGB have different shapes")
    packed = torch.as_tensor(payload["packed_masks"])
    if packed.dtype != torch.uint8 or packed.ndim != 3:
        raise ValueError("packed automatic masks must be uint8 [M,H,ceil(W/8)]")
    if tuple(packed.shape[1:]) != (shape[0], math.ceil(shape[1] / 8)):
        raise ValueError("packed automatic-mask geometry differs")
    masks = unpack_masks(packed, width=shape[1])
    count = int(masks.shape[0])
    expected_shapes = {
        "scores": (count,),
        "stability": (count,),
        "seed_xy": (count, 2),
        "prompt_index": (count,),
        "candidate_index": (count,),
        "boxes_xyxy": (count, 4),
        "proposal_area_fraction": (count,),
    }
    for key, expected in expected_shapes.items():
        value = torch.as_tensor(payload[key])
        if tuple(value.shape) != expected:
            raise ValueError(f"automatic-mask {key} shape differs")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"automatic-mask {key} is non-finite")
    scores = torch.as_tensor(payload["scores"]).float()
    stability = torch.as_tensor(payload["stability"]).float()
    if count == 0 or not bool(((scores >= 0) & (scores <= 1)).all()):
        raise ValueError("automatic-mask scores are empty or outside [0,1]")
    if not bool(((stability >= 0) & (stability <= 1)).all()):
        raise ValueError("automatic-mask stability lies outside [0,1]")
    boxes = torch.as_tensor(payload["boxes_xyxy"]).long()
    areas = torch.as_tensor(payload["proposal_area_fraction"]).float()
    for index, mask in enumerate(masks):
        y, x = np.where(mask)
        if x.size == 0:
            raise ValueError("automatic-mask proposal is empty")
        expected_box = torch.tensor(
            [x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=torch.long
        )
        if not torch.equal(boxes[index], expected_box):
            raise ValueError("automatic-mask box differs from packed support")
        expected_area = float(mask.mean())
        if abs(float(areas[index]) - expected_area) > 1e-6:
            raise ValueError("automatic-mask area differs from packed support")
    return masks


def preflight_inputs(preregistration_path: Path) -> tuple[dict, list[dict]]:
    """Validate every input hash and contract before any model is loaded."""

    preregistration = validate_preregistration(_load_json(preregistration_path))
    inputs = dict(preregistration["inputs"])
    source_root = _resolved(inputs["source_image_root"])
    mask_root = _resolved(inputs["mask_root"])
    manifest_path = _resolved(inputs["mask_manifest"])
    if manifest_path.parent != mask_root:
        raise ValueError("automatic-mask manifest is outside its declared root")
    manifest = _load_json(manifest_path)
    expected_generation = dict(inputs["required_mask_generation_contract"])
    if manifest.get("generation_contract") != expected_generation:
        raise ValueError("automatic-mask root generation contract differs")
    if _resolved(manifest.get("output_root", "")) != mask_root:
        raise ValueError("automatic-mask manifest output root differs")
    records = _manifest_records(manifest)
    selected: list[dict] = []
    for stem in inputs["source_image_stems"]:
        if stem not in records:
            raise ValueError(f"pre-registered source image is absent: {stem}")
        record = records[stem]
        image_path = _resolved(record.get("image", ""))
        mask_path = _resolved(record.get("output", ""))
        if image_path.parent != source_root or image_path.stem != stem:
            raise ValueError(f"source image escapes its pre-registered root: {stem}")
        if mask_path != mask_root / f"{stem}.pt":
            raise ValueError(f"automatic-mask payload path differs: {stem}")
        if sha256_file(image_path) != str(record.get("source_image_sha256", "")):
            raise ValueError(f"source image SHA differs: {stem}")
        if sha256_file(mask_path) != str(record.get("output_sha256", "")):
            raise ValueError(f"automatic-mask payload SHA differs: {stem}")
        payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"automatic-mask payload is not an object: {stem}")
        masks = _validate_mask_payload(
            payload,
            image_path=image_path,
            expected_generation_contract=expected_generation,
        )
        if int(record.get("masks", -1)) != int(masks.shape[0]):
            raise ValueError(f"automatic-mask manifest count differs: {stem}")
        selected.append(
            {
                "stem": stem,
                "image_path": image_path,
                "mask_path": mask_path,
                "source_image_sha256": str(record["source_image_sha256"]),
                "mask_payload_sha256": str(record["output_sha256"]),
                "payload": payload,
                "masks": masks,
            }
        )
    checkpoint = _resolved(inputs["radio_checkpoint"])
    if sha256_file(checkpoint) != str(inputs["radio_checkpoint_sha256"]):
        raise ValueError("official RADIO checkpoint SHA differs")
    if _resolved(preregistration["output_root"]) == source_root:
        raise ValueError("teacher output cannot overwrite source RGBs")
    return preregistration, selected


def expanded_box(
    box_xyxy: Iterable[int], *, height: int, width: int, expansion: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box_xyxy)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    expanded_w = max(1.0, (x1 - x0) * float(expansion))
    expanded_h = max(1.0, (y1 - y0) * float(expansion))
    left = max(0, int(math.floor(center_x - 0.5 * expanded_w)))
    top = max(0, int(math.floor(center_y - 0.5 * expanded_h)))
    right = min(width, int(math.ceil(center_x + 0.5 * expanded_w)))
    bottom = min(height, int(math.ceil(center_y + 0.5 * expanded_h)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def build_crop_pairs(
    image: torch.Tensor,
    masks: np.ndarray,
    boxes_xyxy: torch.Tensor,
    *,
    context_expansion: float,
    crop_resolution: int,
    masked_background_rgb: Iterable[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [M,3,R,R] masked crops and expanded context crops."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("source RGB must be [3,H,W]")
    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(image.shape[1:]):
        raise ValueError("mask/source RGB geometry differs")
    boxes = torch.as_tensor(boxes_xyxy).long()
    if tuple(boxes.shape) != (masks.shape[0], 4):
        raise ValueError("proposal boxes must be [M,4]")
    fill = torch.tensor(list(masked_background_rgb), dtype=image.dtype)[:, None, None]
    masked_crops: list[torch.Tensor] = []
    context_crops: list[torch.Tensor] = []
    height, width = (int(value) for value in image.shape[1:])
    for index, box in enumerate(boxes.tolist()):
        x0, y0, x1, y1 = box
        tight = image[:, y0:y1, x0:x1]
        support = torch.from_numpy(masks[index, y0:y1, x0:x1]).bool()[None]
        masked = torch.where(support, tight, fill)
        cx0, cy0, cx1, cy1 = expanded_box(
            box, height=height, width=width, expansion=context_expansion
        )
        context = image[:, cy0:cy1, cx0:cx1]
        masked_crops.append(
            F.interpolate(
                masked[None],
                size=(crop_resolution, crop_resolution),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
        context_crops.append(
            F.interpolate(
                context[None],
                size=(crop_resolution, crop_resolution),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(masked_crops), torch.stack(context_crops)


def build_geometric_topology(
    masks: np.ndarray,
    *,
    containment_threshold: float,
    maximum_child_parent_area_ratio: float,
    sibling_maximum_iou: float,
) -> dict[str, torch.Tensor]:
    """Build directional containment and shared-direct-parent candidates."""

    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("masks must be non-empty [M,H,W]")
    count = values.shape[0]
    areas = values.reshape(count, -1).sum(axis=1, dtype=np.int64)
    intersection = np.zeros((count, count), dtype=np.int64)
    for left in range(count):
        for right in range(left, count):
            value = np.logical_and(values[left], values[right]).sum(dtype=np.int64)
            intersection[left, right] = value
            intersection[right, left] = value
    parent = np.full(count, -1, dtype=np.int32)
    parent_containment = np.zeros(count, dtype=np.float32)
    all_edges: list[tuple[int, int]] = []
    for child in range(count):
        candidates: list[tuple[int, int, float]] = []
        for whole in range(count):
            if child == whole or areas[child] >= areas[whole]:
                continue
            containment = float(intersection[child, whole] / max(areas[child], 1))
            area_ratio = float(areas[child] / max(areas[whole], 1))
            if (
                containment >= float(containment_threshold)
                and area_ratio <= float(maximum_child_parent_area_ratio)
            ):
                all_edges.append((child, whole))
                candidates.append((int(areas[whole]), whole, containment))
        if candidates:
            _area, direct_parent, containment = min(candidates)
            parent[child] = direct_parent
            parent_containment[child] = containment
    siblings: list[tuple[int, int]] = []
    for left in range(count):
        for right in range(left + 1, count):
            if parent[left] < 0 or parent[left] != parent[right]:
                continue
            union = areas[left] + areas[right] - intersection[left, right]
            iou = float(intersection[left, right] / max(union, 1))
            if iou <= float(sibling_maximum_iou):
                siblings.append((left, right))
    return {
        "direct_parent_index": torch.from_numpy(parent),
        "direct_parent_containment": torch.from_numpy(parent_containment),
        "candidate_part_of_edges": torch.tensor(all_edges, dtype=torch.int32).reshape(-1, 2),
        "candidate_sibling_edges": torch.tensor(siblings, dtype=torch.int32).reshape(-1, 2),
    }


@torch.inference_mode()
def encode_in_batches(
    runtime: OfficialCropSummaryRuntime,
    crops: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    descriptors = []
    for start in range(0, crops.shape[0], int(batch_size)):
        descriptors.append(runtime.encode(crops[start : start + batch_size].to(device)).cpu())
    output = torch.cat(descriptors).float()
    if output.ndim != 2 or output.shape[1] != 1536:
        raise ValueError("official SigLIP2 descriptor shape differs")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("official SigLIP2 descriptor is non-finite")
    norm = torch.linalg.vector_norm(output, dim=1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=1e-4):
        raise ValueError("official SigLIP2 descriptor is not unit normalized")
    return output.half()


def build_frame_teacher(
    record: dict,
    runtime: OfficialCropSummaryRuntime,
    *,
    descriptor_contract: dict,
    topology_contract: dict,
    device: torch.device,
) -> dict:
    payload = record["payload"]
    image = pil_to_tensor(Image.open(record["image_path"]).convert("RGB")).float().div_(255.0)
    masked_crops, context_crops = build_crop_pairs(
        image,
        record["masks"],
        torch.as_tensor(payload["boxes_xyxy"]),
        context_expansion=float(descriptor_contract["context_expansion"]),
        crop_resolution=int(descriptor_contract["crop_resolution"]),
        masked_background_rgb=descriptor_contract["masked_background_rgb"],
    )
    batch_size = int(descriptor_contract["batch_size"])
    masked_descriptors = encode_in_batches(
        runtime, masked_crops, batch_size=batch_size, device=device
    )
    context_descriptors = encode_in_batches(
        runtime, context_crops, batch_size=batch_size, device=device
    )
    topology = build_geometric_topology(
        record["masks"],
        containment_threshold=float(topology_contract["containment_threshold"]),
        maximum_child_parent_area_ratio=float(
            topology_contract["maximum_child_parent_area_ratio"]
        ),
        sibling_maximum_iou=float(topology_contract["sibling_maximum_iou"]),
    )
    boxes = torch.as_tensor(payload["boxes_xyxy"]).int()
    context_boxes = torch.tensor(
        [
            expanded_box(
                box,
                height=int(image.shape[1]),
                width=int(image.shape[2]),
                expansion=float(descriptor_contract["context_expansion"]),
            )
            for box in boxes.tolist()
        ],
        dtype=torch.int32,
    ).reshape(-1, 4)
    return {
        "schema": TEACHER_SCHEMA,
        "masked_crop_descriptor": masked_descriptors,
        "context_crop_descriptor": context_descriptors,
        "tight_boxes_xyxy": boxes,
        "context_boxes_xyxy": context_boxes,
        "proposal_area_fraction": torch.as_tensor(payload["proposal_area_fraction"]).float(),
        "sam_quality": torch.as_tensor(payload["scores"]).float(),
        "sam_stability": torch.as_tensor(payload["stability"]).float(),
        "sam_seed_xy": torch.as_tensor(payload["seed_xy"]).float(),
        "sam_prompt_index": torch.as_tensor(payload["prompt_index"]).int(),
        "sam_candidate_index": torch.as_tensor(payload["candidate_index"]).to(torch.int8),
        "topology": topology,
        "metadata": {
            "teacher_space": "official_siglip2_crop_summary",
            "text_compatibility": "official_siglip2_g_text_space",
            "query_free": True,
            "source_only": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_vocabulary_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "source_image": str(record["image_path"]),
            "source_image_sha256": record["source_image_sha256"],
            "automatic_mask_payload": str(record["mask_path"]),
            "automatic_mask_payload_sha256": record["mask_payload_sha256"],
            "descriptor_contract": descriptor_contract,
            "topology_contract": topology_contract,
            "topology_semantics": (
                "geometric candidates only; sibling candidates are not semantic negatives "
                "and part-of candidates are not semantic assertions"
            ),
        },
    }


def run(preregistration_path: Path, *, device_name: str) -> dict:
    preregistration, records = preflight_inputs(preregistration_path)
    output_root = _resolved(preregistration["output_root"])
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"teacher manifest already exists: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = dict(preregistration["inputs"])
    device = torch.device(device_name)
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=inputs["radio_checkpoint"],
        radio_repo=inputs["radio_repo"],
        version=inputs["radio_version"],
        device=device,
    )
    if runtime.radio_checkpoint_sha256 != inputs["radio_checkpoint_sha256"]:
        raise ValueError("loaded official RADIO checkpoint identity differs")
    outputs = []
    for record in records:
        teacher = build_frame_teacher(
            record,
            runtime,
            descriptor_contract=dict(preregistration["descriptor_contract"]),
            topology_contract=dict(preregistration["topology_contract"]),
            device=device,
        )
        output = output_root / f"{record['stem']}.pt"
        if output.exists():
            raise FileExistsError(f"teacher output exists: {output}")
        _atomic_torch_save(teacher, output)
        outputs.append(
            {
                "stem": record["stem"],
                "source_image": str(record["image_path"]),
                "source_image_sha256": record["source_image_sha256"],
                "automatic_mask_payload": str(record["mask_path"]),
                "automatic_mask_payload_sha256": record["mask_payload_sha256"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "proposal_count": int(record["masks"].shape[0]),
                "candidate_part_of_edges": int(
                    teacher["topology"]["candidate_part_of_edges"].shape[0]
                ),
                "candidate_sibling_edges": int(
                    teacher["topology"]["candidate_sibling_edges"].shape[0]
                ),
            }
        )
    report = {
        "schema": MANIFEST_SCHEMA,
        "candidate_id": preregistration["candidate_id"],
        "status": "sparse_source_view_teacher_materialized",
        "scope_limitation": (
            "descriptor-contract sentinel only; its grid12 single-image proposals "
            "are not a completed automatic multiscale hierarchy"
        ),
        "scene": preregistration["scene"],
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": sha256_file(preregistration_path),
        "source_contract": preregistration["source_contract"],
        "descriptor_contract": preregistration["descriptor_contract"],
        "topology_contract": preregistration["topology_contract"],
        "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
        "mask_generation_contract": inputs["required_mask_generation_contract"],
        "outputs": outputs,
        "benchmark_metrics_computed": False,
        "method_promotion_eligible": False,
    }
    _atomic_json_write(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.preregistration), device_name=args.device), indent=2))


if __name__ == "__main__":
    main()
