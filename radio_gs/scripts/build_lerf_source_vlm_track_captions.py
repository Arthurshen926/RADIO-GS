#!/usr/bin/env python3
"""Caption source-only physical object tracks from multiple registered views.

The compiler opens only source RGB and query-independent SAM proposal boxes.
It never opens LERF evaluation RGB, masks, text queries, or metrics.  Captions
remain candidate language authority until a separate source-only quality gate
accepts their cross-view consistency and sibling discriminability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_payload,
    write_frozen_json,
    write_torch_noclobber,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expanded_box(box: torch.Tensor, width: int, height: int, context: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) for value in box.tolist())
    dx, dy = (x1 - x0) * context, (y1 - y0) * context
    result = (
        max(0, int(x0 - dx)), max(0, int(y0 - dy)),
        min(width, int(x1 + dx + 0.999)), min(height, int(y1 + dy + 0.999)),
    )
    if result[2] - result[0] < 2 or result[3] - result[1] < 2:
        raise ValueError("source proposal crop is empty")
    return result


def _select_track_proposals(
    episodes: dict[str, Any], membership: dict[str, Any], object_id: int, views: int,
) -> list[int]:
    selected_episode = torch.as_tensor(episodes["episode_object_id"]).long() == int(object_id)
    proposals = torch.unique(torch.cat((
        torch.as_tensor(episodes["episode_query_proposal"])[selected_episode],
        torch.as_tensor(episodes["episode_target_proposal"])[selected_episode],
    )).long())
    proposal_view = torch.as_tensor(membership["proposal_view_indices"]).long()
    score = torch.as_tensor(membership["proposal_scores"]).float()
    stability = torch.as_tensor(membership["proposal_stability"]).float()
    area = torch.as_tensor(membership["proposal_area_fraction"]).float()
    quality = score[proposals] * stability[proposals] * (1.0 - area[proposals]).clamp_min(0.05)
    order = proposals[torch.argsort(quality, descending=True, stable=True)]
    output: list[int] = []
    used_views: set[int] = set()
    for proposal in order.tolist():
        view = int(proposal_view[proposal])
        if view in used_views:
            continue
        output.append(proposal); used_views.add(view)
        if len(output) == views:
            break
    if not output:
        raise ValueError(f"physical object track {object_id} has no proposal")
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= float(args.background_context_alpha) <= 1.0:
        raise ValueError("background context alpha must lie in [0,1]")
    episodes, episodes_sha, episodes_path = load_torch_payload(
        args.episodes, expected_sha256=args.expected_episodes_sha256,
        label="source physical object episodes",
    )
    membership, membership_sha, membership_path = load_torch_payload(
        args.membership, expected_sha256=args.expected_membership_sha256,
        label="source query-independent SAM memberships",
    )
    if episodes.get("schema") != "radio_gs.lerf_cross_view_object_episodes.v2":
        raise ValueError("source physical object episode schema differs")
    metadata = membership.get("metadata", {})
    if any(bool(metadata.get(key, False)) for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened", "text_queries_opened",
    )):
        raise ValueError("membership provenance opened a forbidden evaluation channel")
    records = metadata.get("source_records")
    if not isinstance(records, list) or not records:
        raise ValueError("source membership records are absent")

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True,
    ).eval()
    device = torch.device(args.device)
    if device.type == "cuda":
        model = model.half().to(device)
    else:
        model = model.to(device)

    proposal_view = torch.as_tensor(membership["proposal_view_indices"]).long()
    proposal_box = torch.as_tensor(membership["proposal_boxes_xyxy"]).long()
    objects: list[dict[str, Any]] = []
    prompt = args.task
    for object_id in torch.unique(torch.as_tensor(episodes["episode_object_id"]).long()).tolist():
        captions: list[dict[str, Any]] = []
        for proposal in _select_track_proposals(episodes, membership, object_id, args.views_per_track):
            view = int(proposal_view[proposal])
            record = records[view]
            image_path = Path(record["source_image"]).resolve()
            if _sha256(image_path) != record["source_image_sha256"]:
                raise ValueError(f"source image hash differs: {image_path}")
            image = Image.open(image_path).convert("RGB")
            crop_box = _expanded_box(proposal_box[proposal], image.width, image.height, args.context_fraction)
            if args.mask_background:
                mask_path = Path(record["mask_cache"]).resolve()
                if _sha256(mask_path) != record["mask_cache_sha256"]:
                    raise ValueError(f"source mask cache hash differs: {mask_path}")
                mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
                packed = torch.as_tensor(mask_payload["packed_masks"]).cpu().numpy()
                width = int(mask_payload["mask_shape"][1])
                masks = np.unpackbits(packed, axis=-1, bitorder="little")[..., :width].astype(bool)
                view_proposals = torch.where(proposal_view == view)[0]
                local_proposal = int(proposal - int(view_proposals.min()))
                if local_proposal < 0 or local_proposal >= masks.shape[0]:
                    raise ValueError("global and per-view proposal axes differ")
                rgb = np.asarray(image).copy()
                background = rgb[~masks[local_proposal]].astype(np.float32)
                alpha = float(args.background_context_alpha)
                rgb[~masks[local_proposal]] = np.rint(
                    (1.0 - alpha) * background + alpha * float(args.background_value)
                ).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(rgb)
            crop = image.crop(crop_box)
            inputs = processor(text=prompt, images=crop, return_tensors="pt")
            inputs = {
                key: value.to(device=device, dtype=torch.float16 if device.type == "cuda" and value.is_floating_point() else None)
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                    do_sample=False,
                )
            decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(decoded, task=prompt, image_size=crop.size)
            caption = parsed[prompt] if isinstance(parsed, dict) else parsed
            captions.append({
                "caption": str(caption).strip(), "proposal_index": int(proposal),
                "source_view_index": view, "frame_id": int(record["frame_id"]),
                "source_image": str(image_path), "source_image_sha256": record["source_image_sha256"],
                "proposal_box_xyxy": proposal_box[proposal].tolist(), "crop_box_xyxy": list(crop_box),
            })
        objects.append({"object_id": int(object_id), "views": captions})

    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.lerf_source_vlm_track_captions.v1", "schema_version": 1,
        "scene": membership.get("scene"), "objects": objects,
        "metadata": {
            "source_only": True, "benchmark_vocabulary_opened": False,
            "benchmark_images_opened": False, "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False, "text_queries_opened": False,
            "status": "candidate_language_authority_requires_source_quality_gate",
            "model_id": args.model_id, "model_revision": args.model_revision,
            "task": prompt, "views_per_track": args.views_per_track,
            "context_fraction": args.context_fraction,
            "mask_background": args.mask_background,
            "background_value": args.background_value,
            "background_context_alpha": args.background_context_alpha,
            "inputs": {
                "episodes": {"path": str(episodes_path), "sha256": episodes_sha},
                "membership": {"path": str(membership_path), "sha256": membership_sha},
            },
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "candidate_source_vlm_track_captions_complete",
        "scene": payload["scene"], "objects": len(objects),
        "captions": sum(len(item["views"]) for item in objects),
        "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--expected-episodes-sha256", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--model-revision", default="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e")
    parser.add_argument("--task", default="<MORE_DETAILED_CAPTION>")
    parser.add_argument("--views-per-track", type=int, default=3)
    parser.add_argument("--context-fraction", type=float, default=0.15)
    parser.add_argument("--mask-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--background-value", type=int, default=127)
    parser.add_argument("--background-context-alpha", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=3)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
