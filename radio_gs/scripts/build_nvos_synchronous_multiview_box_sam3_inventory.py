#!/usr/bin/env python3
"""Build a sealed all-view NVOS inventory from official SAM3 box regions.

The frozen field owns query identity and supplies one coarse box plus signed
selection evidence in every registered view.  Official SAM3 owns object extent:
all geometric-prompt proposals are produced before one is selected by the same
target-blind signed-evidence rule used by the retained Method-v1 full8 result.
The selected per-view observations are then consumed by the exact-adjoint
positive/unknown fusion stage.  No target mask or metric is reachable here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest
from radio_gs.querying.synchronous_multiview_candidate_marginal import (
    QueryAbstention,
)
from radio_gs.scripts.build_nvos_synchronous_multiview_sam3_inventory import (
    FROZEN_SAM3_SHA256,
    _atomic_json,
    _atomic_numpy,
    _load_array,
    _load_bound,
    _sha256,
    validate_plan,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import (
    choose_candidate_by_signed_points,
    mask_to_box,
)


INVENTORY_TYPE = "nvos_synchronous_multiview_box_sam3_exact_adjoint_inventory_v1"


def _candidate_digest(plan_sha256: str) -> str:
    return hashlib.sha256(
        f"{INVENTORY_TYPE}:{plan_sha256}".encode("utf-8")
    ).hexdigest()


@torch.inference_mode()
def predict_box_view(
    processor: Any,
    record: Mapping[str, Any],
    *,
    device: str,
    box_padding_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb_path = _load_bound(record.get("rgb", {}), label="registered RGB")
    probability = _load_array(
        record.get("projected_probability", {}), label="projected probability"
    ).astype(np.float32, copy=False)
    visibility = _load_array(
        record.get("visibility", {}), label="projected visibility"
    ).astype(bool, copy=False)
    if probability.shape != visibility.shape:
        raise QueryAbstention("projected probability and visibility axes differ")
    native_record = record.get("native_signed_margin")
    if isinstance(native_record, Mapping):
        native_path = _load_bound(native_record, label="native signed field prompt")
        signed_margin = np.load(native_path, allow_pickle=False).astype(
            np.float32, copy=False
        )
        if signed_margin.ndim != 2 or not bool(np.isfinite(signed_margin).all()):
            raise QueryAbstention("native signed field prompt is malformed")
        # Match the retained full8 box compiler exactly: threshold in the
        # native 47x62 render domain, then nearest-resize the binary support.
        coarse = resize_mask_nearest(
            signed_margin >= 0.0, probability.shape
        ).astype(bool)
        coarse &= visibility
        signed_evidence_semantics = "sealed_native_margin_exact_replay"
    else:
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        signed_margin = np.log(clipped) - np.log1p(-clipped)
        # Unsupported pixels are unknown and cannot enlarge the coarse box.
        coarse = (probability >= 0.5) & visibility
        signed_evidence_semantics = "transported_visible_field_log_odds"
    box = mask_to_box(coarse, padding_pixels=int(box_padding_pixels))
    if box is None:
        raise QueryAbstention("registered view has no field-owned coarse box")

    height, width = probability.shape
    image = Image.open(rgb_path).convert("RGB")
    original_size = list(image.size)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    amp_dtype = torch.bfloat16 if str(device).startswith("cuda") else None
    with sam3_autocast_context(device, amp_dtype):
        state = processor.set_image(image)
        output = processor.add_geometric_prompt(box, True, dict(state))
    masks = output.get("masks")
    if masks is None:
        logits = output.get("masks_logits")
        masks = None if logits is None else logits.float() > 0.0
    if masks is None:
        raise QueryAbstention("official SAM3 box call returned no proposal mask")
    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    scores = output.get("scores")
    scores_np = (
        scores.detach().float().cpu().numpy()
        if torch.is_tensor(scores)
        else np.asarray(scores if scores is not None else [], dtype=np.float32)
    )
    selected, report = choose_candidate_by_signed_points(
        signed_margin, coarse, masks_np, scores=scores_np
    )
    if report.get("accepted") is not True:
        raise QueryAbstention(
            f"official SAM3 box proposals abstained: {report.get('fallback_reason')}"
        )
    selected = np.asarray(selected, dtype=np.float32)
    if selected.shape != probability.shape:
        raise QueryAbstention("selected SAM3 box proposal axes differ")
    return selected, {
        "rgb": {"path": str(rgb_path), "sha256": _sha256(rgb_path)},
        "rgb_original_size_wh": original_size,
        "sam_size_wh": [width, height],
        "box_prompt_cxcywh_norm": list(box),
        "box_padding_pixels": int(box_padding_pixels),
        "proposal_selection": "frozen_signed_field_evidence_then_coarse_overlap_then_sam_score",
        "signed_evidence_semantics": signed_evidence_semantics,
        **report,
    }


def run(args: argparse.Namespace, *, processor: Any | None = None) -> dict[str, Any]:
    plan_path = Path(args.plan).expanduser().resolve(strict=True)
    if _sha256(plan_path) != str(args.expected_plan_sha256):
        raise ValueError("candidate/view plan SHA-256 differs")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidates, canonical_views = validate_plan(plan, expected_candidates=10)
    # Candidate identities in the input plan describe point trials.  Box-SAM
    # has one selected region observation per view, so only the sealed static
    # view cohort is reused and a new single-candidate identity is minted.
    first_views = candidates[0]["views"]
    view_by_digest = {str(row["view_digest"]): row for row in first_views}
    if tuple(sorted(view_by_digest)) != canonical_views:
        raise QueryAbstention("registered view cohort is incomplete")

    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if _sha256(checkpoint) != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
    output = Path(args.output_dir).expanduser().resolve()
    inventory_path = output / "inventory.json"
    if inventory_path.exists():
        raise FileExistsError(inventory_path)
    if processor is None:
        set_requested_cuda_device(args.device)
        processor = _load_sam3_model(
            checkpoint_path=str(checkpoint),
            device=args.device,
            confidence_threshold=0.0,
            dtype="float32",
            resolution=int(args.resolution),
            point_only=False,
            build_on_cpu=True,
        )

    digest = _candidate_digest(str(args.expected_plan_sha256))
    views: list[dict[str, Any]] = []
    for view_digest in canonical_views:
        record = dict(view_by_digest[view_digest])
        if record.get("projected_probability_semantics") == "sealed_native_signed_field_prompt":
            record["native_signed_margin"] = dict(
                plan.get("inputs", {}).get("signed_field_prompt", {})
            )
        probability, details = predict_box_view(
            processor,
            record,
            device=args.device,
            box_padding_pixels=int(args.box_padding_pixels),
        )
        probability_path = output / "probabilities" / digest / f"{view_digest}.npy"
        probability_sha = _atomic_numpy(probability_path, probability)
        receipt_path = output / "receipts" / digest / f"{view_digest}.json"
        receipt = {
            "schema_version": 1,
            "artifact_type": "nvos_synchronous_multiview_box_sam3_cell_v1",
            "candidate_digest": digest,
            "view_digest": view_digest,
            "probability": {"path": str(probability_path), "sha256": probability_sha},
            "assignment": dict(record["assignment"]),
            "log_precision": float(record["log_precision"]),
            "official_sam3": details,
            "target_mask_opened": False,
            "target_metric_opened": False,
        }
        receipt_sha = _atomic_json(receipt_path, receipt)
        views.append(
            {
                "view_digest": view_digest,
                "probability": {"path": str(probability_path), "sha256": probability_sha},
                "assignment": dict(record["assignment"]),
                "log_precision": float(record["log_precision"]),
                "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
            }
        )
    inventory = {
        "schema_version": 1,
        "artifact_type": INVENTORY_TYPE,
        "scene_id": plan.get("scene_id"),
        "num_gaussians": int(plan["num_gaussians"]),
        "plan": {"path": str(plan_path), "sha256": str(args.expected_plan_sha256)},
        "official_sam3_checkpoint": {
            "path": str(checkpoint),
            "sha256": str(args.expected_checkpoint_sha256),
        },
        "candidate_count": 1,
        "view_count": len(canonical_views),
        "candidate_digests": [digest],
        "view_digests": list(canonical_views),
        "candidates": [
            {"candidate_digest": digest, "candidate_logit": 0.0, "views": views}
        ],
        "all_candidate_view_predictions_sealed": True,
        "candidate_selection": "frozen_signed_evidence_no_target_metric",
        "view_selection": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    _atomic_json(inventory_path, inventory)
    return {**inventory, "inventory_path": str(inventory_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=FROZEN_SAM3_SHA256)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--box-padding-pixels", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps({"inventory": result["inventory_path"], "view_count": result["view_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
