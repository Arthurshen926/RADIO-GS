#!/usr/bin/env python3
"""Run every sealed NVOS candidate/view through frozen official SAM3.

The input plan is produced before target RGB is opened.  It contains exactly
K candidates and the same complete registered-view cohort for every candidate:
one projected signed-field probability, visibility, explicit source-scribble
authority, RGB identity, exact compositor assignment, and source-frozen
precision per cell.  This program performs no candidate or view selection.
Missing signed evidence abstains and no target mask or metric is ever read.

The resulting inventory is consumed by
``materialize_nvos_synchronous_candidate_marginal.py``, which applies the exact
adjoint and robust candidate/view marginal on the frozen Gaussian carrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import string
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.querying.synchronous_multiview_candidate_marginal import (
    QueryAbstention,
)
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    deterministic_signed_point_trials,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    sam3_autocast_context,
    set_requested_cuda_device,
)


PLAN_TYPE = "nvos_synchronous_multiview_candidate_plan_v1"
INVENTORY_TYPE = "nvos_synchronous_multiview_sam3_exact_adjoint_inventory_v1"
FROZEN_SAM3_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    expected = str(record.get("sha256", ""))
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 differs: {path}")
    return path


def _load_array(record: Mapping[str, Any], *, label: str) -> np.ndarray:
    path = _load_bound(record, label=label)
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2:
        raise QueryAbstention(f"{label} must be one two-dimensional raster")
    return np.asarray(value)


def _atomic_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def validate_plan(
    plan: Mapping[str, Any], *, expected_candidates: int
) -> tuple[list[Mapping[str, Any]], tuple[str, ...]]:
    candidates = plan.get("candidates")
    if (
        plan.get("artifact_type") != PLAN_TYPE
        or not isinstance(candidates, list)
        or len(candidates) != int(expected_candidates)
        or int(plan.get("num_gaussians", 0)) <= 0
        or plan.get("all_candidate_view_inputs_sealed") is not True
        or plan.get("target_mask_opened") is not False
        or plan.get("target_metric_opened") is not False
    ):
        raise ValueError("candidate/view plan contract differs")
    candidate_digests = [str(row.get("candidate_digest", "")) for row in candidates]
    hexdigits = set(string.hexdigits.lower())
    if len(set(candidate_digests)) != len(candidate_digests) or any(
        len(value) != 64 or not set(value.lower()).issubset(hexdigits)
        for value in candidate_digests
    ):
        raise QueryAbstention("candidate cohort identity is incomplete or non-unique")
    canonical_views: tuple[str, ...] | None = None
    trial_ranks: list[int] = []
    verified_assignments: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        rank = candidate.get("trial_rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise QueryAbstention("candidate systematic trial rank is absent")
        trial_ranks.append(rank)
        views = candidate.get("views")
        if not isinstance(views, list) or not views:
            raise QueryAbstention("candidate lacks a complete registered-view cohort")
        digests = tuple(sorted(str(view.get("view_digest", "")) for view in views))
        if len(set(digests)) != len(digests) or any(
            len(value) != 64 or not set(value.lower()).issubset(hexdigits)
            for value in digests
        ):
            raise QueryAbstention("view cohort identity is incomplete or non-unique")
        if canonical_views is None:
            canonical_views = digests
        elif digests != canonical_views:
            raise QueryAbstention("candidate registered-view cohorts differ")
        for view in views:
            if view.get("candidate_trial_rank") != rank:
                raise QueryAbstention("candidate/view systematic trial rank differs")
            if not np.isfinite(float(view.get("log_precision", np.nan))):
                raise QueryAbstention("view precision is absent or nonfinite")
            assignment = view.get("assignment")
            if not isinstance(assignment, Mapping):
                raise QueryAbstention("exact assignment lineage is absent")
            view_digest = str(view.get("view_digest", ""))
            identity = dict(assignment)
            prior = verified_assignments.get(view_digest)
            if prior is None:
                _load_bound(identity, label="exact assignment")
                verified_assignments[view_digest] = identity
            elif identity != prior:
                raise QueryAbstention(
                    "exact assignment lineage differs across candidates"
                )
    if canonical_views is None:
        raise QueryAbstention("registered-view cohort is empty")
    if sorted(trial_ranks) != list(range(int(expected_candidates))):
        raise QueryAbstention("candidate systematic trial cohort is incomplete")
    if int(expected_candidates) != int(FROZEN_POLICY.trials):
        raise QueryAbstention("candidate cohort differs from frozen SAM trial policy")
    return candidates, canonical_views


@torch.inference_mode()
def predict_one_candidate_view(
    processor: Any,
    record: Mapping[str, Any],
    *,
    candidate_digest: str,
    points_per_sign: int,
    device: str,
    prepared_view: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute one mandatory candidate/view cell with explicit signed points."""

    rgb_path = _load_bound(record.get("rgb", {}), label="registered RGB")
    projected = _load_array(
        record.get("projected_probability", {}), label="projected probability"
    ).astype(np.float32, copy=False)
    visibility = _load_array(
        record.get("visibility", {}), label="projected visibility"
    ).astype(bool, copy=False)
    positive = _load_array(
        record.get("positive_authority", {}), label="positive authority"
    ).astype(bool, copy=False)
    negative = _load_array(
        record.get("negative_authority", {}), label="negative authority"
    ).astype(bool, copy=False)
    if int(points_per_sign) != int(FROZEN_POLICY.positive_points_per_trial) or (
        int(points_per_sign) != int(FROZEN_POLICY.negative_points_per_trial)
    ):
        raise QueryAbstention("point count differs from frozen SAM trial policy")
    rank = record.get("candidate_trial_rank")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank < int(FROZEN_POLICY.trials)
    ):
        raise QueryAbstention("candidate systematic trial rank is invalid")
    # Missing renderer support stays unknown.  On visible support, however,
    # the signed field is itself an authorized query-specific positive/negative
    # identity prior (it was built from the official signed source prompt).
    # Keep its continuous log-odds instead of reducing it to the much narrower
    # transported scribble-dominance masks.  Those masks remain bound in the
    # plan and enforce the source observation clamp upstream.
    clipped = np.clip(projected, 1e-6, 1.0 - 1e-6)
    signed_log_odds = np.log(clipped) - np.log1p(-clipped)
    positive_score = np.maximum(signed_log_odds, 0.0)
    negative_score = np.maximum(-signed_log_odds, 0.0)
    positive_score[~visibility] = 0.0
    negative_score[~visibility] = 0.0
    point_trials, label_trials = deterministic_signed_point_trials(
        positive_score,
        negative_score,
        image_shape=projected.shape,
        policy=FROZEN_POLICY,
    )
    points = point_trials[rank]
    labels = label_trials[rank]
    height, width = map(int, projected.shape)
    amp_dtype = torch.bfloat16 if str(device).startswith("cuda") else None
    if prepared_view is None:
        image = Image.open(rgb_path).convert("RGB")
        original_size = list(image.size)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        with sam3_autocast_context(device, amp_dtype):
            state = processor.set_image(image)
    else:
        if (
            str(prepared_view.get("rgb_path")) != str(rgb_path)
            or list(prepared_view.get("sam_size_wh", [])) != [width, height]
        ):
            raise QueryAbstention("candidate view differs from prepared RGB embedding")
        state = prepared_view["state"]
        original_size = list(prepared_view["rgb_original_size_wh"])
    masks, quality, low_resolution = processor.model.predict_inst(
        state,
        point_coords=points.astype(np.float32, copy=False),
        point_labels=labels.astype(np.int32, copy=False),
        multimask_output=False,
    )
    masks = np.asarray(masks)
    quality = np.asarray(quality, dtype=np.float32).reshape(-1)
    if masks.shape != (1, height, width) or quality.shape != (1,):
        raise QueryAbstention("official SAM3 output axes differ")
    if not bool(np.isfinite(masks).all()) or not bool(np.isfinite(quality).all()):
        raise QueryAbstention("official SAM3 output is nonfinite")
    probability = np.asarray(masks[0], dtype=np.float32)
    if bool(((probability < 0) | (probability > 1)).any()):
        raise QueryAbstention("official SAM3 probability leaves [0,1]")
    receipt = {
        "rgb": {"path": str(rgb_path), "sha256": _sha256(rgb_path)},
        "rgb_original_size_wh": original_size,
        "sam_size_wh": [width, height],
        "point_coordinates_xy": points.tolist(),
        "point_labels": labels.tolist(),
        "candidate_trial_rank": rank,
        "point_trial_policy": "weighted_farthest_visible_signed_field_log_odds_v1",
        "quality": float(quality[0]),
        "low_resolution_shape": list(np.asarray(low_resolution).shape),
        "multimask_output": False,
        "candidate_selection": "none_exactly_one_prediction",
    }
    return probability, receipt


def prepare_registered_view(
    processor: Any, record: Mapping[str, Any], *, device: str
) -> dict[str, Any]:
    """Encode one captured RGB once, then reuse it for all K candidates."""

    rgb_path = _load_bound(record.get("rgb", {}), label="registered RGB")
    projected = _load_array(
        record.get("projected_probability", {}), label="projected probability"
    )
    height, width = map(int, projected.shape)
    image = Image.open(rgb_path).convert("RGB")
    original_size = list(image.size)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    amp_dtype = torch.bfloat16 if str(device).startswith("cuda") else None
    with sam3_autocast_context(device, amp_dtype):
        state = processor.set_image(image)
    return {
        "state": state,
        "rgb_path": str(rgb_path),
        "rgb_original_size_wh": original_size,
        "sam_size_wh": [width, height],
    }


def run(args: argparse.Namespace, *, processor: Any | None = None) -> dict[str, Any]:
    plan_path = Path(args.plan).expanduser().resolve(strict=True)
    if _sha256(plan_path) != str(args.expected_plan_sha256):
        raise ValueError("candidate/view plan SHA-256 differs")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidates, canonical_views = validate_plan(
        plan, expected_candidates=int(args.expected_candidates)
    )
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
            dtype="bfloat16",
            resolution=int(args.resolution),
            point_only=True,
            build_on_cpu=True,
        )

    candidate_outputs = {
        str(candidate["candidate_digest"]): {
            "candidate_digest": str(candidate["candidate_digest"]),
            "candidate_logit": float(candidate["candidate_logit"]),
            "views": [],
        }
        for candidate in candidates
    }
    for view_digest in canonical_views:
        cohort: list[tuple[str, Mapping[str, Any]]] = []
        for candidate in candidates:
            candidate_digest = str(candidate["candidate_digest"])
            matches = [
                row
                for row in candidate["views"]
                if str(row["view_digest"]) == view_digest
            ]
            if len(matches) != 1:
                raise QueryAbstention("candidate/view Cartesian cell is ambiguous")
            cohort.append((candidate_digest, matches[0]))
        first = cohort[0][1]
        static_identity = (
            dict(first["rgb"]),
            dict(first["assignment"]),
            float(first["log_precision"]),
        )
        if any(
            (
                dict(view["rgb"]),
                dict(view["assignment"]),
                float(view["log_precision"]),
            )
            != static_identity
            for _, view in cohort[1:]
        ):
            raise QueryAbstention("registered view static authority differs by candidate")
        cached = []
        for candidate_digest, _view in cohort:
            probability_path = (
                output / "probabilities" / candidate_digest / f"{view_digest}.npy"
            )
            receipt_path = output / "receipts" / candidate_digest / f"{view_digest}.json"
            cached.append(probability_path.is_file() and receipt_path.is_file())
        prepared_view = None
        if not all(cached):
            prepared_view = prepare_registered_view(processor, first, device=args.device)
        for (candidate_digest, view), _cell_cached in zip(cohort, cached):
            probability_path = (
                output / "probabilities" / candidate_digest / f"{view_digest}.npy"
            )
            receipt_path = output / "receipts" / candidate_digest / f"{view_digest}.json"
            if probability_path.exists() or receipt_path.exists():
                if not probability_path.is_file() or not receipt_path.is_file():
                    raise QueryAbstention("candidate/view resume cell is partial")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("candidate_digest") != candidate_digest
                    or receipt.get("view_digest") != view_digest
                    or receipt.get("probability", {}).get("sha256")
                    != _sha256(probability_path)
                ):
                    raise QueryAbstention("candidate/view resume identity differs")
            else:
                probability, details = predict_one_candidate_view(
                    processor,
                    view,
                    candidate_digest=candidate_digest,
                    points_per_sign=int(args.points_per_sign),
                    device=args.device,
                    prepared_view=prepared_view,
                )
                probability_sha = _atomic_numpy(probability_path, probability)
                receipt = {
                    "schema_version": 1,
                    "artifact_type": "nvos_synchronous_multiview_sam3_cell_v1",
                    "candidate_digest": candidate_digest,
                    "view_digest": view_digest,
                    "probability": {
                        "path": str(probability_path),
                        "sha256": probability_sha,
                    },
                    "assignment": dict(view["assignment"]),
                    "log_precision": float(view["log_precision"]),
                    "official_sam3": details,
                    "target_mask_opened": False,
                    "target_metric_opened": False,
                }
                _atomic_json(receipt_path, receipt)
            candidate_outputs[candidate_digest]["views"].append(
                {
                    "view_digest": view_digest,
                    "probability": {
                        "path": str(probability_path),
                        "sha256": _sha256(probability_path),
                    },
                    "assignment": dict(view["assignment"]),
                    "log_precision": float(view["log_precision"]),
                    "receipt": {
                        "path": str(receipt_path),
                        "sha256": _sha256(receipt_path),
                    },
                }
            )
        del prepared_view

    inventory_candidates = [
        candidate_outputs[str(candidate["candidate_digest"])]
        for candidate in candidates
    ]

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
        "candidate_count": len(inventory_candidates),
        "view_count": len(canonical_views),
        "candidate_digests": sorted(
            str(row["candidate_digest"]) for row in inventory_candidates
        ),
        "view_digests": list(canonical_views),
        "candidates": inventory_candidates,
        "all_candidate_view_predictions_sealed": True,
        "candidate_selection": False,
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
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--points-per-sign", type=int, default=3)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=FROZEN_SAM3_SHA256
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "inventory": result["inventory_path"],
                "candidate_count": result["candidate_count"],
                "view_count": result["view_count"],
                "target_mask_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
