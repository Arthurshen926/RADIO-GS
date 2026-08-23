#!/usr/bin/env python3
"""Stream exact-adjoint NVOS candidate/view observations into one posterior.

The expensive RGB/SAM stage is deliberately outside this program.  Its sealed
inventory provides one probability raster and one exact compositor assignment
for every candidate/view pair.  This program lifts those rasters with ``W.T``,
robustly fuses registered views in log-odds space, checkpoints each completed
candidate, and finally marginalizes candidates in probability space.

Exact assignments are query-independent and identical across candidates, so
each registered view is hash-verified and loaded once.  Only one candidate's
``V x N`` tensor is resident at a time.  This is the resume boundary needed by
the million-row NVOS carriers; a complete ``K x V x N`` allocation is never
made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.querying.synchronous_multiview_candidate_marginal import (
    QueryAbstention,
    fuse_positive_unknown_views,
    marginalize_synchronous_multiview_candidates,
)
from radio_gs.scripts.build_nvos_two_round_exact_consensus import (
    exact_adjoint_probability,
)


ARTIFACT_TYPE = "nvos_synchronous_multiview_sam3_exact_adjoint_inventory_v1"
BOX_ARTIFACT_TYPE = "nvos_synchronous_multiview_box_sam3_exact_adjoint_inventory_v1"
OUTPUT_TYPE = "nvos_synchronous_multiview_candidate_marginal_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(path: str | Path, expected_sha256: str, *, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if len(str(expected_sha256)) != 64 or _sha256(source) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs: {source}")
    return source


def _write_torch_noclobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(value), temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _write_json_noclobber(path: Path, value: Mapping[str, Any]) -> str:
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
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _load_probability(record: Mapping[str, Any]) -> torch.Tensor:
    path = _load_bound(record.get("path", ""), record.get("sha256", ""), label="SAM3 probability")
    value = np.load(path, allow_pickle=False)
    probability = torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(-1)
    if not probability.numel() or not bool(torch.isfinite(probability).all()):
        raise ValueError("SAM3 probability is empty or nonfinite")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("SAM3 probability leaves [0,1]")
    return probability


def _load_assignment(record: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    path = _load_bound(record.get("path", ""), record.get("sha256", ""), label="exact assignment")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("exact assignment must be a mapping")
    required = {"gaussian_ids", "pixel_ids", "weights"}
    if not required.issubset(value):
        raise ValueError("exact assignment lacks compositor triplets")
    return value


def lift_candidate_views(
    view_records: Sequence[Mapping[str, Any]],
    *,
    num_gaussians: int,
    assignment_cache: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Lift one complete candidate, optionally reusing sealed view assignments."""

    lifted: list[torch.Tensor] = []
    precision: list[float] = []
    digests: list[str] = []
    torch_device = torch.device(device)
    for record in view_records:
        probability = _load_probability(record["probability"]).to(torch_device)
        view_digest = str(record["view_digest"])
        assignment = (
            _load_assignment(record["assignment"])
            if assignment_cache is None
            else assignment_cache.get(view_digest)
        )
        if assignment is None:
            raise QueryAbstention("registered view assignment cache is incomplete")
        primitive, visible_mass = exact_adjoint_probability(
            assignment["gaussian_ids"],
            assignment["pixel_ids"],
            assignment["weights"],
            probability,
            num_gaussians=int(num_gaussians),
        )
        # Invisible rows are neutral evidence.  Precision is a scalar authority
        # fixed upstream from query-independent view reliability.
        primitive[visible_mass <= 0] = 0.5
        lifted.append(primitive)
        precision.append(float(record["log_precision"]))
        digests.append(view_digest)
        del probability, primitive, visible_mass
    if not lifted:
        raise QueryAbstention("candidate has no registered view")
    return (
        torch.stack(lifted, dim=0),
        torch.tensor(precision, dtype=torch.float32, device=torch_device),
        tuple(digests),
    )


def fuse_one_candidate(
    candidate_view_probability: torch.Tensor,
    view_log_precision: torch.Tensor,
    *,
    candidate_digest: str,
    view_digests: Sequence[str],
    view_huber_delta: float = 2.0,
    view_fusion: str = "robust_log_odds",
) -> torch.Tensor:
    """Fuse one candidate with one explicitly recorded observation model."""

    if view_fusion == "positive_unknown_noisy_or":
        return fuse_positive_unknown_views(
            candidate_view_probability, view_log_precision
        ).cpu()
    if view_fusion != "robust_log_odds":
        raise ValueError("unknown candidate view fusion")

    result = marginalize_synchronous_multiview_candidates(
        candidate_view_probability[None],
        view_log_precision[None],
        torch.zeros(1, dtype=candidate_view_probability.dtype),
        candidate_digests=[candidate_digest],
        view_digests=view_digests,
        expected_candidates=1,
        view_huber_delta=float(view_huber_delta),
    )
    return result.candidate_field[0].cpu()


def run(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = _load_bound(
        args.inventory, args.expected_inventory_sha256, label="candidate/view inventory"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidates = inventory.get("candidates")
    expected_candidates = int(args.expected_candidates)
    num_gaussians = int(inventory.get("num_gaussians", 0))
    if (
        inventory.get("artifact_type") not in {ARTIFACT_TYPE, BOX_ARTIFACT_TYPE}
        or not isinstance(candidates, list)
        or len(candidates) != expected_candidates
        or num_gaussians <= 0
        or inventory.get("all_candidate_view_predictions_sealed") is not True
        or inventory.get("target_mask_opened") is not False
        or inventory.get("target_metric_opened") is not False
    ):
        raise ValueError("candidate/view inventory contract differs")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    first_views = candidates[0].get("views", [])
    if not isinstance(first_views, list) or not first_views:
        raise QueryAbstention("candidate has no registered view cohort")
    assignment_cache: dict[str, Mapping[str, torch.Tensor]] = {}
    assignment_identity: dict[str, Mapping[str, Any]] = {}
    for record in first_views:
        view_digest = str(record.get("view_digest", ""))
        if not view_digest or view_digest in assignment_cache:
            raise QueryAbstention("registered view assignment identity is ambiguous")
        assignment_identity[view_digest] = dict(record["assignment"])
        loaded = _load_assignment(record["assignment"])
        assignment_cache[view_digest] = {
            name: torch.as_tensor(loaded[name]).to(args.device)
            for name in ("gaussian_ids", "pixel_ids", "weights")
        }
    canonical_views = tuple(sorted(assignment_cache))
    for candidate in candidates[1:]:
        records = candidate.get("views", [])
        if not isinstance(records, list) or tuple(
            sorted(str(record.get("view_digest", "")) for record in records)
        ) != canonical_views:
            raise QueryAbstention("candidate registered-view cohort differs")
        for record in records:
            if dict(record["assignment"]) != assignment_identity[str(record["view_digest"])]:
                raise QueryAbstention("candidate exact assignment differs by view")
    fields: list[torch.Tensor] = []
    logits: list[float] = []
    candidate_digests: list[str] = []
    candidate_records: list[dict[str, str]] = []
    for rank, candidate in enumerate(candidates):
        digest = str(candidate.get("candidate_digest", ""))
        checkpoint = output / "candidates" / f"candidate_{rank:03d}.pt"
        if checkpoint.is_file():
            cached = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                cached.get("candidate_digest") != digest
                or cached.get("view_fusion") != args.view_fusion
            ):
                raise ValueError("resume candidate identity differs")
            field = torch.as_tensor(cached["candidate_field"]).float().reshape(-1)
        else:
            views, precision, view_digests = lift_candidate_views(
                candidate.get("views", []),
                num_gaussians=num_gaussians,
                assignment_cache=assignment_cache,
                device=args.device,
            )
            field = fuse_one_candidate(
                views,
                precision,
                candidate_digest=digest,
                view_digests=view_digests,
                view_huber_delta=args.view_huber_delta,
                view_fusion=args.view_fusion,
            )
            _write_torch_noclobber(
                checkpoint,
                {
                    "candidate_digest": digest,
                    "candidate_field": field,
                    "view_fusion": args.view_fusion,
                    "num_views": int(views.shape[0]),
                    "num_gaussians": num_gaussians,
                },
            )
            del views, precision
        if field.shape != (num_gaussians,):
            raise ValueError("candidate checkpoint carrier differs")
        fields.append(field)
        logits.append(float(candidate["candidate_logit"]))
        candidate_digests.append(digest)
        candidate_records.append({"path": str(checkpoint), "sha256": _sha256(checkpoint)})

    order = sorted(range(expected_candidates), key=lambda index: candidate_digests[index])
    weights = torch.softmax(torch.tensor([logits[index] for index in order]), dim=0)
    stacked = torch.stack([fields[index] for index in order], dim=0)
    probability = torch.einsum("k,kn->n", weights, stacked).clamp(0, 1)
    result_path = output / "primitive_posterior.pt"
    result_sha = _write_torch_noclobber(
        result_path,
        {
            "artifact_type": OUTPUT_TYPE,
            "scene_id": inventory.get("scene_id"),
            "probability": probability,
            "candidate_probability": weights,
            "candidate_digests": [candidate_digests[index] for index in order],
            "view_fusion": args.view_fusion,
            "num_gaussians": num_gaussians,
        },
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": OUTPUT_TYPE,
        "scene_id": inventory.get("scene_id"),
        "inventory": {"path": str(inventory_path), "sha256": args.expected_inventory_sha256},
        "inventory_artifact_type": inventory.get("artifact_type"),
        "candidate_checkpoints": candidate_records,
        "result": {"path": str(result_path), "sha256": result_sha},
        "resident_tensor_bound": (
            "all_registered_exact_assignments_plus_one_candidate_times_"
            "all_views_times_num_gaussians"
        ),
        "assignment_io": "one_hash_verified_load_per_registered_view",
        "adjoint_device": str(args.device),
        "view_fusion": args.view_fusion,
        "resume_boundary": "one_completed_candidate",
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    receipt_path = output / "receipt.json"
    _write_json_noclobber(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--view-huber-delta", type=float, default=2.0)
    parser.add_argument(
        "--view-fusion",
        choices=("robust_log_odds", "positive_unknown_noisy_or"),
        default="robust_log_odds",
    )
    parser.add_argument("--device", default="cpu")
    report = run(parser.parse_args(argv))
    print(json.dumps({"receipt": report["receipt"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
