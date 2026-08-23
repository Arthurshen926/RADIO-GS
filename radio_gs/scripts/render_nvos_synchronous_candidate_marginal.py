#!/usr/bin/env python3
"""Render one sealed NVOS synchronous primitive posterior before GT access."""

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

from radio_gs.scripts.build_nvos_two_round_exact_consensus import (
    exact_forward_probability,
)
from radio_gs.scripts.build_nvos_synchronous_multiview_candidate_plan import (
    PLAN_TYPE,
)
from radio_gs.scripts.materialize_nvos_synchronous_candidate_marginal import (
    OUTPUT_TYPE,
)


RECEIPT_TYPE = "nvos_synchronous_multiview_target_prediction_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(path: str | Path, expected: str, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if len(str(expected)) != 64 or _sha256(source) != str(expected):
        raise ValueError(f"{label} SHA-256 differs")
    return source


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
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def target_frame_id(plan: Mapping[str, Any]) -> str:
    manifest_record = plan.get("inputs", {}).get("dataset_manifest", {})
    manifest_path = _bound(
        manifest_record.get("path", ""),
        manifest_record.get("sha256", ""),
        "dataset manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = str(plan.get("scene_id", ""))
    scenes = [row for row in manifest.get("scenes", []) if row.get("scene_id") == scene_id]
    if len(scenes) != 1 or len(scenes[0].get("evaluation_frame_ids", [])) != 1:
        raise ValueError("NVOS target frame authority differs")
    return str(scenes[0]["evaluation_frame_ids"][0])


def target_view_record(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = plan.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate plan is empty")
    frame_id = target_frame_id(plan)
    records = [
        row
        for row in candidates[0].get("views", [])
        if str(row.get("frame_id")) == frame_id
    ]
    if len(records) != 1:
        raise ValueError("candidate plan target view differs")
    return records[0]


def render(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = _bound(args.plan, args.expected_plan_sha256, "candidate plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("artifact_type") != PLAN_TYPE
        or plan.get("target_mask_opened") is not False
        or plan.get("target_metric_opened") is not False
    ):
        raise ValueError("candidate plan contract differs")
    marginal_path = _bound(
        args.marginal, args.expected_marginal_sha256, "primitive marginal"
    )
    marginal = torch.load(marginal_path, map_location="cpu", weights_only=False)
    probability = torch.as_tensor(marginal.get("probability")).float().reshape(-1)
    num_gaussians = int(plan.get("num_gaussians", 0))
    if (
        marginal.get("artifact_type") != OUTPUT_TYPE
        or marginal.get("scene_id") != plan.get("scene_id")
        or probability.shape != (num_gaussians,)
        or not bool(torch.isfinite(probability).all())
        or bool(((probability < 0) | (probability > 1)).any())
    ):
        raise ValueError("primitive marginal contract differs")
    view = target_view_record(plan)
    assignment_record = view.get("assignment", {})
    assignment_path = _bound(
        assignment_record.get("path", ""),
        assignment_record.get("sha256", ""),
        "target exact assignment",
    )
    assignment = torch.load(assignment_path, map_location="cpu", weights_only=False)
    height, width = int(assignment["height"]), int(assignment["width"])
    device = torch.device(args.device)
    rendered, mass = exact_forward_probability(
        torch.as_tensor(assignment["gaussian_ids"]).to(device),
        torch.as_tensor(assignment["pixel_ids"]).to(device),
        torch.as_tensor(assignment["weights"]).to(device),
        probability.to(device),
        height=height,
        width=width,
        unsupported_fallback=torch.zeros(height * width, device=device),
    )
    output = Path(args.output_dir).expanduser().resolve()
    prediction_path = output / "target_probability.npy"
    prediction_sha = _atomic_numpy(prediction_path, rendered.cpu().numpy())
    receipt = {
        "schema_version": 1,
        "artifact_type": RECEIPT_TYPE,
        "scene_id": str(plan["scene_id"]),
        "target_frame_id": str(view["frame_id"]),
        "plan": {"path": str(plan_path), "sha256": str(args.expected_plan_sha256)},
        "marginal": {
            "path": str(marginal_path),
            "sha256": str(args.expected_marginal_sha256),
        },
        "exact_assignment": {
            "path": str(assignment_path),
            "sha256": str(assignment_record["sha256"]),
        },
        "prediction": {"path": str(prediction_path), "sha256": prediction_sha},
        "shape": [height, width],
        "alpha_supported_fraction": float((mass > 0).double().mean()),
        "threshold": 0.5,
        "unsupported_fallback": 0.0,
        "render_device": str(device),
        "prediction_sealed_before_target_ground_truth": True,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    receipt_path = output / "prediction_receipt.json"
    receipt_sha = _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": receipt_sha}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--marginal", required=True)
    parser.add_argument("--expected-marginal-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(render(build_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
