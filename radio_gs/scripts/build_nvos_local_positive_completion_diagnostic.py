"""Build the preregistered full8 source-only NVOS v2 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.nvos_local_positive_completion import (
    method_contract,
    source_only_loo_diagnostic,
)


SCENE_ORDER = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
V1_ACCEPTED = frozenset({"horns_left"})


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mask(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L")) > 0
    return torch.from_numpy(array.copy()).bool().contiguous()


def _atomic_write_no_clobber(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different diagnostic: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _completion(path: Path, expected_sha256: str, scene_id: str) -> tuple[torch.Tensor, str]:
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"{scene_id} completion SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{scene_id} completion is not a mapping")
    authority = payload.get("authority")
    tensors = payload.get("tensors")
    if (
        payload.get("artifact_type") != "radio_gs.nvos_sam3_reference_completion"
        or not isinstance(authority, Mapping)
        or authority.get("scene_id") != scene_id
        or not isinstance(tensors, Mapping)
        or "trial_masks" not in tensors
    ):
        raise ValueError(f"{scene_id} completion authority differs")
    digests = {
        str(name): tensor_sha256(torch.as_tensor(value))
        for name, value in sorted(tensors.items())
    }
    if payload.get("tensor_sha256") != digests:
        raise ValueError(f"{scene_id} completion tensor hashes differ")
    if payload.get("tensor_bundle_sha256") != _canonical_sha256(digests):
        raise ValueError(f"{scene_id} completion tensor bundle differs")
    return torch.as_tensor(tensors["trial_masks"]).bool().contiguous(), str(
        payload["tensor_bundle_sha256"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--source-authority-correction", required=True)
    parser.add_argument("--source-authority-correction-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    preregistration_path = Path(args.preregistration).expanduser().resolve()
    source_authority_path = Path(args.source_authority_correction).expanduser().resolve()
    if _file_sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("manifest SHA-256 differs")
    if _file_sha256(preregistration_path) != args.preregistration_sha256:
        raise ValueError("preregistration SHA-256 differs")
    if (
        _file_sha256(source_authority_path)
        != args.source_authority_correction_sha256
    ):
        raise ValueError("source-authority correction SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    source_authority = json.loads(source_authority_path.read_text(encoding="utf-8"))
    if tuple(preregistration["frozen_dataset_authority"]["scene_order"]) != SCENE_ORDER:
        raise ValueError("preregistered scene order differs")
    if (
        source_authority.get("artifact_type")
        != "nvos_local_majority_positive_completion_v2_source_authority_correction"
        or source_authority["lineage"]["preregistration_sha256"]
        != args.preregistration_sha256
        or source_authority["lineage"]["manifest_sha256"] != args.manifest_sha256
        or source_authority["failed_attempt"]["diagnostic_output_written"] is not False
        or source_authority["authority_policy"]["method_or_gate_change"] is not False
        or source_authority["authority_policy"]["target_information_used"] is not False
    ):
        raise ValueError("source-authority correction lineage differs")
    if tuple(source_authority["scribble_assets"]) != SCENE_ORDER:
        raise ValueError("source-authority correction scene order differs")
    scene_records = {str(scene["scene_id"]): scene for scene in manifest["scenes"]}
    if tuple(scene_records) != SCENE_ORDER:
        raise ValueError("manifest scene order differs")

    per_scene: dict[str, object] = {}
    pooled_proposal = 0
    pooled_intersection = 0
    pooled_confidence = 0.0
    pooled_true_confidence = 0.0
    rejected_retained = []
    for scene_id in SCENE_ORDER:
        completion_authority = preregistration["frozen_completion_assets"][scene_id]
        completion_path = Path(completion_authority["path"]).expanduser().resolve()
        receipt_path = completion_path.parent.parent / "receipts" / f"{scene_id}.json"
        if _file_sha256(receipt_path) != completion_authority["receipt_sha256"]:
            raise ValueError(f"{scene_id} completion receipt SHA-256 differs")
        trial_masks, tensor_bundle_sha256 = _completion(
            completion_path, completion_authority["sha256"], scene_id
        )
        scene = scene_records[scene_id]
        prompt = scene["prompt"]
        positive_path = Path(prompt["positive_path"]).expanduser().resolve()
        negative_path = Path(prompt["negative_path"]).expanduser().resolve()
        declared = source_authority["scribble_assets"][scene_id]
        if (
            _file_sha256(positive_path) != declared["positive_sha256"]
            or _file_sha256(negative_path) != declared["negative_sha256"]
        ):
            raise ValueError(f"{scene_id} raw scribble SHA-256 differs")
        diagnostic = source_only_loo_diagnostic(
            trial_masks,
            positive_scribble=_mask(positive_path),
            negative_scribble=_mask(negative_path),
        )
        summary = diagnostic["summary"]
        pooled_proposal += int(summary["pooled_proposal_pixels"])
        pooled_intersection += int(summary["pooled_intersection_pixels"])
        pooled_confidence += float(summary["pooled_confidence_mass"])
        pooled_true_confidence += float(summary["pooled_true_confidence_mass"])
        v1_decision = "accept" if scene_id in V1_ACCEPTED else "reject"
        if (
            v1_decision == "reject"
            and float(diagnostic["full_fit"]["proposal_confidence_mass"]) > 0
        ):
            rejected_retained.append(scene_id)
        per_scene[scene_id] = {
            "v1_gate_decision": v1_decision,
            "completion": {
                "path": str(completion_path),
                "sha256": completion_authority["sha256"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": completion_authority["receipt_sha256"],
                "tensor_bundle_sha256": tensor_bundle_sha256,
            },
            "prompt_frame_id": prompt["frame_id"],
            "positive_scribble_sha256": declared["positive_sha256"],
            "negative_scribble_sha256": declared["negative_sha256"],
            "diagnostic": diagnostic,
        }

    pooled_hard_precision = (
        pooled_intersection / pooled_proposal if pooled_proposal else 1.0
    )
    pooled_weighted_precision = (
        pooled_true_confidence / pooled_confidence if pooled_confidence else 1.0
    )
    structural = all(
        float(value["diagnostic"]["full_fit"]["nonproposal_completion_confidence_mass"])
        == 0.0
        and value["diagnostic"]["safety"]["absence_used_as_negative_evidence"]
        is False
        for value in per_scene.values()
    )
    gates = {
        "structural_invariants_all_pass": structural,
        "pooled_confidence_weighted_precision_at_least_pooled_hard_majority_precision": (
            pooled_weighted_precision >= pooled_hard_precision
        ),
        "at_least_one_v1_rejected_scene_retains_non_scribble_positive_confidence": bool(
            rejected_retained
        ),
        "all_metrics_finite": all(
            np.isfinite(
                [
                    value["diagnostic"]["summary"][
                        "mean_loo_hard_majority_precision"
                    ],
                    value["diagnostic"]["summary"][
                        "mean_loo_confidence_weighted_precision"
                    ],
                ]
            ).all()
            for value in per_scene.values()
        ),
        "no_numeric_margin_or_target_threshold": True,
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_local_majority_positive_completion_source_diagnostic_v2",
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": args.preregistration_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": args.manifest_sha256,
            "protocol_hash": manifest["protocol_hash"],
        },
        "source_authority_correction": {
            "path": str(source_authority_path),
            "sha256": args.source_authority_correction_sha256,
            "method_or_gate_change": False,
            "target_information_used": False,
        },
        "method_contract": method_contract(),
        "scene_order": list(SCENE_ORDER),
        "per_scene": per_scene,
        "pooled": {
            "proposal_pixels": pooled_proposal,
            "intersection_pixels": pooled_intersection,
            "confidence_mass": pooled_confidence,
            "true_confidence_mass": pooled_true_confidence,
            "hard_majority_precision": pooled_hard_precision,
            "confidence_weighted_precision": pooled_weighted_precision,
            "precision_gain_from_margin_weighting": (
                pooled_weighted_precision - pooled_hard_precision
            ),
            "v1_rejected_scenes_retaining_positive_confidence": rejected_retained,
        },
        "admissibility_gate": {
            **gates,
            "overall": all(gates.values()),
            "authorized_next_step": (
                "separate_target_sentinel_preregistration_only"
                if all(gates.values())
                else "retain_v1_global_abstention"
            ),
        },
        "safety": {
            "source_rgb_opened": False,
            "source_scribbles_opened": True,
            "source_completion_trials_opened": True,
            "target_rgb_opened": False,
            "target_score_rendered": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output = Path(args.output).expanduser().resolve()
    _atomic_write_no_clobber(output, encoded)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _file_sha256(output),
                "admissible": payload["admissibility_gate"]["overall"],
                "pooled_hard_majority_precision": pooled_hard_precision,
                "pooled_confidence_weighted_precision": pooled_weighted_precision,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
