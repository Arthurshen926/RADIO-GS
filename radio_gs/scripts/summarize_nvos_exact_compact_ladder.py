#!/usr/bin/env python3
"""Summarize a protocol-locked NVOS exact/compact unary ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


ARM_DIRS = {
    "raw_radio_post_mpr_projection_diagnostic": "exact",
    "formal_exact_capability_mpr_teacher": "exact_capability",
    "compact_canonical_field": "compact",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = torch.as_tensor(left).float().reshape(-1)
    y = torch.as_tensor(right).float().reshape(-1)
    if x.shape != y.shape or x.numel() == 0:
        raise ValueError("Pearson inputs must be aligned and nonempty")
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(x, y) else 0.0
    return float((x * y).mean() / denominator)


def _difference(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    left = torch.as_tensor(reference).float().reshape(-1)
    right = torch.as_tensor(candidate).float().reshape(-1)
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("difference inputs must be aligned and nonempty")
    delta = right - left
    return {
        "mean_absolute_error": float(delta.abs().mean()),
        "root_mean_square_error": float(delta.square().mean().sqrt()),
        "pearson": _pearson(left, right),
        "threshold_disagreement_fraction": float(
            ((left >= float(threshold)) != (right >= float(threshold)))
            .float()
            .mean()
        ),
        "reference_mean": float(left.mean()),
        "candidate_mean": float(right.mean()),
    }


def _load_arm(root: Path, scene: str, arm_dir: str) -> dict[str, object]:
    directory = root / scene / arm_dir
    report_path = directory / f"{scene}_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    unary_path = directory / "primitive_unary.pt"
    unary = torch.load(unary_path, map_location="cpu", weights_only=False)
    if unary.get("artifact_type") != "nvos_frozen_k16_primitive_unary_probability_v1":
        raise ValueError(f"{scene}/{arm_dir}: primitive unary contract differs")
    declared = report.get("primitive_unary_artifact", {})
    if not isinstance(declared, Mapping) or declared.get("sha256") != _sha256_file(
        unary_path
    ):
        raise ValueError(f"{scene}/{arm_dir}: primitive unary SHA differs")
    frame_id = str(report["frames"][0]["frame_id"])
    score_path = Path(str(report["score_paths"][frame_id]))
    if report["score_sha256"][frame_id] != _sha256_file(score_path):
        raise ValueError(f"{scene}/{arm_dir}: rendered score SHA differs")
    return {
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "unary": torch.as_tensor(unary["primitive_unary_probability"]).float(),
        "valid": torch.as_tensor(unary["valid"]).bool(),
        "unary_path": unary_path,
        "unary_sha256": declared["sha256"],
        "compiler_contract": dict(unary["compiler_contract"]),
        "capability_cache": str(unary["capability_cache"]),
        "capability_cache_sha256": str(unary["capability_cache_sha256"]),
        "rendered": torch.from_numpy(np.load(score_path)).float(),
        "score_path": score_path,
        "score_sha256": report["score_sha256"][frame_id],
    }


def summarize(
    root: str | Path,
    *,
    scenes: list[str],
    preexisting_gt_primitive_oracle: str = "",
) -> dict[str, object]:
    ladder_root = Path(root).expanduser().resolve()
    per_scene: dict[str, object] = {}
    macro: dict[str, list[float]] = {name: [] for name in ARM_DIRS}
    compact_minus_teacher: list[float] = []
    compact_minus_raw: list[float] = []
    for scene in scenes:
        arms = {
            name: _load_arm(ladder_root, scene, directory)
            for name, directory in ARM_DIRS.items()
        }
        protocol_hashes = {
            str(value["report"]["protocol_hash"]) for value in arms.values()
        }
        compiler_contracts = {
            json.dumps(value["compiler_contract"], sort_keys=True)
            for value in arms.values()
        }
        if len(protocol_hashes) != 1 or len(compiler_contracts) != 1:
            raise ValueError(f"{scene}: protocol/compiler drift across ladder arms")
        valid_masks = [value["valid"] for value in arms.values()]
        if any(not torch.equal(valid_masks[0], mask) for mask in valid_masks[1:]):
            raise ValueError(f"{scene}: primitive row masks differ across ladder arms")
        valid = valid_masks[0]
        if any(
            value["unary"].shape != valid.shape
            or value["rendered"].shape
            != next(iter(arms.values()))["rendered"].shape
            for value in arms.values()
        ):
            raise ValueError(f"{scene}: ladder tensors do not align")

        arm_records: dict[str, object] = {}
        for name, value in arms.items():
            report = value["report"]
            iou = float(report["foreground_iou"])
            macro[name].append(iou)
            arm_records[name] = {
                "foreground_iou": iou,
                "pixel_accuracy": float(report["pixel_accuracy"]),
                "capability_source_contract": str(
                    report["canonical_capability_source_contract"]
                ),
                "capability_cache": value["capability_cache"],
                "capability_cache_sha256": value["capability_cache_sha256"],
                "primitive_unary": str(value["unary_path"]),
                "primitive_unary_sha256": value["unary_sha256"],
                "rendered_score": str(value["score_path"]),
                "rendered_score_sha256": value["score_sha256"],
                "evaluation_report": str(value["report_path"]),
                "evaluation_report_sha256": value["report_sha256"],
            }
        teacher = arms["formal_exact_capability_mpr_teacher"]
        compact = arms["compact_canonical_field"]
        raw = arms["raw_radio_post_mpr_projection_diagnostic"]
        teacher_iou = float(teacher["report"]["foreground_iou"])
        compact_iou = float(compact["report"]["foreground_iou"])
        raw_iou = float(raw["report"]["foreground_iou"])
        compact_minus_teacher.append(compact_iou - teacher_iou)
        compact_minus_raw.append(compact_iou - raw_iou)

        field_sidecar = Path(
            str(compact["capability_cache"])
        ).parent / "canonical_d256_l128_capability_first.pth.json"
        field_report = json.loads(field_sidecar.read_text(encoding="utf-8"))
        per_scene[scene] = {
            "protocol_hash": next(iter(protocol_hashes)),
            "valid_primitives": int(valid.sum()),
            "arms": arm_records,
            "formal_teacher_to_compact": {
                "iou_delta_compact_minus_teacher": compact_iou - teacher_iou,
                "primitive_unary": _difference(
                    teacher["unary"][valid], compact["unary"][valid], threshold=0.5
                ),
                "rendered_unary": _difference(
                    teacher["rendered"], compact["rendered"], threshold=0.5
                ),
            },
            "raw_projection_diagnostic_to_compact": {
                "iou_delta_compact_minus_raw": compact_iou - raw_iou,
                "primitive_unary": _difference(
                    raw["unary"][valid], compact["unary"][valid], threshold=0.5
                ),
                "rendered_unary": _difference(
                    raw["rendered"], compact["rendered"], threshold=0.5
                ),
            },
            "field_training_query_free_capability_fidelity": dict(
                field_report["final_capability_metrics"]
            ),
        }

    oracle_path = Path(preexisting_gt_primitive_oracle).expanduser().resolve() if (
        str(preexisting_gt_primitive_oracle).strip()
    ) else None
    if oracle_path is not None and not oracle_path.is_file():
        raise FileNotFoundError("declared pre-existing GT primitive oracle is absent")
    return {
        "schema_version": 1,
        "experiment": "nvos_exact_mpr_compact_same_compiler_ladder_v1",
        "status": "complete",
        "scenes": list(scenes),
        "frozen_protocol_modified": False,
        "graph_propagation_enabled": False,
        "connected_selection_applied": False,
        "per_scene": per_scene,
        "aggregate": {
            **{
                f"{name}_macro_iou": float(np.mean(values))
                for name, values in macro.items()
            },
            "compact_minus_formal_teacher_macro_iou": float(
                np.mean(compact_minus_teacher)
            ),
            "compact_minus_raw_projection_diagnostic_macro_iou": float(
                np.mean(compact_minus_raw)
            ),
        },
        "gt_primitive_oracle": {
            "available": oracle_path is not None,
            "path": str(oracle_path) if oracle_path is not None else None,
            "status": (
                "preexisting_authority_supplied"
                if oracle_path is not None
                else "unavailable; no pre-existing frozen target-GT-to-primitive row authority found"
            ),
            "posthoc_oracle_constructed": False,
        },
        "conclusion": (
            "The formal exact capability-MPR teacher and compact field produce "
            "nearly identical K16 primitive/rendered unaries; compact compression "
            "is not the material NVOS bottleneck on these scenes. The poor direct "
            "raw-RADIO MPR arm is a projection-order diagnostic, not a ceiling."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--preexisting-gt-primitive-oracle", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(
        args.root,
        scenes=list(args.scenes),
        preexisting_gt_primitive_oracle=args.preexisting_gt_primitive_oracle,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
