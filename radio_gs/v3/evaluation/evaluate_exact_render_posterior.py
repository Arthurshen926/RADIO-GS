"""Evaluate the real source text posterior after exact Gaussian rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.query.calibrated_posterior import load_null_calibrated_posterior
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import (
    sha256_file,
    validate_source_only_inputs,
)
from radio_gs.v3.training.run_instance_upper_bound import load_episodes


def _mask_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    known: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    score = torch.as_tensor(probability).float().reshape(-1)
    truth = torch.as_tensor(target).bool().reshape(-1)
    authority = torch.as_tensor(known).bool().reshape(-1)
    if not 0 < threshold < 1 or not bool(authority.any()):
        raise ValueError("exact-render posterior threshold or authority differs")
    selected = score >= threshold
    intersection = (selected & truth & authority).sum().float()
    union = ((selected | truth) & authority).sum().float()
    return {
        "mask_iou": float(intersection / union.clamp_min(1)),
        "brier": float((score[authority] - truth[authority].float()).square().mean()),
        "foreground_probability": float(score[authority].mean()),
        "foreground_fraction": float(selected[authority].float().mean()),
        "peak_probability": float(score[authority].max()),
    }


def _mean(records: list[dict[str, float]]) -> dict[str, float] | None:
    if not records:
        return None
    return {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in records[0]
    }


def _input_paths(evidence: dict, evidence_path: Path) -> dict[str, Path]:
    metadata = evidence.get("metadata", {})
    if (
        evidence.get("schema") != "radio_gs.sugm_v3.clean_posterior_evidence.v2"
        or not metadata.get("source_only")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
        or metadata.get("unknown_pairs_used_as_negative")
    ):
        raise ValueError("exact-render posterior evidence lineage differs")
    paths = {"evidence": evidence_path}
    for name in (
        "scene_state",
        "membership",
        "authority",
        "text_embeddings",
        "text_negatives",
    ):
        receipt = metadata.get("inputs", {}).get(name, {})
        path = Path(receipt.get("path", "")).resolve(strict=True)
        if sha256_file(path) != receipt.get("sha256"):
            raise ValueError(f"exact-render posterior {name} receipt differs")
        paths[name] = path
    return paths


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.residue not in (0, 3):
        raise ValueError("exact-render posterior is source-heldout only")
    if args.threshold != 0.5:
        raise ValueError("Gate-4 requires the preregistered global 0.5 threshold")
    evidence_path = Path(args.evidence).resolve(strict=True)
    calibrator_path = Path(args.calibrator).resolve(strict=True)
    evidence = torch.load(evidence_path, map_location="cpu")
    paths = _input_paths(evidence, evidence_path)
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text = torch.load(paths["text_embeddings"], map_location="cpu")
    validate_source_only_inputs(membership, authority)
    if len({evidence["scene"], membership["scene"], authority["scene"]}) != 1:
        raise ValueError("exact-render posterior scene axes differ")
    interface = load_query_interface(
        paths["scene_state"],
        device=args.device,
        text_negative_path=paths["text_negatives"],
        text_logit_scale=10.0,
    )
    calibrator = load_null_calibrated_posterior(calibrator_path, device=args.device)
    episodes, supports = load_episodes(membership, authority)
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)
    if states.shape != (len(episodes), len(authority["query_names"])):
        raise ValueError("exact-render posterior authority axes differ")
    valid = torch.tensor([rows.numel() > 0 for rows, _weights in supports])
    lookup = {str(name).casefold(): index for index, name in enumerate(text["queries"])}
    embeddings = torch.as_tensor(text["embeddings"]).float()
    positive_metrics = {"uncalibrated": [], "calibrated": []}
    empty_metrics = {"uncalibrated": [], "calibrated": []}
    gaussian_metrics = {"uncalibrated": [], "calibrated": []}
    identity_exact = True
    examples = []
    selected_views = torch.unique(views[views % 4 == args.residue])
    for column, raw_name in enumerate(authority["query_names"]):
        name = str(raw_name)
        token_index = lookup.get(name.casefold())
        if token_index is None:
            raise ValueError(f"exact-render posterior lacks query token: {name}")
        packet = QueryPacket("text", embeddings[token_index])
        identity, _null, _unknown = interface.semantic_text_evidence(packet)
        base, returned_identity = interface.posterior_from_packet(
            packet,
            scale=args.scale,
            topk=args.topk,
            temperature=args.temperature,
            posterior_chunk_size=args.posterior_chunk_size,
            text_anchor_policy="positive",
        )
        uncalibrated, _boundary = interface.refine_instance_with_boundary(
            base,
            maximum_logit_residual=interface.maximum_boundary_logit_residual,
        )
        calibrated, calibrated_identity, calibrated_instance = (
            interface.calibrated_posterior_from_packet(
                packet,
                calibrator,
                scale=args.scale,
                topk=args.topk,
                temperature=args.temperature,
                posterior_chunk_size=args.posterior_chunk_size,
            )
        )
        identity_exact &= (
            torch.equal(identity, returned_identity)
            and torch.equal(identity, calibrated_identity)
            and torch.equal(uncalibrated, calibrated_instance)
        )
        gaussian_metrics["uncalibrated"].append({
            "foreground_probability": float(uncalibrated.mean()),
            "foreground_fraction": float((uncalibrated >= args.threshold).float().mean()),
            "peak_probability": float(uncalibrated.max()),
        })
        gaussian_metrics["calibrated"].append({
            "foreground_probability": float(calibrated.mean()),
            "foreground_fraction": float((calibrated >= args.threshold).float().mean()),
            "peak_probability": float(calibrated.max()),
        })
        for view in selected_views.tolist():
            in_view = (views == int(view)) & valid
            positive = torch.where(in_view & (states[:, column] == 1))[0]
            negative = torch.where(in_view & (states[:, column] == 0))[0]
            if not negative.numel():
                continue
            target = (
                torch.stack([episodes[int(index)].target for index in positive]).any(0)
                if positive.numel()
                else torch.zeros_like(episodes[int(negative[0])].target)
            )
            negative_mask = torch.stack([
                episodes[int(index)].target for index in negative
            ]).any(0) & ~target
            if not bool(negative_mask.any()):
                continue
            known = target | negative_mask
            representative = episodes[int(positive[0] if positive.numel() else negative[0])]
            predictions = {}
            for label, posterior in (
                ("uncalibrated", uncalibrated),
                ("calibrated", calibrated),
            ):
                predictions[label] = interface.render_posterior(
                    posterior,
                    representative.gaussian_ids.to(posterior.device),
                    representative.pixel_ids.to(posterior.device),
                    representative.contribution_weights.to(posterior.device),
                    num_pixels=target.numel(),
                ).cpu()
            cohort = "positive" if bool(target.any()) else "empty"
            destination = positive_metrics if cohort == "positive" else empty_metrics
            for label, prediction in predictions.items():
                destination[label].append(_mask_metrics(
                    prediction,
                    target.flatten(),
                    known.flatten(),
                    threshold=args.threshold,
                ))
            examples.append({
                "query": name,
                "view": int(view),
                "cohort": cohort,
                "known_pixels": int(known.sum()),
                "positive_pixels": int(target.sum()),
            })
    if not positive_metrics["calibrated"] and not empty_metrics["calibrated"]:
        raise ValueError("exact-render posterior lacks any explicit heldout cohort")
    payload = {
        "schema": "radio_gs.sugm_v3.exact_render_posterior_source_evaluation.v1",
        "scene": evidence["scene"],
        "residue": args.residue,
        "identity_bitwise_preserved": identity_exact,
        "positive_examples": len(positive_metrics["calibrated"]),
        "empty_examples": len(empty_metrics["calibrated"]),
        "cohort_availability": {
            "positive": bool(positive_metrics["calibrated"]),
            "empty": bool(empty_metrics["calibrated"]),
            "missing_is_not_imputed": True,
        },
        "positive": {name: _mean(values) for name, values in positive_metrics.items()},
        "empty": {name: _mean(values) for name, values in empty_metrics.items()},
        "gaussian_query_mass": {
            name: _mean(values) for name, values in gaussian_metrics.items()
        },
        "examples": examples,
        "method": {
            "topk": args.topk,
            "scale": args.scale,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "posterior_chunk_size": args.posterior_chunk_size,
            "deployment_order": "clean_D128_positive_anchors_then_D48_then_signed_D16_then_null_calibrator_then_exact_MPR",
            "unknown": "excluded_from_negative_authority",
            "empty_target": "explicit_negative_source_masks_with_no_positive_proposal_in_view",
        },
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    payload["inputs"]["calibrator"] = {
        "path": str(calibrator_path),
        "sha256": sha256_file(calibrator_path),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--residue", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--posterior-chunk-size", type=int, default=65536)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run(args)
    write_frozen_json(Path(args.output).resolve(), payload)
    print({
        "scene": payload["scene"],
        "residue": payload["residue"],
        "identity_bitwise_preserved": payload["identity_bitwise_preserved"],
        "positive_examples": payload["positive_examples"],
        "empty_examples": payload["empty_examples"],
        "positive": payload["positive"],
        "empty": payload["empty"],
        "gaussian_query_mass": payload["gaussian_query_mass"],
    })


if __name__ == "__main__":
    main()


__all__ = ["_mask_metrics", "run"]
