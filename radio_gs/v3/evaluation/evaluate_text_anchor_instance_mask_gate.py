"""Gate-2 exact-render mask test for fixed clean-D128 anchors plus D48."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.source_heldout import evaluate_source_heldout
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import (
    proposal_supports,
    sha256_file,
    validate_source_only_inputs,
)
from radio_gs.v3.training.run_instance_upper_bound import load_episodes


def _mean_metrics(values: list) -> dict[str, float]:
    return {
        name: sum(float(getattr(value, name)) for value in values) / len(values)
        for name in ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")
    }


def _gate_decision(
    zero: dict[str, float],
    random: dict[str, float],
    candidate: dict[str, float],
    *,
    identity_exact: bool,
) -> tuple[bool, list[str]]:
    failures = []
    if not identity_exact:
        failures.append("candidate changed clean D128 identity scores or anchors")
    for label, baseline in (("zero", zero), ("random", random)):
        if candidate["mask_iou"] <= baseline["mask_iou"]:
            failures.append(f"text mask IoU did not improve over {label} D48")
        if candidate["brier"] >= baseline["brier"]:
            failures.append(f"text mask Brier did not decrease from {label} D48")
    return not failures, failures


def _render_text_mask(interface, rows, weights, episode, *, temperature: float):
    support = interface.model.instance_view(episode.scale, rows)
    prototype = pool_prototype(support, weights)
    unique, inverse = torch.unique(
        episode.gaussian_ids, sorted=True, return_inverse=True
    )
    embedding = interface.model.instance_view(
        episode.scale, unique.to(interface.model.memory.device)
    )
    hit = embedding[inverse.to(embedding.device)]
    probability = membership_from_prototype(
        hit, prototype, temperature=temperature
    )
    return interface.render_posterior(
        probability,
        torch.arange(probability.numel(), device=probability.device),
        episode.pixel_ids.to(probability.device),
        episode.contribution_weights.to(probability.device),
        num_pixels=episode.target.numel(),
    ).cpu()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--residue", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("text mask Gate-2 is source-heldout only")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("clean_parent", args.clean_parent),
            ("candidate", args.candidate),
            ("membership", args.membership),
            ("authority", args.authority),
            ("text_embeddings", args.text_embeddings),
        )
    }
    candidate_payload = torch.load(paths["candidate"], map_location="cpu")
    gate1 = candidate_payload.get("metadata", {}).get("instance_gate1", {})
    if (
        gate1.get("status") != "passed"
        or gate1.get("parent", {}).get("sha256") != sha256_file(paths["clean_parent"])
    ):
        raise ValueError("text mask Gate-2 requires a passed Gate-1 candidate")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text = torch.load(paths["text_embeddings"], map_location="cpu")
    validate_source_only_inputs(membership, authority)
    episodes, supports = load_episodes(membership, authority)
    zero = load_query_interface(paths["clean_parent"], device=args.device)
    random = load_query_interface(paths["clean_parent"], device=args.device)
    candidate = load_query_interface(paths["candidate"], device=args.device)
    proposal_views = torch.as_tensor(authority["proposal_view_indices"]).long()
    train = (proposal_views % 4 == 1) | (proposal_views % 4 == 2)
    membership_proposals = torch.as_tensor(membership["proposal_indices"]).long()
    train_entries = train[membership_proposals]
    train_rows = torch.unique(
        torch.as_tensor(membership["row_indices"]).long()[train_entries], sorted=True
    )
    seed = int(gate1["seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(
        train_rows.numel(), 48, generator=generator
    ) / 48**0.5
    random.model.memory[:, 448:496].zero_()
    random.model.memory[train_rows.to(random.model.memory.device), 448:496] = initial.to(
        random.model.memory.device
    )
    lookup = {
        str(name).casefold(): index for index, name in enumerate(text["queries"])
    }
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)
    valid = torch.tensor([rows.numel() > 0 for rows, _weights in supports])
    view_values = torch.unique(proposal_views[proposal_views % 4 == args.residue])
    metrics = {"zero_D48": [], "random_D48": [], "candidate_D48": []}
    identity_exact = True
    examples = 0
    for column, name in enumerate(authority["query_names"]):
        token_index = lookup.get(str(name).casefold())
        if token_index is None:
            raise ValueError(f"text mask Gate-2 lacks query token: {name}")
        token = torch.as_tensor(text["embeddings"])[token_index].float()
        packet = QueryPacket("text", token)
        zero_rows, zero_weights, zero_identity = zero.compile_identity_anchors(
            packet, topk=args.topk, text_anchor_policy="positive"
        )
        rows, weights, identity = candidate.compile_identity_anchors(
            packet, topk=args.topk, text_anchor_policy="positive"
        )
        identity_exact &= (
            torch.equal(zero_identity, identity)
            and torch.equal(zero_rows, rows)
            and torch.equal(zero_weights, weights)
        )
        random_rows, random_weights, random_identity = random.compile_identity_anchors(
            packet, topk=args.topk, text_anchor_policy="positive"
        )
        identity_exact &= (
            torch.equal(zero_identity, random_identity)
            and torch.equal(zero_rows, random_rows)
            and torch.equal(zero_weights, random_weights)
        )
        for view in view_values.tolist():
            in_view = (proposal_views == int(view)) & valid
            positive = torch.where(in_view & (states[:, column] == 1))[0]
            negative = torch.where(in_view & (states[:, column] == 0))[0]
            if not positive.numel() or not negative.numel():
                continue
            target = torch.stack([
                episodes[int(index)].target for index in positive
            ]).any(0)
            negative_mask = torch.stack([
                episodes[int(index)].target for index in negative
            ]).any(0) & ~target
            if not bool(target.any()) or not bool(negative_mask.any()):
                continue
            known = target | negative_mask
            unknown = ~known
            episode = episodes[int(positive[0])]
            predictions = {
                "zero_D48": _render_text_mask(
                    zero, zero_rows, zero_weights, episode, temperature=args.temperature
                ),
                "random_D48": _render_text_mask(
                    random, random_rows, random_weights, episode, temperature=args.temperature
                ),
                "candidate_D48": _render_text_mask(
                    candidate, rows, weights, episode, temperature=args.temperature
                ),
            }
            for label, prediction in predictions.items():
                metrics[label].append(evaluate_source_heldout(
                    prediction,
                    target.flatten(),
                    known.flatten(),
                    unknown.flatten(),
                    torch.zeros_like(prediction),
                    torch.zeros_like(target.flatten()),
                ))
            examples += 1
    if not examples:
        raise ValueError("text mask Gate-2 has no heldout mask examples")
    summaries = {name: _mean_metrics(values) for name, values in metrics.items()}
    passed, failures = _gate_decision(
        summaries["zero_D48"],
        summaries["random_D48"],
        summaries["candidate_D48"],
        identity_exact=identity_exact,
    )
    payload = {
        "schema": "radio_gs.sugm_v3.text_anchor_instance_mask_gate.v1",
        "scene": membership["scene"],
        "residue": args.residue,
        "examples": examples,
        "identity_bitwise_preserved": identity_exact,
        "metrics": summaries,
        "delta_vs_zero": {
            name: summaries["candidate_D48"][name] - summaries["zero_D48"][name]
            for name in summaries["candidate_D48"]
        },
        "delta_vs_random": {
            name: summaries["candidate_D48"][name] - summaries["random_D48"][name]
            for name in summaries["candidate_D48"]
        },
        "gate": {
            "passed": passed,
            "failures": failures,
            "rule": "bitwise_clean_D128_identity_and_text_exact_render_IoU_up_Brier_down_vs_zero_and_seeded_random_D48",
        },
        "method": {
            "topk": args.topk,
            "temperature": args.temperature,
            "deployment_order": "D128_positive_anchors_then_D48_gaussian_sigmoid_then_exact_MPR_render",
            "unknown": "excluded_from_negative_authority",
            "calibrator": "disabled",
            "boundary": "disabled",
        },
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()


__all__ = ["_gate_decision"]
