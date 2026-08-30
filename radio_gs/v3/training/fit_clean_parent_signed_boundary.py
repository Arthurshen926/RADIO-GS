"""Gate-3 training of fresh query-signed D16 on a passed D48 candidate."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.evaluation.source_heldout import evaluate_source_heldout
from radio_gs.v3.memory.structured_memory import LowRankPrivateBranchMemory, SharedPrivateLayout
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.training.fit_clean_parent_instance import (
    TextMaskAuthority,
    _compile_text_mask_authority,
)
from radio_gs.v3.training.instance_upper_bound import (
    mask_boundary,
    proposal_supports,
    sha256_file,
    validate_source_only_inputs,
)
from radio_gs.v3.training.rendered_mask import render_membership
from radio_gs.v3.training.run_instance_upper_bound import (
    load_episodes,
    training_support_for_heldout,
)


def _signed_hit_prediction(
    model: LowRankPrivateBranchMemory,
    head: nn.Linear,
    support: tuple[torch.Tensor, torch.Tensor],
    episode,
    *,
    temperature: float,
    maximum_logit_residual: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_rows, support_weights = support
    support_count = support_rows.numel()
    combined = torch.cat((support_rows, episode.gaussian_ids))
    unique, inverse = torch.unique(combined, sorted=True, return_inverse=True)
    device = model.memory.device
    embedding = model.instance_view(episode.scale, unique.to(device))
    prototype = pool_prototype(
        embedding[inverse[:support_count].to(device)], support_weights.to(device)
    )
    hit_embedding = embedding[inverse[support_count:].to(device)]
    base = membership_from_prototype(
        hit_embedding, prototype, temperature=temperature
    )
    unique_raw = head(model.boundary_view(unique.to(device))).squeeze(-1)
    raw = unique_raw[inverse[support_count:].to(device)]
    magnitude = raw.tanh().square()
    logit = torch.logit(base.clamp(1e-5, 1 - 1e-5))
    residual = (
        float(maximum_logit_residual) * magnitude * logit.tanh()
    )
    refined = torch.sigmoid(logit + residual)
    return base, refined, magnitude


def _render_hits(value: torch.Tensor, episode) -> torch.Tensor:
    return render_membership(
        value,
        torch.arange(value.numel(), device=value.device),
        episode.pixel_ids.to(value.device),
        episode.contribution_weights.to(value.device),
        num_pixels=episode.target.numel(),
    )


def signed_boundary_objective(
    model: LowRankPrivateBranchMemory,
    head: nn.Linear,
    support: tuple[torch.Tensor, torch.Tensor],
    episode,
    target: torch.Tensor,
    known: torch.Tensor,
    *,
    temperature: float,
    maximum_logit_residual: float,
    body_preservation_weight: float,
) -> torch.Tensor:
    base, refined, magnitude = _signed_hit_prediction(
        model,
        head,
        support,
        episode,
        temperature=temperature,
        maximum_logit_residual=maximum_logit_residual,
    )
    base_image = _render_hits(base, episode).clamp(1e-6, 1 - 1e-6)
    refined_image = _render_hits(refined, episode).clamp(1e-6, 1 - 1e-6)
    magnitude_image = _render_hits(magnitude, episode)
    device = refined.device
    truth = torch.as_tensor(target, device=device).bool().flatten()
    authority = torch.as_tensor(known, device=device).bool().flatten()
    boundary = mask_boundary(truth.reshape(target.shape)).flatten().to(device)
    # The source mask owns its narrow inside/outside boundary band even when
    # the broader instance contract correctly leaves the surrounding raster
    # unknown.  No pixels beyond this morphology-derived band are promoted to
    # background authority.
    authority = authority | boundary
    band = boundary & authority
    inner = band & truth
    outer = band & ~truth
    if not bool(inner.any()) or not bool(outer.any()):
        raise ValueError("signed D16 boundary authority lacks an inside or outside band")
    boundary_bce = -0.5 * (
        refined_image[inner].log().mean()
        + (1.0 - refined_image[outer]).log().mean()
    )
    boundary_brier = (
        refined_image[band] - truth[band].float()
    ).square().mean()
    body = authority & ~boundary
    edge_score = magnitude_image.clamp(1e-6, 1 - 1e-6)
    if not bool(body.any()):
        raise ValueError("signed D16 boundary authority lacks non-boundary pixels")
    edge_bce = -0.5 * (
        edge_score[band].log().mean()
        + (1.0 - edge_score[body]).log().mean()
    )
    body_preservation = (
        (refined_image[body] - base_image[body]).square().mean()
        if bool(body.any())
        else refined_image.new_zeros(())
    )
    off_boundary = (
        magnitude_image[body].square().mean()
        if bool(body.any())
        else refined_image.new_zeros(())
    )
    return (
        boundary_bce
        + boundary_brier
        + edge_bce
        + float(body_preservation_weight) * (body_preservation + off_boundary)
    )


def _text_target(item: TextMaskAuthority) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.stack(item.positive_masks).any(0)
    negative = torch.stack(item.negative_masks).any(0) & ~target
    return target, target | negative


@torch.no_grad()
def _heldout_text_authority(
    *,
    clean_parent_path: Path,
    authority: dict,
    text_payload: dict,
    episodes: list,
    supports,
    residue: int,
    topk: int,
) -> list[TextMaskAuthority]:
    # Reuse the exact compiler contract by presenting the held-out residue as
    # a temporary train residue only inside this source-only evaluator.
    from radio_gs.v3.query.interface import load_query_interface
    from radio_gs.v3.query.packet import QueryPacket

    interface = load_query_interface(clean_parent_path, device="cpu")
    lookup = {str(name).casefold(): i for i, name in enumerate(text_payload["queries"])}
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)
    valid = torch.tensor([rows.numel() > 0 for rows, _weights in supports])
    output = []
    for column, name in enumerate(authority["query_names"]):
        token = torch.as_tensor(text_payload["embeddings"])[lookup[str(name).casefold()]].float()
        rows, weights, _identity = interface.compile_identity_anchors(
            QueryPacket("text", token), topk=topk, text_anchor_policy="positive"
        )
        for view in torch.unique(views[views % 4 == residue]).tolist():
            in_view = (views == int(view)) & valid
            positive = torch.where(in_view & (states[:, column] == 1))[0]
            negative = torch.where(in_view & (states[:, column] == 0))[0]
            if not positive.numel() or not negative.numel():
                continue
            positive_masks = tuple(episodes[int(index)].target for index in positive)
            negative_masks = tuple(episodes[int(index)].target for index in negative)
            target = torch.stack(positive_masks).any(0)
            negative_pixels = torch.stack(negative_masks).any(0) & ~target
            if not bool(target.any()) or not bool(negative_pixels.any()):
                continue
            output.append(TextMaskAuthority(
                query_name=str(name),
                anchor_rows=rows,
                anchor_weights=weights,
                representative=episodes[int(positive[0])],
                positive_masks=positive_masks,
                negative_masks=negative_masks,
            ))
    del interface
    if not output:
        raise ValueError("signed D16 heldout text cohort is empty")
    return output


@torch.no_grad()
def _evaluate_items(
    model,
    head,
    items,
    *,
    temperature: float,
    maximum_logit_residual: float,
) -> tuple[dict[str, float], dict[str, float]]:
    baseline_values = []
    candidate_values = []
    for support, episode, target, known in items:
        base, refined, magnitude = _signed_hit_prediction(
            model,
            head,
            support,
            episode,
            temperature=temperature,
            maximum_logit_residual=maximum_logit_residual,
        )
        base_image = _render_hits(base, episode).cpu()
        refined_image = _render_hits(refined, episode).cpu()
        edge_image = _render_hits(magnitude, episode).cpu()
        boundary = mask_boundary(target).flatten()
        evaluation_known = known.flatten() | boundary
        unknown = ~evaluation_known
        args = (
            target.flatten(), evaluation_known, unknown,
        )
        baseline_values.append(evaluate_source_heldout(
            base_image, *args, torch.zeros_like(base_image), boundary
        ))
        candidate_values.append(evaluate_source_heldout(
            refined_image, *args, edge_image, boundary
        ))
    def mean(values):
        return {
            name: sum(float(getattr(value, name)) for value in values) / len(values)
            for name in ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")
        }
    return mean(baseline_values), mean(candidate_values)


def _gate(baseline, candidate, *, tolerance: float) -> tuple[bool, list[str]]:
    failures = []
    for cohort in ("oracle", "text"):
        before, after = baseline[cohort], candidate[cohort]
        if after["boundary_f"] <= before["boundary_f"]:
            failures.append(f"{cohort}: Boundary-F did not increase")
        if after["mask_iou"] < before["mask_iou"] - tolerance:
            failures.append(f"{cohort}: body IoU regressed beyond tolerance")
        if after["brier"] > before["brier"] + tolerance:
            failures.append(f"{cohort}: body Brier regressed beyond tolerance")
    return not failures, failures


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("instance_candidate", args.instance_candidate),
            ("gate2_report", args.gate2_report),
            ("membership", args.membership),
            ("authority", args.authority),
            ("text_embeddings", args.text_embeddings),
        )
    }
    candidate = torch.load(paths["instance_candidate"], map_location="cpu")
    gate2 = __import__("json").loads(paths["gate2_report"].read_text())
    metadata = candidate.get("metadata", {})
    gate1 = metadata.get("instance_gate1", {})
    if (
        gate1.get("status") != "passed"
        or gate2.get("schema") != "radio_gs.sugm_v3.text_anchor_instance_mask_gate.v1"
        or not gate2.get("gate", {}).get("passed")
        or gate2.get("inputs", {}).get("candidate", {}).get("sha256")
        != sha256_file(paths["instance_candidate"])
    ):
        raise ValueError("signed D16 requires passed Gate 1 and Gate 2")
    clean_parent_path = Path(gate1["parent"]["path"]).resolve(strict=True)
    if sha256_file(clean_parent_path) != gate1["parent"]["sha256"]:
        raise ValueError("signed D16 clean parent receipt differs")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    validate_source_only_inputs(membership, authority)
    episodes, supports = load_episodes(membership, authority)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    evaluation = [item for item in valid if item.view_index % 4 == 3]
    train_proposals = {item.proposal_index for item in training}
    oracle_training = [
        item for item in training
        if bool((item.known & item.target).any())
        and bool((item.known & ~item.target).any())
    ]
    text_training = _compile_text_mask_authority(
        clean_parent_path=clean_parent_path,
        authority=authority,
        text_payload=text_payload,
        episodes=episodes,
        supports=supports,
        topk=args.text_topk,
    )
    text_evaluation = _heldout_text_authority(
        clean_parent_path=clean_parent_path,
        authority=authority,
        text_payload=text_payload,
        episodes=episodes,
        supports=supports,
        residue=3,
        topk=args.text_topk,
    )
    latent = torch.as_tensor(candidate["latent"]).float()
    layout = SharedPrivateLayout()
    if bool(latent[:, layout.slices["boundary"]].any()):
        raise ValueError("signed D16 parent boundary block is not zero")
    model = LowRankPrivateBranchMemory(latent, layout=layout)
    current = model.state_dict()
    global_state = candidate["global_state_dict"]
    model.load_state_dict({
        "memory": latent,
        **{key: torch.as_tensor(global_state[key]) for key in current if key != "memory"},
    }, strict=True)
    device = torch.device(args.device)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_owned_training_blocks("boundary")
    owned = model.owned_training_parameter("boundary")
    train_hit_rows = torch.unique(torch.cat([
        item.gaussian_ids for item in training
    ]), sorted=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.no_grad():
        owned.zero_()
        owned[train_hit_rows.to(device)] = (
            torch.randn(train_hit_rows.numel(), 16, generator=generator) / 4.0
        ).to(device)
    head = nn.Linear(16, 1).to(device)
    with torch.no_grad():
        head.weight.copy_((torch.randn(1, 16, generator=generator) / 4.0).to(device))
        head.bias.zero_()
    optimizer = torch.optim.AdamW(
        [owned, *head.parameters()], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    best_loss = float("inf")
    best_boundary = None
    best_head = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        oracle_batch = rng.sample(
            oracle_training, k=min(args.oracle_masks_per_step, len(oracle_training))
        )
        text_batch = rng.sample(
            text_training, k=min(args.text_masks_per_step, len(text_training))
        )
        values = []
        for item in oracle_batch:
            loss = signed_boundary_objective(
                model, head, supports[item.proposal_index], item,
                item.target, item.known,
                temperature=args.temperature,
                maximum_logit_residual=args.maximum_logit_residual,
                body_preservation_weight=args.body_preservation_weight,
            )
            values.append(float(loss.detach()))
            (loss / (2 * len(oracle_batch))).backward()
        for item in text_batch:
            target, known = _text_target(item)
            loss = signed_boundary_objective(
                model, head, (item.anchor_rows, item.anchor_weights),
                item.representative, target, known,
                temperature=args.temperature,
                maximum_logit_residual=args.maximum_logit_residual,
                body_preservation_weight=args.body_preservation_weight,
            )
            values.append(float(loss.detach()))
            (loss / (2 * len(text_batch))).backward()
        torch.nn.utils.clip_grad_norm_([owned, *head.parameters()], 5.0)
        optimizer.step()
        value = sum(values) / len(values)
        snapshot = (step + 1) % args.snapshot_interval == 0 or step + 1 == args.steps
        if snapshot:
            print({"step": step + 1, "signed_boundary_loss": value}, flush=True)
            if value < best_loss:
                best_loss = value
                best_boundary = owned.detach().cpu().clone()
                best_head = {key: tensor.detach().cpu().clone() for key, tensor in head.state_dict().items()}
    if best_boundary is None or best_head is None:
        raise RuntimeError("signed D16 produced no checkpoint")
    with torch.no_grad():
        owned.copy_(best_boundary.to(device))
    head.load_state_dict(best_head)
    oracle_items = []
    for item in evaluation:
        support = training_support_for_heldout(
            item.proposal_index, train_proposals, supports, authority
        )
        if support is not None and bool((item.known & ~item.target).any()):
            oracle_items.append((support, item, item.target, item.known))
    text_items = []
    for item in text_evaluation:
        target, known = _text_target(item)
        text_items.append(((item.anchor_rows, item.anchor_weights), item.representative, target, known))
    baseline, final = {}, {}
    baseline["oracle"], final["oracle"] = _evaluate_items(
        model, head, oracle_items,
        temperature=args.temperature,
        maximum_logit_residual=args.maximum_logit_residual,
    )
    baseline["text"], final["text"] = _evaluate_items(
        model, head, text_items,
        temperature=args.temperature,
        maximum_logit_residual=args.maximum_logit_residual,
    )
    passed, failures = _gate(
        baseline, final, tolerance=args.body_metric_tolerance
    )
    deployed = model.deployment_memory()
    deltas = {
        name: float((deployed[:, columns] - latent[:, columns]).abs().max())
        for name, columns in layout.slices.items()
    }
    if any(deltas[name] != 0 for name in ("shared", "semantic", "instance")):
        raise RuntimeError("signed D16 changed a protected parent block")
    output = Path(args.output).resolve()
    output_global = dict(global_state)
    output_global["boundary_head.weight"] = best_head["weight"]
    output_global["boundary_head.bias"] = best_head["bias"]
    payload = {
        **candidate,
        "latent": deployed,
        "global_state_dict": output_global,
        "metadata": {
            **metadata,
            "instance_boundary_private_trained": True,
            "signed_boundary_gate3": {
                "status": "passed" if passed else "failed",
                "instance_parent": {
                    "path": str(paths["instance_candidate"]),
                    "sha256": sha256_file(paths["instance_candidate"]),
                },
                "gate2_report": {
                    "path": str(paths["gate2_report"]),
                    "sha256": sha256_file(paths["gate2_report"]),
                },
                "residual": "max_logit_residual_times_D16_boundary_magnitude_times_tanh_D48_instance_logit",
                "maximum_logit_residual": args.maximum_logit_residual,
                "non_boundary_policy": "zero_residual_penalty",
                "anchor_participation": False,
                "seed": args.seed,
            },
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.signed_boundary_gate.report.v1",
        "scene": candidate["scene"],
        "steps": args.steps,
        "best_source_train_loss": best_loss,
        "baseline_metrics": baseline,
        "candidate_metrics": final,
        "delta": {
            cohort: {
                name: final[cohort][name] - baseline[cohort][name]
                for name in final[cohort]
            }
            for cohort in final
        },
        "gate": {"passed": passed, "failures": failures},
        "protected_block_max_abs_delta": deltas,
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--instance-candidate", required=True)
    value.add_argument("--gate2-report", required=True)
    value.add_argument("--membership", required=True)
    value.add_argument("--authority", required=True)
    value.add_argument("--text-embeddings", required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--cpu-threads", type=int, default=24)
    value.add_argument("--steps", type=int, default=100)
    value.add_argument("--oracle-masks-per-step", type=int, default=2)
    value.add_argument("--text-masks-per-step", type=int, default=2)
    value.add_argument("--text-topk", type=int, default=8)
    value.add_argument("--temperature", type=float, default=0.15)
    value.add_argument("--maximum-logit-residual", type=float, default=1.0)
    value.add_argument("--body-preservation-weight", type=float, default=0.25)
    value.add_argument("--body-metric-tolerance", type=float, default=0.01)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--snapshot-interval", type=int, default=20)
    value.add_argument("--seed", type=int, default=20260829)
    value.add_argument("--output", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if (
        min(
            args.cpu_threads, args.steps, args.oracle_masks_per_step,
            args.text_masks_per_step, args.text_topk, args.snapshot_interval,
        ) <= 0
        or min(
            args.temperature, args.maximum_logit_residual,
            args.body_preservation_weight, args.learning_rate,
        ) <= 0
        or args.body_metric_tolerance < 0
    ):
        raise ValueError("signed D16 budgets differ")
    print(run(args))


if __name__ == "__main__":
    main()


__all__ = ["_gate", "signed_boundary_objective"]
