"""Gate-1 training of a fresh direct D48 on the sealed clean parent."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.memory.structured_memory import (
    SharedPrivateLayout,
    StructuredSharedPrivateMemory,
)
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import (
    sha256_file,
    validate_source_only_inputs,
)
from radio_gs.v3.training.run_instance_upper_bound import evaluate, load_episodes
from radio_gs.v3.training.run_structured_source_mapping import (
    compact_episode_objective,
    compact_relation_contrastive_loss,
    relation_training_edges,
)
from radio_gs.v3.training.rendered_mask import render_membership


@dataclass(frozen=True)
class TextMaskAuthority:
    query_name: str
    anchor_rows: torch.Tensor
    anchor_weights: torch.Tensor
    representative: object
    positive_masks: tuple[torch.Tensor, ...]
    negative_masks: tuple[torch.Tensor, ...]


@torch.no_grad()
def _compile_text_mask_authority(
    *,
    clean_parent_path: Path,
    authority: dict,
    text_payload: dict,
    episodes: list,
    supports,
    topk: int,
) -> list[TextMaskAuthority]:
    interface = load_query_interface(clean_parent_path, device="cpu")
    lookup = {
        str(name).casefold(): index
        for index, name in enumerate(text_payload["queries"])
    }
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)
    train = (views % 4 == 1) | (views % 4 == 2)
    valid = torch.tensor([rows.numel() > 0 for rows, _weights in supports])
    output: list[TextMaskAuthority] = []
    for column, name in enumerate(authority["query_names"]):
        token_index = lookup.get(str(name).casefold())
        if token_index is None:
            raise ValueError(f"fresh D48 text authority lacks query token: {name}")
        positive = torch.where(train & valid & (states[:, column] == 1))[0]
        negative = torch.where(train & valid & (states[:, column] == 0))[0]
        if not positive.numel() or not negative.numel():
            continue
        token = torch.as_tensor(text_payload["embeddings"])[token_index].float()
        rows, weights, _identity = interface.compile_identity_anchors(
            QueryPacket("text", token),
            topk=topk,
            text_anchor_policy="positive",
        )
        for view in torch.unique(views[positive]).tolist():
            positive_view = positive[views[positive] == int(view)]
            negative_view = negative[views[negative] == int(view)]
            if not positive_view.numel() or not negative_view.numel():
                continue
            representative = episodes[int(positive_view[0])]
            positive_masks = tuple(
                episodes[int(index)].target for index in positive_view
            )
            negative_masks = tuple(
                episodes[int(index)].target for index in negative_view
            )
            positive_pixels = torch.stack(positive_masks).any(0)
            negative_pixels = torch.stack(negative_masks).any(0) & ~positive_pixels
            if not bool(positive_pixels.any()) or not bool(negative_pixels.any()):
                continue
            output.append(TextMaskAuthority(
                query_name=str(name),
                anchor_rows=rows.cpu(),
                anchor_weights=weights.cpu(),
                representative=representative,
                positive_masks=positive_masks,
                negative_masks=negative_masks,
            ))
    del interface
    if not output:
        raise ValueError("fresh D48 has no source-train text mask authority")
    return output


def compact_text_mask_objective(
    model: StructuredSharedPrivateMemory,
    item: TextMaskAuthority,
    *,
    temperature: float,
    unknown_growth_weight: float,
) -> torch.Tensor:
    """Train the exact D128-anchor -> D48 -> sigmoid -> render deployment path."""

    episode = item.representative
    anchor_count = item.anchor_rows.numel()
    combined = torch.cat((item.anchor_rows, episode.gaussian_ids))
    unique, inverse = torch.unique(combined, sorted=True, return_inverse=True)
    embedding = model(episode.scale, unique.to(model.memory.device))
    prototype = pool_prototype(
        embedding[inverse[:anchor_count].to(embedding.device)],
        item.anchor_weights.to(embedding.device),
    )
    hit_embedding = embedding[inverse[anchor_count:].to(embedding.device)]
    hit_probability = membership_from_prototype(
        hit_embedding, prototype, temperature=temperature
    )
    prediction = render_membership(
        hit_probability,
        torch.arange(hit_probability.numel(), device=embedding.device),
        episode.pixel_ids.to(embedding.device),
        episode.contribution_weights.to(embedding.device),
        num_pixels=episode.target.numel(),
    ).clamp(1e-6, 1 - 1e-6)
    target = torch.stack(item.positive_masks).any(0).flatten().to(embedding.device)
    negative = torch.stack(item.negative_masks).any(0).flatten().to(embedding.device)
    negative &= ~target
    if not bool(target.any()) or not bool(negative.any()):
        raise ValueError("text mask authority requires positive and negative pixels")
    balanced_bce = -0.5 * (
        prediction[target].log().mean()
        + (1.0 - prediction[negative]).log().mean()
    )
    known = target | negative
    truth = target[known].float()
    score = prediction[known]
    dice = 1.0 - (2.0 * (score * truth).sum() + 1.0) / (
        score.sum() + truth.sum() + 1.0
    )
    brier = (score - truth).square().mean()
    unknown = ~known
    growth = (
        F.relu(prediction[unknown] - 0.5).square().mean()
        if bool(unknown.any())
        else prediction.new_zeros(())
    )
    return balanced_bce + dice + brier + float(unknown_growth_weight) * growth


def _validate_parent(
    parent: dict,
    *,
    parent_path: Path,
    membership_path: Path,
    authority_path: Path,
) -> tuple[torch.Tensor, SharedPrivateLayout]:
    metadata = parent.get("metadata", {})
    contract = metadata.get("clean_parent_contract", {})
    if (
        parent.get("schema") != "radio_gs.sugm_v3.unknown_aware_scene_state.v1"
        or contract.get("version") != "clean_parent_v1"
        or contract.get("child_initialization")
        != "D48_D16_and_all_child_global_paths_exact_zero"
        or not metadata.get("source_only")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
    ):
        raise ValueError("instance training requires the sealed clean parent")
    if contract.get("membership", {}).get("sha256") != sha256_file(membership_path):
        raise ValueError("clean parent membership differs")
    if contract.get("authority", {}).get("sha256") != sha256_file(authority_path):
        raise ValueError("clean parent relation authority differs")
    layout_record = dict(contract["layout"])
    if "visual" in layout_record:
        layout_record["shared"] = layout_record.pop("visual")
    layout = SharedPrivateLayout(**layout_record)
    latent = torch.as_tensor(parent["latent"]).float()
    slices = layout.slices
    if tensor_sha256(latent[:, slices["shared"]]) != contract["d320_tensor_sha256"]:
        raise ValueError("clean parent D320 receipt differs")
    if tensor_sha256(latent[:, slices["semantic"]]) != contract["d128_tensor_sha256"]:
        raise ValueError("clean parent D128 receipt differs")
    if bool(latent[:, slices["instance"]].any()) or bool(
        latent[:, slices["boundary"]].any()
    ):
        raise ValueError("clean parent private blocks are not zero")
    global_state = parent.get("global_state_dict", {})
    child_keys = (
        "visual_to_instance.weight",
        "context_to_boundary.weight",
        "scale_adapter.weight",
        "scale_adapter.bias",
        "instance_down.weight",
        "instance_up.weight",
        "boundary_down.weight",
        "boundary_up.weight",
        "boundary_head.weight",
        "boundary_head.bias",
    )
    if any(key not in global_state or bool(torch.as_tensor(global_state[key]).any()) for key in child_keys):
        raise ValueError("clean parent has a nonzero child global path")
    if not parent_path.is_file():
        raise ValueError("clean parent path differs")
    return latent, layout


def _initialize_supported_rows(
    parameter: torch.Tensor,
    training,
    supports,
    *,
    seed: int,
) -> int:
    rows = torch.unique(
        torch.cat([supports[item.proposal_index][0] for item in training]), sorted=True
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randn(
        rows.numel(), parameter.shape[1], generator=generator
    ) / parameter.shape[1] ** 0.5
    with torch.no_grad():
        parameter.zero_()
        parameter[rows.to(parameter.device)] = initial.to(parameter.device)
    return int(rows.numel())


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(int(args.cpu_threads))
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("clean_parent", args.clean_parent),
            ("membership", args.membership),
            ("authority", args.authority),
            ("text_embeddings", args.text_embeddings),
        )
    }
    parent = torch.load(paths["clean_parent"], map_location="cpu")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    validate_source_only_inputs(membership, authority)
    latent, layout = _validate_parent(
        parent,
        parent_path=paths["clean_parent"],
        membership_path=paths["membership"],
        authority_path=paths["authority"],
    )
    if len({parent["scene"], membership["scene"], authority["scene"]}) != 1:
        raise ValueError("instance training scene axes differ")
    episodes, supports = load_episodes(membership, authority)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    evaluation = [item for item in valid if item.view_index % 4 == 3]
    train_proposals = {item.proposal_index for item in training}
    if not training or not evaluation:
        raise ValueError("instance source split is empty")
    left, right, labels, same_edges, different_edges = relation_training_edges(
        training, authority
    )
    text_training = _compile_text_mask_authority(
        clean_parent_path=paths["clean_parent"],
        authority=authority,
        text_payload=text_payload,
        episodes=episodes,
        supports=supports,
        topk=args.text_topk,
    )
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = StructuredSharedPrivateMemory(latent, layout=layout).to(device)
    current = model.state_dict()
    clean_globals = parent["global_state_dict"]
    compatible = {
        key: torch.as_tensor(clean_globals[key])
        for key in current
        if key != "memory" and key in clean_globals
        and torch.as_tensor(clean_globals[key]).shape == current[key].shape
    }
    model.load_state_dict({"memory": latent.to(device), **compatible}, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    baseline, baseline_count = evaluate(
        model, evaluation, supports, authority, train_proposals, args.temperature
    )
    model.enable_owned_training_blocks("instance")
    owned = model.owned_training_parameter("instance")
    initialized_rows = _initialize_supported_rows(
        owned, training, supports, seed=args.seed
    )
    random_initial, random_initial_count = evaluate(
        model, evaluation, supports, authority, train_proposals, args.temperature
    )
    optimizer = torch.optim.AdamW(
        [owned], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    best_loss = float("inf")
    best_instance = None
    text_loss_sum = 0.0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        selected = rng.sample(
            training, k=min(args.episodes_per_step, len(training))
        )
        rendered_values = []
        for item in selected:
            loss = compact_episode_objective(
                model,
                supports[item.proposal_index],
                item,
                temperature=args.temperature,
                unknown_growth_weight=args.unknown_growth_weight,
            )
            rendered_values.append(float(loss.detach()))
            (loss / len(selected)).backward()
        each = min(
            args.relation_edges_per_step // 2,
            len(same_edges),
            len(different_edges),
        )
        edge_indices = torch.tensor(
            rng.sample(same_edges, each) + rng.sample(different_edges, each),
            dtype=torch.long,
        )
        relation_value = 0.0
        for start in range(0, edge_indices.numel(), args.relation_backward_chunk):
            chunk = edge_indices[start : start + args.relation_backward_chunk]
            relation_loss = compact_relation_contrastive_loss(
                model,
                supports,
                left[chunk],
                right[chunk],
                labels[chunk],
                temperature=args.relation_temperature,
            )
            fraction = chunk.numel() / edge_indices.numel()
            (args.relation_weight * fraction * relation_loss).backward()
            relation_value += fraction * float(relation_loss.detach())
        selected_text = rng.sample(
            text_training,
            k=min(args.text_masks_per_step, len(text_training)),
        )
        text_values = []
        for item in selected_text:
            text_loss = compact_text_mask_objective(
                model,
                item,
                temperature=args.temperature,
                unknown_growth_weight=args.unknown_growth_weight,
            )
            text_values.append(float(text_loss.detach()))
            (
                args.text_coupling_weight * text_loss / len(selected_text)
            ).backward()
        text_value = sum(text_values) / len(text_values)
        text_loss_sum += text_value
        torch.nn.utils.clip_grad_norm_([owned], 5.0)
        optimizer.step()
        value = (
            sum(rendered_values) / len(rendered_values)
            + args.relation_weight * relation_value
            + args.text_coupling_weight * text_value
        )
        snapshot = (step + 1) % args.snapshot_interval == 0 or step + 1 == args.steps
        if snapshot:
            print(
                {
                    "step": step + 1,
                    "instance_loss": value,
                    "render_loss": sum(rendered_values) / len(rendered_values),
                    "relation_loss": relation_value,
                    "text_mask_loss": text_value,
                },
                flush=True,
            )
            if value < best_loss:
                best_loss = value
                best_instance = owned.detach().cpu().clone()
    if best_instance is None:
        raise RuntimeError("fresh D48 training produced no checkpoint")
    with torch.no_grad():
        owned.copy_(best_instance.to(device))
    candidate, candidate_count = evaluate(
        model, evaluation, supports, authority, train_proposals, args.temperature
    )
    deployed = model.deployment_memory()
    slices = layout.slices
    protected_deltas = {
        name: float((deployed[:, columns] - latent[:, columns]).abs().max())
        for name, columns in slices.items()
    }
    if (
        protected_deltas["shared"] != 0
        or protected_deltas["semantic"] != 0
        or protected_deltas["boundary"] != 0
    ):
        raise RuntimeError("fresh D48 training changed a protected block")
    gate_passed = (
        candidate_count == baseline_count
        and candidate["mask_iou"] > baseline["mask_iou"]
        and candidate["brier"] < baseline["brier"]
        and candidate["mask_iou"] > random_initial["mask_iou"]
        and candidate["brier"] < random_initial["brier"]
    )
    parent_metadata = parent["metadata"]
    output = Path(args.output).resolve()
    payload = {
        **parent,
        "latent": deployed,
        "metadata": {
            **parent_metadata,
            "private_architecture": "fresh_direct_D48_joint_oracle_and_text_exact_render",
            "instance_boundary_private_trained": False,
            "instance_private_trained": True,
            "instance_gate1": {
                "status": "passed" if gate_passed else "failed",
                "parent": {
                    "path": str(paths["clean_parent"]),
                    "sha256": sha256_file(paths["clean_parent"]),
                },
                "authority": {
                    "path": str(paths["authority"]),
                    "sha256": sha256_file(paths["authority"]),
                },
                "initialization": "seeded_random_D48_on_source_train_membership_rows_only",
                "initialized_rows": initialized_rows,
                "selection": "source_train_loss_only_dev_opened_once_after_selection",
                "text_training_contract": "fixed_clean_D128_positive_anchors_to_D48_sigmoid_then_exact_MPR_render",
                "text_topk": args.text_topk,
                "text_masks_per_step": args.text_masks_per_step,
                "text_coupling_weight": args.text_coupling_weight,
                "text_embeddings": {
                    "path": str(paths["text_embeddings"]),
                    "sha256": sha256_file(paths["text_embeddings"]),
                },
                "low_rank_private_branch": "disabled_exact_zero",
                "scale_adapter": "disabled_exact_zero",
                "boundary_D16": "disabled_exact_zero",
                "seed": args.seed,
            },
            "source_only": True,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.clean_parent_instance_gate.report.v2",
        "scene": parent["scene"],
        "steps": args.steps,
        "best_source_train_loss": best_loss,
        "mean_text_mask_train_loss": text_loss_sum / args.steps,
        "text_mask_authorities": len(text_training),
        "evaluation_proposals": candidate_count,
        "baseline_evaluation_proposals": baseline_count,
        "random_initial_evaluation_proposals": random_initial_count,
        "zero_D48_baseline": baseline,
        "random_D48_initial": random_initial,
        "candidate_metrics": candidate,
        "delta_vs_zero": {
            name: candidate[name] - baseline[name] for name in candidate
        },
        "delta_vs_random_initial": {
            name: candidate[name] - random_initial[name] for name in candidate
        },
        "gate": {
            "passed": gate_passed,
            "rule": "dev_mask_iou_strictly_improves_and_brier_strictly_decreases_vs_zero_D48_and_seeded_random_initial_D48",
        },
        "protected_block_max_abs_delta": protected_deltas,
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--clean-parent", required=True)
    value.add_argument("--membership", required=True)
    value.add_argument("--authority", required=True)
    value.add_argument("--text-embeddings", required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--cpu-threads", type=int, default=24)
    value.add_argument("--steps", type=int, default=400)
    value.add_argument("--episodes-per-step", type=int, default=4)
    value.add_argument("--temperature", type=float, default=0.15)
    value.add_argument("--unknown-growth-weight", type=float, default=0.25)
    value.add_argument("--relation-edges-per-step", type=int, default=32)
    value.add_argument("--relation-temperature", type=float, default=0.1)
    value.add_argument("--relation-weight", type=float, default=1.0)
    value.add_argument("--relation-backward-chunk", type=int, default=16)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--text-topk", type=int, default=8)
    value.add_argument("--text-masks-per-step", type=int, default=2)
    value.add_argument("--text-coupling-weight", type=float, default=1.0)
    value.add_argument("--snapshot-interval", type=int, default=25)
    value.add_argument("--seed", type=int, default=20260829)
    value.add_argument("--output", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if (
        min(
            args.steps,
            args.episodes_per_step,
            args.relation_edges_per_step,
            args.relation_backward_chunk,
            args.snapshot_interval,
            args.cpu_threads,
            args.text_topk,
            args.text_masks_per_step,
        )
        <= 0
        or args.temperature <= 0
        or args.relation_temperature <= 0
        or args.learning_rate <= 0
        or args.unknown_growth_weight < 0
        or args.relation_weight < 0
        or args.text_coupling_weight <= 0
    ):
        raise ValueError("fresh D48 budgets differ")
    print(run(args))


if __name__ == "__main__":
    main()


__all__ = [
    "TextMaskAuthority",
    "_initialize_supported_rows",
    "_validate_parent",
    "compact_text_mask_objective",
    "run",
]
