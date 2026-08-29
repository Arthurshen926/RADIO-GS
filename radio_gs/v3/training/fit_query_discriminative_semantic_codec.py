"""Fit one shared D1536->D128 codec from fresh ternary source authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.structured_initialization import fixed_jl_projection


def _parse_scene(value: str) -> tuple[Path, Path]:
    parts = value.split("::")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("scene must be AUTHORITY::SIGLIP_TEACHER")
    return tuple(Path(item).resolve(strict=True) for item in parts)  # type: ignore[return-value]


def _ternary_losses(
    visual: torch.Tensor,
    text: torch.Tensor,
    null: torch.Tensor,
    states: torch.Tensor,
    *,
    logit_scale: float = 10.0,
) -> dict[str, torch.Tensor]:
    cosine = visual @ text.T
    null_score = (visual @ null.T).max(1).values
    response = (cosine - null_score[:, None]) * logit_scale
    positive = states == 1
    negative = states == 0
    positive_loss = F.softplus(-response[positive]).mean()
    negative_loss = F.softplus(response[negative]).mean()
    listwise_terms = []
    for query in range(states.shape[1]):
        pos = cosine[positive[:, query], query]
        neg = cosine[negative[:, query], query]
        if pos.numel() and neg.numel():
            listwise_terms.append(F.softplus(neg.max() - pos.max() + 0.05))
    rows = positive.any(1) & negative.any(1)
    if bool(rows.any()):
        pos_best = cosine.masked_fill(~positive, -torch.inf).max(1).values
        neg_best = cosine.masked_fill(~negative, -torch.inf).max(1).values
        sibling = F.softplus(neg_best[rows] - pos_best[rows] + 0.05).mean()
    else:
        sibling = cosine.new_zeros(())
    listwise = (
        torch.stack(listwise_terms).mean()
        if listwise_terms else cosine.new_zeros(())
    )
    return {
        "response": positive_loss + negative_loss,
        "listwise": listwise,
        "sibling": sibling,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", type=_parse_scene, required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--canonical-negatives", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.scene) < 2 or args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("query-discriminative codec budget differs")
    device = torch.device(args.device)
    text_path = Path(args.text_embeddings).resolve(strict=True)
    negative_path = Path(args.canonical_negatives).resolve(strict=True)
    text_payload = torch.load(text_path, map_location="cpu")
    negative_payload = torch.load(negative_path, map_location="cpu")
    names = [str(value) for value in text_payload["queries"]]
    lookup = {name.casefold(): index for index, name in enumerate(names)}
    text = torch.as_tensor(text_payload["embeddings"]).float()
    negatives = torch.as_tensor(negative_payload["embeddings"]).float()
    visuals, states, receipts = [], [], []
    for authority_path, teacher_path in args.scene:
        authority = torch.load(authority_path, map_location="cpu")
        teacher = torch.load(teacher_path, map_location="cpu")
        if authority.get("schema") != "radio_gs.sugm_v3.native_language_authority.v3":
            raise ValueError("semantic codec requires native language authority v3")
        columns = [lookup[str(value).casefold()] for value in authority["query_names"]]
        local = torch.as_tensor(authority["query_state"]).to(torch.int8)
        views = torch.as_tensor(authority["proposal_view_indices"]).long()
        train = (views % 4 == 1) | (views % 4 == 2)
        global_state = torch.full((int(train.sum()), len(names)), -1, dtype=torch.int8)
        global_state[:, columns] = local[train]
        descriptor = torch.as_tensor(teacher["descriptors"])[train].float()
        context = torch.as_tensor(teacher["context_descriptors"])[train].float()
        visuals.extend((descriptor, context))
        states.extend((global_state, global_state.clone()))
        receipts.append({
            "scene": authority["scene"],
            "train_proposals": int(train.sum()),
            "positive_pairs": int((global_state == 1).sum()),
            "negative_pairs": int((global_state == 0).sum()),
            "authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
            "teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
        })
    raw_visual = torch.cat(visuals).to(device)
    ternary = torch.cat(states).to(device)
    if not bool((ternary == 1).any()) or not bool((ternary == 0).any()):
        raise ValueError("semantic codec lacks explicit positive or negative authority")
    raw_text = text.to(device)
    raw_null = negatives.to(device)
    torch.manual_seed(args.seed)
    basis = nn.Parameter(fixed_jl_projection(1536, 128, args.seed).to(device))
    optimizer = torch.optim.AdamW([basis], lr=args.learning_rate, weight_decay=1e-4)
    native_visual = F.normalize(raw_visual, dim=-1, eps=1e-8)
    native_text = F.normalize(raw_text, dim=-1, eps=1e-8)
    history = []
    for epoch in range(args.epochs):
        visual = F.normalize(raw_visual @ basis, dim=-1, eps=1e-8)
        projected_text = F.normalize(raw_text @ basis, dim=-1, eps=1e-8)
        null = F.normalize(raw_null @ basis, dim=-1, eps=1e-8)
        losses = _ternary_losses(visual, projected_text, null, ternary)
        distill = F.mse_loss(visual @ projected_text.T, native_visual @ native_text.T)
        gram = basis.T @ basis
        orthogonal = (gram - torch.eye(128, device=device)).square().mean()
        loss = (
            losses["response"] + losses["listwise"] + losses["sibling"]
            + 0.25 * distill + 0.01 * orthogonal
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            history.append({
                "epoch": epoch + 1, "loss": float(loss.detach()),
                **{name: float(value.detach()) for name, value in losses.items()},
                "distill": float(distill.detach()), "orthogonal": float(orthogonal.detach()),
            })
    payload = {
        "schema": "radio_gs.sugm_v3.query_discriminative_semantic_codec.v1",
        "state_dict": {"siglip_mean": torch.zeros(1536), "siglip_basis": basis.detach().cpu()},
        "history": history,
        "scene_receipts": receipts,
        "metadata": {
            "source_only": True, "source_train_residues": [1, 2],
            "objective": "null_response_plus_listwise_plus_sibling_margin_plus_native_distillation",
            "unknown_pairs_excluded": True, "historical_field_opened": False,
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "shared_across_scenes": True, "gaussian_indexed_state_added": 0,
            "text_embeddings": {"path": str(text_path), "sha256": sha256_file(text_path)},
            "canonical_negatives": {"path": str(negative_path), "sha256": sha256_file(negative_path)},
            "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({"output": str(output), "sha256": sha256_file(output), "history": history[-1]})


if __name__ == "__main__":
    main()


__all__ = ["_ternary_losses"]
