#!/usr/bin/env python3
"""Train a query-free, boundary-gated 2-D observation residual."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.field import BoundaryConditionedScreenResidual, load_canonical_field_checkpoint
from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_masked_render_losses
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint, project_feature_map_with_adaptor
from radio_gs.rendering.coefficient_renderer import render_boundary_conditioned_radio
from radio_gs.scripts.audit_canonical_capability_fidelity import _dataset, _parse_frame_ids, _sha256_tensor_rows
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.canonical_field_losses import normalized_render_reconstruction_loss


@torch.no_grad()
def _metrics(residual, field, model, renderer, dataset, lookup, frames, adaptors, device, alpha):
    sums = {"raw_radio": 0.0, "dino_v3": 0.0, "sam3": 0.0, "boundary_margin": 0.0}
    count = 0
    residual.eval()
    for frame in frames:
        sample = dataset[lookup[frame]]
        out = render_boundary_conditioned_radio(renderer, model, field, residual,
            sample["pose_w2c"].to(device), feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2])
        pred, target = out["feature_map"][None].float(), sample["radio_features"].to(device)[None].float()
        valid = out["alpha_map"] >= alpha
        raw = F.cosine_similarity(pred, target, dim=1)[0]
        sums["raw_radio"] += float(raw[valid].mean())
        condition = residual.conditions(out["rgb"][None], out["depth_map"][None], out["alpha_map"][None]).amax(1)[0]
        q = torch.quantile(condition[valid], .8) if bool(valid.any()) else condition.new_tensor(1.)
        boundary, interior = valid & (condition >= q), valid & (condition < q)
        sums["boundary_margin"] += float(raw[boundary].mean() - raw[interior].mean()) if bool(boundary.any() and interior.any()) else 0.
        for name, adaptor in adaptors.items():
            pp, tt = project_feature_map_with_adaptor(pred, adaptor), project_feature_map_with_adaptor(target, adaptor)
            sums[name] += float((pp * tt).sum(1)[0][valid].mean())
        count += 1
    return {key: value / count for key, value in sums.items()}


def train(args):
    device = torch.device(args.device); torch.manual_seed(args.seed)
    config = load_config(args.config)
    model, _, renderer, _, _, _, _ = load_render_pipeline(args.config, args.geometry_checkpoint,
        device, strict_checkpoint_contract=True, load_ply_rgb_features=False)
    field, payload = load_canonical_field_checkpoint(args.field_checkpoint, map_location="cpu")
    if payload.get("geometry_fingerprint", {}).get("xyz_sha256") != _sha256_tensor_rows(model.get_xyz()):
        raise ValueError("field/geometry fingerprint mismatch")
    field = field.to(device).eval(); field.requires_grad_(False)
    training = list(map(int, payload.get("mpr_cache_metadata", {}).get("selected_frame_indices", [])))
    excluded = set(map(int, payload.get("mpr_cache_metadata", {}).get("excluded_frame_ids", [])))
    validation = _parse_frame_ids(args.validation_frame_ids)
    if not training or not validation or not set(validation).issubset(excluded):
        raise ValueError("training provenance or excluded validation frames unavailable")
    included = _parse_frame_ids(args.include_frame_ids) if args.include_frame_ids else None
    dataset = _dataset(config, renderer, included); lookup = {int(f): i for i, f in enumerate(dataset.frame_indices)}
    if (set(training) | set(validation)) - set(lookup): raise ValueError("requested frames unavailable")
    residual = BoundaryConditionedScreenResidual(rank=args.rank, hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale).to(device)
    if sum(p.numel() for p in residual.parameters()) > args.maximum_parameters:
        raise ValueError("boundary residual exceeds low-capacity budget")
    adaptors = {name: load_radio_adaptor_from_checkpoint(args.radio_checkpoint, name,
        kind="feature_projection").to(device).eval() for name in ("dino_v3", "sam3")}
    for module in adaptors.values(): module.requires_grad_(False)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial = _metrics(residual, field, model, renderer, dataset, lookup, validation, adaptors, device, args.alpha_threshold)
    best, best_state, best_step = initial, copy.deepcopy(residual.state_dict()), 0
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for step in range(args.steps):
        frame = training[int(torch.randint(len(training), (), generator=generator))]
        sample = dataset[lookup[frame]]; residual.train(); optimizer.zero_grad(set_to_none=True)
        out = render_boundary_conditioned_radio(renderer, model, field, residual,
            sample["pose_w2c"].to(device), feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2])
        pred, target = out["feature_map"][None], sample["radio_features"].to(device)[None]
        valid = out["alpha_map"][None] >= args.alpha_threshold
        reconstruction = normalized_render_reconstruction_loss(pred, target, out["alpha_map"][None],
            alpha_threshold=args.alpha_threshold, cosine_weight=1., huber_weight=args.huber_weight)
        alignment, local, _ = compute_radio_adaptor_masked_render_losses(pred, target, adaptors, valid)
        regs = residual.regularization(); energy = out["screen_delta"].square().mean()
        loss = reconstruction + args.alignment_weight*alignment + args.local_weight*local + args.energy_weight*energy + args.basis_weight*regs["basis_orthogonality"]
        loss.backward(); torch.nn.utils.clip_grad_norm_(residual.parameters(), 5.); optimizer.step()
        if step == 0 or (step+1) % args.eval_every == 0:
            current = _metrics(residual, field, model, renderer, dataset, lookup, validation, adaptors, device, args.alpha_threshold)
            selected = (
                current["boundary_margin"]
                > best["boundary_margin"] + float(args.minimum_boundary_gain)
                and all(
                    current[k] >= initial[k] - args.maximum_drop
                    for k in ("raw_radio", "dino_v3", "sam3")
                )
            )
            if selected: best, best_state, best_step = current, copy.deepcopy(residual.state_dict()), step+1
            history.append({"step": step+1, "loss": float(loss), "metrics": current, "selected": selected}); print(json.dumps(history[-1]), flush=True)
    residual.load_state_dict(best_state); residual.cpu().eval()
    architecture = {"feature_dim": residual.feature_dim, "rank": residual.rank, "hidden_dim": residual.hidden_dim, "residual_scale": residual.residual_scale}
    result = {"schema_version": 1, "kind": "boundary_conditioned_screen_residual", "architecture": architecture,
        "state_dict": residual.state_dict(), "training_provenance": {"training_frames": training, "validation_frames": validation,
        "benchmark_images_opened": False, "benchmark_masks_opened": False, "text_queries_opened": False},
        "selection": {"initial": initial, "best": best, "best_step": best_step, "history": history},
        "invariants": {"canonical_field_frozen": True, "primitive_query_unchanged": True,
        "screen_only": True, "support_exactly_zero_without_observable_discontinuity": True}}
    output=Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(result, output)
    report={"output":str(output.resolve()),"initial":initial,"best":best,"gain":{k:best[k]-initial[k] for k in initial},"best_step":best_step,"parameters":sum(p.numel() for p in residual.parameters())}
    output.with_suffix(output.suffix+".json").write_text(json.dumps(report,indent=2)); return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","geometry-checkpoint","field-checkpoint","validation-frame-ids","output"): p.add_argument("--"+name,required=True)
    p.add_argument("--radio-checkpoint",default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    p.add_argument("--include-frame-ids", default="", help="Optional registered-observation allowlist.")
    p.add_argument("--device",default="cuda:0"); p.add_argument("--steps",type=int,default=128); p.add_argument("--eval-every",type=int,default=16)
    p.add_argument("--rank",type=int,default=8); p.add_argument("--hidden-dim",type=int,default=16); p.add_argument("--residual-scale",type=float,default=.1); p.add_argument("--maximum-parameters",type=int,default=25000)
    p.add_argument("--learning-rate",type=float,default=2e-3); p.add_argument("--weight-decay",type=float,default=1e-5); p.add_argument("--huber-weight",type=float,default=.25)
    p.add_argument("--alignment-weight",type=float,default=.5); p.add_argument("--local-weight",type=float,default=.2); p.add_argument("--energy-weight",type=float,default=1e-3); p.add_argument("--basis-weight",type=float,default=1e-4)
    p.add_argument("--alpha-threshold",type=float,default=.02); p.add_argument("--maximum-drop",type=float,default=.002); p.add_argument("--minimum-boundary-gain",type=float,default=1e-5); p.add_argument("--seed",type=int,default=42)
    print(json.dumps(train(p.parse_args()),indent=2))
if __name__ == "__main__": main()
