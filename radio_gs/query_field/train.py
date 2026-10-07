"""Scene-specific rendered membership fitting with protected native channels."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .contracts import Support
from .field import CapabilityField
from .observations import training_episode
from .storage import save_field, sha256
from .geometry import relaxed_subset_alpha


def partial_membership_loss(predicted, positive, negative):
    """Equal positive/explicit-negative mass; unknown entries receive no gradient."""
    terms = []
    if bool(positive.any()):
        terms.append(-predicted[positive].clamp_min(1e-6).log().mean())
    if bool(negative.any()):
        terms.append(-(1-predicted[negative]).clamp_min(1e-6).log().mean())
    if not terms:
        raise ValueError("no known supervision")
    return torch.stack(terms).mean()


def fit(observations, output, config, device="cuda"):
    started = time.monotonic()
    root = Path(observations)
    manifest = json.loads((root / "observations.json").read_text())
    objective = config.get("render_objective", "visible_membership")
    if objective not in {"visible_membership", "subset_alpha"}:
        raise ValueError("unknown member rendering objective")
    anchor_domain = config.get('inferred_anchor_domain', 'all_observed')
    if anchor_domain not in {'all_observed', 'training_eligible'}:
        raise ValueError('unknown inferred anchor domain')
    if sha256(root / "initial.pt") != manifest["initial"]["sha256"]:
        raise ValueError("initial native field changed")
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(config["seed"])
    projection = torch.randn(initial["local"].shape[1], config["group_dim"])
    projection = torch.linalg.qr(projection, mode="reduced").Q
    group = F.normalize(initial["local"].float() @ projection, dim=-1)
    train_frames = manifest["splits"]["train_frames"]
    views = []
    examples = []
    anchor_eligible = torch.zeros(len(initial['observed']), dtype=torch.bool)
    for frame in train_frames:
        entry = next(r for r in manifest["frames"] if r["frame"] == frame)
        path = root / f"view_{frame:05d}.pt"
        if sha256(path) != entry["compiled"]["sha256"]:
            raise ValueError("source observation changed")
        view = torch.load(path, map_location="cpu", weights_only=True)
        view["operator"] = view["operator"].to(device)
        view["camera"] = SimpleNamespace(intrinsic=view["intrinsic"],world_to_camera=view["world_to_camera"],
            height=view["original_hw"][0],width=view["original_hw"][1])
        views.append(view)
        for index in range(len(view["masks"])):
            episode = training_episode(view, index, initial['observed'])
            if episode is not None:
                pos, neg, candidate_pixels = episode
                anchor_eligible[view['best_element'][candidate_pixels]] = True
                examples.append((len(views)-1, index, pos.to(device), neg.to(device), candidate_pixels))
    if not examples:
        raise ValueError("no reliable positive/negative training episodes")
    physical_scale = float(torch.cat([v["scales_m"] for v in views]).median())
    field = CapabilityField(initial["geometry"], initial["semantic"], initial["local"], group,
              initial["observed"], default_scale=physical_scale, threshold=config["threshold"],
              candidate_count=config["candidate_count"],
              candidate_radius=config["candidate_radius_m"]*manifest["world_units_per_metre"],
              anchor_eligible=anchor_eligible if anchor_domain=='training_eligible' else None,
              spatial_metric=config.get('spatial_metric', 'none'),
              world_units_per_metre=manifest['world_units_per_metre']).to(device)
    baseline_path = Path(output) / "untrained"
    trained_path = Path(output) / "trained"
    baseline_manifest = save_field(field, baseline_path, dict(observations_sha256=sha256(root / "observations.json"),
                                    supervision="native source features; random local projection for grouping"))
    @torch.no_grad()
    def fixed_training_diagnostic():
        losses = []
        for view_index, mask_index, positive, negative, pixels in examples:
            view = views[view_index]
            anchor = view["best_element"][pixels[len(pixels)//2]].to(device)
            support = Support(anchor.reshape(1), torch.ones(1, device=device))
            member = field.member_logits(support, float(view["scales_m"][mask_index])).sigmoid()
            rendered = torch.sparse.mm(view["operator"], member[:,None])[:,0]
            losses.append(float(partial_membership_loss(rendered,positive,negative)))
        return sum(losses)/len(losses)
    initial_training_loss = fixed_training_diagnostic()
    optimizer = torch.optim.Adam(field.parameters(), lr=config["learning_rate"])
    history = []
    visits = torch.zeros(len(examples), dtype=torch.long)
    for step in range(config["steps"]):
        position = int(torch.randint(len(examples), (1,)))
        visits[position] += 1
        view_index, mask_index, positive, negative, pixels = examples[position]
        view = views[view_index]
        pixel = pixels[int(torch.randint(len(pixels), (1,)))]
        anchor = view["best_element"][pixel].to(device)
        support = Support(anchor.reshape(1), torch.ones(1, device=device))
        member = field.member_logits(support, float(view["scales_m"][mask_index])).sigmoid()
        if objective == "subset_alpha":
            height,width = manifest["raster_hw"]
            rendered = relaxed_subset_alpha(field, member, view["camera"],height,width,manifest["world_units_per_metre"])
        else:
            rendered = torch.sparse.mm(view["operator"], member[:, None])[:, 0]
        loss = partial_membership_loss(rendered, positive, negative)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.)
        optimizer.step()
        if step % 25 == 0 or step == config["steps"]-1:
            row = dict(step=step+1, loss=float(loss.detach()), frame=train_frames[view_index], region=mask_index)
            history.append(row)
            print("fit", row, flush=True)
    geometry_unchanged = all(torch.equal(getattr(field, k).cpu(), v.float()) for k,v in initial["geometry"].items())
    native_unchanged = torch.equal(field.semantic.codes.cpu(), initial["semantic"]) and torch.equal(field.local.codes.cpu(), initial["local"])
    if not geometry_unchanged or not native_unchanged:
        raise RuntimeError("group fitting changed protected geometry/native coordinates")
    final_training_loss = fixed_training_diagnostic()
    trained_manifest = save_field(field, trained_path, dict(
            observations_sha256=sha256(root / "observations.json"),
            supervision="source SAM partial masks rendered through frozen Gaussian geometry",
            render_objective=objective,
            extra_3d_labels=False, final_benchmark_queries_used=False, relation_language_training=False))
    report = dict(config=config, train_frames=train_frames, training_episodes=len(examples),
                  render_objective=objective,
                  inferred_anchor_domain=anchor_domain,eligible_input_anchors=int(anchor_eligible.sum()),
                  fixed_training_diagnostic_domain="full-scene normalized membership; identical diagnostic across objectives",
                  development_frames=manifest["splits"]["development_frames"],
                  default_scale_m=physical_scale, geometry_unchanged=geometry_unchanged,
                  native_coordinates_unchanged=native_unchanged, history=history,
                  unique_training_episodes_sampled=int((visits>0).sum()),
                  fixed_training_partial_loss_before=initial_training_loss,
                  fixed_training_partial_loss_after=final_training_loss,
                  seconds=time.monotonic()-started,
                  untrained_deployment_bytes=baseline_manifest["files"]["field.pt"]["bytes"],
                  trained_deployment_bytes=trained_manifest["files"]["field.pt"]["bytes"],
                  benchmark_performance_established=False, compact_student=False)
    (Path(output) / "training.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
