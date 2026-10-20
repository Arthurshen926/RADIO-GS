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
from .storage import save_field, sha256, deployment_bytes
from .geometry import relaxed_subset_alpha
from .support_training import TrainingRays, episode_support


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


def fit(observations, output, config, device="cuda", *, save_function=None, legacy_v1=False):
    """Fit only with an explicit storage writer or an explicit historical v1 request."""
    if save_function is None and not legacy_v1:
        raise ValueError("new fits require an explicit save_function for shared v2 storage; "
                         "use legacy_v1=True only for historical reconstruction")
    if save_function is not None and legacy_v1:
        raise ValueError("save_function and legacy_v1 are mutually exclusive")
    save = save_field if legacy_v1 else save_function
    started = time.monotonic()
    root = Path(observations)
    manifest = json.loads((root / "observations.json").read_text())
    objective = config.get("render_objective", "visible_membership")
    support_policy = config.get("support_policy", "single")
    if support_policy not in {"single", "mixed_unsigned", "mixed_signed", "paired_signed"}:
        raise ValueError("unknown source support policy")
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
    negative_pixels = []
    anchor_eligible = torch.zeros(len(initial['observed']), dtype=torch.bool)
    for frame in train_frames:
        entry = next(r for r in manifest["frames"] if r["frame"] == frame)
        path = root / f"view_{frame:05d}.pt"
        if sha256(path) != entry["compiled"]["sha256"]:
            raise ValueError("source observation changed")
        view = torch.load(path, map_location="cpu", weights_only=True)
        if support_policy != "single":
            view["training_rays"] = TrainingRays(view["operator"])
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
                eligible_negative = torch.where(neg & (view['best_element'] >= 0))[0]
                eligible_negative = eligible_negative[initial['observed'][view['best_element'][eligible_negative]]]
                negative_pixels.append(eligible_negative)
    if not examples:
        raise ValueError("no reliable positive/negative training episodes")
    physical_scale = float(torch.cat([v["scales_m"] for v in views]).median())
    field = CapabilityField(initial["geometry"], initial["semantic"], initial["local"], group,
              initial["observed"], default_scale=physical_scale, threshold=config["threshold"],
              candidate_count=config["candidate_count"],
              candidate_radius=config["candidate_radius_m"]*manifest["world_units_per_metre"],
              anchor_eligible=anchor_eligible if anchor_domain=='training_eligible' else None,
              spatial_metric=config.get('spatial_metric', 'none'),
              local_metric=config.get('local_metric', 'none'),
              support_aggregation=config.get('support_aggregation', 'normalized_prototype'),
              negative_rule=config.get('negative_rule', 'product'),
              world_units_per_metre=manifest['world_units_per_metre']).to(device)
    baseline_path = Path(output) / "untrained"
    trained_path = Path(output) / "trained"
    baseline_manifest = save(field, baseline_path, dict(observations_sha256=sha256(root / "observations.json"),
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
    support_generator = torch.Generator().manual_seed(config["seed"]+1)
    support_counts = {}
    support_trace = []
    condition_evaluations = 0
    visits = torch.zeros(len(examples), dtype=torch.long)
    for step in range(config["steps"]):
        position = int(torch.randint(len(examples), (1,)))
        visits[position] += 1
        view_index, mask_index, positive, negative, pixels = examples[position]
        view = views[view_index]
        pixel = pixels[int(torch.randint(len(pixels), (1,)))]
        def rendered_loss(member):
            if objective == "subset_alpha":
                height, width = manifest["raster_hw"]
                rendered = relaxed_subset_alpha(field, member, view["camera"], height, width, manifest["world_units_per_metre"])
            else:
                rendered = torch.sparse.mm(view["operator"], member[:, None])[:, 0]
            return partial_membership_loss(rendered, positive, negative)
        optimizer.zero_grad(set_to_none=True)
        anchor_loss = None
        if support_policy == 'paired_signed':
            anchor = view['best_element'][pixel].to(device)
            anchor_member = field.support_membership(Support(anchor.reshape(1), torch.ones(1, device=device)),
                                                     float(view['scales_m'][mask_index]))
            anchor_loss = rendered_loss(anchor_member)
            (anchor_loss * .5).backward()
            anchor_loss = anchor_loss.detach()
            del anchor_member
            condition_evaluations += 1
        support, exclusion, support_kind = episode_support(support_policy, step, view, pixels, pixel,
            negative_pixels[position], support_generator, device)
        member = field.support_membership(support, float(view["scales_m"][mask_index]), exclusion)
        support_counts[support_kind] = support_counts.get(support_kind, 0)+1
        support_trace.append(dict(frame=train_frames[view_index], region=mask_index, first_pixel=int(pixel),
                                 kind=support_kind, positive_rows=len(support.indices),
                                 negative_rows=0 if exclusion is None else len(exclusion.indices)))
        condition_evaluations += 1
        if exclusion is not None:
            condition_evaluations += (len(exclusion.components.unique())
                if field.support_aggregation == 'expected_energy' and exclusion.components is not None else 1)
        loss = rendered_loss(member)
        if anchor_loss is None:
            loss.backward()
        else:
            (loss * .5).backward()
            loss = .5 * (loss.detach() + anchor_loss)
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
    trained_manifest = save(field, trained_path, dict(
            observations_sha256=sha256(root / "observations.json"),
            supervision="source SAM partial masks rendered through frozen Gaussian geometry",
            render_objective=objective,
            extra_3d_labels=False, final_benchmark_queries_used=False, relation_language_training=False))
    report = dict(config=config, train_frames=train_frames, training_episodes=len(examples),
                  render_objective=objective,
                  support_policy=support_policy, support_counts=support_counts, support_trace=support_trace,
                  member_forward_evaluations=condition_evaluations,
                  rendered_losses=config['steps']*(2 if support_policy == 'paired_signed' else 1),
                  paired_anchor_episodes=config['steps'] if support_policy == 'paired_signed' else 0,
                  inferred_anchor_domain=anchor_domain,eligible_input_anchors=int(anchor_eligible.sum()),
                  fixed_training_diagnostic_domain="full-scene normalized membership; identical diagnostic across objectives",
                  development_frames=manifest["splits"]["development_frames"],
                  default_scale_m=physical_scale, geometry_unchanged=geometry_unchanged,
                  native_coordinates_unchanged=native_unchanged, history=history,
                  unique_training_episodes_sampled=int((visits>0).sum()),
                  fixed_training_partial_loss_before=initial_training_loss,
                  fixed_training_partial_loss_after=final_training_loss,
                  seconds=time.monotonic()-started,
                  untrained_deployment_bytes=deployment_bytes(baseline_manifest),
                  trained_deployment_bytes=deployment_bytes(trained_manifest),
                  benchmark_performance_established=False, compact_student=False)
    (Path(output) / "training.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
