"""Matched-budget single-latent controls using existing legal teacher observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .bounded_experiment import BoundedExperiment, tree_bytes
from .contracts import Support
from .functional_training import FunctionalSource, preference_loss, projected_query_scores
from .storage import sha256, deployment_bytes
from .unified_compact import UnifiedCompactField


@torch.no_grad()
def initialize(initial, config, device):
    """Equal teacher mass joint PCA, followed by jointly trainable single codes.

    PCA is construction-only. Neither its encoder nor teacher features are kept
    in the exported student. Initial features exclude development views.
    """
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    observed = torch.where(initial["observed"])[0]
    selected = observed[torch.randperm(len(observed), generator=generator)[:config["pca_rows"]]]

    def joint(rows):
        return torch.cat([F.normalize(initial[c][rows].to(device).float(), dim=-1)
                          for c in ("semantic", "local")], dim=-1) / 2 ** .5

    examples = joint(selected)
    mean = examples.mean(0)
    _, _, basis = torch.pca_lowrank(examples - mean, q=config["latent_dim"], center=False, niter=3)
    latent = torch.cat([(joint(slice(i, i + 4096)) - mean) @ basis
                        for i in range(0, len(initial["observed"]), 4096)])
    field = UnifiedCompactField({k: v.to(device) for k, v in initial["geometry"].items()},
                                initial["observed"].to(device), latent,
                                semantic_dim=initial["semantic"].shape[1],
                                local_dim=initial["local"].shape[1], hidden_dim=config["hidden_dim"])
    field.to(device)
    start = 0
    for channel in ("semantic", "local"):
        end = start + initial[channel].shape[1]
        field.heads[channel].weight.copy_(basis[start:end] * 2 ** .5)
        field.heads[channel].bias.copy_(mean[start:end] * 2 ** .5)
        start = end
    return field


def sampled_support(operator, rows):
    """All contributors of the sampled positive ray; no fabricated exact point."""
    operator = operator.coalesce()
    mask = operator.indices()[0] == 0
    return Support(rows.to(device=operator.device, dtype=torch.long)[operator.indices()[1, mask]],
                   operator.values()[mask])


def functional_episode(field, source, index, generator, pairs, device, *, inferred=False):
    rows, operator, queries, teacher = source.sample(index, generator, pairs, device)
    rows_gpu = rows.to(device)
    scores = {c: projected_query_scores(getattr(field, c).decode(rows_gpu), queries[c], operator)
              for c in ("semantic", "local")}
    function = {c: preference_loss(scores[c], teacher[c]) for c in scores}
    if inferred:
        with torch.no_grad():
            identity = field.semantic.score(queries["semantic"]).flatten().masked_fill(~field.observed, -torch.inf)
            ids = identity.topk(field.candidate_count).indices
            support = Support(ids, (10 * identity[ids]).softmax(0))
    else:
        support = sampled_support(operator, rows)
    member = field.member_logits(support, field.default_scale, rows_gpu, explicit=not inferred).sigmoid()
    rendered = torch.sparse.mm(operator, member[:, None]).flatten()
    # Only sampled explicit positives and explicit negatives are included.
    target = torch.cat([torch.ones(pairs, device=device), torch.zeros(pairs, device=device)])
    member_loss = F.binary_cross_entropy(rendered.clamp(1e-6, 1 - 1e-6), target)
    return function, member_loss, scores, teacher, rendered, target


@torch.no_grad()
def validate(field, source, config, device):
    generator = torch.Generator().manual_seed(config["seed"] + 7)
    count = min(config["validation_episodes"], len(source.examples))
    indices = np.linspace(0, len(source.examples) - 1, count, dtype=int).tolist()
    values = {c: {"kl": [], "pair_agreement": []} for c in ("semantic", "local")}
    member = []
    intersections = unions = 0
    for index in indices:
        losses, mloss, scores, teacher, predicted, target = functional_episode(
            field, source, index, generator, config["pairs"], device)
        for c in values:
            a, b = scores[c].chunk(2)
            ta, tb = teacher[c].chunk(2)
            values[c]["kl"].append(float(losses[c]))
            values[c]["pair_agreement"].append(float(((a > b) == (ta > tb)).float().mean()))
        member.append(float(mloss))
        intersections += int(((predicted > .5) & target.bool()).sum())
        unions += int(((predicted > .5) | target.bool()).sum())
    metrics = {c: {k: float(np.mean(v)) for k, v in row.items()} for c, row in values.items()}
    return dict(native=metrics, known_ray_membership_bce=float(np.mean(member)),
                sampled_known_domain_iou=intersections / max(unions, 1), episodes=count,
                frames=[v["frame"] for v in source.views],
                selection_score=float(np.mean([metrics[c]["kl"] for c in metrics]) +
                                      config["membership_weight"] * np.mean(member)),
                complete_object_accuracy=False, benchmark_accuracy=False)


def cpu_state(field):
    return {k: v.detach().cpu().clone() for k, v in field.state_dict().items()}


def run(config, arm, study_root, *, device="cpu", run_name=None):
    if arm not in {"reconstruction", "functional"} or arm not in config["arms"]:
        raise ValueError("unknown controlled objective")
    source_root = Path(config["observations"]).resolve(strict=True)
    manifest_path = source_root / "observations.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("benchmark_queries_used") is not False or manifest.get("semantic_labels_used") is not False:
        raise ValueError("pilot requires source-only unlabeled observations")
    if manifest.get("feature_development_excluded") is not True:
        raise ValueError("development features must be excluded from field initialization")
    if set(manifest["splits"]["train_frames"]) & set(manifest["splits"]["development_frames"]):
        raise ValueError("source frame split overlap")
    store = BoundedExperiment(study_root, run_name or f'{manifest["scene"]}_{arm}', config["storage"])
    # Include best, latest, a replacement latest, export and small reports.
    peak = 3 * config["checkpoint_budget_bytes"] + config["export_budget_bytes"]
    store.preflight(peak)
    source_files = [Path(__file__), Path(__file__).with_name("unified_compact.py"),
                    Path(__file__).with_name("bounded_experiment.py"),
                    Path(__file__).with_name("functional_training.py"),
                    Path(__file__).with_name("observations.py"),
                    Path(__file__).with_name("storage.py"), Path(__file__).with_name("shared_storage.py")]
    design = dict(config=config, arm=arm, source_manifest_sha256=sha256(manifest_path),
                  source_files={str(p.resolve()): sha256(p) for p in source_files},
                  teacher_source_manifest=manifest, device=device, torch_version=str(torch.__version__),
                  shared_across_scenes=False, complete_multimode_teacher=False,
                  scientific_scope="two-teacher native function and SAM partial membership pilot",
                  model_selection="fixed source-development KL plus partial membership BCE, no benchmark labels")
    store.json("design.json", design)
    design_sha = sha256(store.root / "design.json")
    start = time.monotonic()
    initial_path = source_root / "initial.pt"
    if sha256(initial_path) != manifest["initial"]["sha256"]:
        raise ValueError("initial feature authority changed")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True, mmap=True)
    field = initialize(initial, config, device)
    field.world_units_per_metre = float(manifest["world_units_per_metre"])
    train = FunctionalSource(source_root, initial["observed"], "train")
    dev = FunctionalSource(source_root, initial["observed"], "development")
    optimizer = torch.optim.Adam(field.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    available = torch.where(initial["observed"])[0]
    history = []
    teacher_gradient_audit = None
    baseline = validate(field, dev, config, device)
    best_score, best_step = baseline["selection_score"], 0
    store.tensor("best.pt", dict(model=cpu_state(field), step=0, validation=baseline, design_sha256=design_sha),
                 max_bytes=config["checkpoint_budget_bytes"])
    print(json.dumps(dict(arm=arm, step=0, validation=baseline)), flush=True)
    for step in range(1, config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        rows = available[torch.randint(len(available), (config["reconstruction_rows"],), generator=generator)]
        reconstruction = []
        for c in ("semantic", "local"):
            decoded = F.normalize(getattr(field, c).decode(rows.to(device)), dim=-1)
            target = F.normalize(initial[c][rows].to(device).float(), dim=-1)
            reconstruction.append((1 - (decoded * target).sum(1)).mean())
        index = int(torch.randint(len(train.examples), (1,), generator=generator))
        function, membership, _, _, _, _ = functional_episode(
            field, train, index, generator, config["pairs"], device, inferred=step % 2 == 0)
        reconstruction_loss = torch.stack(reconstruction).mean()
        function_loss = torch.stack(list(function.values())).mean()
        loss = reconstruction_loss + config["membership_weight"] * membership
        if arm == "functional":
            loss = loss + config["functional_weight"] * function_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite student objective")
        if step == 1 and config.get("audit_teacher_gradients", False):
            # Audit real training objectives on the SAME latent, not merely
            # the existence of teacher checkpoints or decoder modules.
            objectives = dict(semantic=reconstruction[0] / 2,
                              local=reconstruction[1] / 2,
                              membership=config["membership_weight"] * membership)
            if arm == "functional":
                for channel in ("semantic", "local"):
                    objectives[channel] = objectives[channel] + config["functional_weight"] * function[channel] / 2
            teacher_gradient_audit = {}
            for channel, objective in objectives.items():
                gradient, = torch.autograd.grad(objective, field.latent, retain_graph=True)
                norm = float(gradient.norm())
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("teacher has no finite nonzero latent gradient: " + channel)
                teacher_gradient_audit[channel] = dict(latent_gradient_norm=norm,
                    objective=float(objective.detach()), nonzero_rows=int((gradient.abs().sum(1) > 0).sum()))
                del gradient
            store.json("teacher_gradient_audit.json", dict(step=step, teachers=teacher_gradient_audit,
                same_parameter="latent", all_teachers_active=True, proves_capability_transfer=False))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.)
        optimizer.step()
        if step % config["validation_every"] == 0 or step == config["steps"]:
            field.eval()
            validation = validate(field, dev, config, device)
            row = dict(step=step, reconstruction=float(reconstruction_loss.detach()),
                       function=float(function_loss.detach()), membership=float(membership.detach()),
                       validation=validation, seconds=time.monotonic() - start)
            history.append(row)
            state = cpu_state(field)
            if validation["selection_score"] < best_score:
                best_score, best_step = validation["selection_score"], step
                store.tensor("best.pt", dict(model=state, step=step, validation=validation, design_sha256=design_sha),
                             max_bytes=config["checkpoint_budget_bytes"], replace=True)
            store.tensor("latest.pt", dict(model=state, optimizer=optimizer.state_dict(), step=step,
                         sampling_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                         design_sha256=design_sha, best_step=best_step, best_score=best_score),
                         max_bytes=config["checkpoint_budget_bytes"], replace=True)
            store.json("progress.json", dict(history=history, best_step=best_step), replace=True)
            print(json.dumps(dict(arm=arm, **row)), flush=True)
            field.train()
    best = torch.load(store.root / "best.pt", map_location="cpu", weights_only=True)
    field.load_state_dict(best["model"], strict=True)
    field.cpu().eval()
    if any(not torch.equal(getattr(field, k), v) for k, v in initial["geometry"].items()):
        raise ValueError("frozen geometry changed")
    exported = store.export(field, dict(design_sha256=design_sha, observations_sha256=sha256(manifest_path),
                            arm=arm, selected_step=best_step, benchmark_queries_used=False,
                            extra_3d_labels=False), max_new_bytes=config["export_budget_bytes"])
    store.retire_checkpoints_after_verified_export(exported["state_identity"])
    report = dict(arm=arm, scene=manifest["scene"], latent_dim=config["latent_dim"],
                  elements=len(field.latent), single_latent=True, initial_validation=baseline,
                  selected_validation=best["validation"], selected_step=best_step, history=history,
                  geometry_unchanged=True, complete_field_payload_bytes=deployment_bytes(exported),
                  geometry_bytes=exported["geometry_bytes"], feature_and_network_bytes=exported["semantic_increment_bytes"],
                  new_blob_bytes=exported["new_blob_bytes"], teacher_initial_file_bytes=initial_path.stat().st_size,
                  state_identity=exported["state_identity"], inference_replay_verified=True,
                  query_encoder_cost_excluded=True, benchmark_performance_established=False,
                  complete_multimode_teacher=False, cross_scene_member_generalization=False,
                  teacher_gradient_audit=teacher_gradient_audit,
                  seconds=time.monotonic() - start, study_actual_bytes_before_report=tree_bytes(store.study))
    store.json("training.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=["reconstruction", "functional"], required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", help="Explicit new attempt name; existing runs are never overwritten")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.15, torch.device(args.device))
    print(json.dumps(run(json.loads(Path(args.config).read_text()), args.arm,
                         args.study_root, device=args.device, run_name=args.run_name), indent=2))


if __name__ == "__main__":
    main()
