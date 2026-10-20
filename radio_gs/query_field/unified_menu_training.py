"""Matched-capacity transfer of complete-menu teacher functions into one latent.

Both arms see the same retained teacher, Gaussian domain, initial student,
query stream and auxiliary supervision. Only nonlinear integration order differs.
Training menus contain source region descriptors, never benchmark class names.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .bounded_experiment import BoundedExperiment
from .functional_training import FunctionalSource
from .shared_storage import load_field_v2
from .storage import deployment_bytes, sha256
from .unified_compact import GEOMETRY
from .unified_modes import UnifiedModesField
from .unified_training import cpu_state, functional_episode, validate as validate_membership


class MeasureTeacher:
    """Construction-only sparse teacher. Never attached to the student module."""
    def __init__(self, root, expected_manifest_sha256, base, device):
        root = Path(root)
        if sha256(root / "manifest.json") != expected_manifest_sha256:
            raise ValueError("teacher authority changed")
        self.manifest = json.loads((root / "manifest.json").read_text())
        provenance = self.manifest["provenance"]
        if provenance.get("query_free") is not True or provenance.get("labels_used") is not False:
            raise ValueError("teacher must be query-free and label-free")
        if set(provenance["legal_frames"]) & set(provenance["excluded_frames"]):
            raise ValueError("teacher frame leakage")
        if sha256(root / "field.pt") != self.manifest["files"]["field.pt"]["sha256"]:
            raise ValueError("teacher payload changed")
        state = torch.load(root / "field.pt", map_location="cpu", weights_only=True, mmap=True)
        if any(not torch.equal(state[k], getattr(base, k).cpu()) for k in GEOMETRY):
            raise ValueError("teacher and student Gaussian identity differ")
        prefix = "semantic_measure."
        self.atoms = F.normalize(state[prefix + "descriptors"].to(device).float(), dim=-1)
        self.transfer = sparse.csr_matrix((state[prefix + "mass"].numpy(), state[prefix + "index"].numpy(),
                                          state[prefix + "pointer"].numpy()),
                                         shape=(len(base.latent), len(self.atoms)))
        self.supported = state[prefix + "supported"].clone()
        self.fallback = F.normalize(state["semantic.codes"].float(), dim=-1)
        self.device = device
        self.headers = []
        held = []
        for receipt in provenance["source_headers"]:
            path = Path(receipt["path"])
            frame = int(path.stem.split("_")[-1])
            if frame not in provenance["excluded_frames"]:
                continue
            if sha256(path) != receipt["sha256"]:
                raise ValueError("development query source changed")
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data.get("query_independent") is not True or data["frame"] != frame:
                raise ValueError("invalid development query source")
            held.append(data["descriptors"].float())
            self.headers.append(receipt)
        if len(self.headers) != len(provenance["excluded_frames"]):
            raise ValueError("incomplete held-view descriptor bank")
        # Exclude exact descriptor duplicates, beyond the frame-level split.
        def fingerprint(row):
            return hashlib.sha256(row.half().contiguous().numpy().tobytes()).digest()
        train_keys = {fingerprint(x) for x in state[prefix + "descriptors"]}
        held = torch.cat(held)
        keep = [i for i, row in enumerate(held) if fingerprint(row) not in train_keys]
        self.dev_queries = F.normalize(held[keep].to(device), dim=-1)
        self.duplicate_dev_queries_removed = len(held) - len(keep)
        if len(self.dev_queries) < 2:
            raise ValueError("no held-out native query directions")

    @torch.no_grad()
    def probabilities(self, rows, queries, temperature=100.):
        rows = torch.as_tensor(rows, dtype=torch.long, device="cpu")
        a = self.transfer[rows.numpy()].tocsr()
        # Compact this batch's columns to bound transient teacher memory/work.
        used, columns = np.unique(a.indices, return_inverse=True)
        operator = torch.sparse_csr_tensor(torch.from_numpy(a.indptr.astype(np.int64)),
                     torch.from_numpy(columns.astype(np.int64)), torch.from_numpy(a.data),
                     size=(len(rows), len(used)), device=self.device)
        q = F.normalize(queries.to(self.atoms), dim=-1)
        response = (temperature * (self.atoms[torch.as_tensor(used, device=self.device)] @ q.T)).softmax(-1)
        entropy = -(response * response.clamp_min(torch.finfo(response.dtype).tiny).log()).sum(-1)
        reliability = (1 - entropy / math.log(len(queries))).clamp(0, 1)
        ordinary = operator @ response
        mass = operator @ reliability[:, None]
        numerator = operator @ (reliability[:, None] * response)
        result = torch.where(mass > 0, numerator / mass.clamp_min(torch.finfo(mass.dtype).tiny), ordinary)
        fallback = (temperature * (self.fallback[rows].to(self.device) @ q.T)).softmax(-1)
        return torch.where(self.supported[rows].to(self.device)[:, None], result, fallback)


def initialize(base, config, integration):
    torch.manual_seed(config["seed"])
    spec = {k: v for k, v in base.export_spec().items() if k not in {"architecture", "latent_dim"}}
    field = UnifiedModesField({k: getattr(base, k) for k in GEOMETRY}, base.observed, base.latent,
                **spec, mode_count=config["mode_count"], integration=integration,
                temperature=config["temperature"]).to(base.latent.device)
    current = field.state_dict()
    current.update(base.state_dict())
    field.load_state_dict(current, strict=True)
    with torch.no_grad():
        for head in field.mode_heads:
            head.weight.copy_(base.heads["semantic"].weight)
            head.bias.copy_(base.heads["semantic"].bias)
            # Break permutation symmetry without adding per-row mode storage.
            head.weight.add_(config["mode_initial_noise"] * torch.randn_like(head.weight))
        field.mode_mass.weight.zero_()
        field.mode_mass.bias.zero_()
    return field


def menu(bank, generator, sizes):
    size = min(sizes[int(torch.randint(len(sizes), (1,), generator=generator))], len(bank))
    return bank[torch.randperm(len(bank), generator=generator)[:size].to(bank.device)]


def divergence(prediction, target):
    return (target * (target.clamp_min(1e-8).log() - prediction.clamp_min(1e-8).log())).sum(-1).mean()


@torch.no_grad()
def validate(field, base, teacher, dev, config, device):
    generator = torch.Generator().manual_seed(config["seed"] + 77)
    available = torch.where(base.observed.cpu())[0]
    kl, agreement, l1, oracle_kl = [], [], [], []
    for _ in range(config["validation_menus"]):
        rows = available[torch.randperm(len(available), generator=generator)[:config["validation_rows"]]]
        queries = menu(teacher.dev_queries, generator, config["menu_sizes"])
        target = teacher.probabilities(rows, queries, config["temperature"])
        predicted = field.category_probabilities(queries, rows.to(device))
        kl.append(float(divergence(predicted, target)))
        agreement.append(float((target.argmax(-1) == predicted.argmax(-1)).float().mean()))
        l1.append(float((target - predicted).abs().sum(-1).mean()))
        if hasattr(teacher, "compressed_mass"):
            from .unified_dictionary import integrate
            oracle = integrate(teacher.compressed_atoms, teacher.compressed_mass[rows].to(device), queries,
                               "reliable", config["temperature"])
            oracle_kl.append(float(divergence(oracle, target)))
    rows = available[torch.randperm(len(available), generator=generator)[:config["validation_rows"]]].to(device)
    fidelity = {c: float(F.cosine_similarity(getattr(field, c).decode(rows),
                                           getattr(base, c).decode(rows), dim=-1).mean())
                for c in ("semantic", "local")}
    membership = validate_membership(field, dev, config, device)
    score = (float(np.mean(kl)) + config["membership_weight"] * membership["known_ray_membership_bce"]
             + config["preservation_weight"] * (1 - float(np.mean(list(fidelity.values())))))
    return dict(menu_kl=float(np.mean(kl)), menu_top1_agreement=float(np.mean(agreement)),
                menu_l1=float(np.mean(l1)), native_fidelity_to_base=fidelity, membership=membership,
                native_basis_oracle_menu_kl=float(np.mean(oracle_kl)) if oracle_kl else None,
                selection_score=score, held_view_query_directions=True, held_out_scene=False)


def run(config, arm, study_root, device="cpu", run_name=None):
    if arm not in {"mean", "reliable", "reliable_no_mass"} or arm not in config["arms"]:
        raise ValueError("unknown matched-capacity arm")
    integration = "mean" if arm == "mean" else "reliable"
    store = BoundedExperiment(study_root, run_name or f"scene0062_00_menu_{arm}", config["storage"])
    store.preflight(3 * config["checkpoint_budget_bytes"] + config["export_budget_bytes"])
    base, base_manifest = load_field_v2(Path(study_root) / config["base_run"] / "field", device,
                                       store_root=Path(study_root) / "blobs")
    if base_manifest["state_identity"] != config["base_state_identity"]:
        raise ValueError("warm-start student changed")
    base.requires_grad_(False)
    teacher = MeasureTeacher(config["teacher"], config["teacher_manifest_sha256"], base, device)
    source = Path(config["observations"])
    manifest = json.loads((source / "observations.json").read_text())
    base_design = json.loads((Path(study_root) / config["base_run"] / "design.json").read_text())
    if base_design["source_manifest_sha256"] != sha256(source / "observations.json"):
        raise ValueError("auxiliary observation authority changed")
    if set(manifest["splits"]["development_frames"]) != set(teacher.manifest["provenance"]["excluded_frames"]):
        raise ValueError("teacher and auxiliary development splits differ")
    decoder = config.get("decoder", "free_modes")
    if arm == "reliable_no_mass" and decoder != "native_dictionary":
        raise ValueError("mass ablation requires the fixed native basis")
    mass_weight = config.get("mass_weight", 0.) if arm != "reliable_no_mass" else 0.
    if decoder == "native_dictionary":
        from .unified_dictionary import compress_teacher, project_teacher, initialize as initialize_dictionary
        if "basis_artifact" in config:
            receipt = config["basis_artifact"]
            path = Path(study_root) / receipt["run"] / receipt["file"]
            if sha256(path) != receipt["sha256"]:
                raise ValueError("frozen native basis changed")
            teacher.compressed_atoms = torch.load(path, map_location=device, weights_only=True)["atoms"]
            if len(teacher.compressed_atoms) != config["atom_count"]:
                raise ValueError("frozen native basis has wrong capacity")
            teacher.compressed_mass = project_teacher(teacher, teacher.compressed_atoms)
        else:
            teacher.compressed_atoms, teacher.compressed_mass = compress_teacher(teacher, config["atom_count"], config["seed"])
        field = initialize_dictionary(base, config, integration, teacher.compressed_atoms, teacher.compressed_mass)
    elif decoder == "free_modes":
        field = initialize(base, config, integration)
    else:
        raise ValueError("unknown compact decoder")
    train = FunctionalSource(source, base.observed.cpu(), "train")
    dev = FunctionalSource(source, base.observed.cpu(), "development")
    source_paths = sorted(Path(__file__).parent.glob("unified_*.py")) + [Path(__file__).with_name(k) for k in
                    ("functional_training.py", "observations.py", "storage.py", "shared_storage.py", "bounded_experiment.py")]
    design = dict(config=config, arm=arm, source_files={str(p): sha256(p) for p in source_paths},
                  teacher_provenance=teacher.manifest["provenance"], teacher_bytes=teacher.manifest["files"]["field.pt"]["bytes"],
                  query_development_headers=teacher.headers, train_query_count=len(teacher.atoms),
                  development_query_count=len(teacher.dev_queries),
                  exact_duplicate_dev_queries_removed=teacher.duplicate_dev_queries_removed,
                  observed=int(base.observed.sum()), teacher_supported=int(teacher.supported.sum()),
                  availability="same original student domain for both arms and teacher comparison; no coverage expansion",
                  parameter_count=sum(p.numel() for p in field.parameters()),
                  source_manifest_sha256=sha256(source / "observations.json"),
                  objective="T0 reliable complete-menu KL + base native preservation + SAM partial membership BCE"
                            + (" + query-independent native mixture mass KL" if mass_weight > 0 else ""),
                  decoder=decoder, mass_weight=mass_weight, persistent_source_mass=False,
                  selection="fixed held-view menu KL + partial BCE + native preservation, no target labels",
                  source_views_semantic=len(teacher.manifest["provenance"]["legal_frames"]),
                  source_views_auxiliary=len(manifest["splits"]["train_frames"]),
                  benchmark_queries_used=False, independent_final_test=False, shared_across_scenes=False)
    store.json("design.json", design)
    design_sha = sha256(store.root / "design.json")
    optimizer = torch.optim.Adam(field.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    available = torch.where(base.observed.cpu())[0]
    history = []
    start = time.monotonic()
    baseline = validate(field, base, teacher, dev, config, device)
    best_score, best_step = baseline["selection_score"], 0
    store.tensor("best.pt", dict(model=cpu_state(field), step=0, validation=baseline, design_sha256=design_sha),
                 max_bytes=config["checkpoint_budget_bytes"])
    print(json.dumps(dict(arm=arm, step=0, validation=baseline)), flush=True)
    for step in range(1, config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        rows = available[torch.randint(len(available), (config["training_rows"],), generator=generator)]
        queries = menu(teacher.atoms, generator, config["menu_sizes"])
        target = teacher.probabilities(rows, queries, config["temperature"])
        prediction = field.category_probabilities(queries, rows.to(device))
        function = divergence(prediction, target)
        mass_loss = (divergence(field.mode_probabilities(rows.to(device)), teacher.compressed_mass[rows].to(device))
                     if decoder == "native_dictionary" else function.new_zeros(()))
        preservation = torch.stack([1 - F.cosine_similarity(getattr(field, c).decode(rows.to(device)),
                       getattr(base, c).decode(rows.to(device)), dim=-1).mean() for c in ("semantic", "local")]).mean()
        index = int(torch.randint(len(train.examples), (1,), generator=generator))
        _, membership, _, _, _, _ = functional_episode(field, train, index, generator, config["pairs"],
                                                       device, inferred=step % 2 == 0)
        loss = (function + config["preservation_weight"] * preservation + config["membership_weight"] * membership
                + mass_weight * mass_loss)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite menu transfer loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.)
        optimizer.step()
        if step % config["validation_every"] == 0 or step == config["steps"]:
            validation = validate(field, base, teacher, dev, config, device)
            row = dict(step=step, menu_kl=float(function.detach()), mass_kl=float(mass_loss.detach()),
                       preservation=float(preservation.detach()),
                       membership=float(membership.detach()), validation=validation, seconds=time.monotonic() - start)
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
    best = torch.load(store.root / "best.pt", map_location="cpu", weights_only=True)
    field.load_state_dict(best["model"], strict=True)
    if any(not torch.equal(getattr(field, k).cpu(), getattr(base, k).cpu()) for k in GEOMETRY):
        raise ValueError("frozen geometry changed")
    # Prespecified intervention: same learned state, swap only integration order.
    field.integration = "reliable" if integration == "mean" else "mean"
    counterfactual = validate(field, base, teacher, dev, config, device)
    field.integration = integration
    field.cpu().eval()
    exported = store.export(field, dict(design_sha256=design_sha, arm=arm, selected_step=best_step,
                          teacher_manifest_sha256=config["teacher_manifest_sha256"], benchmark_queries_used=False),
                          max_new_bytes=config["export_budget_bytes"])
    store.retire_checkpoints_after_verified_export(exported["state_identity"])
    report = dict(arm=arm, initial_validation=baseline, selected_validation=best["validation"],
                  counterfactual_integration_validation=counterfactual, selected_step=best_step,
                  state_identity=exported["state_identity"], complete_field_payload_bytes=deployment_bytes(exported),
                  feature_and_network_bytes=exported["semantic_increment_bytes"], new_blob_bytes=exported["new_blob_bytes"],
                  history=history, seconds=time.monotonic() - start, single_gaussian_latent=True,
                  teacher_state_in_student=False, geometry_unchanged=True, query_encoder_cost_excluded=True,
                  complete_v10_transfer=False, five_benchmarks_complete=False)
    store.json("training.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True, choices=["mean", "reliable", "reliable_no_mass"])
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.15, torch.device(args.device))
    print(json.dumps(run(json.loads(Path(args.config).read_text()), args.arm, args.study_root,
                         args.device, args.run_name), indent=2))


if __name__ == "__main__":
    main()
