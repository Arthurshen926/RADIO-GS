"""Audit whether retrieved training supports agree with their source SAM targets.

This evaluates the *final* student on training-source queries, not the unrecorded
supports at every optimizer step. Invisible and unlabelled support is unknown,
never negative. No benchmark inputs or annotations are loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .bounded_experiment import BoundedExperiment
from .functional_training import FunctionalSource
from .shared_storage import load_field_v2
from .storage import sha256
from .unified_five_benchmark import read


def support_evidence(matrix, positive, negative, ids, weights):
    """Visible-footprint fractions on reliable known domains, not object GT."""
    columns = matrix[:, ids]
    total = np.asarray(columns.sum(0)).ravel()
    pos = np.asarray(columns[positive].sum(0)).ravel()
    neg = np.asarray(columns[negative].sum(0)).ravel()
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    visible = total > 0
    positive_mass = float(weights @ np.divide(pos, total, out=np.zeros_like(pos), where=visible))
    negative_mass = float(weights @ np.divide(neg, total, out=np.zeros_like(neg), where=visible))
    return dict(positive_mass=positive_mass, negative_mass=negative_mass,
                unknown_mass=max(0., 1. - positive_mass - negative_mass),
                invisible_mass=float(weights @ ~visible),
                negative_majority=negative_mass > .5,
                positive_majority=positive_mass > .5)


@torch.no_grad()
def audit(field, source, count=24):
    indices = np.linspace(0, len(source.examples) - 1, min(count, len(source.examples)), dtype=int)
    examples = [source.examples[i] for i in indices]
    queries = torch.stack([source.views[vi]["descriptors"][source.views[vi]["examples"][ri]["region"]]
                           for vi, ri in examples])
    scores = field.semantic.score(queries).masked_fill(~field.observed[:, None], -torch.inf)
    values, candidates = scores.topk(min(field.candidate_count, int(field.observed.sum())), dim=0)
    weights = (10 * values).softmax(0).cpu().numpy()
    candidates = candidates.cpu().numpy()
    rows = []
    for i, (vi, ri) in enumerate(examples):
        view = source.views[vi]
        example = view["examples"][ri]
        evidence = support_evidence(view["operator"], example["positive"].numpy(),
                                    example["negative"].numpy(), candidates[:, i], weights[:, i])
        rows.append(dict(frame=view["frame"], region=example["region"],
                         candidates=candidates[:, i].tolist(), **evidence))
    return rows


def run(config, device="cuda"):
    root = Path(config["study_root"])
    plan = read(root / "cohort/plan.json")
    store = BoundedExperiment(root, "condition_audit", config["storage"])
    store.json("design.json", dict(source_sha256=sha256(__file__),
        plan_sha256=sha256(root / "cohort/plan.json"), all_fields=True, episodes_per_field=24,
        query_source="training-source SAM region semantic descriptors", weights="frozen final baseline",
        exact_optimizer_history=False, no_parameter_update=True, benchmark_data_used=False,
        unknown_is_negative=False))
    reports = []
    for entry in plan["fields"]:
        field, manifest = load_field_v2(entry["field"], device, store_root=root / "blobs")
        source = FunctionalSource(entry["observations"], field.observed, "train")
        rows = audit(field, source)
        result = dict(run=entry["run"], family=entry["family"], state_identity=manifest["state_identity"], rows=rows)
        store.json(entry["run"] + ".json", result)
        reports.append(result)
        print(entry["run"], "negative majority", sum(r["negative_majority"] for r in rows), "/", len(rows), flush=True)
        del field, source
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    summary = {}
    for family in ["lerf", "nvos", "scannet", "uqis"]:
        rows = [r for d in reports if d["family"] == family for r in d["rows"]]
        summary[family] = dict(episodes=len(rows), fields=sum(d["family"] == family for d in reports),
            negative_majority=sum(r["negative_majority"] for r in rows),
            positive_majority=sum(r["positive_majority"] for r in rows),
            means={k: float(np.mean([r[k] for r in rows])) for k in
                   ["positive_mass", "negative_mass", "unknown_mass", "invisible_mass"]})
    store.json("summary.json", dict(families=summary, fields=len(reports),
        source_only=True, exact_optimizer_history=False,
        interpretation="Known-negative source footprint proves some conditioning-target mismatches; invisible mass is inconclusive. This does not measure full instance accuracy or establish the causal effect of training repair."))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/query_field/unified_five_benchmark.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(read(args.config), args.device)
