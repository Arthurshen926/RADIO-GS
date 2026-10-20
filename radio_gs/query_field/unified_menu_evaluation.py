"""Seal same-domain complete-menu predictions before ScanNet label scoring.

The declared 19-class menu is held fixed before the existing 19/15/10 reporting
subsets are selected, matching the repository's category evaluation contract.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from radio_gs.querying.observation_readout import positive_support_rank
from .bounded_experiment import BoundedExperiment, tree_bytes
from .contracts import QueryPacket
from .shared_storage import load_field_v2
from .storage import sha256
from .unified_menu_training import MeasureTeacher


def deny_files(*args, **kwargs):
    raise AssertionError("student inference attempted filesystem access")


@torch.no_grad()
def predict(field, vectors, *, modes):
    # Warm kernels using only the already-loaded field and current input.
    field.semantic.score(vectors[:1])
    with ExitStack() as stack:
        for target in ("builtins.open", "io.open", "torch.load"):
            stack.enter_context(patch(target, deny_files))
        native = field.category_probabilities(vectors) if modes else (100 * field.semantic.score(vectors)).softmax(-1)
        members = torch.stack([field.query(QueryPacket("text", task="union", semantic=q)).membership_scores
                               for q in vectors], dim=1)
    return native.cpu(), members.cpu()


def validate_pair(config, study, runs):
    if set(runs) != set(config["arms"]) or not {"mean", "reliable"} <= set(runs):
        raise ValueError("both predefined integration arms must finish before evaluation")
    designs = {a: json.loads((study / r / "design.json").read_text()) for a, r in runs.items()}
    left = designs["mean"]
    for right in designs.values():
        for key in ("config", "source_files", "parameter_count", "observed", "source_manifest_sha256", "teacher_provenance"):
            if left[key] != right[key]:
                raise ValueError("integration control is confounded: " + key)
    if left["config"] != config:
        raise ValueError("evaluation configuration changed")
    return {a: json.loads((study / r / "training.json").read_text()) for a, r in runs.items()}


@torch.no_grad()
def evaluate(config, study_root, runs, queries_path, evaluation_path, *, query_comparison="comparison_attempt2",
             name="menu_comparison", device="cpu"):
    study = Path(study_root)
    reports = validate_pair(config, study, runs)
    query_root = study / query_comparison
    encoding = json.loads((query_root / "query_encoding.json").read_text())
    parent_design = json.loads((query_root / "design.json").read_text())
    if parent_design["queries_sha256"] != sha256(queries_path):
        raise ValueError("native query menu differs from encoded authority")
    if encoding["query_vectors_sha256"] != sha256(query_root / "query_vectors.pt"):
        raise ValueError("native query vectors changed")
    packet = torch.load(query_root / "query_vectors.pt", map_location="cpu", weights_only=True)
    if packet["queries"] != json.loads(Path(queries_path).read_text())["queries"]:
        raise ValueError("encoded query metadata differs")
    vectors, queries = packet["vectors"].to(device), packet["queries"]
    meta = BoundedExperiment(study, name, config["storage"])
    meta.json("design.json", dict(config=config, runs=runs, primary="reliable versus matched-capacity mean; native complete-menu",
              diagnostic="typed positive member corroboration, fixed inherited rule",
              query_vectors_sha256=encoding["query_vectors_sha256"], query_vectors_parent=str(query_root),
              query_encoder_parameter_bytes=encoding["parameter_bytes"], current_query_vectors_reused=True,
              training_reports={a: sha256(study / r / "training.json") for a, r in runs.items()},
              code_sha256=sha256(__file__), evaluation_sha256=sha256(evaluation_path),
              class_subsets="competition on full declared menu, then restrict reporting classes",
              independent_final_test=False, availability="same original 46199-row student support for all arms"))
    base, base_manifest = load_field_v2(study / config["base_run"] / "field", device, store_root=study / "blobs")
    if base_manifest["state_identity"] != config["base_state_identity"]:
        raise ValueError("base export changed")
    stores = {}

    def seal(arm, native, member, binding, source_denied):
        store = BoundedExperiment(study, f"{name}_{arm}", config["storage"])
        payload = dict(xyz=base.xyz.cpu(), observed=base.observed.cpu(), queries=queries,
                       native=native, member=member, typed=positive_support_rank(native, member))
        store.tensor("predictions.pt", payload, max_bytes=32 * 1024**2)
        store.json("prediction.json", dict(predictions_sha256=sha256(store.root / "predictions.pt"),
                   binding=binding, labels_opened=False, student_source_access_denied=source_denied,
                   shared_domain_with_base=True, current_query_vectors_sha256=encoding["query_vectors_sha256"]))
        stores[arm] = store

    dictionary_atoms = None
    for arm, run_name in runs.items():
        field, manifest = load_field_v2(study / run_name / "field", device, store_root=study / "blobs")
        if manifest["state_identity"] != reports[arm]["state_identity"]:
            raise ValueError("selected student export changed")
        if not torch.equal(field.observed, base.observed) or not torch.equal(field.xyz, base.xyz):
            raise ValueError("student changed evaluation domain")
        native, members = predict(field, vectors, modes=True)
        seal(arm, native, members, manifest["state_identity"], True)
        if hasattr(field, "semantic_atoms"):
            if dictionary_atoms is not None and not torch.equal(dictionary_atoms, field.semantic_atoms):
                raise ValueError("matched control decoder bases differ")
            dictionary_atoms = field.semantic_atoms.detach().clone()
        del field
    native, members = predict(base, vectors, modes=False)
    seal("base_v1", native, members, base_manifest["state_identity"], True)
    # Offline capability reference, not part of any student's query path. The
    # same base member is used only for its explicitly labelled typed diagnostic.
    teacher = MeasureTeacher(config["teacher"], config["teacher_manifest_sha256"], base, device)
    reference = torch.cat([teacher.probabilities(torch.arange(i, min(i + 1024, len(base.latent))), vectors,
                          config["temperature"]).cpu() for i in range(0, len(base.latent), 1024)])
    seal("teacher_same_domain", reference, members, config["teacher_manifest_sha256"], False)
    if dictionary_atoms is not None:
        from .unified_dictionary import integrate, project_teacher
        source_mass = project_teacher(teacher, dictionary_atoms)
        oracle = integrate(dictionary_atoms, source_mass.to(device), vectors, "reliable", config["temperature"]).cpu()
        seal("basis_oracle", oracle, members, "frozen student basis with uncompressed teacher mass; not deployable student", False)
    meta.json("prediction_barrier.json", {a: sha256(s.root / "predictions.pt") for a, s in stores.items()})
    from .public_scannet import score
    results = {}
    with tempfile.TemporaryDirectory(prefix="radiogs-menu-score-") as temporary:
        for arm, store in stores.items():
            report = score(store.root, evaluation_path, Path(temporary) / (arm + ".json"))
            store.json("score.json", report)
            results[arm] = report["metrics"]
    summary = dict(metrics=results, field_bytes={a: r["complete_field_payload_bytes"] for a, r in reports.items()},
                   text_encoder_parameter_bytes=encoding["parameter_bytes"],
                   total_text_deployment_bytes={a: r["complete_field_payload_bytes"] + encoding["parameter_bytes"]
                                                for a, r in reports.items()},
                   full_teacher_file_bytes=teacher.manifest["files"]["field.pt"]["bytes"],
                   teacher_reference_is_not_full_v10=True, teacher_typed_uses_base_member=True,
                   matched_capacity_and_source=True, independent_final_test=False,
                   five_benchmarks_complete=False, all_predictions_sealed_before_score=True,
                   storage_bytes_before_summary=tree_bytes(study))
    meta.json("summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--mean-run", required=True)
    parser.add_argument("--reliable-run", required=True)
    parser.add_argument("--no-mass-run")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--name", default="menu_comparison")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.15, torch.device(args.device))
    runs = dict(mean=args.mean_run, reliable=args.reliable_run)
    if args.no_mass_run:
        runs["reliable_no_mass"] = args.no_mass_run
    result = evaluate(json.loads(Path(args.config).read_text()), args.study_root,
                      runs, args.queries, args.evaluation,
                      name=args.name, device=args.device)
    print(json.dumps({a: {k: {s: v["miou"] for s, v in splits.items()} for k, splits in values.items()}
                      for a, values in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
