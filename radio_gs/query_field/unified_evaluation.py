"""Seal compact-student ScanNet predictions before opening evaluation labels.

This is one scene's development comparison, not a five-task result or a v10
recovery claim. All arms share freshly encoded native text vectors.
"""
import argparse
import gc
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .adapters import NativeQueryEncoder
from .bounded_experiment import BoundedExperiment, tree_bytes
from .contracts import QueryPacket
from .shared_storage import load_field_v2
from .storage import sha256


@torch.no_grad()
def evaluate(config, study_root, runs, queries_path, evaluation_path, *, device="cpu", comparison_name="comparison"):
    study = Path(study_root)
    source = Path(config["observations"]).resolve(strict=True)
    source_manifest = json.loads((source / "observations.json").read_text())
    queries = json.loads(Path(queries_path).read_text())["queries"]
    # Training and model selection must have completed before public queries.
    reports = {arm: json.loads((study / name / "training.json").read_text()) for arm, name in runs.items()}
    if set(runs) != {"reconstruction", "functional"}:
        raise ValueError("both predefined controlled arms must be sealed")
    designs = {arm: json.loads((study / name / "design.json").read_text()) for arm, name in runs.items()}
    a, b = designs["reconstruction"], designs["functional"]
    if a["config"] != b["config"] or a["source_files"] != b["source_files"]:
        raise ValueError("comparison differs in code or configuration, beyond its objective")
    if a["config"] != config or a["source_manifest_sha256"] != sha256(source / "observations.json"):
        raise ValueError("evaluation does not match the frozen source configuration")
    meta = BoundedExperiment(study, comparison_name, config["storage"])
    meta.json("design.json", dict(runs=runs, queries_sha256=sha256(queries_path),
                                 evaluation_sha256=sha256(evaluation_path),
                                 training_reports={arm: sha256(study / name / "training.json") for arm, name in runs.items()},
                                 code_sha256=sha256(__file__), query_encoder=source_manifest["native_semantic_space"],
                                 source_development_not_independent_test=True,
                                 primary="native category competition for compression; membership diagnostic separately"))
    encoder = NativeQueryEncoder(source_manifest["native_semantic_space"], device)
    vectors = torch.stack([encoder.text(q["text"], task="union").semantic.float().cpu() for q in queries])
    encoder_bytes = sum(p.numel() * p.element_size() for p in encoder._text[1].parameters())
    meta.tensor("query_vectors.pt", dict(queries=queries, vectors=vectors), max_bytes=1_000_000)
    meta.json("query_encoding.json", dict(parameter_bytes=encoder_bytes,
                query_vectors_sha256=sha256(meta.root / "query_vectors.pt"),
                current_input_only=True, encoder_storage_dedup_is_not_model_compression=True))
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stores = {}
    for arm, name in runs.items():
        field, manifest = load_field_v2(study / name / "field", device, store_root=study / "blobs")
        if manifest["state_identity"] != reports[arm]["state_identity"]:
            raise ValueError("selected student export changed")
        store = BoundedExperiment(study, f"{comparison_name}_{arm}", config["storage"])
        native, members = [], []
        for vector in vectors:
            result = field.query(QueryPacket("text", task="union", semantic=vector.to(device)))
            native.append(result.semantic_scores.cpu())
            members.append(result.membership_scores.cpu())
        payload = dict(xyz=field.xyz.cpu(), observed=field.observed.cpu(), queries=queries,
                       native=torch.stack(native, 1), member=torch.stack(members, 1))
        store.tensor("predictions.pt", payload, max_bytes=32 * 1024**2)
        store.json("prediction.json", dict(predictions_sha256=sha256(store.root / "predictions.pt"),
                    state_identity=manifest["state_identity"], query_sha256=sha256(meta.root / "query_vectors.pt"),
                    labels_opened=False, field_source_access_at_query=False))
        stores[arm] = store
        del field, payload, native, members
    # Same-source uncompressed mean-feature reference, not the stronger
    # multi-observation T0 measure and not the historical v10 baseline.
    if sha256(source / "initial.pt") != source_manifest["initial"]["sha256"]:
        raise ValueError("uncompressed reference changed")
    initial = torch.load(source / "initial.pt", map_location="cpu", weights_only=True, mmap=True)
    native = F.normalize(initial["semantic"].float(), dim=1) @ F.normalize(vectors, dim=1).T
    reference = BoundedExperiment(study, f"{comparison_name}_native_reference", config["storage"])
    reference.tensor("predictions.pt", dict(xyz=initial["geometry"]["xyz"], observed=initial["observed"],
                     queries=queries, native=native, member=native), max_bytes=32 * 1024**2)
    reference.json("prediction.json", dict(predictions_sha256=sha256(reference.root / "predictions.pt"),
                   labels_opened=False, member_aliases_native_no_member_model=True))
    stores["native_mean_reference"] = reference
    meta.json("prediction_barrier.json", {arm: sha256(store.root / "predictions.pt") for arm, store in stores.items()})
    # Reuse the established scorer, with all predictions already sealed.
    from .public_scannet import score
    results = {}
    with tempfile.TemporaryDirectory(prefix="radiogs-compact-score-") as temporary:
        for arm, store in stores.items():
            report = score(store.root, evaluation_path, Path(temporary) / (arm + ".json"))
            store.json("score.json", report)
            results[arm] = report["metrics"]
    summary = dict(scene=source_manifest["scene"], metrics=results,
                   field_bytes={arm: report["complete_field_payload_bytes"] for arm, report in reports.items()},
                   current_text_encoder_parameter_bytes=encoder_bytes,
                   query_encoder_included_deployment_bytes={arm: report["complete_field_payload_bytes"] + encoder_bytes
                                                           for arm, report in reports.items()},
                   storage_bytes_before_summary=tree_bytes(study),
                   independent_final_test=False, five_benchmarks_complete=False,
                   full_v10_capability_transfer=False, native_reference_is_mean_feature_only=True)
    meta.json("summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--reconstruction-run", required=True)
    parser.add_argument("--functional-run", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--comparison-name", default="comparison")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.2, torch.device(args.device))
    print(json.dumps(evaluate(json.loads(Path(args.config).read_text()), args.study_root,
          {"reconstruction": args.reconstruction_run, "functional": args.functional_run},
          args.queries, args.evaluation, device=args.device, comparison_name=args.comparison_name), indent=2))


if __name__ == "__main__":
    main()
