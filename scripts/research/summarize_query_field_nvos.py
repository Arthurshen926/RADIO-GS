"""Seal the fixed fern signed-prompt study, retaining all arms and limitations."""
import json
import os
from pathlib import Path
import subprocess

import torch

from radio_gs.query_field.storage import sha256


def receipt(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def main():
    root = Path("output/query_field_20261007/nvos_fern_v1")
    read = lambda path: json.loads(Path(path).read_text())
    config = read("configs/query_field/nvos_fern_prediction.json")
    evaluation = read(root/"public/evaluation.json")
    training = read(root/"models/training.json")
    cold = read(root/"cold_replay.json")
    if len(evaluation["results"]) != len(config["jobs"]) or len(cold["comparisons"]) != len(config["jobs"]):
        raise ValueError("incomplete fixed cohort")
    if not cold["original_open_rejected"] or not all(all(row["exact"].values()) for row in cold["comparisons"]):
        raise ValueError("source-denied replay failed")
    if cold["configuration_sha256"] != sha256("configs/query_field/nvos_fern_prediction.json"):
        raise ValueError("configuration changed after cold replay")
    observations = read(root/"observations/observations.json")
    initial = torch.load(root/"observations/initial.pt", map_location="cpu", weights_only=True)
    protected = {}
    for stage in ["untrained", "trained"]:
        state = torch.load(root/"models"/stage/"field.pt", map_location="cpu", weights_only=True)
        expected = dict(initial["geometry"], observed=initial["observed"])
        expected.update({"semantic.codes": initial["semantic"], "local.codes": initial["local"]})
        protected[stage] = {key: torch.equal(value, state[key]) for key, value in expected.items()}
        if not all(protected[stage].values()):
            raise ValueError("protected coordinates changed")
        del state
    del initial
    baseline_path = Path("output/query_conditioned_membership_20261006/image_convergence/system/summary.json")
    baseline = next(row for row in read(baseline_path)["rows"]
                    if row["job"]["benchmark"] == "nvos" and row["job"]["name"] == "fern")
    rows = []
    for result in evaluation["results"]:
        p = read(root/"public"/result["job"]/"prediction.json")
        rows.append(dict(**result, scale_m=p["scale_m"], prompt_agreement=p["prompt_agreement"],
                         registration=p["registration"], negative_observed_mass_fraction=p["negative_observed_mass_fraction"],
                         observed_positive_mass_fraction=p["diagnostics"]["observed_positive_mass_fraction"],
                         reference_gap_pp=100*(baseline["metrics"]["iou"]-result["retained_response"]["foreground_iou"])))
    # Capture changes made concurrently in the original worktree without touching it.
    command = lambda *args: subprocess.check_output(["git", "-C", "/root/RADIO-GS", *args],
                                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    index = Path(command("rev-parse", "--git-path", "index").decode().strip())
    if not index.is_absolute():
        index = Path("/root/RADIO-GS")/index
    original = dict(branch=command("branch", "--show-current").decode().strip(),
                    head=command("rev-parse", "HEAD").decode().strip(), index_sha256=sha256(index),
                    original_index_matches_frozen=sha256(index)=="917366f03337f78208f976ee26f4c58f198d2ce78ff64b29f9d7c94c06ad621c",
                    note="Original source worktree is externally dirty; no claim of complete byte equality.")
    # Check deployment payloads and compiled source records against their own receipts.
    for stage in ["untrained", "trained"]:
        directory = root/"models"/stage
        if sha256(directory/"field.pt") != read(directory/"manifest.json")["files"]["field.pt"]["sha256"]:
            raise ValueError("field payload no longer matches manifest")
    for frame in observations["frames"]:
        record = frame["compiled"]
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("compiled observation changed")
    artifacts = [root/"availability.json", root/"source_authority.json", root/"scene.json", root/"cameras.pt",
                 root/"native_regions/encoding.json", root/"observations/observations.json", root/"models/training.json",
                 root/"models/untrained/manifest.json", root/"models/trained/manifest.json",
                 root/"public/evaluation.json", root/"cold_replay.json", root/"development_trained.json",
                 root/"tests_corrected.log", root/"tests_scorer.log", root/"retained_readout_check.json", baseline_path]
    artifacts += [root/"public"/j["name"]/"prediction.json" for j in config["jobs"]]
    for job in config["jobs"]:
        directory = root/"public"/job["name"]
        if sha256(directory/"predictions.pt") != read(directory/"prediction.json")["predictions_sha256"]:
            raise ValueError("public predictions changed")
    code = sorted(Path("radio_gs/query_field").glob("*.py"))
    code += sorted(Path("configs/query_field").glob("nvos_fern*.json"))
    code += [Path("scripts/research")/f"{name}_query_field_nvos.py" for name in ["prepare", "run", "summarize"]]
    code += [Path("tests/test_query_field_signed_raster.py")]
    code += [Path("scripts/research/evaluate_reference_instance.py"),
             Path("radio_gs/evaluation/promptable_segmentation.py"),
             Path("radio_gs/training/retrieval_metrics.py")]
    report = dict(schema="radio_gs.query_field_nvos_evidence.v1", source_commit_before_work="de2face",
                  construction=dict(elements=observations["elements"], observed=observations["observed_elements"],
                                    splits=observations["splits"], episodes=training["training_episodes"],
                                    protected_equal=protected, bytes=training["trained_deployment_bytes"],
                                    training_seconds=training["seconds"]),
                  results=rows, saved_v10_fern=baseline["metrics"], original_worktree=original,
                  code=[receipt(p) for p in code], artifacts=[receipt(p) for p in artifacts],
                  public_labels_opened_only_after_all_predictions_and_cold_replay=True,
                  aborted_attempt="CUDA pooling last-bit member mismatch; no target score; retained under public_atomic_aborted",
                  current_candidate_changed=False, independent_final_test=False, model_promotion=False,
                  caveats=["one availability-selected historically opened case, not full NVOS",
                           "same retained response operator, different prompt/source-access/source-budget conditions",
                           "actual subset >0.5 is separate from retained response and old selected-projection >10/255",
                           "source feature mean fusion is not proven query-preserving distillation",
                           "deployment is high-capacity and excludes shared query encoder weights"])
    path = Path("docs/experiments/2026-10-07-native-signed-prompt-evidence.json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(results=rows, construction=report["construction"]), indent=2))


if __name__ == "__main__":
    main()
