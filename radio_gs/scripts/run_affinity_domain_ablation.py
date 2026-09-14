"""Replay the sealed affinity cohort with one explicitly recorded feature ablation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--feature-mode", choices=("full", "geometry_only"), required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference_report).read_text())
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    config = reference["training_configuration"]
    noise = reference["fragment_noise_contract"]
    command = [sys.executable, "-u", "-m", "radio_gs.v4.training.train_scannet_fragment_affinity",
               "--cohort-manifest", reference["cohort"]["path"],
               "--allow-instance-oracle-training", "--device", "cuda:0",
               "--feature-mode", args.feature_mode, "--model-kind", "pair_mlp",
               "--cpu-threads", "4", "--output-checkpoint", str(output / "model.pt"),
               "--output", str(output / "report.json")]
    for key in ("seed", "step_count", "batch_size", "hidden_dimension", "learning_rate", "weight_decay"):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    for key, flag in (("fragment_noise_mode", "fragment-noise-mode"), ("parts_per_object", "parts-per-object"),
                      ("retained_fraction", "retained-fragment-fraction"), ("light_merge_fraction", "light-merge-fraction"),
                      ("pair_scope", "pair-scope")):
        command.extend(["--" + flag, str(noise[key])])
    for item in reference["scene_cache_receipts"]:
        command.extend(["--scene-cache", item["path"]])
    for split in ("training", "validation"):
        for scene in reference[split + "_scene_ids"]:
            command.extend(["--" + split + "-scene", scene])
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "run.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (output / "exit_code.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
