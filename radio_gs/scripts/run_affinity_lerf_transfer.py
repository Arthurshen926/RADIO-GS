"""Frozen four-scene transfer of affinity checkpoints, with separate processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    root, data, output = Path(args.reference_root), Path(args.data_root), Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    commands = []

    def execute(module, flags, log_name):
        command = [sys.executable, "-u", "-m", module]
        for flag, value in flags.items():
            command.append("--" + flag.replace("_", "-"))
            command.extend(map(str, value if isinstance(value, list) else [value]))
        commands.append(command)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (output / log_name).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    for scene in ("figurines", "ramen", "teatime", "waldo_kitchen"):
        old = torch.load(root / "v4_3_object_hypotheses_sam_like_all_views_affinity_20260904" / (scene + ".pt"), map_location="cpu", weights_only=False)
        prior = json.loads((root / "v4_3_object_hypothesis_text_sam_like_all_views_affinity_20260904" / (scene + ".json")).read_text())
        state = {"scene_state": old["scene_state"], "expected_scene_state_sha256": old["scene_state_sha256"], "scene_label": scene}
        memory = output / (scene + ".pt")
        config = dict(old["association_configuration"])
        if config.pop("maximum_fragment_hypotheses", 2) != 2:
            raise ValueError("reference requires an unsupported assignment capacity")
        execute("radio_gs.v4.contracts.build_lerf_object_hypotheses", {
            **state, **config,
            "fragment_memory": old["fragment_memory"],
            "expected_fragment_memory_sha256": old["fragment_memory_sha256"],
            "appearance_source": "fragment_language_manifest",
            "fragment_language_manifest": prior["appearance_audit"]["fragment_language_manifest"],
            "maximum_object_prototypes": prior["query_contract"]["object_prototypes_retained"],
            "learned_affinity_checkpoint": checkpoint,
            "expected_learned_affinity_checkpoint_sha256": sha256_file(checkpoint),
            "cpu_threads": 4, "output": memory,
        }, scene + "_build.log")
        common = {**state, "hypothesis_memory": memory, "expected_hypothesis_memory_sha256": sha256_file(memory),
                  "scene_root": data / scene, "label_root": data / "label", "raster_shape": prior["raster_shape"], "cpu_threads": 4}
        ceiling = output / (scene + "_ceiling.json")
        execute("radio_gs.v4.evaluation.lerf_object_hypothesis_ceiling", {**common, "output": ceiling}, scene + "_ceiling.log")
        execute("radio_gs.v4.evaluation.lerf_object_hypothesis_text_evaluator", {
            **common, "hypothesis_ceiling": ceiling,
            "text_query_cache": prior["text_query_cache"], "negative_text_query_cache": prior["negative_text_query_cache"],
            "query_temperature": prior["query_contract"]["temperature"],
            "query_set_mass": prior["query_contract"]["query_set_mass"],
            "consensus_fragments": prior["query_contract"]["distinct_fragment_consensus"],
            "null_similarity": prior["query_contract"]["null_similarity"],
            "pixel_threshold": prior["pixel_threshold"], "device": "cuda:0", "output": output / (scene + "_text.json"),
        }, scene + "_text.log")
        print(scene + " completed", flush=True)
    (output / "completed.json").write_text(json.dumps({"completed": True, "assignment_formula_unchanged": True,
        "geometry_only_scope": "learned_grouping_only; fixed assignment still uses appearance",
        "development_only": True}) + "\n")


if __name__ == "__main__":
    main()
