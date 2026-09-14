"""Bind repaired scene memories to audited query banks without stale metadata."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import canonicalize_siglip2_query, load_text_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--scene-memory-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    audit_path = args.audit_directory / "report.json"
    audit_sha = sha256_file(audit_path)
    scenes = []
    for scene in ("figurines", "ramen", "teatime", "waldo_kitchen"):
        memory_path = args.scene_memory_root / scene / "sealed/manifest.json"
        record = json.loads(memory_path.read_text())
        record["scene_memory_manifest_sha256"] = sha256_file(memory_path)
        for kind in ("text_query_cache", "negative_text_query_cache"):
            source = args.audit_directory / (scene + "_" + kind + ".pt")
            old = torch.load(source, map_location="cpu", weights_only=False)
            payload = {"schema": "radio_gs.verified_siglip2_query_cache.v1", "text_encoder": "siglip2",
                "model_name": "google/siglip2-giant-opt-patch16-384", "prompt_templates": ["{query}"],
                "text_canonicalization": "official_c_radio_siglip2_g",
                "queries": old["queries"], "canonical_queries": [canonicalize_siglip2_query(q) for q in old["queries"]],
                "embeddings": old["embeddings"], "reencoding_audit_sha256": audit_sha,
                "reencoded_bank_sha256": sha256_file(source)}
            target = (args.output_directory / (scene + "_" + kind + ".pt")).resolve()
            torch.save(payload, target)
            load_text_cache(target, payload["queries"], torch.device("cpu"))
            record[kind] = str(target)
            record[kind + "_sha256"] = sha256_file(target)
        scenes.append(record)
    (args.output_directory / "manifest.json").write_text(json.dumps({"schema": "radio_gs.lerf_repaired_inputs.v1",
        "quality_accepted": False, "scenes": scenes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
