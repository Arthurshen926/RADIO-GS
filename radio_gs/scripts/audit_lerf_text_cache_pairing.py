"""Compare deployed query banks with freshly encoded official SigLIP2 text."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.scripts.eval_lerf_grounding import encode_text_siglip2, _canonicalize_siglip2_text
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    records, queries = [], []
    for scene in ("figurines", "ramen", "teatime", "waldo_kitchen"):
        report = json.loads((args.reference_root / (scene + "_text.json")).read_text())
        for kind in ("text_query_cache", "negative_text_query_cache"):
            path = Path(report[kind])
            if sha256_file(path) != report[kind + "_sha256"]:
                raise ValueError("query cache hash differs")
            data = torch.load(path, map_location="cpu", weights_only=False)
            records.append((scene, kind, path, data))
            queries.extend(data["queries"])
    queries = list(dict.fromkeys(queries))
    fresh = encode_text_siglip2(queries, torch.device("cuda:0")).cpu()
    lookup = dict(zip(queries, fresh))
    results = []
    for scene, kind, path, data in records:
        expected = torch.stack([lookup[q] for q in data["queries"]])
        cached = torch.nn.functional.normalize(data["embeddings"].float(), dim=-1)
        cosine = torch.nn.functional.cosine_similarity(expected, cached)
        result = {"scene": scene, "kind": kind, "cache_sha256": sha256_file(path),
                  "minimum_cosine": float(cosine.min()), "max_absolute_error": float((cached-expected).abs().max()),
                  "per_query_cosine": dict(zip(data["queries"], cosine.tolist()))}
        results.append(result)
        torch.save({"schema": "radio_gs.reencoded_siglip2_query_cache.v1", "text_encoder": "siglip2",
                    "model_name": "google/siglip2-giant-opt-patch16-384", "queries": data["queries"], "prompt_templates": ["{query}"],
                    "original_cache_sha256": sha256_file(path),
                    "embeddings": expected, "canonical_queries": [_canonicalize_siglip2_text(q) for q in data["queries"]],
                    "text_canonicalization": "official_c_radio_siglip2_g"}, args.output_directory / (scene + "_" + kind + ".pt"))
        print(scene, kind, result["minimum_cosine"], result["max_absolute_error"], flush=True)
    (args.output_directory / "report.json").write_text(json.dumps({"results": results, "queries_used_for_training": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
