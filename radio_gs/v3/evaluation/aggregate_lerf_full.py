"""Aggregate complete 2D and dual-readout 3D LERF result receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).resolve(strict=True)
    rows, inputs = [], []
    for scene in args.scene:
        paths = {
            "2d": root / "lerf2d" / f"{scene}_eval" / "lerf_ovs_results.json",
            "3d_strict": root / "lerf3d_strict" / scene / scene / "lerf_direct_3d_selection_results.json",
            "3d_top2": root / "lerf3d_top2" / scene / scene / "lerf_direct_3d_selection_results.json",
        }
        payloads = {}
        for key, path in paths.items():
            path = path.resolve(strict=True)
            with path.open() as handle:
                payloads[key] = json.load(handle)
            inputs.append({"role": key, "scene": scene, "path": str(path), "sha256": sha256_file(path)})
        two = payloads["2d"]["aggregates"]["rendered"]
        row = {
            "scene": scene,
            "samples": int(two["sample_count"]),
            "lerf2d_miou": float(two["sample_micro_miou"]),
            "lerf2d_localization": float(two["localization_accuracy"]),
        }
        for role in ("3d_strict", "3d_top2"):
            scene_result = payloads[role]["scene"]
            result = scene_result["results"][scene_result["best_by_miou"]]
            row[f"lerf_{role}_miou"] = float(result["miou"])
            row[f"lerf_{role}_acc025"] = float(result["acc025"])
            row[f"lerf_{role}_acc050"] = float(result["acc050"])
        rows.append(row)
    keys = [key for key in rows[0] if key not in ("scene", "samples")]
    total = sum(row["samples"] for row in rows)
    payload = {
        "schema": "radio_gs.sugm_v3.lerf_full_evaluation.v1",
        "scenes": rows,
        "scene_macro": {key: sum(row[key] for row in rows) / len(rows) for key in keys},
        "sample_micro": {key: sum(row[key] * row["samples"] for row in rows) / total for key in keys},
        "sample_count": total,
        "same_gaussian_posterior_for_2d_and_3d": True,
        "strict_3d_rule": "absolute_probability_above_0.5",
        "relaxed_3d_rule": "fixed_top_2_percent",
        "benchmark_used_for_method_selection": False,
        "inputs": inputs,
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
