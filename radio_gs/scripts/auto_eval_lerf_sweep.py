#!/usr/bin/env python3
"""Run a temperature sweep for a trained LERF checkpoint and freeze the best result."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from radio_gs.config import load_config


DEFAULT_SCENE_TEMPS = {
    "figurines": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
    "ramen": [20.0, 25.0, 30.0, 35.0, 40.0],
    "teatime": [15.0, 20.0, 25.0, 28.0, 30.0, 35.0],
    "waldo_kitchen": [10.0, 12.0, 15.0, 18.0, 20.0, 25.0],
}


def parse_temps(text: str | None, scene: str) -> list[float]:
    if text:
        return [float(part.strip()) for part in text.split(",") if part.strip()]
    return DEFAULT_SCENE_TEMPS.get(scene, [15.0, 20.0, 25.0, 30.0, 35.0])


def temp_label(temp: float) -> str:
    if float(temp).is_integer():
        return str(int(temp))
    return str(temp).replace(".", "p")


def read_rendered_metrics(result_path: Path, scene: str) -> dict[str, float]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = payload["scenes"][scene]["rendered"]
    return {
        "loc_acc": float(metrics["loc_acc"]),
        "miou": float(metrics["miou"]),
        "loc_total": int(metrics["loc_total"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature sweep for LERF grounding evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--temps", default=None, help="Comma-separated temperature list")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--gpu", type=int, default=0, help="Local CUDA device index")
    parser.add_argument("--heatmap_upsample", type=int, default=4)
    parser.add_argument("--scoring", default="softmax_scene", choices=["softmax_scene", "cosine", "relevancy"])
    parser.add_argument(
        "--text_embedding_cache",
        default="checkpoints/siglip2_lerf_text_embeddings.pt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    scene = args.scene or getattr(cfg, "scene", "")
    if not scene:
        raise ValueError("Scene is required either via --scene or config.scene")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temps = parse_temps(args.temps, scene)

    results: list[dict[str, object]] = []
    best_entry: dict[str, object] | None = None

    for temp in temps:
        label = temp_label(temp)
        run_dir = output_root / f"T{label}"
        cmd = [
            sys.executable,
            "-m",
            "radio_gs.scripts.eval_lerf_grounding",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--scene",
            scene,
            "--output_dir",
            str(run_dir),
            "--text_embedding_cache",
            args.text_embedding_cache,
            "--scoring",
            args.scoring,
            "--relevancy_temp",
            str(temp),
            "--heatmap_upsample",
            str(args.heatmap_upsample),
            "--gpu",
            str(args.gpu),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        result_path = run_dir / "lerf_ovs_results.json"
        metrics = read_rendered_metrics(result_path, scene)
        entry: dict[str, object] = {
            "temp": temp,
            "output_dir": str(run_dir),
            **metrics,
        }
        results.append(entry)

        if best_entry is None:
            best_entry = entry
        else:
            best_pair = (float(best_entry["loc_acc"]), float(best_entry["miou"]))
            cur_pair = (metrics["loc_acc"], metrics["miou"])
            if cur_pair > best_pair:
                best_entry = entry

    assert best_entry is not None

    best_dir = output_root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_src = Path(str(best_entry["output_dir"])) / "lerf_ovs_results.json"
    shutil.copy2(best_src, best_dir / "lerf_ovs_results.json")

    summary_payload = {
        "scene": scene,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "scoring": args.scoring,
        "heatmap_upsample": args.heatmap_upsample,
        "results": results,
        "best": best_entry,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# LERF Sweep Summary: {scene}",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- scoring: `{args.scoring}`",
        f"- heatmap upsample: `{args.heatmap_upsample}`x",
        "",
        "| Temp | LocAcc | mIoU | Samples | Output |",
        "|---|---:|---:|---:|---|",
    ]
    for entry in results:
        lines.append(
            f"| {entry['temp']} | {float(entry['loc_acc']):.4f} | {float(entry['miou']):.4f} | "
            f"{int(entry['loc_total'])} | `{entry['output_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Best",
            "",
            f"- best temp: `{best_entry['temp']}`",
            f"- best LocAcc: `{float(best_entry['loc_acc']):.4f}`",
            f"- best mIoU: `{float(best_entry['miou']):.4f}`",
            f"- frozen JSON: `{best_dir / 'lerf_ovs_results.json'}`",
            "",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Sweep summary saved to {output_root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
