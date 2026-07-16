#!/usr/bin/env python3
"""Run one resumable, protocol-locked canonical promptable-NVS scene.

This runner deliberately consumes an already audited prompt queue instead of
rediscovering RGB/camera correspondences.  It skips a stage only when the
declared terminal artifact exists and records every exact command and return
code.  No benchmark mask is opened before the final registered-prompt stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


PYTHON = "/root/miniconda3/envs/cybersim_agent/bin/python"
RADIO_CHECKPOINT = "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _scene(queue: dict[str, Any], scene_id: str) -> dict[str, Any]:
    rows = [row for row in queue["scenes"] if row["scene_id"] == scene_id]
    if len(rows) != 1:
        raise ValueError(f"expected one queue scene {scene_id!r}; found {len(rows)}")
    return rows[0]


def _run_stage(
    *, name: str, command: list[str], terminal: Path, log_root: Path,
    records: list[dict[str, Any]],
) -> None:
    if terminal.is_file():
        records.append({"stage": name, "status": "reused", "terminal": str(terminal)})
        return
    terminal.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root / f"{name}.log"
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    record = {
        "stage": name, "status": "complete" if result.returncode == 0 else "failed",
        "returncode": result.returncode, "seconds": time.time() - started,
        "command": command, "log": str(log), "terminal": str(terminal),
    }
    records.append(record)
    if result.returncode != 0 or not terminal.is_file():
        raise RuntimeError(f"stage {name} failed or lacks terminal artifact: {terminal}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    queue_path = Path(args.queue_plan).expanduser().resolve()
    queue = json.loads(queue_path.read_text())
    scene = _scene(queue, args.scene_id)
    scene_root = Path(scene["config"]).parent
    legacy_field = Path(scene["artifacts"]["feature_field_checkpoint"]["path"])
    geometry_ply = Path(scene["artifacts"]["geometry_ply"]["path"])
    geometry_checkpoint = geometry_ply.parents[2] / "chkpnt30000.pth"
    out = Path(args.output_root).expanduser().resolve() / args.scene_id
    logs = out / "logs"
    status_path = out / "status.json"
    records: list[dict[str, Any]] = []

    terminals = {
        "geometry": geometry_ply,
        "feature_extraction": scene_root / "radio_features" / "frame_manifest.json",
        "feature_field": legacy_field,
    }
    for name in ("geometry", "feature_extraction", "feature_field"):
        try:
            _run_stage(name=name, command=list(scene["commands"][name]),
                       terminal=terminals[name], log_root=logs, records=records)
        finally:
            _write_json(status_path, {"scene_id": args.scene_id, "records": records})
    if args.stop_after == "legacy":
        return {"scene_id": args.scene_id, "status": "legacy_complete", "records": records}

    raw = out / "raw_radio.pt"; responsibility = out / "responsibility.pt"
    base_mpr = [
        PYTHON, str(project / "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py"),
        "--config", scene["config"], "--checkpoint", str(legacy_field),
        "--observation-contract", "canonical-mpr-v1", "--max-views", "120",
        "--device", args.device,
    ]
    excluded = [str(value) for value in scene.get("excluded_image_stems", [])]
    numeric_excluded: list[str] = []
    for value in excluded:
        try:
            numeric_excluded.append(str(int(value)))
        except ValueError:
            # Audited NVOS queues exclude RGB basenames before geometry and
            # feature extraction.  Their resulting feature IDs are compact
            # integers and contain no row corresponding to the held-out stem,
            # so forwarding the basename to this integer-only MPR option is
            # both invalid and unnecessary.
            continue
    if numeric_excluded:
        # Query/prompt frames must be excluded before lifting whenever the
        # audited queue declares them.  SPIn's registered reference image is
        # part of the reconstruction protocol, whereas NVOS declares unseen
        # prompt frames here; the runner must preserve that distinction.
        base_mpr += ["--exclude-frame-ids", ",".join(numeric_excluded)]
    _run_stage(name="mpr_raw", command=base_mpr + [
        "--feature-space", "radio", "--save-responsibility-cache", str(responsibility),
        "--output", str(raw),
    ], terminal=raw, log_root=logs, records=records)
    dino = out / "dino_v3.pt"; sam = out / "sam3.pt"
    for name, space, terminal in (("mpr_dino", "dino_v3", dino), ("mpr_sam3", "sam3", sam)):
        _run_stage(name=name, command=base_mpr + [
            "--feature-space", space, "--radio-checkpoint", RADIO_CHECKPOINT,
            "--responsibility-cache", str(responsibility), "--output", str(terminal),
        ], terminal=terminal, log_root=logs, records=records)
    field = out / "canonical_d256_l128_capability_first.pth"
    _run_stage(name="canonical_field", command=[
        PYTHON, str(project / "radio_gs/scripts/train_canonical_radio_field.py"),
        "--mpr-cache", str(raw), "--observation-contract", "canonical-mpr-v1",
        "--radio-checkpoint", RADIO_CHECKPOINT, "--output", str(field),
        "--device", args.device, "--coefficient-dim", "256", "--local-dim", "128",
        "--primitive-fusion", "--official-capability-loss",
        "--dino-mpr-cache", str(dino), "--sam3-mpr-cache", str(sam),
        "--epochs", "20", "--min-epochs", "5", "--target-cosine", "0.985",
        "--seed", "0",
    ], terminal=field, log_root=logs, records=records)
    capability = out / "official_dino_sam3_views.pt"
    _run_stage(name="capability_views", command=[
        PYTHON, str(project / "radio_gs/scripts/build_canonical_capability_views.py"),
        "--field-checkpoint", str(field), "--mpr-cache", str(raw),
        "--radio-checkpoint", RADIO_CHECKPOINT, "--output", str(capability),
        "--batch-size", "2048", "--device", args.device,
    ], terminal=capability, log_root=logs, records=records)
    graph = out / "shared_support_graph_k16.pt"
    _run_stage(name="support_graph", command=[
        PYTHON, str(project / "radio_gs/scripts/build_canonical_support_graph.py"),
        "--capability-cache", str(capability), "--output", str(graph),
        "--neighbors", "16", "--topology-mode", "symmetric_union",
    ], terminal=graph, log_root=logs, records=records)
    evaluation = out / "eval_full_mask_random_walker" / f"{args.scene_id}_evaluation.json"
    _run_stage(name="registered_full_mask", command=[
        PYTHON, str(project / "radio_gs/scripts/eval_nvos_gaussian_first.py"),
        "--manifest", queue["manifest"], "--queue-root", str(Path(args.queue_plan).parent),
        "--scene-id", args.scene_id, "--output-dir", str(evaluation.parent),
        "--device", args.device, "--region-space", "sam3", "--support-mode", "canonical_support",
        "--prototype-count", "4", "--canonical-capability-cache", str(capability),
        "--canonical-support-graph", str(graph), "--canonical-field-sha256", _sha256(field),
        "--graph-policy", "legacy", "--component-graph-policy", "same",
        "--feature-calibration", "none", "--score-calibration", "none",
        "--solver-type", "confidence_random_walker", "--laplacian-weight", "1.0",
        "--solver-iterations", "12", "--solver-residual", "0.30",
        "--solver-support-threshold", "0.50",
    ], terminal=evaluation, log_root=logs, records=records)
    result = {
        "scene_id": args.scene_id, "status": "complete", "protocol_hash": queue["protocol_hash"],
        "queue_plan_sha256": _sha256(queue_path), "field_sha256": _sha256(field),
        "canonical_mainline": "canonical-mpr-v3", "records": records,
    }
    _write_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-plan", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after", choices=("legacy", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
