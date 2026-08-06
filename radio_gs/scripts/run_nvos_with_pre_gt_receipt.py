"""Run a frozen graph-disabled NVOS evaluator with an external pre-GT receipt.

The evaluator's graph-enabled receipt option is intentionally incompatible
with the frozen graph-disabled unary candidate.  This supervisor leaves the
evaluator unchanged, delays only the exact target-mask open via LD_PRELOAD,
and atomically seals hashes of every prediction-side artifact before release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = _file_sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _primitive_tensor_hash(path: Path) -> tuple[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("primitive unary artifact must be a mapping")
    for key in ("primitive_unary_probability", "gaussian_scores"):
        value = payload.get(key)
        if torch.is_tensor(value):
            return key, tensor_sha256(value.detach().cpu().contiguous())
    raise ValueError("primitive unary artifact lacks a recognized score tensor")


def _atomic_write_no_clobber(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different receipt: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _create_release(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(descriptor)


def _wait_for_blocked_target_open(
    child: subprocess.Popen[Any], marker: Path, deadline: float
) -> None:
    while not marker.is_file():
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(
                f"evaluator exited before its target-mask open was gated: {return_code}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out before target-mask open was gated")
        time.sleep(0.05)


def _build_receipt_payload(
    *,
    scene_id: str,
    command: list[str],
    target: Path,
    output_dir: Path,
    primitive: Path,
    score: Path,
    completion: Path,
    completion_sha256: str,
    completion_receipt: Path,
    completion_receipt_sha256: str,
    source_gate: Path,
    source_gate_sha256: str,
    evaluator: Path,
    evaluator_sha256: str,
) -> dict[str, Any]:
    if not primitive.is_file() or not score.is_file():
        raise RuntimeError("GT open was attempted before primitive/score persistence")
    primitive_tensor_name, primitive_tensor_sha256 = _primitive_tensor_hash(primitive)
    stage_records = []
    stage_root = output_dir / "stage_scores"
    if stage_root.is_dir():
        for path in sorted(stage_root.rglob("*.npy")):
            stage_records.append(
                {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            )
    candidate_config = {"evaluator_argv": command}
    return {
        "schema_version": 1,
        "artifact_type": "nvos_graph_disabled_pre_gt_prediction_receipt_v1",
        "scene_id": str(scene_id),
        "candidate_config": candidate_config,
        "candidate_config_sha256": _canonical_sha256(candidate_config),
        "evaluator": {"path": str(evaluator), "sha256": evaluator_sha256},
        "source_completion": {
            "path": str(completion),
            "sha256": completion_sha256,
            "receipt_path": str(completion_receipt),
            "receipt_sha256": completion_receipt_sha256,
        },
        "source_only_gate": {
            "path": str(source_gate),
            "sha256": source_gate_sha256,
        },
        "primitive_unary": {
            "path": str(primitive),
            "file_sha256": _file_sha256(primitive),
            "score_tensor_name": primitive_tensor_name,
            "score_tensor_sha256": primitive_tensor_sha256,
        },
        "rendered_prediction": {
            "path": str(score),
            "sha256": _file_sha256(score),
        },
        "stage_predictions": stage_records,
        "safety": {
            "exact_target_ground_truth_path": str(target),
            "target_open_attempt_observed_and_blocked": True,
            "sealed_before_target_ground_truth_open": True,
            "evaluator_modified_for_receipt": False,
            "frozen_graph_disabled_method_changed": False,
        },
    }


def _wait_for_child_success(child: subprocess.Popen[Any], deadline: float) -> None:
    return_code = child.wait(timeout=max(1.0, deadline - time.monotonic()))
    if return_code != 0:
        raise RuntimeError(f"evaluator failed after GT release: {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--target-ground-truth", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primitive-unary", required=True)
    parser.add_argument("--rendered-score", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--blocked-marker", required=True)
    parser.add_argument("--release-path", required=True)
    parser.add_argument("--interposer", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--completion-receipt", required=True)
    parser.add_argument("--completion-receipt-sha256", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--source-gate-sha256", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an evaluator command is required after --")

    target = Path(args.target_ground_truth).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    output_dir = Path(args.output_dir).expanduser().resolve()
    primitive = Path(args.primitive_unary).expanduser().resolve()
    score = Path(args.rendered_score).expanduser().resolve()
    receipt = Path(args.receipt_output).expanduser().resolve()
    marker = Path(args.blocked_marker).expanduser().resolve()
    release = Path(args.release_path).expanduser().resolve()
    interposer = Path(args.interposer).expanduser().resolve()
    if not interposer.is_file():
        raise FileNotFoundError(interposer)
    if any(path.exists() for path in (receipt, marker, release)):
        raise FileExistsError("receipt, blocker marker, or release file already exists")
    completion = _require_file(
        args.completion, args.completion_sha256, "source completion"
    )
    completion_receipt = _require_file(
        args.completion_receipt,
        args.completion_receipt_sha256,
        "source completion receipt",
    )
    source_gate = _require_file(args.source_gate, args.source_gate_sha256, "source gate")
    evaluator = _require_file(args.evaluator, args.evaluator_sha256, "evaluator")

    environment = os.environ.copy()
    previous_preload = environment.get("LD_PRELOAD", "").strip()
    environment["LD_PRELOAD"] = (
        f"{interposer}:{previous_preload}" if previous_preload else str(interposer)
    )
    environment["RADIO_GS_GATED_TARGET_PATH"] = str(target)
    environment["RADIO_GS_GT_BLOCKED_MARKER_PATH"] = str(marker)
    environment["RADIO_GS_GT_RELEASE_PATH"] = str(release)
    child = subprocess.Popen(command, env=environment)
    deadline = time.monotonic() + float(args.timeout_seconds)
    try:
        _wait_for_blocked_target_open(child, marker, deadline)
        payload = _build_receipt_payload(
            scene_id=args.scene_id,
            command=command,
            target=target,
            output_dir=output_dir,
            primitive=primitive,
            score=score,
            completion=completion,
            completion_sha256=args.completion_sha256,
            completion_receipt=completion_receipt,
            completion_receipt_sha256=args.completion_receipt_sha256,
            source_gate=source_gate,
            source_gate_sha256=args.source_gate_sha256,
            evaluator=evaluator,
            evaluator_sha256=args.evaluator_sha256,
        )
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_write_no_clobber(receipt, encoded)
        _create_release(release)
        _wait_for_child_success(child, deadline)
        print(
            json.dumps(
                {
                    "scene_id": args.scene_id,
                    "receipt": str(receipt),
                    "receipt_sha256": _file_sha256(receipt),
                    "candidate_config_sha256": payload["candidate_config_sha256"],
                    "primitive_score_tensor_sha256": payload["primitive_unary"][
                        "score_tensor_sha256"
                    ],
                    "rendered_prediction_sha256": payload["rendered_prediction"]["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise


if __name__ == "__main__":
    main()
