#!/usr/bin/env python3
"""Fail-closed launcher for the LangSplatV2 LERF-2D camera fix."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path("/tmp/radio-gs-gpu0.lock")
UPSTREAM_COMMIT = "1667303d5c111a5b62f69b9b8991d80045e92b5f"
PATCH_SHA256 = "a0ba52f843fdc21a0135f71b2ebe2edb5112c4ef48235a080c3c8828a5b285f3"
AUDIT_EVAL_DIFF_SHA256 = "c65abd0c79f06ecc56df6e4a8a5093c8203ba785ebd3ff898cb39246221b8994"
AUDIT_FULL_DIFF_SHA256 = "31f14de37bb17650526b68f9dc6bd0a904c9322e7be1db2186557e217bbee608"
SCENES = ("figurines", "teatime", "ramen", "waldo_kitchen")


class ProtocolError(RuntimeError):
    """Raised before GPU work when provenance or inputs are not exact."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(checkout: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _validate_upstream(
    checkout: Path,
    *,
    allow_recorded_audit_checkout: bool = False,
) -> dict[str, str]:
    head = _git(checkout, "rev-parse", "HEAD").decode().strip()
    if head != UPSTREAM_COMMIT:
        raise ProtocolError(
            f"LangSplatV2 checkout must be pinned to {UPSTREAM_COMMIT}; found {head}"
        )
    patch_path = (
        ROOT
        / "reproductions"
        / "langsplatv2"
        / "patches"
        / "0001-exact-label-camera-resolution.patch"
    )
    packaged_patch_sha = _sha256_file(patch_path)
    if packaged_patch_sha != PATCH_SHA256:
        raise ProtocolError(
            f"packaged patch hash mismatch: {packaged_patch_sha} != {PATCH_SHA256}"
        )
    changed_paths = [
        line
        for line in _git(checkout, "diff", "--name-only").decode().splitlines()
        if line
    ]
    eval_diff = _git(checkout, "diff", "--", "eval_lerf.py")
    eval_diff_sha = _sha256_bytes(eval_diff)
    full_diff_sha = _sha256_bytes(_git(checkout, "diff"))
    if changed_paths == ["eval_lerf.py"] and eval_diff_sha == PATCH_SHA256:
        checkout_mode = "packaged_minimal_patch"
    elif (
        allow_recorded_audit_checkout
        and changed_paths == [
            "eval_lerf.py",
            "utils/loss_utils.py",
            "utils/vq_utils.py",
        ]
        and eval_diff_sha == AUDIT_EVAL_DIFF_SHA256
        and full_diff_sha == AUDIT_FULL_DIFF_SHA256
    ):
        checkout_mode = "recorded_20260731_audit_checkout"
    else:
        raise ProtocolError(
            "checkout does not match the packaged minimal patch"
            + (
                " or the explicitly allowed recorded audit checkout"
                if allow_recorded_audit_checkout
                else ""
            )
            + f"; paths={changed_paths}, eval_diff_sha256={eval_diff_sha}, "
            f"full_diff_sha256={full_diff_sha}"
        )
    return {
        "upstream_commit": head,
        "checkout_mode": checkout_mode,
        "packaged_patch_sha256": packaged_patch_sha,
        "checkout_eval_diff_sha256": eval_diff_sha,
        "checkout_full_diff_sha256": full_diff_sha,
    }


def _read_cfg_args(path: Path) -> dict[str, object]:
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ProtocolError(f"{path}: expected Namespace(...)")
    if not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace":
        raise ProtocolError(f"{path}: expected Namespace(...)")
    if expression.args:
        raise ProtocolError(f"{path}: positional Namespace arguments are unsupported")
    config: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ProtocolError(f"{path}: expanded Namespace arguments are unsupported")
        config[keyword.arg] = ast.literal_eval(keyword.value)
    return config


def _checkpoint_provenance(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cohort_configs: list[dict[str, object]] = []
    for level in (1, 2, 3):
        level_dir = Path(f"{args.scene}_{args.index}_{level}")
        relative = level_dir / f"chkpnt{args.checkpoint}.pth"
        checkpoint = args.checkpoint_root / relative
        if not checkpoint.exists():
            raise ProtocolError(f"missing checkpoint: {checkpoint}")
        cfg_path = args.checkpoint_root / level_dir / "cfg_args"
        if not cfg_path.exists():
            raise ProtocolError(f"missing checkpoint config: {cfg_path}")
        config = _read_cfg_args(cfg_path)
        expected_source = (args.data_root / args.scene).resolve()
        configured_source = Path(str(config.get("source_path", ""))).resolve()
        if configured_source != expected_source:
            raise ProtocolError(
                f"{cfg_path}: source_path={configured_source} != {expected_source}"
            )
        if config.get("eval") is not True:
            raise ProtocolError(f"{cfg_path}: eval must be exactly True")
        if config.get("feature_level") != level:
            raise ProtocolError(
                f"{cfg_path}: feature_level={config.get('feature_level')} != {level}"
            )
        if Path(str(config.get("model_path", ""))).name != level_dir.name:
            raise ProtocolError(
                f"{cfg_path}: model_path does not identify cohort directory {level_dir}"
            )
        cohort_configs.append(
            {
                key: value
                for key, value in config.items()
                if key not in {"feature_level", "model_path"}
            }
        )
        target_hash = _sha256_file(checkpoint)
        record: dict[str, object] = {
            "level": level,
            "target": str(checkpoint),
            "target_size_bytes": checkpoint.stat().st_size,
            "target_sha256": target_hash,
            "cfg_args": str(cfg_path),
            "cfg_args_sha256": _sha256_file(cfg_path),
            "cfg_source_path": str(config["source_path"]),
            "cfg_eval": config["eval"],
            "cfg_feature_level": config["feature_level"],
        }
        if args.checkpoint_source_root is not None:
            source = args.checkpoint_source_root / relative
            if not source.exists():
                raise ProtocolError(f"missing staged-checkpoint source: {source}")
            source_hash = _sha256_file(source)
            if source.stat().st_size != checkpoint.stat().st_size or source_hash != target_hash:
                raise ProtocolError(
                    f"staged checkpoint differs from source for level {level}: "
                    f"{source} -> {checkpoint}"
                )
            source_cfg_path = args.checkpoint_source_root / level_dir / "cfg_args"
            if not source_cfg_path.exists():
                raise ProtocolError(f"missing staged-config source: {source_cfg_path}")
            source_cfg_hash = _sha256_file(source_cfg_path)
            if source_cfg_hash != record["cfg_args_sha256"]:
                raise ProtocolError(
                    f"staged cfg_args differs from source for level {level}: "
                    f"{source_cfg_path} -> {cfg_path}"
                )
            record.update(
                {
                    "source": str(source),
                    "source_size_bytes": source.stat().st_size,
                    "source_sha256": source_hash,
                    "source_target_match": True,
                    "source_cfg_args": str(source_cfg_path),
                    "source_cfg_args_sha256": source_cfg_hash,
                    "source_target_cfg_match": True,
                }
            )
        records.append(record)
    reference_config = cohort_configs[0]
    for level, config in enumerate(cohort_configs[1:], start=2):
        if config != reference_config:
            raise ProtocolError(
                f"checkpoint cfg_args are not one cohort after excluding "
                f"feature_level/model_path; level 1 != level {level}"
            )
    return records


def _validate_inputs(args: argparse.Namespace) -> list[dict[str, object]]:
    source = args.data_root / args.scene
    label_root = args.label_root / args.scene
    if not (source / "sparse" / "0" / "images.bin").exists():
        raise ProtocolError(f"missing COLMAP source for {args.scene}: {source}")
    if not label_root.exists():
        raise ProtocolError(f"missing label directory: {label_root}")
    return _checkpoint_provenance(args)


def _command(args: argparse.Namespace) -> list[str]:
    return [
        "bash",
        str(ROOT / "radio_gs" / "scripts" / "run_repo_python.sh"),
        str(args.upstream / "eval_lerf.py"),
        "-s",
        str(args.data_root / args.scene),
        "-m",
        str(args.checkpoint_root / f"{args.scene}_{args.index}_1"),
        "--dataset_name",
        args.scene,
        "--index",
        str(args.index),
        "--ckpt_root_path",
        str(args.checkpoint_root),
        "--output_dir",
        str(args.output_root),
        "--mask_thresh",
        str(args.mask_thresh),
        "--json_folder",
        str(args.label_root),
        "--checkpoint",
        str(args.checkpoint),
        "--include_feature",
        "--topk",
        str(args.topk),
        "--quick_render",
        "--quiet",
    ]


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "RADIO_GS_LD_LIBRARY_PATH": str(args.ld_library_path),
            "RADIO_GS_SITE_PACKAGES": str(args.site_packages),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(args.upstream),
                    str(args.upstream / "submodules" / "segment-anything-langsplat"),
                ]
            ),
        }
    )
    if args.python is not None:
        environment["RADIO_GS_PYTHON"] = str(args.python)
    return environment


def build_arg_parser() -> argparse.ArgumentParser:
    runtime_root = ROOT / "output" / "protocol_audit_20260731" / "runtime"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("/root/baselines/LangSplatV2"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/pool/sqy/3d_understanding/lerf_ovs"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("/mnt/pool/sqy/3d_understanding/lerf_ovs/label"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "output" / "baselines" / "langsplatv2" / "lerf_compat_20260518",
    )
    parser.add_argument(
        "--checkpoint-source-root",
        type=Path,
        default=None,
        help=(
            "When --checkpoint-root is a local staging tree, hash the corresponding "
            "source checkpoints here and require exact source/target equality."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "output"
            / "protocol_audit_20260731"
            / "langsplatv2_lerf2d_view_fix"
        ),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--mask-thresh", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument(
        "--ld-library-path",
        type=Path,
        default=runtime_root / "libcuda535",
    )
    parser.add_argument(
        "--site-packages",
        default=os.pathsep.join(
            [
                str(runtime_root / "python_site"),
                str(runtime_root / "langsplatv2_site"),
                "/root/miniconda3/envs/iclpose/lib/python3.9/site-packages",
            ]
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-recorded-audit-checkout",
        action="store_true",
        help=(
            "Accept only the exact extra runtime/visualization diff hashes recorded "
            "for /root/baselines/LangSplatV2 on 2026-07-31."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    provenance = _validate_upstream(
        args.upstream,
        allow_recorded_audit_checkout=args.allow_recorded_audit_checkout,
    )
    checkpoints = _validate_inputs(args)
    command = _command(args)
    environment = _environment(args)
    locked_command = ["flock", str(LOCK_PATH), "-c", shlex.join(command)]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "protocol": {
            "camera_mapping": "exact label-image stem across train+test camera union",
            "mixed_camera_roles_allowed": True,
            "mask_threshold": args.mask_thresh,
            "checkpoint": args.checkpoint,
            "topk": args.topk,
            "quick_render": True,
        },
        "provenance": provenance,
        "checkpoints": checkpoints,
        "command": locked_command,
        "cwd": str(args.upstream),
        "dry_run": args.dry_run,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / f"{args.scene}_launcher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(shlex.join(locked_command))
    if args.dry_run:
        return 0
    subprocess.run(
        locked_command,
        check=True,
        cwd=args.upstream,
        env=environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
