#!/usr/bin/env python3
"""Build fixed VALA/OpenGaFF ScanNet-8 aggregate reports from evaluator JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALA8_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)
SCAN_SPLITS = ("19", "15", "10")
PROTOCOL_ARG_KEYS = (
    "prepared_root",
    "config",
    "checkpoint",
    "class_splits",
    "query_mode",
    "k",
    "candidate_k",
    "max_points",
    "opacity_filter_mode",
    "logit_calibration",
    "logit_calibration_alpha",
    "gaussian_index_position_mode",
    "prompt_templates",
    "compact_feature_key",
    "text_embedding_cache",
    "projection_weights",
    "summary_head_weights",
    "radio_checkpoint",
    "use_summary_head",
    "use_point_summary_adapter",
    "point_summary_adapter_blend_alpha",
)


def _round4(value: float) -> float:
    return round(float(value), 4)


def _source_args(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("source_args", payload.get("args", {}))
    return args if isinstance(args, dict) else {}


def _canonical_protocol_value(value: Any, scenes: tuple[str, ...] = VALA8_SCENES) -> str:
    text = "None" if value is None else str(value)
    for scene in scenes:
        text = text.replace(scene, "{scene}")
    return text


def _protocol_signature(payload: dict[str, Any]) -> dict[str, str]:
    args = _source_args(payload)
    return {
        key: _canonical_protocol_value(args.get(key))
        for key in PROTOCOL_ARG_KEYS
        if key in args
    }


def _parse_expected_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"expected non-empty KEY in {raw!r}")
    return key, value


def _validate_expected_args(args: dict[str, Any], expected: dict[str, str]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for key, value in expected.items():
        actual = _canonical_protocol_value(args.get(key))
        target = _canonical_protocol_value(value)
        if actual != target:
            raise ValueError(f"source arg {key!r} mismatch: got {actual!r}, expected {target!r}")
        checked[key] = actual
    return checked


def _assert_compatible_protocols(payloads: list[dict[str, Any]]) -> dict[str, str]:
    if not payloads:
        return {}
    reference = _protocol_signature(payloads[0])
    for index, payload in enumerate(payloads[1:], start=2):
        current = _protocol_signature(payload)
        for key in sorted(set(reference) | set(current)):
            if reference.get(key) != current.get(key):
                raise ValueError(
                    f"input {index} protocol arg {key!r} differs: "
                    f"{current.get(key)!r} != {reference.get(key)!r}"
                )
    return reference


def _scene_split_metrics(scene_payload: dict[str, Any], split: str) -> dict[str, float]:
    metrics = scene_payload["splits"][split]
    return {
        "miou": float(metrics["miou"]),
        "macc": float(metrics["macc"]),
    }


def build_vala8_summary(
    payload: dict[str, Any],
    *,
    scenes: tuple[str, ...] = VALA8_SCENES,
    label: str,
    require_exact_scene_set: bool = False,
    expected_source_args: dict[str, str] | None = None,
) -> dict[str, Any]:
    all_scenes = payload.get("scenes", {})
    missing = [scene for scene in scenes if scene not in all_scenes]
    if missing:
        raise KeyError(f"Missing required VALA8 scenes: {', '.join(missing)}")
    if require_exact_scene_set:
        extras = sorted(set(all_scenes) - set(scenes))
        if extras:
            raise ValueError(f"Unexpected non-VALA8 scenes in paper-facing report: {', '.join(extras)}")

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_payload = all_scenes[scene]
        row = {"scene": scene}
        for split in SCAN_SPLITS:
            row[split] = _scene_split_metrics(scene_payload, split)
        rows.append(row)

    macro = {
        split: {
            "miou": _round4(sum(row[split]["miou"] for row in rows) / len(rows)),
            "macc": _round4(sum(row[split]["macc"] for row in rows) / len(rows)),
        }
        for split in SCAN_SPLITS
    }
    source_args = _source_args(payload)
    expected_checks = _validate_expected_args(source_args, expected_source_args or {})
    return {
        "label": label,
        "protocol": "VALA/OpenGaFF ScanNet-8 candidate split",
        "scene_count": len(rows),
        "scenes": list(scenes),
        "source_timestamp": payload.get("timestamp"),
        "source_args": source_args,
        "source_protocol_checks": {
            "require_exact_scene_set": bool(require_exact_scene_set),
            "protocol_signature": _protocol_signature(payload),
            "expected_source_args": expected_checks,
        },
        "macro": macro,
        "rows": rows,
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {summary['label']}",
        "",
        f"Protocol: {summary['protocol']}",
        f"Scenes: {', '.join(summary['scenes'])}",
        "",
        "| Split | mIoU | mAcc |",
        "|---|---:|---:|",
    ]
    for split in SCAN_SPLITS:
        metrics = summary["macro"][split]
        lines.append(f"| {split} classes | {metrics['miou']:.4f} | {metrics['macc']:.4f} |")
    lines.extend(["", "## Per-Scene", "", "| Scene | split19 | split15 | split10 |", "|---|---:|---:|---:|"])
    for row in summary["rows"]:
        lines.append(
            "| {scene} | {m19:.4f}/{a19:.4f} | {m15:.4f}/{a15:.4f} | {m10:.4f}/{a10:.4f} |".format(
                scene=row["scene"],
                m19=row["19"]["miou"],
                a19=row["19"]["macc"],
                m15=row["15"]["miou"],
                a15=row["15"]["macc"],
                m10=row["10"]["miou"],
                a10=row["10"]["macc"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One combined ScanNet evaluator JSON or multiple per-scene evaluator JSONs",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--require_exact_scene_set",
        action="store_true",
        help="Fail if the input JSON contains scenes outside the fixed VALA/OpenGaFF-8 list.",
    )
    parser.add_argument(
        "--expect_arg",
        action="append",
        default=[],
        type=_parse_expected_arg,
        metavar="KEY=VALUE",
        help="Require a source/evaluator arg value before writing the paper-facing report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inputs = [Path(path) for path in args.input]
    payload = json.loads(inputs[0].read_text(encoding="utf-8"))
    if len(inputs) > 1:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
        _assert_compatible_protocols(payloads)
        merged_scenes: dict[str, Any] = {}
        for item in payloads:
            merged_scenes.update(item.get("scenes", {}))
        payload = {
            "timestamp": payload.get("timestamp"),
            "args": _source_args(payload),
            "scenes": merged_scenes,
        }
    summary = build_vala8_summary(
        payload,
        label=args.label,
        require_exact_scene_set=args.require_exact_scene_set,
        expected_source_args=dict(args.expect_arg),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(f"Saved {output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
