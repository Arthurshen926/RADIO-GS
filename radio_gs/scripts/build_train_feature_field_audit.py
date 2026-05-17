"""Build a reproducibility audit for the train_feature_field entry point."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "radio_gs" / "scripts" / "train_feature_field.py"
DEFAULT_MARKDOWN = REPO_ROOT / "output" / "radio_gs" / "reports" / "train_feature_field_audit.md"
DEFAULT_JSON = REPO_ROOT / "output" / "radio_gs" / "reports" / "train_feature_field_audit.json"
DEFAULT_LATEX = REPO_ROOT / "paper" / "train_feature_field_audit_table.tex"
TRAINING_SUPPORT_GLOB = REPO_ROOT / "radio_gs" / "training"


REQUIRED_MANIFEST_SYMBOLS = {
    "_write_run_manifest",
    "_get_git_metadata",
    "_artifact_paths",
    "_append_metrics_history",
    "_write_experiment_report",
}
SPLIT_TOKENS = {
    "resolve_split_feature_dir",
    "resolve_split_frame_ids",
    "resolve_split_pose_source",
    "train_frame_ids",
    "val_frame_ids",
}
TRAINING_LOCK_TOKENS = {"_acquire_training_lock", "_release_training_lock", "training.lock"}
TEST_PATTERNS = {
    "frame_order_and_direct_point_visibility": ("direct_point", "sample_multiview"),
    "split_config_generation": ("train_frame_ids_path", "val_frame_ids_path"),
    "provenance_freeze": ("verify_submission_provenance", "submission_freeze"),
    "checkpoint_io": ("load_trusted_checkpoint", "checkpoint"),
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _extract_defined_symbols(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _load_support_texts(script_path: Path) -> tuple[str, list[str]]:
    """Return importable training-module text used by the repository entry point."""
    if script_path.resolve() != DEFAULT_TRAIN_SCRIPT.resolve():
        return "", []
    if not TRAINING_SUPPORT_GLOB.exists():
        return "", []
    parts: list[str] = []
    paths: list[str] = []
    for path in sorted(TRAINING_SUPPORT_GLOB.glob("*.py")):
        parts.append(path.read_text(encoding="utf-8"))
        paths.append(_rel(path))
    return "\n".join(parts), paths


def _status(all_present: bool) -> str:
    return "pass" if all_present else "missing"


def _check(
    check_id: str,
    *,
    status: str,
    severity: str,
    evidence: str,
    recommendation: str,
) -> dict[str, str]:
    return {
        "id": check_id,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _scan_tests(tests_root: Path) -> list[dict[str, Any]]:
    if not tests_root.exists():
        return [
            {
                "id": name,
                "status": "missing",
                "evidence": "tests root not found",
                "matched_files": [],
            }
            for name in TEST_PATTERNS
        ]
    test_files = sorted(tests_root.glob("test_*.py"))
    payload: list[dict[str, Any]] = []
    for name, patterns in TEST_PATTERNS.items():
        matches: list[str] = []
        for path in test_files:
            text = path.read_text(encoding="utf-8")
            if any(pattern in text for pattern in patterns):
                matches.append(_rel(path))
        payload.append(
            {
                "id": name,
                "status": "pass" if matches else "missing",
                "evidence": f"{len(matches)} matching test files",
                "matched_files": matches,
            }
        )
    return payload


def analyze_train_script(
    train_script: str | Path = DEFAULT_TRAIN_SCRIPT,
    *,
    tests_root: str | Path = REPO_ROOT / "tests",
    line_risk_threshold: int = 4000,
) -> dict[str, Any]:
    script_path = Path(train_script)
    text = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
    support_text, support_paths = _load_support_texts(script_path)
    analysis_text = text + "\n" + support_text
    symbols = _extract_defined_symbols(analysis_text)
    line_count = _line_count(text)
    checks: list[dict[str, str]] = []

    checks.append(
        _check(
            "script_size",
            status="risk" if line_count > int(line_risk_threshold) else "pass",
            severity="medium",
            evidence=f"{line_count} lines; threshold {int(line_risk_threshold)}",
            recommendation="Split training data, losses, trainer loop, and checkpointing into importable modules before release.",
        )
    )

    missing_manifest = sorted(REQUIRED_MANIFEST_SYMBOLS - symbols)
    checks.append(
        _check(
            "run_manifest",
            status=_status(not missing_manifest),
            severity="high",
            evidence="present" if not missing_manifest else f"missing {missing_manifest}",
            recommendation="Keep run_manifest, git metadata, artifact paths, experiment report, and metrics history in every training run.",
        )
    )

    missing_split_tokens = sorted(token for token in SPLIT_TOKENS if token not in analysis_text)
    checks.append(
        _check(
            "split_resolution",
            status=_status(not missing_split_tokens),
            severity="high",
            evidence="train/val split tokens present" if not missing_split_tokens else f"missing {missing_split_tokens}",
            recommendation="Keep explicit train/val feature, pose, and frame-id resolution to prevent view/feature leakage.",
        )
    )

    trusted = "load_trusted_checkpoint" in analysis_text
    checks.append(
        _check(
            "trusted_checkpoint_io",
            status="pass" if trusted else "missing",
            severity="high",
            evidence="load_trusted_checkpoint referenced" if trusted else "load_trusted_checkpoint not referenced",
            recommendation="Load model checkpoints through the trusted checkpoint helper; keep raw torch.load limited to feature/text/cache tensors.",
        )
    )

    missing_lock_tokens = sorted(token for token in TRAINING_LOCK_TOKENS if token not in text)
    checks.append(
        _check(
            "training_lock",
            status=_status(not missing_lock_tokens),
            severity="medium",
            evidence="lock acquisition/release present" if not missing_lock_tokens else f"missing {missing_lock_tokens}",
            recommendation="Keep per-output training lock to prevent concurrent writers from corrupting run artifacts.",
        )
    )

    raw_torch_load_count = analysis_text.count("torch.load(")
    tensor_cache_loader = "load_training_tensor_cache" in analysis_text
    wrapped_tensor_cache_loads = tensor_cache_loader and raw_torch_load_count <= 1
    raw_load_status = "pass" if raw_torch_load_count == 0 or wrapped_tensor_cache_loads else "risk"
    if wrapped_tensor_cache_loads:
        raw_load_evidence = (
            f"{raw_torch_load_count} raw torch.load site behind or replaced by "
            "load_training_tensor_cache"
        )
    else:
        raw_load_evidence = f"{raw_torch_load_count} raw torch.load sites"
    checks.append(
        _check(
            "raw_tensor_load_sites",
            status=raw_load_status,
            severity="medium",
            evidence=raw_load_evidence,
            recommendation="Document each raw torch.load as feature/text/cache tensor loading or move it behind typed loader helpers.",
        )
    )

    test_coverage = _scan_tests(Path(tests_root))
    if any(row["status"] == "missing" for row in checks):
        overall = "missing"
    elif any(row["status"] == "risk" for row in checks) or any(row["status"] == "missing" for row in test_coverage):
        overall = "risk"
    else:
        overall = "pass"

    open_items = []
    if line_count > int(line_risk_threshold):
        open_items.append("Split train_feature_field.py into data/loss/trainer/checkpoint modules.")
    if raw_load_status == "risk":
        open_items.append("Wrap or document raw torch.load feature/text/cache sites.")
    if any(row["status"] == "missing" for row in test_coverage):
        open_items.append("Add missing leakage/provenance tests listed in test_coverage.")

    return {
        "script_path": _rel(script_path),
        "line_count": line_count,
        "support_modules": support_paths,
        "overall_status": overall,
        "checks": checks,
        "test_coverage": test_coverage,
        "open_items": open_items,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Train Feature Field Audit",
        "",
        f"- Script: `{summary.get('script_path', '')}`",
        f"- Line count: `{summary.get('line_count', 0)}`",
        f"- Overall status: `{summary.get('overall_status', 'unknown')}`",
        "",
        "## Static Checks",
        "",
        "| Check | Status | Severity | Evidence | Recommendation |",
        "|---|---|---|---|---|",
    ]
    for row in summary.get("checks", []):
        lines.append(
            "| {id} | {status} | {severity} | {evidence} | {recommendation} |".format(
                id=row.get("id", ""),
                status=row.get("status", ""),
                severity=row.get("severity", ""),
                evidence=str(row.get("evidence", "")).replace("|", "/"),
                recommendation=str(row.get("recommendation", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Test Coverage Signals",
            "",
            "| Coverage target | Status | Evidence | Matched files |",
            "|---|---|---|---|",
        ]
    )
    for row in summary.get("test_coverage", []):
        lines.append(
            "| {id} | {status} | {evidence} | {files} |".format(
                id=row.get("id", ""),
                status=row.get("status", ""),
                evidence=str(row.get("evidence", "")).replace("|", "/"),
                files=", ".join(row.get("matched_files", [])),
            )
        )
    lines.extend(["", "## Open Items", ""])
    for item in summary.get("open_items", []):
        lines.append(f"- {item}")
    if not summary.get("open_items"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Auditability checks for the training entry point. Status `risk' means the guard exists but needs release hardening.}",
        "\\label{tab:train_feature_field_audit}",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Check & Status & Severity & Evidence \\\\",
        "\\midrule",
    ]
    for row in summary.get("checks", []):
        lines.append(
            "{id} & {status} & {severity} & {evidence} \\\\".format(
                id=_latex_escape(str(row.get("id", ""))),
                status=_latex_escape(str(row.get("status", ""))),
                severity=_latex_escape(str(row.get("severity", ""))),
                evidence=_latex_escape(str(row.get("evidence", ""))),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    markdown_path: str | Path = DEFAULT_MARKDOWN,
    json_path: str | Path = DEFAULT_JSON,
    latex_path: str | Path = DEFAULT_LATEX,
) -> dict[str, Path]:
    markdown_out = Path(markdown_path)
    json_out = Path(json_path)
    latex_out = Path(latex_path)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    latex_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(build_markdown(summary), encoding="utf-8")
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latex_out.write_text(build_latex_table(summary), encoding="utf-8")
    return {"markdown": markdown_out, "json": json_out, "latex": latex_out}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_script", default=str(DEFAULT_TRAIN_SCRIPT))
    parser.add_argument("--tests_root", default=str(REPO_ROOT / "tests"))
    parser.add_argument("--line_risk_threshold", type=int, default=4000)
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_LATEX))
    args = parser.parse_args(argv)

    summary = analyze_train_script(
        args.train_script,
        tests_root=args.tests_root,
        line_risk_threshold=args.line_risk_threshold,
    )
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
