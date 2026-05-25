#!/usr/bin/env python3
"""Guard paper-facing claims against selector and leaderboard drift."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import yaml


CONTEXT_TABLE = Path("paper/lerf_direct_3d_context_table.tex")
VPR_PROTOCOL_CARD = Path("paper/vpr_protocol_card.tex")
SELECTION_TABLE = Path("paper/lerf_direct_3d_selection_table.tex")
SCANNET_CONTEXT_TABLE = Path("paper/scannet_published_context_table.tex")
FINAL_ROWS = Path("paper/artifacts/final_rows.yaml")
NARRATIVE_PATHS = (
    Path("paper/radio_gs_draft.tex"),
    Path("docs/paper_draft_current.md"),
    Path("docs/submission_status.md"),
)
DEFAULT_PATHS = (
    CONTEXT_TABLE,
    VPR_PROTOCOL_CARD,
    SELECTION_TABLE,
    SCANNET_CONTEXT_TABLE,
    FINAL_ROWS,
    *NARRATIVE_PATHS,
)

MEAN_STD_RE = re.compile(r"mean\s*\+\s*(?:2\.5\s*)?std", re.IGNORECASE)
MEAN_2P5_STD_RE = re.compile(r"mean\s*\+\s*2\.5\s*std", re.IGNORECASE)
DANGEROUS_CLAIM_PATTERNS = (
    (
        "full ScanNet semantic segmentation leaderboard",
        re.compile(r"\bfull\s+ScanNet\s+semantic\s+segmentation\s+leaderboard\b", re.IGNORECASE),
    ),
    (
        "standard ScanNet semantic segmentation leaderboard",
        re.compile(
            r"\b(?:fully\s+)?standard\s+ScanNet\s+semantic\s+segmentation\s+leaderboard\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ScanNet SOTA",
        re.compile(r"\bScanNet\b.{0,80}\bSOTA\b", re.IGNORECASE),
    ),
    (
        "global SOTA",
        re.compile(r"\bglobal\b.{0,80}\bSOTA\b", re.IGNORECASE),
    ),
    (
        "direct 3D SOTA",
        re.compile(r"\bdirect[- ]3D\b.{0,80}\bSOTA\b", re.IGNORECASE),
    ),
    (
        "pure 3D segmentation SOTA",
        re.compile(r"\bpure\s+3D\s+segmentation\b.{0,80}\bSOTA\b", re.IGNORECASE),
    ),
    (
        "primitive-level SOTA",
        re.compile(r"\bprimitive-level\b.{0,80}\bSOTA\b", re.IGNORECASE),
    ),
    (
        "exact OpenGaFF ScanNet reproduction",
        re.compile(r"\bexact\s+OpenGaFF\s+ScanNet\s+reproduction\b", re.IGNORECASE),
    ),
    (
        "OpenGaFF-style ScanNet setting without subset qualifier",
        re.compile(
            r"\bunder\s+the\s+OpenGaFF-style\s+point-cloud\s+understanding\s+setting\b",
            re.IGNORECASE,
        ),
    ),
)
SAFE_CONTEXT_RE = re.compile(
    r"\b("
    r"not|cannot|avoid|avoids|prevent|prevents|preventing|overclaim|overclaiming|"
    r"caveat|diagnostic|pending|rather\s+than|must\s+stay|should\s+remain|"
    r"not\s+presented|not\s+as"
    r")\b",
    re.IGNORECASE,
)
CAGS_CONTEXT_RE = re.compile(r"\bCAGS\b|\\cite\{sun2025cags\}")


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_texts(root: Path, paths: Iterable[str | Path], issues: list[str]) -> dict[Path, str]:
    texts: dict[Path, str] = {}
    for path in paths:
        rel_path = Path(path)
        resolved = _resolve(root, rel_path)
        try:
            texts[rel_path] = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(f"{rel_path}: cannot read file: {exc}")
    return texts


def _check_vpr_context_row(text: str, issues: list[str]) -> None:
    if CAGS_CONTEXT_RE.search(text):
        issues.append(f"{CONTEXT_TABLE}: CAGS must not be promoted in the OpenGaFF-aligned context table")
    vpr_rows = [line.strip() for line in text.splitlines() if r"\method{} + VPR" in line]
    if not vpr_rows:
        issues.append(f"{CONTEXT_TABLE}: missing VPR context row")
        return
    for line in vpr_rows:
        if "thr0p25" not in line:
            issues.append(f"{CONTEXT_TABLE}: VPR context row must use fixed thr0p25: {line}")
        if MEAN_STD_RE.search(line):
            issues.append(f"{CONTEXT_TABLE}: VPR context row must not promote mean+std: {line}")


def _check_vpr_protocol_card(text: str, issues: list[str]) -> None:
    selection_lines = [line.strip() for line in text.splitlines() if "Selection" in line]
    if not selection_lines:
        issues.append(f"{VPR_PROTOCOL_CARD}: missing Selection row")
        return
    for line in selection_lines:
        lowered = line.lower()
        if "thr0p25" not in line or "fixed global" not in lowered:
            issues.append(f"{VPR_PROTOCOL_CARD}: Selection row must state fixed global thr0p25: {line}")
        if MEAN_2P5_STD_RE.search(line):
            issues.append(f"{VPR_PROTOCOL_CARD}: Selection row must not cite mean+2.5std: {line}")
        if "mean+std" in lowered and "diagnostic only" not in lowered:
            issues.append(f"{VPR_PROTOCOL_CARD}: mean+std must be marked diagnostic only: {line}")


def _check_selection_table(text: str, issues: list[str]) -> None:
    if MEAN_2P5_STD_RE.search(text):
        issues.append(f"{SELECTION_TABLE}: promoted selection table must not cite mean+2.5std")
    if not re.search(r"CTF-GS\s*&\s*thr0p25\s+mIoU\b", text):
        issues.append(f"{SELECTION_TABLE}: missing CTF-GS thr0p25 mIoU row")
    if not re.search(r"CTF-GS\s*&\s*thr0p25\s+Acc@0\.25\b", text):
        issues.append(f"{SELECTION_TABLE}: missing CTF-GS thr0p25 Acc@0.25 row")


def _check_scannet_context_table(text: str, issues: list[str]) -> None:
    if CAGS_CONTEXT_RE.search(text):
        issues.append(f"{SCANNET_CONTEXT_TABLE}: CAGS must not be promoted in the OpenGaFF-aligned ScanNet table")
    required_snippets = (
        "LangSplatV2 & 14.75 & 25.47 & 17.09 & 35.68 & 22.83 & 41.52",
        "VALA & 32.11 & 50.05 & 35.10 & 54.77 & 46.21 & 65.61",
        "\\method{} DINO-CV contextual kNN + spatial propagation & \\textbf{38.06} & \\textbf{61.29} & \\textbf{38.71} & \\textbf{63.15} & \\textbf{47.11} & \\textbf{72.00}",
    )
    for snippet in required_snippets:
        if snippet not in text:
            issues.append(f"{SCANNET_CONTEXT_TABLE}: missing synced row snippet: {snippet}")


def _check_final_rows(text: str, issues: list[str]) -> None:
    try:
        payload = yaml.safe_load(text)
        policy = payload["tracks"]["t2_lerf_direct_3d_selection"]["protocol"][
            "main_selector_policy"
        ]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        issues.append(f"{FINAL_ROWS}: cannot read T2 main_selector_policy: {exc}")
        return
    if "thr0p25" not in str(policy) or MEAN_STD_RE.search(str(policy)):
        issues.append(f"{FINAL_ROWS}: T2 main_selector_policy must remain fixed thr0p25: {policy}")


def _line_window(lines: list[str], index: int) -> str:
    start = max(0, index - 2)
    end = min(len(lines), index + 2)
    return " ".join(lines[start:end])


def _check_overclaims(path: Path, text: str, issues: list[str]) -> None:
    lines = text.splitlines()
    reported: set[tuple[int, str]] = set()
    for index, line in enumerate(lines):
        for label, pattern in DANGEROUS_CLAIM_PATTERNS:
            if not pattern.search(line):
                continue
            window = _line_window(lines, index)
            key = (index + 1, label)
            if key in reported:
                continue
            reported.add(key)
            if SAFE_CONTEXT_RE.search(window):
                continue
            issues.append(f"{path}:{index + 1}: unqualified {label} claim: {line.strip()}")


def validate_claims(
    *,
    root: str | Path = ".",
    paths: Iterable[str | Path] | None = None,
) -> list[str]:
    root_path = Path(root)
    selected_paths = tuple(Path(path) for path in paths) if paths is not None else DEFAULT_PATHS
    issues: list[str] = []
    texts = _read_texts(root_path, selected_paths, issues)

    if CONTEXT_TABLE in texts:
        _check_vpr_context_row(texts[CONTEXT_TABLE], issues)
    if VPR_PROTOCOL_CARD in texts:
        _check_vpr_protocol_card(texts[VPR_PROTOCOL_CARD], issues)
    if SELECTION_TABLE in texts:
        _check_selection_table(texts[SELECTION_TABLE], issues)
    if SCANNET_CONTEXT_TABLE in texts:
        _check_scannet_context_table(texts[SCANNET_CONTEXT_TABLE], issues)
    if FINAL_ROWS in texts:
        _check_final_rows(texts[FINAL_ROWS], issues)
    for path, text in texts.items():
        if path in NARRATIVE_PATHS:
            _check_overclaims(path, text, issues)
    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional paper-facing paths to validate instead of the default set",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    issues = validate_claims(root=args.root, paths=args.paths or None)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)
    print("paper claims ok")


if __name__ == "__main__":
    main()
