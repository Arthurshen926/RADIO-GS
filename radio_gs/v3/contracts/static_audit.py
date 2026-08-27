"""AST/static audit preventing historical or benchmark logic entering v3."""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_SCENE_TOKENS = ("figurines", "ramen", "teatime", "waldo_kitchen", "scene0000")
FORBIDDEN_IMPORT_PREFIXES = (
    "radio_gs.querying",
    "radio_gs.models.query_native_",
    "radio_gs.scripts",
)


def audit_v3_tree(root: str | Path) -> list[str]:
    base = Path(root)
    failures: list[str] = []
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base)
        if relative.parts and relative.parts[0] == "legacy_adapter":
            continue
        source = path.read_text(encoding="utf-8")
        if relative != Path("contracts/static_audit.py"):
            lowered = source.lower()
            for token in FORBIDDEN_SCENE_TOKENS:
                if token in lowered:
                    failures.append(f"{relative}: benchmark scene token {token!r}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if (
                relative != Path("contracts/static_audit.py")
                and isinstance(node, ast.Attribute)
                and node.attr == "local_codes"
            ):
                failures.append(
                    f"{relative}: internal pre-fusion local_codes access is forbidden"
                )
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name.startswith(prefix) for prefix in FORBIDDEN_IMPORT_PREFIXES):
                    failures.append(f"{relative}: forbidden historical import {name}")
    return failures


__all__ = ["audit_v3_tree"]
