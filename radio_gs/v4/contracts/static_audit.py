"""Static isolation audit for the clean v4 namespace."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


FORBIDDEN_IMPORT_PREFIXES = ("radio_gs.v3",)
FORBIDDEN_METHOD_TERMS = (
    "anchor expansion",
    "connected component",
    "graph propagation",
)
CONCRETE_SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}", re.IGNORECASE)


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path}: syntax error: {error}")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    failures.append(f"{path}:{node.lineno}: forbidden v3 method import {name}")
        lowered = source.lower()
        if CONCRETE_SCENE_PATTERN.search(source):
            failures.append(f"{path}: concrete scene identity is forbidden in v4 source")
        for term in FORBIDDEN_METHOD_TERMS:
            if term in lowered and path.name != "static_audit.py":
                failures.append(f"{path}: forbidden historical method term {term!r}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).parents[1])
    args = parser.parse_args()
    failures = audit(Path(args.root))
    if failures:
        raise SystemExit("\n".join(failures))
    print("v4 static isolation audit passed")


if __name__ == "__main__":
    main()
