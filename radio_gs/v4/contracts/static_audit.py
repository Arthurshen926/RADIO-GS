"""Static isolation audit for the clean v4 namespace."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


FORBIDDEN_IMPORT_PREFIXES = ("radio_gs.v1", "radio_gs.v2", "radio_gs.v3")
HISTORICAL_NAMESPACES = frozenset({"v1", "v2", "v3"})
QUARANTINED_MODULE_NAMES = frozenset({"lerf_development_pipeline"})
FORBIDDEN_METHOD_TERMS = (
    "anchor expansion",
    "connected component",
    "graph propagation",
)
CONCRETE_SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}", re.IGNORECASE)
CONCRETE_LERF_SCENE_PATTERN = re.compile(
    r"\b(?:figurines|ramen|teatime|waldo(?:[_ -]kitchen)?)\b", re.IGNORECASE
)
FORBIDDEN_DYNAMIC_CALLS = frozenset({"__import__", "eval", "exec"})


def _has_forbidden_prefix(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)


def _assignment_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [target.id for target in targets if isinstance(target, ast.Name)]
    return []


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path}: syntax error: {error}")
            continue
        dynamic_module_aliases = {"builtins", "importlib"}
        dynamic_callable_aliases = set(FORBIDDEN_DYNAMIC_CALLS) | {"import_module"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"builtins", "importlib"}:
                        dynamic_module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if (module, alias.name) in {
                        ("builtins", "__import__"),
                        ("builtins", "eval"),
                        ("builtins", "exec"),
                        ("importlib", "import_module"),
                    }:
                        dynamic_callable_aliases.add(alias.asname or alias.name)
        # Resolve simple alias chains to a fixed point.  Attribute names such as
        # ``model.eval()`` are common and harmless; only attributes rooted at a
        # proven builtins/importlib alias are dynamic-code entry points.
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                assignment_value = None
                if isinstance(node, ast.Assign):
                    assignment_value = node.value
                elif isinstance(node, ast.AnnAssign):
                    assignment_value = node.value
                if assignment_value is None:
                    continue
                assigned_names = set(_assignment_names(node))
                if (
                    isinstance(assignment_value, ast.Name)
                    and assignment_value.id in dynamic_module_aliases
                ):
                    additions = assigned_names - dynamic_module_aliases
                    dynamic_module_aliases.update(additions)
                    changed = changed or bool(additions)
                aliases_dynamic = (
                    isinstance(assignment_value, ast.Name)
                    and assignment_value.id in dynamic_callable_aliases
                ) or (
                    isinstance(assignment_value, ast.Attribute)
                    and isinstance(assignment_value.value, ast.Name)
                    and assignment_value.value.id in dynamic_module_aliases
                    and assignment_value.attr in FORBIDDEN_DYNAMIC_CALLS | {"import_module"}
                )
                if (
                    isinstance(assignment_value, ast.Call)
                    and isinstance(assignment_value.func, ast.Name)
                    and assignment_value.func.id == "getattr"
                    and len(assignment_value.args) >= 2
                    and isinstance(assignment_value.args[0], ast.Name)
                    and assignment_value.args[0].id in dynamic_module_aliases
                    and isinstance(assignment_value.args[1], ast.Constant)
                    and assignment_value.args[1].value
                    in FORBIDDEN_DYNAMIC_CALLS | {"import_module"}
                ):
                    aliases_dynamic = True
                if aliases_dynamic:
                    additions = assigned_names - dynamic_callable_aliases
                    dynamic_callable_aliases.update(additions)
                    changed = changed or bool(additions)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if _has_forbidden_prefix(name):
                    failures.append(f"{path}:{node.lineno}: forbidden historical-method import {name}")
                if (
                    name.split(".")[-1] in QUARANTINED_MODULE_NAMES
                    and path.stem not in QUARANTINED_MODULE_NAMES
                ):
                    failures.append(
                        f"{path}:{node.lineno}: dependency on quarantined development module is forbidden"
                    )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_roots = {alias.name.split(".", 1)[0] for alias in node.names}
                historical_relative = node.level and (
                    module.split(".", 1)[0] in HISTORICAL_NAMESPACES
                    or (not module and bool(imported_roots & HISTORICAL_NAMESPACES))
                )
                historical_from_root = module == "radio_gs" and bool(
                    imported_roots & HISTORICAL_NAMESPACES
                )
                if historical_relative or historical_from_root:
                    failures.append(
                        f"{path}:{node.lineno}: relative/root historical-method import is forbidden"
                    )
                imports_quarantined = (
                    module.split(".")[-1] in QUARANTINED_MODULE_NAMES
                    or bool(imported_roots & QUARANTINED_MODULE_NAMES)
                )
                if imports_quarantined and path.stem not in QUARANTINED_MODULE_NAMES:
                    failures.append(
                        f"{path}:{node.lineno}: dependency on quarantined development module is forbidden"
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dynamic_callable_aliases:
                    failures.append(
                        f"{path}:{node.lineno}: dynamic code/import call {node.func.id} is forbidden"
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in dynamic_module_aliases
                    and node.func.attr in FORBIDDEN_DYNAMIC_CALLS | {"import_module"}
                ):
                    failures.append(
                        f"{path}:{node.lineno}: dynamic code/import attribute call is forbidden"
                    )
                if (
                    isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "getattr"
                    and len(node.func.args) >= 2
                    and isinstance(node.func.args[0], ast.Name)
                    and node.func.args[0].id in dynamic_module_aliases
                    and isinstance(node.func.args[1], ast.Constant)
                    and node.func.args[1].value in FORBIDDEN_DYNAMIC_CALLS | {"import_module"}
                ):
                    failures.append(
                        f"{path}:{node.lineno}: getattr-based dynamic code/import call is forbidden"
                    )
        lowered = source.lower()
        if path.name != "static_audit.py" and (
            CONCRETE_SCENE_PATTERN.search(source)
            or CONCRETE_LERF_SCENE_PATTERN.search(source)
        ):
            failures.append(f"{path}: concrete benchmark scene identity is forbidden in v4 source")
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
