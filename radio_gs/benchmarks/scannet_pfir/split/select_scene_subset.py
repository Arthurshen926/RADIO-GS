"""Deterministically select development or official-validation test scenes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def read_scene_split(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def select_scene_subset(
    scenes: Iterable[str],
    *,
    count: int,
    seed: int = 20260718,
    excluded_scenes: Iterable[str] = (),
    excluded_spaces: Iterable[str] = (),
    unique_spaces: bool = True,
) -> list[str]:
    """Stable hash ordering avoids dependence on filesystem or method output."""

    excluded = set(map(str, excluded_scenes))
    excluded_space_set = set(map(str, excluded_spaces))
    candidates = sorted(
        {
            str(scene)
            for scene in scenes
            if str(scene) not in excluded
            and str(scene).split("_")[0] not in excluded_space_set
        }
    )
    ordered = sorted(
        candidates,
        key=lambda scene: (
            hashlib.sha256(f"{int(seed)}:{scene}".encode()).hexdigest(),
            scene,
        ),
    )
    if not unique_spaces:
        return ordered[: int(count)]
    selected: list[str] = []
    seen_spaces: set[str] = set()
    for scene in ordered:
        space = scene.split("_")[0]
        if space in seen_spaces:
            continue
        selected.append(scene)
        seen_spaces.add(space)
        if len(selected) >= int(count):
            break
    return selected
