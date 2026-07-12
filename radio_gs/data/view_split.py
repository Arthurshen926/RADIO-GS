"""Fail-closed image-view inclusion/exclusion helpers.

Promptable novel-view benchmarks sometimes reserve an RGB frame as the
evaluation target.  These helpers keep that exclusion identical across RGB
3DGS training and frozen-feature extraction.  Matching is deliberately by the
exact basename stem; fuzzy or numeric-nearest matching would make a leaked
target hard to notice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class ViewSplitError(ValueError):
    """Raised when an image split is incomplete, ambiguous, or unsafe."""


def _validate_stem(value: object) -> str:
    stem = str(value).strip()
    if not stem:
        raise ViewSplitError("Image stems must be non-empty")
    if Path(stem).name != stem or "/" in stem or "\\" in stem:
        raise ViewSplitError(f"Image stem must not contain a path: {stem!r}")
    if Path(stem).suffix:
        raise ViewSplitError(
            f"Expected a basename stem without an extension, got {stem!r}"
        )
    return stem


def _stems_from_payload(payload: object, *, source: Path) -> list[str]:
    if isinstance(payload, Mapping):
        for key in ("excluded_image_stems", "exclude_image_stems", "frame_ids"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ViewSplitError(
                f"{source} must contain a list or an excluded_image_stems field"
            )
    if not isinstance(payload, list):
        raise ViewSplitError(f"{source} must contain a JSON list of image stems")
    return [_validate_stem(item) for item in payload]


def load_excluded_image_stems(
    explicit: Sequence[str] | None = None,
    source_file: str | Path | None = None,
) -> tuple[str, ...]:
    """Merge repeated CLI stems with an optional JSON/text stem file."""

    values = [_validate_stem(item) for item in (explicit or ())]
    if source_file:
        path = Path(source_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Excluded-view file not found: {path}")
        if path.suffix.lower() == ".json":
            values.extend(
                _stems_from_payload(
                    json.loads(path.read_text(encoding="utf-8")), source=path
                )
            )
        else:
            text = path.read_text(encoding="utf-8")
            values.extend(
                _validate_stem(token)
                for token in re.split(r"[\s,]+", text)
                if token.strip()
            )
    return tuple(dict.fromkeys(values))


def select_image_indices(
    image_paths: Sequence[str | Path],
    excluded_stems: Iterable[str],
    *,
    min_remaining: int = 1,
) -> tuple[list[int], list[str]]:
    """Return retained indices and excluded filenames after exact validation.

    Every requested exclusion must match exactly one input basename stem.  A
    duplicate source stem or an unknown requested stem is fatal.
    """

    paths = [Path(path) for path in image_paths]
    by_stem: dict[str, int] = {}
    for index, path in enumerate(paths):
        stem = path.stem
        if stem in by_stem:
            raise ViewSplitError(
                f"Duplicate input image stem {stem!r}: "
                f"{paths[by_stem[stem]]} and {path}"
            )
        by_stem[stem] = index

    requested = tuple(dict.fromkeys(_validate_stem(item) for item in excluded_stems))
    unknown = sorted(set(requested) - set(by_stem))
    if unknown:
        raise ViewSplitError(
            f"Excluded image stems were not found by exact match: {unknown}"
        )
    excluded_set = set(requested)
    retained = [index for index, path in enumerate(paths) if path.stem not in excluded_set]
    if len(retained) < int(min_remaining):
        raise ViewSplitError(
            f"View exclusion leaves {len(retained)} images; at least "
            f"{int(min_remaining)} are required"
        )
    excluded_names = [path.name for path in paths if path.stem in excluded_set]
    return retained, excluded_names


__all__ = [
    "ViewSplitError",
    "load_excluded_image_stems",
    "select_image_indices",
]
