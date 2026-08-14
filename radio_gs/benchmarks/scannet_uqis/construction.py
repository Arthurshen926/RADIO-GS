"""Method-independent construction primitives for ScanNet-UQIS.

These functions may inspect evaluator-private annotations while constructing a
release, but return only deterministic geometry/query facts.  They never read
model predictions or benchmark metrics.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import csv
import itertools
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


VIEW_INDEPENDENT_VALUES = frozenset(
    {"indep", "independent", "view_independent", "view-independent"}
)

# Exact lexical rule shipped by ReferIt3D in
# ``referit3d.analysis.utterances.is_explicitly_view_dependent``.  Raw Nr3D
# does not contain a derived ``dep_or_indep`` column, so formal construction
# applies this rule to the released token list and records the rule identity.
REFERIT3D_EXPLICIT_VIEW_DEPENDENT_TOKENS = frozenset(
    {
        "front",
        "behind",
        "back",
        "right",
        "left",
        "facing",
        "leftmost",
        "rightmost",
        "looking",
        "across",
    }
)
REFERIT3D_VIEW_DEPENDENCE_RULE = (
    "referit3d.analysis.utterances.is_explicitly_view_dependent@"
    "725a5d31b8ff3647b6fb5342304093e9568d1341"
)


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def load_reference_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read Nr3D/ScanRefer-style JSON, JSONL, or CSV annotations."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{source}:{number}: annotation must be an object")
            rows.append(dict(value))
        return rows
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        for key in ("annotations", "records", "data"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{source}: expected a JSON array of annotation objects")
    return [dict(row) for row in value]


def _first(row: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def _nr3d_tokens(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Decode the token field in the official Nr3D CSV without executing it."""

    raw = row.get("tokens", row.get("token"))
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as error:
            raise ValueError("malformed Nr3D tokens") from error
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(token, str) for token in raw
    ):
        raise ValueError("Nr3D tokens must be a list of strings")
    return tuple(token.strip().lower() for token in raw if token.strip())


def nr3d_view_independence(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Return the official ReferIt3D lexical classification and rule identity.

    Raw Nr3D rows are authoritative when their token list is present.  The
    explicit dependence column remains supported for already-derived fixtures,
    but formal raw-data construction cannot silently infer from utterance text.
    """

    tokens = _nr3d_tokens(row)
    if tokens is not None:
        dependent = bool(
            set(tokens).intersection(REFERIT3D_EXPLICIT_VIEW_DEPENDENT_TOKENS)
        )
        return not dependent, REFERIT3D_VIEW_DEPENDENCE_RULE
    dependence = str(
        _first(row, ("dep_or_indep", "view_dependence", "view_dependency"), "")
    ).strip().lower()
    if not dependence:
        raise ValueError("Nr3D row has neither tokens nor view-dependence label")
    return dependence in VIEW_INDEPENDENT_VALUES, "explicit_annotation_field"


def _scanrefer_class_is_mentioned(row: Mapping[str, Any]) -> bool:
    object_name = str(row.get("object_name", "")).strip().lower().replace("_", " ")
    expression = str(_first(row, ("description", "utterance", "expression"), ""))
    normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    target = normalize(object_name)
    text = f" {normalize(expression)} "
    return bool(target and f" {target} " in text)


def select_view_independent_expression(
    rows: Iterable[Mapping[str, Any]],
    *,
    scene_id: str,
    official_instance_id: int,
    minimum_tokens: int = 2,
    maximum_tokens: int = 64,
) -> dict[str, Any]:
    """Choose one natural expression without consulting method output.

    ScanNet aggregation represents an official instance as ``objectId + 1``;
    Nr3D/ReferIt3D annotations use the zero-based ``target_id``.  The returned
    record explicitly preserves that conversion.
    """

    object_id = int(official_instance_id) - 1
    if object_id < 0:
        raise ValueError("official_instance_id must be positive")
    eligible: list[dict[str, Any]] = []
    for raw in rows:
        row_scene = str(
            _first(raw, ("scan_id", "scene_id", "scanId", "scene"), "")
        )
        if row_scene != str(scene_id):
            continue
        try:
            row_object = int(
                _first(raw, ("target_id", "object_id", "targetId", "objectId"))
            )
        except (TypeError, ValueError):
            continue
        if row_object != object_id:
            continue
        declared_source = str(_first(raw, ("source", "dataset"), "")).strip().lower()
        is_scanrefer = declared_source == "scanrefer" or (
            "object_name" in raw and "description" in raw and "assignmentid" not in raw
        )
        if is_scanrefer:
            if not _scanrefer_class_is_mentioned(raw):
                continue
            qualification_rule = "scanrefer_bound_target_and_object_name_mention"
        else:
            if not _as_bool(
                _first(raw, ("correct_guess", "is_correct", "correct"), None),
                default=False,
            ):
                continue
            if not _as_bool(
                _first(
                    raw,
                    ("mentions_target_class", "mentions_target", "target_class_mentioned"),
                    None,
                ),
                default=False,
            ):
                continue
            qualification_rule = "nr3d_correct_guess_and_mentions_target_class"
        try:
            view_independent, dependence_rule = nr3d_view_independence(raw)
        except ValueError:
            continue
        if not view_independent:
            continue
        expression = str(
            _first(raw, ("utterance", "description", "expression", "text"), "")
        ).strip()
        token_count = len(expression.split())
        if not minimum_tokens <= token_count <= maximum_tokens:
            continue
        if any(ord(character) < 32 and character not in "\t\n" for character in expression):
            continue
        annotation_id = str(
            _first(raw, ("ann_id", "annotation_id", "assignmentid", "id"), "")
        )
        eligible.append(
            {
                "scene_id": str(scene_id),
                "object_id": object_id,
                "official_instance_id": int(official_instance_id),
                "expression": expression,
                "annotation_id": annotation_id,
                "source": str(_first(raw, ("source", "dataset"), "nr3d")),
                "view_independent": True,
                "view_dependence_rule": dependence_rule,
                "qualification_rule": qualification_rule,
            }
        )
    if not eligible:
        raise ValueError(
            f"{scene_id}/objectId={object_id}: no valid view-independent expression"
        )

    def annotation_key(record: Mapping[str, Any]) -> tuple[int, int | str, str]:
        value = str(record["annotation_id"])
        try:
            return 0, int(value), str(record["expression"])
        except ValueError:
            return 1, value, str(record["expression"])

    return min(eligible, key=annotation_key)


def select_profiled_expression(
    rows: Iterable[Mapping[str, Any]],
    *,
    scene_id: str,
    official_instance_id: int,
    minimum_tokens: int = 2,
    maximum_tokens: int = 64,
) -> dict[str, Any]:
    """Select the v0.2 canonical text and classify its evaluation cohort.

    The released Nr3D ``uses_spatial_lang`` annotation is mandatory here.
    A target enters the Unified-Query Core Cohort when at least one otherwise
    eligible expression is explicitly non-spatial; deterministic selection
    then happens only inside that non-spatial set.  Targets without such an
    expression retain a valid relational utterance in the separate challenge.
    """

    profiled: list[tuple[dict[str, Any], bool]] = []
    for raw in rows:
        declared_source = str(_first(raw, ("source", "dataset"), "nr3d")).strip().lower()
        if declared_source not in {"", "nr3d"} or "uses_spatial_lang" not in raw:
            continue
        try:
            selected = select_view_independent_expression(
                [raw],
                scene_id=scene_id,
                official_instance_id=official_instance_id,
                minimum_tokens=minimum_tokens,
                maximum_tokens=maximum_tokens,
            )
            spatial = _as_bool(raw.get("uses_spatial_lang"), default=True)
        except ValueError:
            continue
        profiled.append((selected, spatial))
    if not profiled:
        raise ValueError(
            f"{scene_id}/objectId={int(official_instance_id) - 1}: "
            "no valid Nr3D expression with spatial-language evidence"
        )

    nonspatial = [record for record, spatial in profiled if not spatial]
    candidates = nonspatial or [record for record, _spatial in profiled]

    def annotation_key(record: Mapping[str, Any]) -> tuple[int, int | str, str]:
        value = str(record["annotation_id"])
        try:
            return 0, int(value), str(record["expression"])
        except ValueError:
            return 1, value, str(record["expression"])

    selected = dict(min(candidates, key=annotation_key))
    relational = not bool(nonspatial)
    selected.update(
        {
            "evaluation_tier": (
                "relational_text_challenge" if relational else "unified_core"
            ),
            "relational_language_required": relational,
            "spatial_language_evidence": "nr3d.uses_spatial_lang",
        }
    )
    return selected


@dataclass(frozen=True)
class QueryFrameCover:
    frame_ids: tuple[str, ...]
    target_to_frame: Mapping[int, str]
    target_scores: Mapping[int, float]


def _frame_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def recompute_union_frame_exclusion(
    frame_ids: Sequence[str],
    camera_to_world_by_frame: Mapping[str, np.ndarray],
    query_frame_ids: Iterable[str],
    *,
    temporal_radius: int = 5,
    translation_m: float = 0.10,
    rotation_deg: float = 8.0,
) -> tuple[str, ...]:
    """Recompute the formal query-frame union exclusion on the full frame order.

    ``frame_ids`` must be the complete sensor sequence, not a sparsely exported
    projection subset.  Temporal neighbors and pose-near frames are unioned
    across every query camera before any mapping observation is exposed.
    """

    ordered = tuple(map(str, frame_ids))
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or int(temporal_radius) < 0
        or float(translation_m) < 0
        or float(rotation_deg) < 0
    ):
        raise ValueError("full frame order and exclusion thresholds are invalid")
    if set(camera_to_world_by_frame) != set(ordered):
        raise ValueError("pose inventory must exactly cover the full frame order")
    poses: dict[str, np.ndarray] = {}
    for frame_id in ordered:
        pose = np.asarray(camera_to_world_by_frame[frame_id], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"{frame_id}: camera pose must have shape [4,4]")
        poses[frame_id] = pose
    queries = tuple(map(str, query_frame_ids))
    if not queries or len(set(queries)) != len(queries) or not set(queries).issubset(ordered):
        raise ValueError("query cameras must be a distinct subset of the full frame order")

    index = {frame_id: position for position, frame_id in enumerate(ordered)}
    excluded: set[str] = set()
    for query_id in queries:
        if not np.isfinite(poses[query_id]).all():
            raise ValueError(f"{query_id}: query camera pose is not finite")
        position = index[query_id]
        excluded.update(
            ordered[
                max(0, position - int(temporal_radius)) :
                min(len(ordered), position + int(temporal_radius) + 1)
            ]
        )
        query_pose = poses[query_id]
        for frame_id in ordered:
            pose = poses[frame_id]
            if not np.isfinite(pose).all():
                continue
            translation = float(np.linalg.norm(query_pose[:3, 3] - pose[:3, 3]))
            relative = query_pose[:3, :3].T @ pose[:3, :3]
            cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            rotation = float(np.degrees(np.arccos(cosine)))
            if translation < float(translation_m) and rotation < float(rotation_deg):
                excluded.add(frame_id)
    return tuple(frame_id for frame_id in ordered if frame_id in excluded)


def select_query_frame_cover(
    target_frame_scores: Mapping[int, Mapping[str, float]],
    *,
    maximum_frames: int = 3,
) -> QueryFrameCover:
    """Find a deterministic minimum-cardinality target-cover of query frames.

    Targets are normally six to eight, so frames are first compressed by their
    target-coverage bit mask.  This leaves at most ``2**targets - 1`` candidates
    and permits an exact search up to the frozen three-frame limit.
    """

    targets = sorted(int(value) for value in target_frame_scores)
    if not targets or maximum_frames <= 0:
        raise ValueError("targets and maximum_frames must be non-empty/positive")
    target_index = {target: index for index, target in enumerate(targets)}
    scores: dict[int, dict[str, float]] = {}
    frames: set[str] = set()
    for target in targets:
        current: dict[str, float] = {}
        for frame_id, raw_score in target_frame_scores[target].items():
            score = float(raw_score)
            if not np.isfinite(score):
                raise ValueError("query-frame scores must be finite")
            current[str(frame_id)] = score
            frames.add(str(frame_id))
        if not current:
            raise ValueError(f"target {target} has no eligible query frame")
        scores[target] = current

    representatives: dict[int, tuple[float, str]] = {}
    for frame_id in frames:
        mask = 0
        utility = 0.0
        for target in targets:
            if frame_id in scores[target]:
                mask |= 1 << target_index[target]
                utility += scores[target][frame_id]
        if mask == 0:
            continue
        previous = representatives.get(mask)
        candidate = (utility, frame_id)
        if previous is None or utility > previous[0] or (
            utility == previous[0] and _frame_key(frame_id) < _frame_key(previous[1])
        ):
            representatives[mask] = candidate

    candidates = sorted(
        [(mask, utility, frame_id) for mask, (utility, frame_id) in representatives.items()],
        key=lambda item: _frame_key(item[2]),
    )
    full = (1 << len(targets)) - 1
    best: tuple[float, tuple[tuple[int, int | str], ...], tuple[str, ...]] | None = None
    for count in range(1, min(int(maximum_frames), len(candidates)) + 1):
        for combination in itertools.combinations(candidates, count):
            covered = 0
            for mask, _utility, _frame_id in combination:
                covered |= mask
            if covered != full:
                continue
            frame_ids = tuple(
                sorted((item[2] for item in combination), key=_frame_key)
            )
            utility = float(sum(item[1] for item in combination))
            key = (-utility, tuple(_frame_key(value) for value in frame_ids), frame_ids)
            if best is None or key < best:
                best = key
        if best is not None:
            break
    if best is None:
        raise ValueError(
            f"no query-frame cover exists within the {maximum_frames}-frame limit"
        )
    selected = best[2]
    assignment: dict[int, str] = {}
    assigned_scores: dict[int, float] = {}
    for target in targets:
        eligible = [frame for frame in selected if frame in scores[target]]
        assignment[target] = min(
            eligible,
            key=lambda frame: (-scores[target][frame], _frame_key(frame)),
        )
        assigned_scores[target] = float(scores[target][assignment[target]])
    return QueryFrameCover(selected, assignment, assigned_scores)


def select_interior_pixel(
    target_mask: np.ndarray,
    *,
    valid_depth: np.ndarray | None = None,
    valid_correspondence: np.ndarray | None = None,
) -> tuple[tuple[int, int], float]:
    """Select the deepest legal target-mask interior with ``(v,u)`` tie-break."""

    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or not bool(mask.any()):
        raise ValueError("target_mask must be a non-empty 2-D mask")
    legal = mask.copy()
    for name, value in (
        ("valid_depth", valid_depth),
        ("valid_correspondence", valid_correspondence),
    ):
        if value is None:
            continue
        constraint = np.asarray(value, dtype=bool)
        if constraint.shape != mask.shape:
            raise ValueError(f"{name} must align with target_mask")
        legal &= constraint
    candidates = np.argwhere(legal)
    if not len(candidates):
        raise ValueError("target mask has no legal interior prompt pixel")
    distance = distance_transform_edt(mask)
    values = distance[candidates[:, 0], candidates[:, 1]]
    maximum = float(values.max())
    row = candidates[np.flatnonzero(values == maximum)[0]]
    v, u = int(row[0]), int(row[1])
    return (u, v), maximum


def backproject_world_point(
    pixel_uv: Sequence[int | float],
    depth_m: float,
    camera_intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Back-project one depth-aligned raster pixel into world coordinates."""

    uv = np.asarray(pixel_uv, dtype=np.float64)
    intrinsic = np.asarray(camera_intrinsics, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    depth = float(depth_m)
    if uv.shape != (2,) or intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("pixel/K/camera_to_world shapes must be [2], [3,3], [4,4]")
    if not (
        np.isfinite(uv).all()
        and np.isfinite(intrinsic).all()
        and np.isfinite(pose).all()
        and np.isfinite(depth)
        and depth > 0
        and intrinsic[0, 0] != 0
        and intrinsic[1, 1] != 0
    ):
        raise ValueError("back-projection inputs must be finite and non-degenerate")
    u, v = uv
    camera = np.array(
        [
            (u - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
            1.0,
        ],
        dtype=np.float64,
    )
    world = pose @ camera
    if abs(float(world[3])) <= 1e-12:
        raise ValueError("camera_to_world produced an invalid homogeneous point")
    return (world[:3] / world[3]).astype(np.float32)


def derive_paired_point_prompt(
    target_mask: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    valid_correspondence: np.ndarray | None = None,
    depth_scale: float = 1000.0,
) -> dict[str, Any]:
    """Derive the one-click 2-D prompt and its exact world-space counterpart."""

    mask = np.asarray(target_mask, dtype=bool)
    depth_values = np.asarray(depth)
    if depth_values.shape != mask.shape or depth_scale <= 0:
        raise ValueError("depth must align with target_mask and depth_scale be positive")
    valid_depth = np.isfinite(depth_values) & (depth_values > 0)
    pixel, interior_distance = select_interior_pixel(
        mask,
        valid_depth=valid_depth,
        valid_correspondence=valid_correspondence,
    )
    u, v = pixel
    depth_m = float(depth_values[v, u]) / float(depth_scale)
    point = backproject_world_point(
        pixel, depth_m, camera_intrinsics, camera_to_world
    )
    return {
        "positive_pixel_uv": [u, v],
        "click_depth_m": depth_m,
        "point_world_xyz": [float(value) for value in point],
        "interior_distance_px": interior_distance,
        "selection_rule": "max_2d_distance_transform_legal_depth_correspondence_vu_tie_break",
    }


def align_depth_to_color_raster(
    depth: np.ndarray,
    depth_intrinsics: np.ndarray,
    color_intrinsics: np.ndarray,
    color_size: Sequence[int],
    *,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """Z-buffer ScanNet depth samples into the registered color raster."""

    values = np.asarray(depth)
    kd = np.asarray(depth_intrinsics, dtype=np.float64)
    kc = np.asarray(color_intrinsics, dtype=np.float64)
    if values.ndim != 2 or kd.shape != (3, 3) or kc.shape != (3, 3):
        raise ValueError("depth/K shapes must be [H,W], [3,3], [3,3]")
    if len(color_size) != 2 or min(map(int, color_size)) <= 0 or depth_scale <= 0:
        raise ValueError("color_size/depth_scale is invalid")
    height, width = values.shape
    yy, xx = np.mgrid[0:height, 0:width]
    z = values.astype(np.float64) / float(depth_scale)
    valid = np.isfinite(z) & (z > 0.0)
    x = (xx - kd[0, 2]) * z / kd[0, 0]
    y = (yy - kd[1, 2]) * z / kd[1, 1]
    u = np.rint(kc[0, 0] * x / np.maximum(z, 1e-12) + kc[0, 2]).astype(np.int64)
    v = np.rint(kc[1, 1] * y / np.maximum(z, 1e-12) + kc[1, 2]).astype(np.int64)
    color_width, color_height = map(int, color_size)
    valid &= (u >= 0) & (v >= 0) & (u < color_width) & (v < color_height)
    flat_index = v[valid] * color_width + u[valid]
    flat_depth = z[valid]
    # Stable depth sort followed by first-per-pixel implements a deterministic
    # nearest-surface z-buffer when several depth pixels round to one RGB pixel.
    order = np.argsort(flat_depth, kind="stable")
    flat_index, flat_depth = flat_index[order], flat_depth[order]
    unique, first = np.unique(flat_index, return_index=True)
    aligned = np.zeros(color_width * color_height, dtype=np.float32)
    aligned[unique] = flat_depth[first].astype(np.float32)
    return aligned.reshape(color_height, color_width)


def build_image_query_crop(
    image: Image.Image,
    target_mask: np.ndarray,
    *,
    padding_fraction: float = 0.15,
    output_size: int = 224,
    fill_rgb: Sequence[int] = (0, 0, 0),
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Build a deterministic padded, background-preserving image query crop."""

    rgb = image.convert("RGB")
    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (rgb.height, rgb.width) or not bool(mask.any()):
        raise ValueError("target mask must be non-empty and align with the RGB raster")
    if not 0.0 <= float(padding_fraction) <= 1.0 or int(output_size) <= 0:
        raise ValueError("crop padding/output size is invalid")
    fill = tuple(int(value) for value in fill_rgb)
    if len(fill) != 3 or any(not 0 <= value <= 255 for value in fill):
        raise ValueError("fill_rgb must be an RGB byte triplet")
    y, x = np.nonzero(mask)
    tight_x0, tight_x1 = int(x.min()), int(x.max()) + 1
    tight_y0, tight_y1 = int(y.min()), int(y.max()) + 1
    pad_x = int(np.ceil((tight_x1 - tight_x0) * float(padding_fraction)))
    pad_y = int(np.ceil((tight_y1 - tight_y0) * float(padding_fraction)))
    box = (
        tight_x0 - pad_x,
        tight_y0 - pad_y,
        tight_x1 + pad_x,
        tight_y1 + pad_y,
    )
    width, height = box[2] - box[0], box[3] - box[1]
    canvas = Image.new("RGB", (width, height), fill)
    source_box = (
        max(0, box[0]),
        max(0, box[1]),
        min(rgb.width, box[2]),
        min(rgb.height, box[3]),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        canvas.paste(
            rgb.crop(source_box),
            (source_box[0] - box[0], source_box[1] - box[1]),
        )
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return canvas.resize((int(output_size), int(output_size)), resampling), box
