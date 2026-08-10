#!/usr/bin/env python3
"""Remove verified empty view records from a sparse responsibility manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
)


SCHEMA = "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
VIEW_SCHEMA = "radio_gs.sparse_exact_marginal_responsibility_view.v1"


def canonicalize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refuses to clobber canonical responsibility authority: {output}"
        )
    payload_raw, source_sha, source = load_json_object(
        args.input,
        expected_sha256=args.expected_input_sha256,
        label="sparse responsibility authority",
    )
    if not isinstance(payload_raw, Mapping):
        raise ValueError("sparse responsibility authority must be a mapping")
    payload = dict(payload_raw)
    views = payload.get("views")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("schema_version") != 1
        or not isinstance(views, list)
        or not views
    ):
        raise ValueError("sparse responsibility authority header differs")
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    last: tuple[int, int] | None = None
    retained_hits = 0
    for raw in views:
        if not isinstance(raw, Mapping):
            raise ValueError("responsibility view record must be a mapping")
        record = dict(raw)
        key = (int(record["frame_index"]), int(record["view_index"]))
        if last is not None and key <= last:
            raise ValueError("responsibility view records are not strictly ordered")
        last = key
        count = int(record["num_hits"])
        if count < 0:
            raise ValueError("responsibility view declares a negative hit count")
        if count > 0:
            retained.append(record)
            retained_hits += count
            continue
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("empty responsibility view path is not relative")
        view_path = (source.parent / relative).resolve()
        if source.parent not in view_path.parents:
            raise ValueError("empty responsibility view escapes authority root")
        view_raw, observed, _ = load_torch_mapping(
            view_path,
            expected_sha256=str(record["sha256"]),
            map_location="cpu",
            label="empty responsibility view",
        )
        if not isinstance(view_raw, Mapping):
            raise ValueError("empty responsibility view must be a mapping")
        gaussian = torch.as_tensor(view_raw.get("gaussian_ids"))
        pixels = torch.as_tensor(view_raw.get("pixel_ids"))
        weights = torch.as_tensor(view_raw.get("base_weights"))
        if (
            view_raw.get("schema") != VIEW_SCHEMA
            or view_raw.get("schema_version") != 1
            or int(view_raw.get("frame_index", -1)) != key[0]
            or int(view_raw.get("view_index", -1)) != key[1]
            or gaussian.numel() != 0
            or pixels.numel() != 0
            or weights.numel() != 0
        ):
            raise ValueError("zero-hit record does not bind an empty view payload")
        removed.append(
            {
                "frame_index": key[0],
                "view_index": key[1],
                "relative_path": relative.as_posix(),
                "sha256": observed,
            }
        )
    if len(removed) != int(args.expected_removed_views):
        raise ValueError("unexpected number of empty responsibility views")
    if retained_hits != int(payload.get("total_hits", -1)):
        raise ValueError("retained nonempty hits differ from authority total")
    payload["views"] = retained
    payload["frame_indices"] = [int(record["frame_index"]) for record in retained]
    written = write_frozen_json(output, payload)
    return {
        "status": "sparse_responsibility_nonempty_manifest_complete",
        "source": {"path": str(source), "sha256": source_sha},
        "output": str(written),
        "retained_views": len(retained),
        "removed_views": removed,
        "total_hits": retained_hits,
        "access_audit": {
            "benchmark_queries_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "target_metrics_computed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-removed-views", type=int, required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(canonicalize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
