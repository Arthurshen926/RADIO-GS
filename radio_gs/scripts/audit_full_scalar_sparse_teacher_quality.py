#!/usr/bin/env python3
"""Publish a caller-SHA-bound source-only sparse teacher quality audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces.full_scalar_sparse_teacher_quality_audit import (
    build_quality_audit,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
)


def _optional_graph(args: argparse.Namespace) -> tuple[object | None, str | None]:
    path = getattr(args, "support_graph", None)
    expected = getattr(args, "expected_support_graph_sha256", None)
    if (path is None) != (expected is None):
        raise ValueError(
            "--support-graph and --expected-support-graph-sha256 are an exact pair"
        )
    if path is None:
        return None, None
    value, observed, _ = load_torch_mapping(
        path,
        expected_sha256=expected,
        map_location="cpu",
        label="quality audit support graph",
    )
    return value, observed


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber sparse teacher quality audit: {output}")
    accepted, accepted_sha, _ = load_torch_mapping(
        args.accepted_region_authority,
        expected_sha256=args.expected_accepted_region_authority_sha256,
        map_location="cpu",
        label="quality audit AcceptedV2 authority",
    )
    teacher, teacher_sha, _ = load_torch_mapping(
        args.teacher_observation_authority,
        expected_sha256=args.expected_teacher_observation_authority_sha256,
        map_location="cpu",
        label="quality audit teacher authority",
    )
    graph, graph_sha = _optional_graph(args)
    payload = build_quality_audit(
        accepted_value=accepted,
        accepted_file_sha256=accepted_sha,
        teacher_value=teacher,
        teacher_file_sha256=teacher_sha,
        support_graph_value=graph,
        support_graph_file_sha256=graph_sha,
    )
    write_frozen_json(output, payload)
    return {
        "status": "materialized",
        "scene_id": payload["scene_id"],
        "conclusion_gate": payload["conclusion_gate"],
        "output": file_record(output),
        "outputs_written": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument(
        "--expected-accepted-region-authority-sha256", required=True
    )
    parser.add_argument("--teacher-observation-authority", required=True)
    parser.add_argument(
        "--expected-teacher-observation-authority-sha256", required=True
    )
    parser.add_argument("--support-graph")
    parser.add_argument("--expected-support-graph-sha256")
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
