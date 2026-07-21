#!/usr/bin/env python3
"""Build and freeze ScanNet-PFIR-Small from official ScanNet assets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from radio_gs.benchmarks.scannet_pfir.protocol import (
    BENCHMARK_VERSION,
    ProtocolConfig,
    build_scene_records,
    discover_frames,
    freeze_manifest,
    sha256_file,
)
from radio_gs.benchmarks.scannet_pfir.split.select_scene_subset import (
    read_scene_split,
    select_scene_subset,
)


def _values(raw: str) -> list[str]:
    path = Path(raw)
    if raw and not any(character.isspace() for character in raw) and path.is_file():
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return [value for value in raw.replace(",", " ").split() if value]


def find_scene_annotations(
    scene_id: str, roots: Iterable[str | Path]
) -> tuple[Path, Path, Path]:
    searched: list[Path] = []
    for source_root in roots:
        root = Path(source_root)
        scene_roots = (root / scene_id, root)
        for scene_root in scene_roots:
            annotation_roots = (scene_root / "instance_annotations", scene_root)
            mesh_candidates = (
                scene_root / f"{scene_id}_vh_clean_2.ply",
                scene_root / f"{scene_id}_vh_clean_2.labels.ply",
            )
            for annotation_root in annotation_roots:
                aggregation = annotation_root / f"{scene_id}.aggregation.json"
                segmentation_candidates = (
                    annotation_root / f"{scene_id}_vh_clean_2.0.010000.segs.json",
                    annotation_root / f"{scene_id}_vh_clean_2.segs.json",
                )
                searched.extend((aggregation, *mesh_candidates, *segmentation_candidates))
                mesh = next((path for path in mesh_candidates if path.is_file()), None)
                segmentation = next(
                    (path for path in segmentation_candidates if path.is_file()), None
                )
                if mesh is not None and aggregation.is_file() and segmentation is not None:
                    return mesh, aggregation, segmentation
    raise FileNotFoundError(
        f"{scene_id}: incomplete mesh/aggregation/segments under "
        + ", ".join(map(str, roots))
    )


def readiness(records: list[dict], reports: list[dict]) -> dict:
    return {
        "valid_query_count": len(records),
        "valid_scene_count": sum(report.get("status") == "ok" for report in reports),
        "same_category_query_count": sum(
            int(record.get("same_category_distractor_count", 0)) > 0
            for record in records
        ),
        "non_structural_category_count": len(
            {int(record["nyu40_class_id"]) for record in records}
        ),
        "requirements": {
            "queries_at_least_200": len(records) >= 200,
            "scenes_at_least_20": sum(
                report.get("status") == "ok" for report in reports
            )
            >= 20,
            "same_category_queries_at_least_50": sum(
                int(record.get("same_category_distractor_count", 0)) > 0
                for record in records
            )
            >= 50,
            "categories_at_least_10": len(
                {int(record["nyu40_class_id"]) for record in records}
            )
            >= 10,
        },
    }


def _build_scene_job(job: tuple) -> tuple[list[dict], dict]:
    (
        scene_id,
        frames_root,
        mesh,
        aggregation,
        segmentation,
        config,
    ) = job
    return build_scene_records(
        scene_id=scene_id,
        frames_root=frames_root,
        mesh_path=mesh,
        aggregation_path=aggregation,
        segmentation_path=segmentation,
        config=config,
    )


def run(args: argparse.Namespace) -> dict:
    frames_root = Path(args.frames_root)
    annotation_roots = [Path(value) for value in args.annotations_root]
    if args.split_file:
        candidates = read_scene_split(args.split_file)
    else:
        candidates = _values(args.scene_names)
    if not candidates:
        raise ValueError("--split-file or --scene-names must provide scenes")
    excluded = _values(args.exclude_scenes)
    finite_frame_counts: dict[str, int] = {}
    eligible_candidates: list[str] = []
    for scene_id in candidates:
        scene_dir = frames_root / scene_id
        if not scene_dir.is_dir():
            finite_frame_counts[scene_id] = 0
            continue
        try:
            count = len(discover_frames(scene_dir))
        except FileNotFoundError:
            count = 0
        finite_frame_counts[scene_id] = count
        if count >= int(args.minimum_finite_frames):
            eligible_candidates.append(scene_id)
    selected = select_scene_subset(
        eligible_candidates,
        count=args.maximum_scenes,
        seed=args.seed,
        excluded_scenes=excluded,
        excluded_spaces=_values(args.exclude_spaces),
    )
    config = ProtocolConfig(
        depth_stride=args.depth_stride,
        min_instance_surface_coverage=args.minimum_surface_coverage,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    reports: list[dict] = []
    processed_valid_scenes = 0
    selection_metadata = {
        "source_split_file": (
            str(Path(args.split_file).resolve()) if args.split_file else None
        ),
        "source_split_file_sha256": (
            sha256_file(args.split_file) if args.split_file else None
        ),
        "candidate_scene_count": len(candidates),
        "finite_frame_eligible_scene_count": len(eligible_candidates),
        "minimum_finite_frames": int(args.minimum_finite_frames),
        "scene_hash_seed": int(args.seed),
        "unique_physical_spaces": True,
        "excluded_scenes": excluded,
        "excluded_spaces": _values(args.exclude_spaces),
    }
    jobs: list[tuple] = []
    precomputed_failures: dict[str, dict] = {}
    for scene_id in selected:
        if not (frames_root / scene_id).is_dir():
            precomputed_failures[scene_id] = {
                "scene_id": scene_id,
                "status": "missing_frames",
            }
            continue
        try:
            mesh, aggregation, segmentation = find_scene_annotations(
                scene_id, annotation_roots
            )
            jobs.append(
                (
                    scene_id,
                    frames_root,
                    mesh,
                    aggregation,
                    segmentation,
                    config,
                )
            )
        except FileNotFoundError as error:
            precomputed_failures[scene_id] = {
                "scene_id": scene_id,
                "status": "unavailable",
                "reason": str(error),
            }

    if int(args.workers) > 1:
        executor = ProcessPoolExecutor(max_workers=int(args.workers))
        futures = {
            job[0]: executor.submit(_build_scene_job, job)
            for job in jobs
        }
        results = (
            (scene_id, futures[scene_id])
            for scene_id in selected
            if scene_id in futures
        )
    else:
        executor = None
        results = (
            (job[0], _build_scene_job(job))
            for job in jobs
        )
    try:
        result_by_scene: dict[str, tuple[list[dict], dict]] = {}
        for scene_id, result in results:
            try:
                result_by_scene[scene_id] = (
                    result.result() if int(args.workers) > 1 else result
                )
            except ValueError as error:
                precomputed_failures[scene_id] = {
                    "scene_id": scene_id,
                    "status": "invalid_annotations",
                    "reason": str(error),
                }
        for scene_id in selected:
            if scene_id in precomputed_failures:
                reports.append(precomputed_failures[scene_id])
                continue
            if scene_id not in result_by_scene:
                continue
            scene_records, report = result_by_scene[scene_id]
            records.extend(scene_records)
            reports.append(report)
            if report.get("status") == "ok":
                processed_valid_scenes += 1
            partial = {
                "benchmark_version": BENCHMARK_VERSION,
                "split_role": args.split_role,
                "protocol_config": asdict(config),
                "queries": records,
                "scene_reports": reports,
                "scene_selection": selection_metadata,
            }
            (output / "construction.partial.json").write_text(
                json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8"
            )
            # Start with 20 scenes; expand only when fewer than 200 queries survive.
            if (
                processed_valid_scenes >= args.initial_scenes
                and len(records) >= args.target_queries
            ):
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    status = readiness(records, reports)
    if args.split_role == "test" and not args.allow_incomplete:
        failed = [key for key, value in status["requirements"].items() if not value]
        if failed:
            raise RuntimeError(
                "formal test benchmark is incomplete: "
                + ", ".join(failed)
                + "; use --allow-incomplete only for a named pilot"
            )
    release = freeze_manifest(
        records,
        output,
        split_role=args.split_role,
        scene_reports=reports,
        config=config,
        selection_metadata=selection_metadata,
    )
    report = {
        **release,
        "readiness": status,
        "selected_scenes": selected,
        "incomplete_pilot": not all(status["requirements"].values()),
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-root",
        default="/mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k",
    )
    parser.add_argument("--annotations-root", action="append", required=True)
    parser.add_argument("--split-file", default="")
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--exclude-scenes", default="")
    parser.add_argument("--exclude-spaces", default="")
    parser.add_argument("--split-role", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-scenes", type=int, default=20)
    parser.add_argument("--maximum-scenes", type=int, default=30)
    parser.add_argument("--target-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--minimum-surface-coverage", type=float, default=0.70)
    parser.add_argument("--minimum-finite-frames", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
