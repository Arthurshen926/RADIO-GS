#!/usr/bin/env python3
"""Prepare or serially execute the audited NVOS LUDVIG-SAM hybrid cohort.

Preparation is CPU-only. Execution is opt-in and delegates every run to
``run_ludvig_sam.py``, which pins GPU 0 and acquires the shared flock itself.
Completed scene/seed pairs are reused exactly once; failed attempts remain
visible and receive a new immutable attempt id on a later invocation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproductions.ludvig.run_ludvig_sam import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DRIVER_LIBRARY_DIR,
    DEFAULT_OUTPUT_ROOT,
    NVOS_IMAGE_SIZE,
    NVOS_SCENES,
    ProtocolError,
    _stage_nvos_pinhole_colmap,
)


COHORT_ID = "nvos_hybrid_8scene_3seed_v1"
EXPECTED_SEEDS = (0, 1, 2)
GEOMETRY_PROTOCOL = "strict_geometry_hybrid_diagnostic"
ATTEMPT_PREFIX = f"{COHORT_ID}_attempt_"
RUNNER = ROOT / "reproductions" / "ludvig" / "run_ludvig_sam.py"
AGGREGATOR = ROOT / "reproductions" / "ludvig" / "aggregate_results.py"


class CohortError(RuntimeError):
    """Raised before execution when the cohort would be ambiguous."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_scene_name(scene: str) -> str:
    return "horns" if scene.startswith("horns_") else scene


def _geometry_path(benchmark_root: Path, scene: str) -> Path:
    return (
        benchmark_root
        / "gaussfm_jobs"
        / "nvos_strict_unseen_v1"
        / "scenes"
        / scene
        / "geometry"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )


def _seed_dir(output_root: Path, scene: str, seed: int) -> Path:
    return (
        output_root
        / "nvos"
        / GEOMETRY_PROTOCOL
        / scene
        / f"seed_{seed}"
    )


def _completed_attempt(seed_dir: Path, scene: str, seed: int) -> dict[str, Any] | None:
    completed: list[dict[str, Any]] = []
    for manifest_path in sorted(seed_dir.rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        if (
            manifest.get("benchmark") != "NVOS"
            or manifest.get("scene") != scene
            or manifest.get("seed") != seed
            or manifest.get("geometry_protocol") != GEOMETRY_PROTOCOL
            or manifest.get("strict_unseen_exact_match") is not False
            or manifest.get("target_rgb_visible_during_uplifting") is not True
        ):
            raise CohortError(
                f"Completed manifest has incompatible protocol fields: {manifest_path}"
            )
        results = list(manifest_path.parent.rglob("protocol_result.json"))
        if len(results) != 1:
            raise CohortError(
                f"Completed manifest must own exactly one result: {manifest_path}"
            )
        result = json.loads(results[0].read_text(encoding="utf-8"))
        completed.append(
            {
                "manifest": str(manifest_path),
                "result": str(results[0]),
                "selected_iou": result["selected_iou"],
                "attempt_id": manifest.get("attempt_id"),
            }
        )
    if len(completed) > 1:
        raise CohortError(
            f"Duplicate completed scene/seed {scene}/{seed}: "
            f"{[item['manifest'] for item in completed]}"
        )
    return completed[0] if completed else None


def _next_attempt_id(seed_dir: Path) -> str:
    used: set[int] = set()
    attempts_dir = seed_dir / "attempts"
    if attempts_dir.exists():
        for path in attempts_dir.iterdir():
            if path.name.startswith(ATTEMPT_PREFIX):
                suffix = path.name[len(ATTEMPT_PREFIX) :]
                if suffix.isdigit():
                    used.add(int(suffix))
    candidate = 1
    while candidate in used:
        candidate += 1
    return f"{ATTEMPT_PREFIX}{candidate}"


def _task_grid() -> list[tuple[str, int]]:
    return [(scene, seed) for scene in NVOS_SCENES for seed in EXPECTED_SEEDS]


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    planning_root = (
        output_root / "nvos" / "cohort_preflight" / COHORT_ID
    )
    camera_audits: dict[str, dict[str, Any]] = {}
    geometry: dict[str, dict[str, Any]] = {}
    for scene in NVOS_SCENES:
        source_scene = (
            benchmark_root
            / "NVOS"
            / "llff_undistorted"
            / f"{_source_scene_name(scene)}_undistort"
        )
        width, height = NVOS_IMAGE_SIZE[scene]
        camera_audits[scene] = _stage_nvos_pinhole_colmap(
            source_scene,
            planning_root / "staging" / scene,
            width,
            height,
        )
        point_cloud = _geometry_path(benchmark_root, scene)
        if not point_cloud.exists():
            raise CohortError(f"Missing strict-geometry point cloud: {point_cloud}")
        geometry[scene] = {
            "point_cloud": str(point_cloud.resolve()),
            "size_bytes": point_cloud.stat().st_size,
            "target_rgb_visible_during_training": False,
        }

    tasks: list[dict[str, Any]] = []
    for scene, seed in _task_grid():
        seed_dir = _seed_dir(output_root, scene, seed)
        completed = _completed_attempt(seed_dir, scene, seed)
        task = {
            "scene": scene,
            "seed": seed,
            "status": "complete" if completed else "pending",
            "reused_completed_attempt": completed,
            "planned_attempt_id": (
                None if completed else _next_attempt_id(seed_dir)
            ),
            "seed_dir": str(seed_dir),
        }
        tasks.append(task)

    complete_count = sum(task["status"] == "complete" for task in tasks)
    return {
        "schema_version": 1,
        "cohort_id": COHORT_ID,
        "generated_at": _utc_now(),
        "benchmark": "NVOS",
        "method": "LUDVIG-SAM",
        "protocol_id": "ludvig_official_online_multiview_v1",
        "geometry_protocol": GEOMETRY_PROTOCOL,
        "scenes": list(NVOS_SCENES),
        "seeds": list(EXPECTED_SEEDS),
        "expected_runs": len(NVOS_SCENES) * len(EXPECTED_SEEDS),
        "complete_runs": complete_count,
        "pending_runs": len(tasks) - complete_count,
        "execution_policy": {
            "serial": True,
            "gpu": 0,
            "lock": "/tmp/radio-gs-gpu0.lock",
            "lock_owned_by_per_run_wrapper": True,
            "immutable_attempts": True,
            "stop_on_error": not args.continue_on_error,
        },
        "visibility": {
            "target_rgb_visible_during_gaussian_splatting_training": False,
            "target_rgb_visible_during_uplifting": True,
            "target_view_2d_foundation_model_calls": True,
            "target_masks_scoring_only": True,
        },
        "eligibility": {
            "full_hybrid_cohort_after_completion": True,
            "three_seed_hybrid_mean_after_completion": True,
            "exact_paper_all_view_protocol": False,
            "strict_unseen_exact_match": False,
        },
        "paper_context": {
            "aggregate_iou_percent": 91.3,
            "aggregation": "seed_mean_per_scene_then_equal_weight_macro_over_8_tasks",
            "paper_geometry": "30k_original_3dgs_all_registered_views",
        },
        "camera_preflight": camera_audits,
        "geometry_preflight": geometry,
        "tasks": tasks,
    }


def _write_plan(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    temporary.replace(path)


def _refresh_counts(plan: dict[str, Any]) -> None:
    plan["complete_runs"] = sum(
        task["status"] == "complete" for task in plan["tasks"]
    )
    plan["pending_runs"] = sum(
        task["status"] == "pending" for task in plan["tasks"]
    )
    plan["failed_runs"] = sum(
        task["status"] == "failed" for task in plan["tasks"]
    )
    plan["running_runs"] = sum(
        task["status"] == "running" for task in plan["tasks"]
    )
    plan["updated_at"] = _utc_now()


def _run_command(args: argparse.Namespace, task: dict[str, Any]) -> list[str]:
    return [
        str(args.python.resolve()),
        str(RUNNER),
        "--benchmark",
        "nvos",
        "--scene",
        task["scene"],
        "--seed",
        str(task["seed"]),
        "--geometry-protocol",
        GEOMETRY_PROTOCOL,
        "--attempt-id",
        task["planned_attempt_id"],
        "--stage-nvos-pinhole",
        "--upstream",
        str(args.upstream.resolve()),
        "--python",
        str(args.python.resolve()),
        "--pythonpath",
        str(args.pythonpath.resolve()),
        "--driver-library-dir",
        str(args.driver_library_dir.resolve()),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--sam-checkpoint",
        str(args.sam_checkpoint.resolve()),
    ]


def execute_plan(args: argparse.Namespace, plan: dict[str, Any], plan_path: Path) -> None:
    for task in plan["tasks"]:
        if task["status"] == "complete":
            continue
        command = _run_command(args, task)
        task["status"] = "running"
        task["started_at"] = _utc_now()
        task["command"] = command
        _refresh_counts(plan)
        _write_plan(plan, plan_path)
        completed = subprocess.run(command, cwd=ROOT)
        task["completed_at"] = _utc_now()
        task["returncode"] = completed.returncode
        task["status"] = "complete" if completed.returncode == 0 else "failed"
        if completed.returncode == 0:
            task["reused_completed_attempt"] = _completed_attempt(
                Path(task["seed_dir"]), task["scene"], task["seed"]
            )
        _refresh_counts(plan)
        _write_plan(plan, plan_path)
        if completed.returncode and not args.continue_on_error:
            raise subprocess.CalledProcessError(completed.returncode, command)
    if all(task["status"] == "complete" for task in plan["tasks"]):
        aggregate_command = [
            str(args.python.resolve()),
            str(AGGREGATOR),
            "--benchmark",
            "nvos",
            "--input-root",
            str(
                args.output_root.resolve()
                / "nvos"
                / GEOMETRY_PROTOCOL
            ),
            "--output",
            str(args.summary.resolve()),
        ]
        subprocess.run(aggregate_command, cwd=ROOT, check=True)
        plan["full_cohort_summary"] = str(args.summary.resolve())
        plan["full_cohort_completed_at"] = _utc_now()
        _write_plan(plan, plan_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--upstream", type=Path, default=Path("/root/baselines/LUDVIG"))
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/cybersim_agent/bin/python"),
    )
    parser.add_argument(
        "--pythonpath",
        type=Path,
        default=Path("/root/baselines/LUDVIG/.reproduction-deps-sm86"),
    )
    parser.add_argument(
        "--driver-library-dir",
        type=Path,
        default=DEFAULT_DRIVER_LIBRARY_DIR,
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("/root/baselines/VALA/ckpts/sam_vit_h_4b8939.pth"),
    )
    parser.add_argument(
        "--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
            / "nvos"
            / f"{COHORT_ID}_plan.json"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
            / "nvos"
            / f"{COHORT_ID}_summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        plan = prepare_plan(args)
    except (ProtocolError, CohortError) as error:
        raise SystemExit(f"cohort protocol error: {error}") from error
    plan_path = args.plan.resolve()
    _refresh_counts(plan)
    _write_plan(plan, plan_path)
    print(plan_path)
    print(
        f"complete={plan['complete_runs']} pending={plan['pending_runs']} "
        f"execute={args.execute}"
    )
    if args.execute:
        execute_plan(args, plan, plan_path)


if __name__ == "__main__":
    main()
