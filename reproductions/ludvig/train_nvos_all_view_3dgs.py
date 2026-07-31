#!/usr/bin/env python3
"""Train the released NVOS fern all-view geometry with pinned original 3DGS.

The LUDVIG repository vendors gaussian-splatting as a normal directory, so it
does not provide a gitlink that can be checked out directly.  The companion
lock file records the official revision whose training entrypoint is byte
identical to LUDVIG's vendored entrypoint.  This launcher verifies that
revision and all of its relevant gitlinks, stages the already-undistorted
NVOS PINHOLE model, and serializes the only GPU section with the shared GPU 0
lock.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any
from urllib.parse import unquote, urlparse

from reproductions.ludvig.run_ludvig_sam import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DRIVER_LIBRARY_DIR,
    LOCK_PATH,
    _driver_library,
    _stage_nvos_pinhole_colmap,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = Path(__file__).with_name("official_3dgs.lock.json")
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "output"
    / "protocol_audit_20260731"
    / "ludvig"
    / "nvos"
    / "released_all_view"
    / "fern"
    / "training"
)
OFFICIAL_3DGS_COMMIT = "f7a116fb1397d9842239127d39dc212f93171f70"
LUDVIG_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
RASTERIZER_COMMIT = "8064f52ca233942bdec2d1a1451c026deedd320b"
SIMPLE_KNN_COMMIT = "44f764299fa305faf6ec5ebd99939e0508331503"
GLM_COMMIT = "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
TRAIN_ENTRYPOINT_SHA256 = (
    "c5a61947e2abcf56bf83451ae9633799d96894910ea2982a01f209c47cec462d"
)
EXPECTED_REGISTERED_IMAGES = 20
TARGET_WIDTH = 1600
TARGET_HEIGHT = 1199


class TrainingProtocolError(RuntimeError):
    """Raised before GPU work when exact training provenance is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head(checkout: Path) -> str:
    return _git(checkout, "rev-parse", "HEAD")


def _require_clean(checkout: Path, label: str) -> None:
    status = _git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise TrainingProtocolError(
            f"{label} checkout has tracked modifications:\n{status}"
        )


def _literal_class_defaults(source: str, class_name: str) -> dict[str, Any]:
    """Extract literal ``self.<name> = value`` defaults from ``__init__``."""

    tree = ast.parse(source)
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise TrainingProtocolError(f"Missing class {class_name}")
    init_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
        None,
    )
    if init_node is None:
        raise TrainingProtocolError(f"Missing {class_name}.__init__")
    defaults: dict[str, Any] = {}
    for node in init_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        try:
            defaults[target.attr] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return defaults


def _validate_source(
    checkout: Path,
    ludvig_checkout: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    if lock.get("commit") != OFFICIAL_3DGS_COMMIT:
        raise TrainingProtocolError("official_3dgs.lock.json has an unknown commit")
    if _git_head(checkout) != OFFICIAL_3DGS_COMMIT:
        raise TrainingProtocolError(
            f"3DGS checkout must be {OFFICIAL_3DGS_COMMIT}; "
            f"found {_git_head(checkout)}"
        )
    _require_clean(checkout, "3DGS")

    submodules = {
        "submodules/diff-gaussian-rasterization": RASTERIZER_COMMIT,
        "submodules/simple-knn": SIMPLE_KNN_COMMIT,
    }
    resolved_submodules: dict[str, str] = {}
    for relative, expected in submodules.items():
        submodule = checkout / relative
        if not submodule.exists():
            raise TrainingProtocolError(f"Missing initialized submodule: {submodule}")
        found = _git_head(submodule)
        if found != expected:
            raise TrainingProtocolError(
                f"{relative} must be {expected}; found {found}"
            )
        _require_clean(submodule, relative)
        resolved_submodules[relative] = found
    glm = checkout / "submodules" / "diff-gaussian-rasterization" / "third_party" / "glm"
    if not glm.exists() or _git_head(glm) != GLM_COMMIT:
        raise TrainingProtocolError(
            f"rasterizer GLM must be initialized at {GLM_COMMIT}"
        )
    _require_clean(glm, "rasterizer third_party/glm")
    resolved_submodules["submodules/diff-gaussian-rasterization/third_party/glm"] = (
        _git_head(glm)
    )

    source_hashes = {}
    for relative, expected in lock["source_sha256"].items():
        source_path = checkout / relative
        found = _sha256(source_path)
        if found != expected:
            raise TrainingProtocolError(
                f"Source hash mismatch for {relative}: expected {expected}, found {found}"
            )
        source_hashes[relative] = found

    train_hash = _sha256(checkout / "train.py")
    if train_hash != TRAIN_ENTRYPOINT_SHA256:
        raise TrainingProtocolError("Pinned official train.py hash changed")
    if _git_head(ludvig_checkout) != LUDVIG_COMMIT:
        raise TrainingProtocolError(
            f"LUDVIG checkout must be {LUDVIG_COMMIT}; "
            f"found {_git_head(ludvig_checkout)}"
        )
    ludvig_train = ludvig_checkout / "gaussiansplatting" / "train.py"
    if _sha256(ludvig_train) != train_hash:
        raise TrainingProtocolError(
            "Official pinned train.py is no longer byte-identical to LUDVIG's "
            "vendored training entrypoint"
        )
    tree_entry = _git(
        ludvig_checkout,
        "ls-tree",
        "HEAD",
        "gaussiansplatting",
    )
    if not tree_entry.startswith("040000 tree "):
        raise TrainingProtocolError(
            "Expected LUDVIG gaussian-splatting to be a vendored tree, not a gitlink"
        )

    argument_source = (checkout / "arguments" / "__init__.py").read_text(
        encoding="utf-8"
    )
    optimization_defaults = _literal_class_defaults(
        argument_source, "OptimizationParams"
    )
    model_defaults = _literal_class_defaults(argument_source, "ModelParams")
    for key, expected in lock["training_defaults"].items():
        if optimization_defaults.get(key) != expected:
            raise TrainingProtocolError(
                f"Official 3DGS default {key} changed: "
                f"expected {expected}, found {optimization_defaults.get(key)}"
            )
    for key, expected in lock["model_defaults"].items():
        source_key = f"_{key}" if f"_{key}" in model_defaults else key
        if model_defaults.get(source_key) != expected:
            raise TrainingProtocolError(
                f"Official 3DGS model default {key} changed: expected "
                f"{expected}, found {model_defaults.get(source_key)}"
            )

    return {
        "repository": lock["repository"],
        "commit": OFFICIAL_3DGS_COMMIT,
        "checkout": str(checkout),
        "tracked_source_clean": True,
        "source_sha256": source_hashes,
        "submodules": resolved_submodules,
        "training_defaults": lock["training_defaults"],
        "model_defaults": lock["model_defaults"],
        "ludvig_commit": LUDVIG_COMMIT,
        "ludvig_vendored_tree_entry": tree_entry,
        "ludvig_train_entrypoint_sha256": train_hash,
        "reconstruction_evidence": lock["selection"],
    }


def _extension_path(dependency_root: Path, package: str) -> Path:
    candidates = sorted((dependency_root / package).glob("_C*.so"))
    if len(candidates) != 1:
        raise TrainingProtocolError(
            f"Expected exactly one compiled {package} extension under "
            f"{dependency_root}; found {len(candidates)}"
        )
    return candidates[0]


def _runtime_environment(
    dependency_root: Path,
    driver_library_dir: Path,
    *,
    expose_gpu: bool,
) -> tuple[dict[str, str], Path]:
    driver_library = _driver_library(driver_library_dir)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if expose_gpu else ""
    environment["PYTHONPATH"] = str(dependency_root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    environment["LD_LIBRARY_PATH"] = (
        str(driver_library_dir)
        + os.pathsep
        + "/usr/local/cuda/lib64"
        + os.pathsep
        + environment.get("LD_LIBRARY_PATH", "")
    )
    return environment, driver_library


def _validate_dependencies(
    checkout: Path,
    python: Path,
    dependency_root: Path,
    driver_library_dir: Path,
    preflight_log: Path,
) -> dict[str, Any]:
    if not python.exists():
        raise TrainingProtocolError(f"Missing Python interpreter: {python}")
    if not dependency_root.is_dir():
        raise TrainingProtocolError(
            f"Missing isolated dependency directory: {dependency_root}"
        )
    extensions = {
        package: _extension_path(dependency_root, package)
        for package in ("diff_gaussian_rasterization", "simple_knn")
    }
    expected_install_sources = {
        "diff_gaussian_rasterization": (
            checkout / "submodules" / "diff-gaussian-rasterization"
        ),
        "simple_knn": checkout / "submodules" / "simple-knn",
    }
    install_sources = {}
    for package, expected_source in expected_install_sources.items():
        dist_info = sorted(
            dependency_root.glob(f"{package}-*.dist-info/direct_url.json")
        )
        if len(dist_info) != 1:
            raise TrainingProtocolError(
                f"Expected one direct_url.json for {package}; found {len(dist_info)}"
            )
        direct_url = json.loads(dist_info[0].read_text(encoding="utf-8"))["url"]
        parsed = urlparse(direct_url)
        if parsed.scheme != "file":
            raise TrainingProtocolError(
                f"{package} was not built from the locked local checkout"
            )
        installed_from = Path(unquote(parsed.path)).resolve()
        if installed_from != expected_source.resolve():
            raise TrainingProtocolError(
                f"{package} was built from {installed_from}, expected "
                f"{expected_source.resolve()}"
            )
        install_sources[package] = {
            "direct_url": direct_url,
            "resolved_source": str(installed_from),
        }
    environment, driver_library = _runtime_environment(
        dependency_root,
        driver_library_dir,
        expose_gpu=False,
    )
    command = [str(python), str(checkout / "train.py"), "--help"]
    completed = subprocess.run(
        command,
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
    )
    preflight_log.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise TrainingProtocolError(
            f"Pinned train.py dependency preflight failed; see {preflight_log}"
        )
    required_help = (
        "--position_lr_init",
        "--densify_until_iter",
        "--save_iterations",
        "--resolution RESOLUTION",
    )
    if any(token not in completed.stdout for token in required_help):
        raise TrainingProtocolError(
            "Pinned train.py help is missing expected original-3DGS arguments"
        )
    runtime_command = [
        str(python),
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({'torch': torch.__version__, "
            "'torch_cuda': torch.version.cuda}))"
        ),
    ]
    runtime_completed = subprocess.run(
        runtime_command,
        cwd=checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_versions = json.loads(runtime_completed.stdout)
    return {
        "python": str(python),
        "dependency_root": str(dependency_root),
        "runtime_versions": runtime_versions,
        "install_sources": install_sources,
        "compiled_extensions": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in extensions.items()
        },
        "driver_library": str(driver_library.resolve()),
        "driver_library_sha256": _sha256(driver_library),
        "cpu_only_train_help_preflight": {
            "command": command,
            "cuda_visible_devices": "",
            "returncode": completed.returncode,
            "log": str(preflight_log),
            "log_sha256": _sha256(preflight_log),
        },
    }


def _training_command(
    python: Path,
    checkout: Path,
    staged_scene: Path,
    model_path: Path,
) -> list[str]:
    return [
        str(python),
        str(checkout / "train.py"),
        "--source_path",
        str(staged_scene),
        "--model_path",
        str(model_path),
        "--iterations",
        "30000",
        "--test_iterations",
        "-1",
        "--save_iterations",
        "30000",
        "--quiet",
    ]


def _parse_namespace(path: Path) -> dict[str, Any]:
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
    ):
        raise TrainingProtocolError(f"Unexpected cfg_args syntax: {path}")
    values = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise TrainingProtocolError(f"Unexpected cfg_args expansion: {path}")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def _parse_ply_vertex_count(path: Path) -> int:
    vertex_count = None
    found_end = False
    with path.open("rb") as handle:
        for _ in range(512):
            line = handle.readline()
            if not line:
                break
            if len(line) > 4096:
                raise TrainingProtocolError(f"Invalid PLY header line in {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            match = re.fullmatch(r"element vertex ([0-9]+)", decoded)
            if match:
                vertex_count = int(match.group(1))
            if decoded == "end_header":
                found_end = True
                break
    if not found_end or vertex_count is None or vertex_count <= 0:
        raise TrainingProtocolError(f"Invalid or empty 3DGS PLY: {path}")
    return vertex_count


def _validate_training_output(run_dir: Path, model_path: Path) -> dict[str, Any]:
    point_cloud = (
        model_path
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    if not point_cloud.is_file():
        raise TrainingProtocolError(f"Missing final 30k point cloud: {point_cloud}")
    cfg_args = _parse_namespace(model_path / "cfg_args")
    if cfg_args.get("eval") is not False:
        raise TrainingProtocolError("All-view training unexpectedly enabled --eval")
    if cfg_args.get("resolution") != -1:
        raise TrainingProtocolError(
            f"Expected original automatic resolution -1, found {cfg_args.get('resolution')}"
        )
    cameras = json.loads((model_path / "cameras.json").read_text(encoding="utf-8"))
    if len(cameras) != EXPECTED_REGISTERED_IMAGES:
        raise TrainingProtocolError(
            f"Expected {EXPECTED_REGISTERED_IMAGES} all-view cameras, "
            f"found {len(cameras)}"
        )
    return {
        "point_cloud": str(point_cloud),
        "point_cloud_sha256": _sha256(point_cloud),
        "point_cloud_size_bytes": point_cloud.stat().st_size,
        "point_cloud_vertices": _parse_ply_vertex_count(point_cloud),
        "cfg_args": cfg_args,
        "registered_all_view_cameras": len(cameras),
        "target_rgb_visible_during_training": True,
        "model_path": str(model_path),
        "run_dir": str(run_dir),
    }


def launch(args: argparse.Namespace) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.attempt_id):
        raise TrainingProtocolError(
            "--attempt-id must contain only letters, digits, '.', '_' or '-'"
        )
    run_dir = args.output_root.resolve() / "attempts" / args.attempt_id
    if run_dir.exists():
        raise TrainingProtocolError(
            f"Refusing to reuse immutable attempt directory: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "training_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "preflighting",
        "method": "original-3DGS",
        "benchmark": "NVOS",
        "scene": "fern",
        "geometry_protocol": "released_all_view",
        "attempt_id": args.attempt_id,
        "created_at": _utc_now(),
        "lock_file": str(LOCK_FILE),
        "lock_file_sha256": _sha256(LOCK_FILE),
        "gpu_lock": str(LOCK_PATH),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        source_provenance = _validate_source(
            args.upstream.resolve(),
            args.ludvig_upstream.resolve(),
            lock,
        )
        preflight_log = run_dir / "dependency_preflight.log"
        dependency_provenance = _validate_dependencies(
            args.upstream.resolve(),
            args.python.resolve(),
            args.dependency_root.resolve(),
            args.driver_library_dir.resolve(),
            preflight_log,
        )
        source_scene = (
            args.benchmark_root.resolve()
            / "NVOS"
            / "llff_undistorted"
            / "fern_undistort"
        )
        staged_scene = run_dir / "staging" / "colmap_pinhole_undistorted"
        camera_audit = _stage_nvos_pinhole_colmap(
            source_scene,
            staged_scene,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        if camera_audit["registered_images"] != EXPECTED_REGISTERED_IMAGES:
            raise TrainingProtocolError(
                f"Expected {EXPECTED_REGISTERED_IMAGES} fern views, found "
                f"{camera_audit['registered_images']}"
            )
        model_path = run_dir / "model"
        command = _training_command(
            args.python.resolve(),
            args.upstream.resolve(),
            staged_scene,
            model_path,
        )
        manifest.update(
            {
                "source_provenance": source_provenance,
                "dependency_provenance": dependency_provenance,
                "camera_audit": camera_audit,
                "training_command": command,
                "effective_training_protocol": {
                    "registered_training_views": EXPECTED_REGISTERED_IMAGES,
                    "held_out_training_views": 0,
                    "eval_split_enabled": False,
                    "iterations": 30000,
                    "resolution_argument": -1,
                    "effective_resolution": [TARGET_WIDTH, TARGET_HEIGHT],
                    "test_iterations_override": [-1],
                    "save_iterations": [30000],
                    "rng_seed": 0,
                    "hyperparameters": lock["training_defaults"],
                    "algorithm_source_modified": False,
                    "environment_compatibility_source_patch": None,
                },
            }
        )
        if args.dry_run:
            manifest["status"] = "dry_run"
            manifest["completed_at"] = _utc_now()
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            return manifest_path

        environment, _driver = _runtime_environment(
            args.dependency_root.resolve(),
            args.driver_library_dir.resolve(),
            expose_gpu=True,
        )
        manifest["status"] = "queued"
        manifest["queued_at"] = _utc_now()
        manifest["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
        queue_started_epoch = time.time()
        wall_started = time.monotonic()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        log_path = run_dir / "stdout_stderr.log"
        gpu_started_marker = run_dir / "gpu_started_at.txt"
        locked_script = (
            "date -u +%Y-%m-%dT%H:%M:%S.%NZ"
            f" > {shlex.quote(str(gpu_started_marker))}; "
            f"exec {shlex.join(command)}"
        )
        locked_command = ["flock", str(LOCK_PATH), "-c", locked_script]
        try:
            with log_path.open("w") as log_handle:
                completed = subprocess.run(
                    locked_command,
                    cwd=args.upstream.resolve(),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["completed_at"] = _utc_now()
            manifest["wall_time_seconds"] = time.monotonic() - wall_started
            manifest["log"] = str(log_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            raise

        completed_epoch = time.time()
        manifest["returncode"] = completed.returncode
        manifest["completed_at"] = _utc_now()
        manifest["wall_time_seconds"] = time.monotonic() - wall_started
        manifest["log"] = str(log_path)
        manifest["log_sha256"] = _sha256(log_path)
        if gpu_started_marker.exists():
            gpu_started_epoch = gpu_started_marker.stat().st_mtime
            manifest["gpu_started_at"] = gpu_started_marker.read_text(
                encoding="utf-8"
            ).strip()
            manifest["queue_wait_seconds"] = max(
                0.0, gpu_started_epoch - queue_started_epoch
            )
            manifest["gpu_wall_time_seconds"] = max(
                0.0, completed_epoch - gpu_started_epoch
            )
        if completed.returncode:
            manifest["status"] = "failed"
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            raise subprocess.CalledProcessError(completed.returncode, command)

        manifest["status"] = "validating"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["training_output"] = _validate_training_output(run_dir, model_path)
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path
    except BaseException as error:
        if manifest.get("status") not in {"failed", "interrupted", "complete"}:
            manifest["status"] = (
                "failed_validation"
                if manifest.get("status") == "validating"
                else "failed_preflight"
            )
            manifest["completed_at"] = _utc_now()
            manifest["error_type"] = type(error).__name__
            manifest["error"] = str(error)
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("/root/baselines/gaussian-splatting-ludvig-audit"),
    )
    parser.add_argument(
        "--ludvig-upstream",
        type=Path,
        default=Path("/root/baselines/LUDVIG"),
    )
    parser.add_argument(
        "--dependency-root",
        type=Path,
        default=Path(
            "/root/baselines/"
            "gaussian-splatting-ludvig-audit-deps/f7a116f-sm86"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/cybersim_agent/bin/python"),
    )
    parser.add_argument(
        "--driver-library-dir",
        type=Path,
        default=DEFAULT_DRIVER_LIBRARY_DIR,
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except TrainingProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
