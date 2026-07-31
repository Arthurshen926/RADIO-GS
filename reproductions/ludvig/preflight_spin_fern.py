#!/usr/bin/env python3
"""CPU-only SPIn-NeRF fern checkpoint-reuse and mask-mapping preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image

from reproductions.ludvig.run_ludvig_sam import (
    DEFAULT_BENCHMARK_ROOT,
    ProtocolError,
    _read_colmap_image_poses,
    _sha256,
    _stage_spin_llff_pinhole_colmap,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "protocol_audit_20260731"
    / "ludvig"
    / "spin"
    / "preflight"
    / "fern_checkpoint_reuse_v1"
    / "preflight_manifest.json"
)
DEFAULT_CONVERTED_SOURCE = (
    DEFAULT_BENCHMARK_ROOT
    / "NVOS"
    / "llff_undistorted"
    / "fern_undistort"
)
SPIN_SOURCE_RELATIVE = Path(
    "SPIn-NeRF/source_images/llff_google_drive/extracted/"
    "nerf_llff_data/fern"
)
SPIN_ANNOTATION_RELATIVE = Path(
    "SPIn-NeRF/multiview_annotations/fern (llff)"
)
EXPECTED_LUDVIG_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
EXPECTED_FRAMES = 20
TARGET_WIDTH = 1600
TARGET_HEIGHT = 1199


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pixel_values(path: Path) -> list[int]:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ProtocolError(f"Expected grayscale foreground mask: {path}")
        histogram = image.histogram()
    return [index for index, count in enumerate(histogram) if count]


def _audit_masks_and_mapping(
    annotation_dir: Path,
    converted_source: Path,
) -> dict[str, Any]:
    if not annotation_dir.is_dir():
        raise ProtocolError(f"Missing SPIn fern annotations: {annotation_dir}")
    main_masks = sorted(
        path
        for path in annotation_dir.glob("image[0-9][0-9][0-9].png")
        if re.fullmatch(r"image[0-9]{3}\.png", path.name)
    )
    cutouts = sorted(annotation_dir.glob("image[0-9][0-9][0-9]_cutout.png"))
    pseudos = sorted(annotation_dir.glob("image[0-9][0-9][0-9]_pseudo.png"))
    if not (
        len(main_masks)
        == len(cutouts)
        == len(pseudos)
        == EXPECTED_FRAMES
    ):
        raise ProtocolError(
            "SPIn fern requires 20 main masks, cutouts, and pseudo images; "
            f"found {len(main_masks)}, {len(cutouts)}, {len(pseudos)}"
        )
    expected_names = [f"image{index:03d}.png" for index in range(EXPECTED_FRAMES)]
    if [path.name for path in main_masks] != expected_names:
        raise ProtocolError("SPIn fern mask names are not image000..image019")

    dimensions = set()
    mask_values = set()
    mask_hashes = {}
    for mask in main_masks:
        with Image.open(mask) as image:
            dimensions.add(image.size)
        values = _pixel_values(mask)
        mask_values.update(values)
        mask_hashes[mask.name] = _sha256(mask)
    if dimensions != {(1008, 756)}:
        raise ProtocolError(f"Unexpected SPIn fern mask dimensions: {dimensions}")
    if mask_values != {0, 1}:
        raise ProtocolError(f"SPIn fern masks are not binary 0/1: {mask_values}")

    coco_path = annotation_dir / "annotations.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco_names = [row["file_name"] for row in coco["images"]]
    if coco_names != expected_names or len(coco["annotations"]) != EXPECTED_FRAMES:
        raise ProtocolError("COCO metadata does not match the 20 SPIn fern masks")
    if {
        (row["width"], row["height"]) for row in coco["images"]
    } != {(1008, 756)}:
        raise ProtocolError("COCO metadata has unexpected SPIn fern dimensions")

    poses = _read_colmap_image_poses(
        converted_source / "dense" / "sparse" / "images.bin"
    )
    camera_names = [Path(name).stem for name in sorted(poses)]
    if len(camera_names) != EXPECTED_FRAMES:
        raise ProtocolError(
            f"Expected 20 registered fern cameras, found {len(camera_names)}"
        )
    mapping = dict(zip(expected_names, camera_names))
    return {
        "annotation_dir": str(annotation_dir.resolve()),
        "main_masks": len(main_masks),
        "cutout_images": len(cutouts),
        "pseudo_images": len(pseudos),
        "mask_dimensions": [1008, 756],
        "mask_values": sorted(mask_values),
        "annotations_json_sha256": _sha256(coco_path),
        "main_mask_sha256": mask_hashes,
        "mapping_policy": (
            "released utils.data.spinnerf_paths: sorted unique mask names "
            "zip sorted COLMAP camera image_name values"
        ),
        "mask_to_camera": mapping,
        "reference_mask": expected_names[0],
        "reference_camera": camera_names[0],
        "target_masks": expected_names[1:],
        "target_cameras": camera_names[1:],
        "reference_scored": False,
        "num_scored_target_frames": EXPECTED_FRAMES - 1,
    }


def _audit_released_code(ludvig_checkout: Path) -> dict[str, Any]:
    if _git_head(ludvig_checkout) != EXPECTED_LUDVIG_COMMIT:
        raise ProtocolError(
            f"LUDVIG must be pinned to {EXPECTED_LUDVIG_COMMIT}"
        )
    seg_script = ludvig_checkout / "script" / "seg.sh"
    data_source = ludvig_checkout / "utils" / "data.py"
    evaluation_source = (
        ludvig_checkout / "evaluation" / "spin_nvos" / "segmentation.py"
    )
    seg_text = seg_script.read_text(encoding="utf-8")
    data_text = data_source.read_text(encoding="utf-8")
    evaluation_text = evaluation_source.read_text(encoding="utf-8")
    required_seg = (
        'it="30000"',
        'src_path="llff_data"',
        'height=1199',
        'width=1600',
    )
    required_data = (
        "mask_names = sorted(set(mask_names))",
        "img_names = sorted([cam.image_name for cam in colmap_cameras])",
        "path=os.path.join(evaluate, mask_names[0])",
        "cam_name=img_names[0]",
    )
    required_eval = (
        "if i > 0:",
        "self.features[:, i : i + 1]",
        "sid_best_iou = np.argmax",
        "maximize_reference_mask_iou_per_scene",
    )
    missing = [
        token
        for token in (*required_seg, *required_data, *required_eval)
        if token not in seg_text + data_text + evaluation_text
    ]
    if missing:
        raise ProtocolError(
            f"Released SPIn protocol source changed; missing tokens: {missing}"
        )
    return {
        "ludvig_commit": EXPECTED_LUDVIG_COMMIT,
        "seg_script": str(seg_script),
        "seg_script_sha256": _sha256(seg_script),
        "data_mapping_source": str(data_source),
        "data_mapping_source_sha256": _sha256(data_source),
        "evaluation_source": str(evaluation_source),
        "evaluation_source_sha256": _sha256(evaluation_source),
        "geometry_iterations": 30000,
        "llff_scene_root_shared_by_nvos_and_spin_fern": True,
        "render_dimensions": [TARGET_WIDTH, TARGET_HEIGHT],
        "calibration": (
            "choose one of three uplifted SAM candidates and its threshold "
            "by reference-mask IoU separately for the scene"
        ),
        "aggregation": (
            "exclude reference frame; mean 19 target IoUs within fern; "
            "scene macro only when combining scenes"
        ),
    }


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"Refusing to overwrite immutable preflight: {output}")
    output.parent.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "preflighting",
        "benchmark": "SPIn-NeRF",
        "scene": "fern",
        "created_at": _utc_now(),
        "cpu_only": True,
        "gpu_work_queued": False,
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        benchmark_root = args.benchmark_root.resolve()
        spin_source = benchmark_root / SPIN_SOURCE_RELATIVE
        annotation_dir = benchmark_root / SPIN_ANNOTATION_RELATIVE
        converted_source = args.converted_source.resolve()
        staging = output.parent / "staging" / "colmap_pinhole_undistorted"
        equivalence = _stage_spin_llff_pinhole_colmap(
            spin_source,
            converted_source,
            staging,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        mask_mapping = _audit_masks_and_mapping(
            annotation_dir,
            converted_source,
        )
        released_code = _audit_released_code(args.ludvig_upstream.resolve())
        manifest.update(
            {
                "status": "ready_pending_exact_nvos_fern_checkpoint",
                "completed_at": _utc_now(),
                "scene_equivalence": equivalence,
                "mask_and_camera_mapping": mask_mapping,
                "released_protocol": released_code,
                "checkpoint_reuse": {
                    "can_reuse_exact_nvos_fern_all_view_30k": True,
                    "additional_spin_fern_training_required": False,
                    "reason": (
                        "SPIn fern raw sparse files and all 20 raw RGBs are "
                        "byte-identical to the raw source of the audited NVOS "
                        "PINHOLE conversion; LUDVIG's released seg.sh uses the "
                        "same llff_data/fern 30k checkpoint for both tasks."
                    ),
                    "checkpoint_status": (
                        "pending; GPU work prohibited while GPU0 is lost"
                    ),
                },
                "planned_evaluation": {
                    "method": "LUDVIG-SAM",
                    "seeds": [0, 1, 2],
                    "paper_fern_iou_percent": 97.0,
                    "metric": "foreground_iou",
                    "oracle_values_aggregated": False,
                    "estimated_runtime_per_seed_minutes": [2, 5],
                    "estimated_total_runtime_minutes": [6, 15],
                    "estimated_peak_gpu_memory_gib": [8, 12],
                    "eligible_for_full_10_scene_row": False,
                    "eligible_for_local_9_scene_row": False,
                    "role": (
                        "single-scene diagnostic of the distinct SPIn "
                        "reference-mask candidate/threshold path"
                    ),
                },
            }
        )
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return output
    except BaseException as error:
        manifest.update(
            {
                "status": "failed_preflight",
                "completed_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument(
        "--converted-source",
        type=Path,
        default=DEFAULT_CONVERTED_SOURCE,
    )
    parser.add_argument(
        "--ludvig-upstream",
        type=Path,
        default=Path("/root/baselines/LUDVIG"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(run(parse_args()))
    except ProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
