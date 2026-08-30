"""Bind a label-free geometry-selected frame cohort to legal source RGB files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.v4.contracts.geometry_receipt import sha256_file


def run(args: argparse.Namespace) -> dict:
    transforms = Path(args.transforms).resolve(strict=True)
    selection_path = Path(args.selection_authority).resolve(strict=True)
    selection = json.loads(selection_path.read_text())
    if selection.get("schema") != "radio_gs.surface_object_memory_v4.geometry_source_view_selection.v1":
        raise ValueError("source-view selection schema differs")
    if selection["geometry_receipt"]["benchmark_labels_opened"] is not False:
        raise ValueError("source-view selection opened benchmark labels")
    selected = selection["selections"].get(str(args.view_count))
    if selected is None:
        raise KeyError("requested view count is absent from selection authority")
    selected_ids = list(map(int, selected["selected_frame_ids"]))
    payload = json.loads(transforms.read_text())
    frames = {int(Path(frame["file_path"]).stem): frame for frame in payload["frames"]}
    root = transforms.parent
    images = []
    for frame_id in selected_ids:
        if frame_id not in frames:
            raise KeyError(f"selected frame {frame_id} is absent from transforms")
        path = (root / frames[frame_id]["file_path"]).with_suffix(".jpg").resolve()
        if not path.is_file() or root not in path.parents:
            raise ValueError(f"selected source RGB is absent or escaped the scene root: {path}")
        images.append({
            "image_id": f"frame_{frame_id:05d}",
            "path": str(path),
            "sha256": sha256_file(path),
            "rgb_role": "registered_source_or_mapping_view",
        })
    report = {
        "schema_version": 1,
        "contract": "sam3-query-free-source-rgb-authority-v1",
        "cohort": "geometry-coverage-greedy-source-only",
        "information_policy": {
            "registered_source_rgb_only": True,
            "query_text_used": False,
            "benchmark_ground_truth_used": False,
            "target_or_evaluation_rgb_used": False,
        },
        "construction": {
            "selection_authority": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "strategy": selection["selection_strategy"],
            },
            "camera_transforms": {"path": str(transforms), "sha256": sha256_file(transforms)},
            "selected_count": len(images),
        },
        "images": images,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"selected_count": len(report["images"])}, indent=2))


if __name__ == "__main__":
    main()
