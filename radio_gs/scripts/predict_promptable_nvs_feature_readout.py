"""Generate NVOS/SPIn-NeRF feature-field scores without evaluating target masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.evaluation.promptable_feature_readout import (
    DEFAULT_FEATURE_PATTERN,
    FEATURE_LAYOUTS,
    PREDICTION_MANIFEST_NAME,
    generate_feature_readout_predictions,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Root used when a frame has no inline feature_path.",
    )
    parser.add_argument(
        "--feature-pattern",
        default=DEFAULT_FEATURE_PATTERN,
        help=(
            "Path below --feature-root. Supported fields: {benchmark}, {scene}, "
            "{scene_id}, {frame_id}, {camera_name}."
        ),
    )
    parser.add_argument("--feature-layout", choices=FEATURE_LAYOUTS, default="auto")
    parser.add_argument(
        "--scene-id",
        action="append",
        dest="scene_ids",
        help="Predict only this frozen scene (repeatable); the full manifest remains bound.",
    )
    parser.add_argument(
        "--radio-sam3-adaptor-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optionally map RADIO 1280d maps to the frozen 1024d RADIO sam3 "
            "feature_projection space. This is not an official SAM decoder."
        ),
    )
    parser.add_argument("--adaptor-device", default="cpu")
    parser.add_argument("--projection-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--require-render-authority",
        action="store_true",
        help="Require every consumed feature to be SHA-bound to a factorized-v2 render.",
    )
    parser.add_argument(
        "--method-name",
        default="GaussFM feature-field prototype readout",
    )
    parser.add_argument("--output-manifest-name", default=PREDICTION_MANIFEST_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The low-level readout also supports synthetic unit manifests.  Keep the
    # benchmark CLI strict about the complete cohort and prompt/target roles,
    # but do not open or hash evaluation ground truth during prediction.  The
    # producer opens the declared prompt assets below when it constructs the
    # prototypes; frozen manifest/file hashes are bound by the separate result
    # authority, and target masks belong exclusively to the evaluator stage.
    validate_dataset_manifest(raw_manifest, check_files=False)
    result = generate_feature_readout_predictions(
        manifest_path,
        args.output_dir,
        feature_root=args.feature_root,
        feature_pattern=args.feature_pattern,
        feature_layout=args.feature_layout,
        radio_sam3_adaptor_checkpoint=args.radio_sam3_adaptor_checkpoint,
        adaptor_device=args.adaptor_device,
        projection_chunk_size=args.projection_chunk_size,
        method_name=args.method_name,
        output_manifest_name=args.output_manifest_name,
        scene_ids=args.scene_ids,
        require_render_authority=args.require_render_authority,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "prediction_manifest": result["prediction_manifest_path"],
                "protocol_hash": result["protocol_hash"],
                "num_scenes": len(result["scenes"]),
                "num_predictions": sum(
                    len(scene["outputs"]) for scene in result["scenes"]
                ),
                "evaluation_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
