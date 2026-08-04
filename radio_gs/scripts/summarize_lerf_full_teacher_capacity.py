#!/usr/bin/env python3
"""Compare preregistered compact and full-MPR LERF score diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.scripts.eval_lerf_adaptive_support_diagnostic import sha256_file


METRICS = (
    "average_precision",
    "oracle_threshold_iou",
    "positive_negative_score_margin",
    "frozen_formal_miou",
    "frozen_formal_positive_coverage",
    "frozen_formal_selected_purity",
    "target_blind_otsu3_miou",
    "target_blind_otsu3_positive_coverage",
    "target_blind_otsu3_selected_purity",
)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SCENE=PATH")
    scene, path = value.split("=", 1)
    if not scene or not path:
        raise argparse.ArgumentTypeError("expected non-empty SCENE=PATH")
    return scene, Path(path)


def _load_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = payload.get("score_quality_diagnostic")
    if not isinstance(audit, dict):
        raise ValueError(f"missing score-quality diagnostic: {path}")
    return audit


def summarize(
    compact_paths: dict[str, Path],
    full_paths: dict[str, Path],
    descriptor_sidecars: dict[str, Path],
) -> dict:
    if set(compact_paths) != set(full_paths) or set(compact_paths) != set(
        descriptor_sidecars
    ):
        raise ValueError("compact/full/descriptor scenes differ")
    scenes = {}
    registrations = set()
    for scene in compact_paths:
        compact_path = compact_paths[scene]
        full_path = full_paths[scene]
        compact = _load_audit(compact_path)
        full = _load_audit(full_path)
        if compact.get("scene") != scene or full.get("scene") != scene:
            raise ValueError(f"scene identity mismatch: {scene}")
        compact_metrics = compact["aggregate_object_mean"]
        full_metrics = full["aggregate_object_mean"]
        compact_receipt = compact["method_receipt_frozen_before_labels"]
        full_receipt = full["method_receipt_frozen_before_labels"]
        registrations.update(
            [
                compact_receipt["experiment_registration_sha256"],
                full_receipt["experiment_registration_sha256"],
            ]
        )
        sidecar_path = descriptor_sidecars[scene]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        metadata = sidecar.get("metadata", {})
        if (
            metadata.get("canonical_radio_source")
            != "frozen_mpr_full_1280_teacher"
            or metadata.get("mpr_radio_features_opened") is not True
            or metadata.get("capacity_diagnostic_only") is not True
        ):
            raise ValueError(f"full descriptor lacks capacity receipt: {scene}")
        registration = metadata.get("experiment_registration", {})
        registrations.add(registration.get("sha256"))
        compact_values = {key: float(compact_metrics[key]) for key in METRICS}
        full_values = {key: float(full_metrics[key]) for key in METRICS}
        scenes[scene] = {
            "objects": int(full["objects"]),
            "compact": compact_values,
            "full_mpr_teacher": full_values,
            "full_minus_compact": {
                key: full_values[key] - compact_values[key] for key in METRICS
            },
            "compact_result": {
                "path": str(compact_path.resolve()),
                "sha256": sha256_file(compact_path),
            },
            "full_result": {
                "path": str(full_path.resolve()),
                "sha256": sha256_file(full_path),
            },
            "full_descriptor_sidecar": {
                "path": str(sidecar_path.resolve()),
                "sha256": sha256_file(sidecar_path),
                "source": metadata["canonical_radio_source"],
                "mpr_cache_sha256": metadata["mpr_cache_sha256"],
            },
        }
    if len(registrations) != 1 or None in registrations:
        raise ValueError("capacity comparison registration hashes differ")
    otsu_deltas = {
        scene: row["full_minus_compact"]["target_blind_otsu3_miou"]
        for scene, row in scenes.items()
    }
    return {
        "artifact_type": "radio_gs_lerf3d_full_teacher_capacity_diagnostic",
        "status": "label_free_representation_source_label_aware_scoring_diagnostic",
        "experiment_registration_sha256": next(iter(registrations)),
        "scenes": scenes,
        "preregistered_confirmation": {
            "criterion": "full teacher improves target-blind Otsu3 on both figurines development and waldo_kitchen independent confirmation",
            "scene_deltas": otsu_deltas,
            "passed": set(scenes) == {"figurines", "waldo_kitchen"}
            and all(value > 0.0 for value in otsu_deltas.values()),
        },
        "claim_boundary": (
            "The full-MPR source is a query-independent, label-free 1280-D teacher "
            "capacity diagnostic under the identical frozen SurfaceRegion/text scorer; "
            "it is not the compact deployable field. AP and oracle IoU remain label-aware "
            "diagnostics and cannot select the method."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="append", type=_named_path, required=True)
    parser.add_argument("--full", action="append", type=_named_path, required=True)
    parser.add_argument("--descriptor-sidecar", action="append", type=_named_path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output exists: {output}")
    report = summarize(dict(args.compact), dict(args.full), dict(args.descriptor_sidecar))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["preregistered_confirmation"], indent=2))


if __name__ == "__main__":
    main()
