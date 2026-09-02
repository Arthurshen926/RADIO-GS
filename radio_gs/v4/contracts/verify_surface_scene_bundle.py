"""Cold-load a v4 scene bundle without construction or query inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.v4.carrier import Camera

from .surface_scene_bundle import SurfaceSceneBundle, projection_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-bundle", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--camera",
        required=True,
        help="torch payload with key/intrinsic/camera_to_world/height/width",
    )
    parser.add_argument("--expected-projection-sha256", required=True)
    args = parser.parse_args()
    bundle = SurfaceSceneBundle.load(
        args.scene_bundle, expected_sha256=args.expected_sha256
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.cold_load_verification.v1",
        "scene_label": bundle.scene_label,
        "element_count": int(bundle.centres.shape[0]),
        "token_count": int(bundle.observed_assignment.token_probability.shape[1]),
        "carrier_content_sha256": bundle.carrier_content_sha256,
        "projection_sha256": None,
        "query_inputs_opened": False,
        "benchmark_inputs_opened": False,
    }
    payload = torch.load(
        Path(args.camera).resolve(strict=True), map_location="cpu", weights_only=False
    )
    camera = Camera(
        str(payload["key"]), payload["intrinsic"], payload["camera_to_world"],
        int(payload["height"]), int(payload["width"]),
    )
    observed_projection = projection_digest(bundle.build_carrier().project(camera))
    if observed_projection != args.expected_projection_sha256:
        raise ValueError("cold-loaded projection digest differs from the expected build digest")
    report["projection_sha256"] = observed_projection
    report["projection_matches_expected"] = True
    report["bundle_content_sha256"] = bundle.content_sha256
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
