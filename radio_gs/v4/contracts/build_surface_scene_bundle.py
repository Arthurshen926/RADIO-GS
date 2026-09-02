"""Build a query-free v4 scene bundle from sealed source-memory tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.v4.object_memory import DenseObjectAssignments

from .geometry_receipt import sha256_file
from .surface_scene_bundle import (
    ElementTokenObservedEvidence,
    SurfaceSceneBundle,
    load_geometry_binding,
)


def _sealed_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sealed input must be ROLE=PATH")
    role, raw_path = value.split("=", 1)
    if not role or not raw_path:
        raise argparse.ArgumentTypeError("sealed input must contain a role and path")
    return role, Path(raw_path).resolve(strict=True)


def run(args: argparse.Namespace) -> dict:
    binding, carrier_payload = load_geometry_binding(
        args.geometry_authority,
        args.surface_carrier,
        verify_all_inputs=args.verify_all_geometry_inputs,
    )
    source_memory_path = Path(args.source_memory).resolve(strict=True)
    source_memory = torch.load(source_memory_path, map_location="cpu", weights_only=False)
    if not isinstance(source_memory, dict):
        raise ValueError("source memory must be a mapping")
    required = {
        "observed_assignment",
        "observed_evidence",
        "local_surface_memory",
        "object_memory",
        "source_frames",
        "source_input_digests",
        "information_policy",
    }
    if not required.issubset(source_memory):
        raise ValueError("source memory is missing required query-free fields")
    observed = source_memory["observed_assignment"]
    if not isinstance(observed, dict):
        raise ValueError("observed_assignment must be a mapping")
    observed_evidence = source_memory["observed_evidence"]
    if not isinstance(observed_evidence, dict):
        raise ValueError("observed_evidence must be a mapping")
    receipt_scene = binding.receipt.get("metadata", {}).get("scene_label")
    if receipt_scene is not None and str(receipt_scene) != args.scene_label:
        raise ValueError("scene label disagrees with geometry authority")
    source_digests = {
        "geometry_authority": binding.authority_sha256,
        "surface_carrier": binding.surface_carrier_sha256,
        "source_memory": sha256_file(source_memory_path),
    }
    upstream_digests = source_memory["source_input_digests"]
    if not isinstance(upstream_digests, dict) or not upstream_digests:
        raise ValueError("source memory must expose its sealed upstream input digests")
    for role, digest in upstream_digests.items():
        if not isinstance(role, str) or not role:
            raise ValueError("source memory upstream digest roles must be non-empty strings")
        source_digests[f"source_memory_upstream:{role}"] = digest
    for role, path in args.sealed_input or []:
        if role in source_digests:
            raise ValueError(f"duplicate sealed input role: {role}")
        source_digests[role] = sha256_file(path)
    bundle = SurfaceSceneBundle(
        scene_label=args.scene_label,
        configuration=binding.configuration,
        centres=carrier_payload["centres"],
        normals=carrier_payload.get("normals"),
        confidence=carrier_payload.get(
            "confidence", torch.ones(len(carrier_payload["centres"]), dtype=torch.float32)
        ),
        observed_assignment=DenseObjectAssignments(
            observed["token_probability"], observed["unknown_probability"]
        ),
        observed_evidence=ElementTokenObservedEvidence.from_payload(observed_evidence),
        local_surface_memory=source_memory["local_surface_memory"],
        object_memory=source_memory["object_memory"],
        source_frames=tuple(source_memory["source_frames"]),
        source_input_digests=source_digests,
        geometry_authority_sha256=binding.authority_sha256,
        source_surface_carrier_sha256=binding.surface_carrier_sha256,
        information_policy=source_memory["information_policy"],
        metadata={
            **dict(source_memory.get("metadata", {})),
            "geometry_authority_path_at_build": binding.authority_path,
            "surface_carrier_path_at_build": binding.surface_carrier_path,
        },
    )
    bundle_sha256 = bundle.save(args.output)
    report = {
        "schema": "radio_gs.surface_object_memory_v4.surface_scene_bundle_receipt.v1",
        "scene_label": args.scene_label,
        "scene_bundle": str(Path(args.output).resolve()),
        "scene_bundle_sha256": bundle_sha256,
        "carrier_content_sha256": bundle.carrier_content_sha256,
        "carrier_configuration": bundle.to_payload()["carrier_configuration"],
        "geometry_authority_sha256": binding.authority_sha256,
        "source_input_digests": source_digests,
        "information_policy": dict(bundle.information_policy),
        "completion_present": False,
    }
    if args.receipt_output:
        receipt_path = Path(args.receipt_output).resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--geometry-authority", required=True)
    parser.add_argument("--surface-carrier", required=True)
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--sealed-input", action="append", type=_sealed_input)
    parser.add_argument("--verify-all-geometry-inputs", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
