"""Isolated instance diagnostic for source-only SAM3 video identity edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.v4.carrier import MeshCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.evaluation.lerf_source_mask_gate import _masks
from radio_gs.v4.evaluation.object_oracle_gate import _load_instance_labels
from radio_gs.v4.evaluation.sam_object_evidence_audit import _proposal_association


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if not args.allow_instance_oracle_diagnostic:
        raise PermissionError("video identity audit requires explicit instance-oracle authorization")
    manifests = [Path(value).resolve(strict=True) for value in args.pair_manifest]
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    segmentation = Path(args.segmentation).resolve(strict=True)
    aggregation = Path(args.aggregation).resolve(strict=True)
    pairs = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        if payload.get("information_policy", {}).get("benchmark_labels_used") is not False:
            raise ValueError("pair manifest is not label-free")
        pairs.extend(payload["pairs"])
    frame_ids = sorted({
        int(value)
        for row in pairs
        for value in (row["source_frame_id"], row["target_frame_id"])
    })
    cameras = _load_cameras(
        transforms,
        [{"frame_id": value} for value in frame_ids],
        args.feature_height,
        args.feature_width,
    )
    camera_by_frame = dict(zip(frame_ids, cameras))
    vertices, triangles = _load_mesh(mesh_path)
    vertex_instances, object_ids = _load_instance_labels(
        segmentation, aggregation, vertices.shape[0]
    )
    lookup = torch.full((int(vertex_instances.max()) + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(object_ids)] = torch.arange(len(object_ids))
    vertex_tokens = lookup[vertex_instances]
    membership = torch.zeros(vertices.shape[0], len(object_ids))
    annotated = vertex_tokens >= 0
    membership[annotated, vertex_tokens[annotated]] = 1
    mesh = MeshCarrier(vertices, triangles)
    oracle_by_frame = {
        frame_id: mesh.render_posterior(membership, camera_by_frame[frame_id])
        for frame_id in frame_ids
    }

    accepted_count = safe_count = correct_count = seeded_count = 0
    tracker_ious = []
    target_collisions = 0
    pair_reports = []
    for pair in pairs:
        source_frame = int(pair["source_frame_id"])
        target_frame = int(pair["target_frame_id"])
        source_masks = _masks(
            Path(pair["source_mask_cache"]), args.feature_height, args.feature_width
        )
        target_masks = _masks(
            Path(pair["target_mask_cache"]), args.feature_height, args.feature_width
        )
        candidates = [
            edge for edge in pair["edges"]
            if float(edge["tracked_to_target_root_iou"]) >= args.minimum_tracker_iou
            and int(edge["target_proposal_index"]) >= 0
        ]
        best_by_target = {}
        for edge in candidates:
            target = int(edge["target_proposal_index"])
            key = (float(edge["tracked_to_target_root_iou"]), -int(edge["source_proposal_index"]))
            previous = best_by_target.get(target)
            if previous is None or key > previous[0]:
                best_by_target[target] = (key, edge)
        accepted = [value[1] for _, value in sorted(best_by_target.items())]
        seeded_count += int(pair["seeded_root_count"])
        accepted_count += len(accepted)
        target_collisions += len(candidates) - len(accepted)
        pair_safe = pair_correct = 0
        for edge in accepted:
            source_index = int(edge["source_proposal_index"])
            target_index = int(edge["target_proposal_index"])
            source = _proposal_association(
                source_masks[source_index : source_index + 1],
                oracle_by_frame[source_frame],
                minimum_purity=args.minimum_purity,
                minimum_margin=args.minimum_margin,
                whole_recall=0.0,
            )
            target = _proposal_association(
                target_masks[target_index : target_index + 1],
                oracle_by_frame[target_frame],
                minimum_purity=args.minimum_purity,
                minimum_margin=args.minimum_margin,
                whole_recall=0.0,
            )
            safe = bool(source["associated"][0] and target["associated"][0])
            correct = safe and int(source["token_id"][0]) == int(target["token_id"][0])
            pair_safe += int(safe)
            pair_correct += int(correct)
            safe_count += int(safe)
            correct_count += int(correct)
            tracker_ious.append(float(edge["tracked_to_target_root_iou"]))
        pair_reports.append({
            "source_frame_id": source_frame,
            "target_frame_id": target_frame,
            "seeded_root_count": int(pair["seeded_root_count"]),
            "accepted_edge_count": len(accepted),
            "safe_diagnostic_edge_count": pair_safe,
            "correct_instance_edge_count": pair_correct,
        })
    receipt = GeometryReceipt(
        carrier="mesh_oracle_video_identity_edge_audit",
        coordinate_convention="source_video_masks_to_instance_diagnostic",
        inputs=(
            *tuple(HashedInput.seal(f"source_pair_manifest_{i}", path) for i, path in enumerate(manifests)),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
            HashedInput.seal("diagnostic_instance_segmentation", segmentation),
            HashedInput.seal("diagnostic_instance_aggregation", aggregation),
        ),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=True,
        metadata={
            "labels_used_only_after_pair_edges_were_sealed": True,
            "method_pair_manifests_query_independent": True,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.sam3_video_pair_association_audit.v1",
        "seeded_root_count": seeded_count,
        "accepted_edge_count": accepted_count,
        "accepted_edge_rate": accepted_count / max(seeded_count, 1),
        "safe_diagnostic_edge_count": safe_count,
        "safe_instance_identity_accuracy": correct_count / max(safe_count, 1),
        "mean_accepted_tracker_to_target_root_iou": sum(tracker_ious) / max(len(tracker_ious), 1),
        "duplicate_target_root_edge_count": target_collisions,
        "per_pair": pair_reports,
        "diagnostic_policy": {
            "minimum_proposal_purity": args.minimum_purity,
            "minimum_proposal_margin": args.minimum_margin,
            "minimum_tracker_to_target_root_iou": args.minimum_tracker_iou,
            "duplicate_target_policy": "keep_highest_tracker_iou_then_lowest_source_index",
        },
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", action="append", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--minimum-purity", type=float, default=0.70)
    parser.add_argument("--minimum-margin", type=float, default=0.20)
    parser.add_argument("--minimum-tracker-iou", type=float, default=0.30)
    parser.add_argument("--allow-instance-oracle-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in (
        "seeded_root_count", "accepted_edge_count", "accepted_edge_rate",
        "safe_diagnostic_edge_count", "safe_instance_identity_accuracy",
        "mean_accepted_tracker_to_target_root_iou", "duplicate_target_root_edge_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
