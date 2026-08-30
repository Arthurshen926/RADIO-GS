"""Source-only SAM-to-token bootstrap followed by an isolated oracle audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from radio_gs.v4.carrier import MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records, _masks
from radio_gs.v4.evaluation.object_oracle_gate import _load_instance_labels
from radio_gs.v4.object_memory import ObservedObjectEvidence, SurfaceTokenBootstrap


@dataclass
class MethodSnapshot:
    view_count: int
    membership: torch.Tensor
    assigned_surface: torch.Tensor
    proposal_positive: list[torch.Tensor]
    proposal_token_ids: list[torch.Tensor]
    proposal_is_root: list[torch.Tensor]
    created_count: int


def _lift_masks(surface, camera, masks: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    projection = surface.project(camera)
    element_ids = projection.element_ids.to(device)
    pixel_ids = projection.pixel_ids.to(device)
    weights = projection.weights.to(device)
    flat = masks.to(device).reshape(masks.shape[0], camera.height * camera.width)
    numerator = torch.zeros(surface.num_elements, masks.shape[0], device=device)
    numerator.index_add_(0, element_ids, flat[:, pixel_ids].T * weights[:, None])
    denominator = torch.zeros(surface.num_elements, device=device)
    denominator.scatter_add_(0, element_ids, weights)
    positive = (numerator / denominator.clamp_min(1e-12)[:, None]).T
    visible = (denominator > 0)[None].expand(masks.shape[0], -1)
    return positive, visible


def _rgb_histogram_descriptors(
    image_path: Path,
    masks: torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Query-free colour moments and histograms inside each source mask."""

    with Image.open(image_path) as image:
        rgb = torch.from_numpy(
            np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)).copy()
        ).to(device=device, dtype=torch.float32) / 255.0
    flat_masks = masks.to(device).reshape(masks.shape[0], height * width).float()
    flat_rgb = rgb.reshape(-1, 3)
    mass = flat_masks.sum(-1, keepdim=True).clamp_min(1e-8)
    mean = flat_masks @ flat_rgb / mass
    second = flat_masks @ flat_rgb.square() / mass
    deviation = (second - mean.square()).clamp_min(0).sqrt()
    histograms = []
    for channel in range(3):
        bins = (flat_rgb[:, channel] * 8).long().clamp(0, 7)
        histograms.append(flat_masks @ F.one_hot(bins, 8).float() / mass)
    return F.normalize(torch.cat([mean, deviation, *histograms], dim=-1), dim=-1, eps=1e-8)


def _sealed_identity_labels(
    paths: list[Path], *, minimum_tracker_iou: float = 0.30
) -> dict[int, dict[int, int]]:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(value: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(first: tuple[int, int], second: tuple[int, int]) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("information_policy", {}).get("benchmark_labels_used") is not False:
            raise ValueError("identity edge manifest is not label-free")
        for pair in payload["pairs"]:
            source_frame = int(pair["source_frame_id"])
            target_frame = int(pair["target_frame_id"])
            best_by_target = {}
            for edge in pair["edges"]:
                iou = float(edge["tracked_to_target_root_iou"])
                target = int(edge["target_proposal_index"])
                if iou < minimum_tracker_iou or target < 0:
                    continue
                key = (iou, -int(edge["source_proposal_index"]))
                if target not in best_by_target or key > best_by_target[target][0]:
                    best_by_target[target] = (key, edge)
            for _, edge in best_by_target.values():
                join(
                    (source_frame, int(edge["source_proposal_index"])),
                    (target_frame, int(edge["target_proposal_index"])),
                )
    roots = sorted({find(value) for value in parent})
    label_by_root = {root: index for index, root in enumerate(roots)}
    output: dict[int, dict[int, int]] = {}
    for frame_proposal in parent:
        frame, proposal = frame_proposal
        output.setdefault(frame, {})[proposal] = label_by_root[find(frame_proposal)]
    return output


@torch.no_grad()
def _run_method_only(
    *,
    surface,
    cameras,
    frame_ids: list[int],
    sam_records: dict[int, dict[str, Any]],
    image_paths: dict[int, Path] | None,
    descriptor_records: dict[int, Path] | None,
    descriptor_channel_weight: torch.Tensor | None,
    requested_counts: list[int],
    feature_height: int,
    feature_width: int,
    device: torch.device,
    minimum_overlap: float,
    null_logit: float,
    temperature: float,
    geometry_weight: float,
    appearance_weight: float,
    association_mode: str,
    batch_size: int,
    batch_birth_overlap: float,
    sealed_identity_labels: dict[int, dict[int, int]],
) -> tuple[list[MethodSnapshot], list[dict[str, Any]]]:
    """Method boundary: this function has no oracle labels or label paths."""

    model = SurfaceTokenBootstrap(
        surface.centres.to(device),
        minimum_overlap=minimum_overlap,
        null_logit=null_logit,
        temperature=temperature,
        geometry_weight=geometry_weight,
        appearance_weight=appearance_weight,
        batch_birth_overlap=batch_birth_overlap,
    )
    assigned_surface = torch.zeros(surface.num_elements, dtype=torch.bool, device=device)
    positives: list[torch.Tensor] = []
    assignments: list[torch.Tensor] = []
    roots: list[torch.Tensor] = []
    snapshots: list[MethodSnapshot] = []
    view_records: list[dict[str, Any]] = []
    created_total = 0
    pending = []

    for view_index, (frame_id, camera) in enumerate(zip(frame_ids, cameras), start=1):
        cache_path = Path(sam_records[frame_id]["output"]).resolve(strict=True)
        payload = torch.load(cache_path, map_location="cpu")
        masks = _masks(cache_path, feature_height, feature_width)
        positive, visible = _lift_masks(surface, camera, masks, device)
        quality = torch.as_tensor(payload["quality"], dtype=torch.float32, device=device)
        parent_index = torch.as_tensor(payload["parent_index"], dtype=torch.long, device=device)
        if quality.shape != (masks.shape[0],) or parent_index.shape != (masks.shape[0],):
            raise ValueError("SAM hierarchy metadata does not align with packed masks")
        identity_ids = torch.full(
            (masks.shape[0],), -1, dtype=torch.long, device=device
        )
        for proposal_index, identity_id in sealed_identity_labels.get(frame_id, {}).items():
            if not 0 <= proposal_index < masks.shape[0]:
                raise ValueError("sealed identity proposal index escaped SAM cache")
            identity_ids[proposal_index] = identity_id
        evidence = ObservedObjectEvidence.from_positive_visibility(
            positive,
            visible,
            view_ids=torch.full((masks.shape[0],), frame_id, dtype=torch.long, device=device),
            quality=quality,
        )
        descriptors = None
        if appearance_weight > 0:
            if descriptor_records is not None:
                descriptor_payload = torch.load(descriptor_records[frame_id], map_location="cpu")
                descriptors = torch.as_tensor(
                    descriptor_payload["descriptor"], dtype=torch.float32, device=device
                )
                if descriptor_channel_weight is not None:
                    descriptors = descriptors * descriptor_channel_weight.to(device)
                if descriptors.shape[0] != masks.shape[0]:
                    raise ValueError("SAM3 descriptor count differs from mask count")
            else:
                if image_paths is None or frame_id not in image_paths:
                    raise KeyError("appearance matching requires source RGB or sealed descriptors")
                descriptors = _rgb_histogram_descriptors(
                    image_paths[frame_id],
                    masks,
                    height=feature_height,
                    width=feature_width,
                    device=device,
                )
        pending.append((frame_id, positive, visible, parent_index, evidence, descriptors, identity_ids))
        should_commit = association_mode == "online" or len(pending) == batch_size or view_index == len(frame_ids)
        if not should_commit:
            continue
        if association_mode == "online":
            item = pending[0]
            results = [model.process_view(
                item[4],
                element_visibility=item[2],
                parent_index=item[3],
                proposal_descriptors=item[5],
            )]
        elif association_mode == "frozen_batch":
            results = model.process_batch(
                [item[4] for item in pending],
                element_visibilities=[item[2] for item in pending],
                parent_indices=[item[3] for item in pending],
                proposal_descriptors=[item[5] for item in pending],
                proposal_identity_ids=[item[6] for item in pending],
            )
        else:
            raise ValueError("association mode must be online or frozen_batch")
        for item, result in zip(pending, results):
            item_frame, item_positive, _, item_parent, _, _, _ = item
            assigned = result.token_ids >= 0
            if bool(assigned.any()):
                assigned_surface |= item_positive[assigned].amax(0) > 1e-6
            positives.append(item_positive.cpu())
            assignments.append(result.token_ids.cpu())
            roots.append((item_parent < 0).cpu())
            created = int(result.created.sum())
            created_total += created
            view_records.append({
                "frame_id": item_frame,
                "proposal_count": int(item_positive.shape[0]),
                "root_proposal_count": int((item_parent < 0).sum()),
                "assigned_proposal_count": int(assigned.sum()),
                "null_proposal_count": int((~assigned).sum()),
                "created_token_count": created,
                "running_token_count_after_commit": model.num_tokens,
                "commit_ending_view_index": view_index,
            })
        pending.clear()
        processed_view_count = len(positives)
        if processed_view_count in requested_counts:
            snapshots.append(MethodSnapshot(
                view_count=processed_view_count,
                membership=model.membership.cpu().clone(),
                assigned_surface=assigned_surface.cpu().clone(),
                proposal_positive=[value.clone() for value in positives],
                proposal_token_ids=[value.clone() for value in assignments],
                proposal_is_root=[value.clone() for value in roots],
                created_count=created_total,
            ))
    return snapshots, view_records


def _diagnose_snapshot(
    snapshot: MethodSnapshot,
    ground_truth_token: torch.Tensor,
    object_count: int,
    *,
    minimum_purity: float,
    minimum_margin: float,
) -> dict[str, Any]:
    """Oracle boundary: labels score a frozen method snapshot but never alter it."""

    annotated = ground_truth_token >= 0
    oracle = torch.zeros(ground_truth_token.shape[0], object_count)
    oracle[annotated, ground_truth_token[annotated]] = 1
    membership = snapshot.membership.float()
    token_object_overlap = membership.T @ oracle
    annotated_token_mass = token_object_overlap.sum(-1)
    total_token_mass = membership.sum(0).clamp_min(1e-12)
    if membership.shape[1]:
        dominant_mass, dominant_object = token_object_overlap.max(-1)
    else:
        dominant_mass = torch.empty(0)
        dominant_object = torch.empty(0, dtype=torch.long)
    valid_tokens = annotated_token_mass > 1e-8
    purity = dominant_mass / annotated_token_mass.clamp_min(1e-12)
    annotated_fraction = annotated_token_mass / total_token_mass

    object_mass = oracle.sum(0).clamp_min(1e-12)
    if membership.shape[1]:
        best_object_overlap = token_object_overlap.max(0).values
        object_token_sum = token_object_overlap.sum(0)
    else:
        best_object_overlap = torch.zeros(object_count)
        object_token_sum = torch.zeros(object_count)
    object_recall = best_object_overlap / object_mass
    observed_objects = object_token_sum > 1e-8
    split_impurity = 1.0 - best_object_overlap / object_token_sum.clamp_min(1e-12)

    proposal_count = safe_count = safe_assigned = correct = 0
    root_count = root_assigned = 0
    for positive, token_ids, is_root in zip(
        snapshot.proposal_positive, snapshot.proposal_token_ids, snapshot.proposal_is_root
    ):
        overlap = positive @ oracle
        mass = positive.sum(-1).clamp_min(1e-12)
        values, ids = overlap.topk(min(2, object_count), dim=-1)
        best = values[:, 0]
        second = values[:, 1] if values.shape[1] > 1 else torch.zeros_like(best)
        safe = (best / mass >= minimum_purity) & ((best - second) / mass >= minimum_margin)
        assigned = token_ids >= 0
        safe_count += int(safe.sum())
        safe_assigned += int((safe & assigned).sum())
        if bool((safe & assigned).any()):
            selected = safe & assigned
            correct += int((dominant_object[token_ids[selected]] == ids[selected, 0]).sum())
        proposal_count += int(token_ids.numel())
        root_count += int(is_root.sum())
        root_assigned += int((is_root & assigned).sum())

    return {
        "selected_view_count": snapshot.view_count,
        "proposal_count": proposal_count,
        "created_token_count": snapshot.created_count,
        "final_token_count": int(membership.shape[1]),
        "proposal_assignment_rate": float(sum(int((x >= 0).sum()) for x in snapshot.proposal_token_ids)) / max(proposal_count, 1),
        "root_assignment_rate": root_assigned / max(root_count, 1),
        "safe_diagnostic_proposal_count": safe_count,
        "safe_diagnostic_assignment_rate": safe_assigned / max(safe_count, 1),
        "safe_diagnostic_association_accuracy": correct / max(safe_assigned, 1),
        "assigned_proposal_surface_union": float(snapshot.assigned_surface.float().mean()),
        "token_membership_surface_union": float((membership.max(-1).values > 1e-6).float().mean()) if membership.shape[1] else 0.0,
        "valid_annotated_token_count": int(valid_tokens.sum()),
        "token_purity_macro": float(purity[valid_tokens].mean()) if bool(valid_tokens.any()) else 0.0,
        "token_purity_mass_weighted": float(dominant_mass.sum() / annotated_token_mass.sum().clamp_min(1e-12)),
        "token_annotated_fraction_mass_weighted": float(annotated_token_mass.sum() / total_token_mass.sum().clamp_min(1e-12)),
        "strongly_merged_token_count_at_purity_below_0_7": int((valid_tokens & (purity < 0.7)).sum()),
        "observed_object_count": int(observed_objects.sum()),
        "object_best_token_recall_macro": float(object_recall.mean()),
        "observed_object_best_token_recall_macro": float(object_recall[observed_objects].mean()) if bool(observed_objects.any()) else 0.0,
        "observed_object_split_impurity_macro": float(split_impurity[observed_objects].mean()) if bool(observed_objects.any()) else 0.0,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_diagnostic:
        raise PermissionError("final association audit requires explicit instance-oracle authorization")
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    selection_path = Path(args.selection_authority).resolve(strict=True)
    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    segmentation_path = Path(args.segmentation).resolve(strict=True)
    aggregation_path = Path(args.aggregation).resolve(strict=True)
    manifests = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    descriptor_manifests = [Path(value).resolve(strict=True) for value in args.descriptor_manifest]
    identity_edge_manifests = [Path(value).resolve(strict=True) for value in args.identity_edge_manifest]
    sealed_identity_labels = _sealed_identity_labels(
        identity_edge_manifests, minimum_tracker_iou=args.identity_minimum_tracker_iou
    )
    descriptor_records: dict[int, Path] | None = None
    if descriptor_manifests:
        descriptor_records = {}
        for manifest in descriptor_manifests:
            payload = json.loads(manifest.read_text())
            for record in payload["records"]:
                descriptor_records[int(record["frame_id"])] = Path(record["output"]).resolve(strict=True)
    descriptor_metric_path = Path(args.descriptor_metric).resolve(strict=True) if args.descriptor_metric else None
    descriptor_channel_weight = None
    if descriptor_metric_path is not None:
        descriptor_channel_weight = torch.as_tensor(
            torch.load(descriptor_metric_path, map_location="cpu")["channel_weight"], dtype=torch.float32
        )
    selection = json.loads(selection_path.read_text())
    authority = json.loads(authority_path.read_text())
    if authority.get("information_policy", {}).get("benchmark_ground_truth_used") is not False:
        raise ValueError("source authority is not label-free")
    selected = list(map(int, selection["selections"][str(args.maximum_view_count)]["selected_frame_ids"]))
    image_paths = {
        int(str(record["image_id"]).removeprefix("frame_")): Path(record["path"]).resolve(strict=True)
        for record in authority["images"]
    }
    maximum_requested = max(args.view_count)
    frame_ids = selected[:maximum_requested]
    sam_records = _load_sam_records(manifests)
    if not set(frame_ids).issubset(sam_records):
        raise ValueError("selected source frames are missing from SAM manifests")
    cameras = _load_cameras(
        transforms,
        [{"frame_id": value} for value in frame_ids],
        args.feature_height,
        args.feature_width,
    )
    vertices, triangles = _load_mesh(mesh_path)
    mesh = MeshCarrier(vertices, triangles)
    surface = SurfaceVoxelCarrier.from_points(
        vertices,
        args.voxel_size,
        normals=mesh.normals,
        maximum_splat_radius=args.maximum_splat_radius,
        surface_band_voxels=args.surface_band_voxels,
        maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
    )
    device = torch.device(args.device)
    snapshots, view_records = _run_method_only(
        surface=surface,
        cameras=cameras,
        frame_ids=frame_ids,
        sam_records=sam_records,
        image_paths=image_paths if args.appearance_weight > 0 and descriptor_records is None else None,
        descriptor_records=descriptor_records,
        descriptor_channel_weight=descriptor_channel_weight,
        requested_counts=sorted(set(args.view_count)),
        feature_height=args.feature_height,
        feature_width=args.feature_width,
        device=device,
        minimum_overlap=args.minimum_overlap,
        null_logit=args.null_logit,
        temperature=args.temperature,
        geometry_weight=args.geometry_weight,
        appearance_weight=args.appearance_weight,
        association_mode=args.association_mode,
        batch_size=args.batch_size,
        batch_birth_overlap=args.batch_birth_overlap,
        sealed_identity_labels=sealed_identity_labels,
    )

    # Diagnostic labels are deliberately opened only after all method snapshots
    # have been frozen and copied back to CPU.
    vertex_instances, object_ids = _load_instance_labels(
        segmentation_path, aggregation_path, vertices.shape[0]
    )
    lookup = torch.full((int(vertex_instances.max()) + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(object_ids)] = torch.arange(len(object_ids))
    vertex_tokens = lookup[vertex_instances]
    from scipy.spatial import cKDTree
    nearest = cKDTree(vertices.numpy()).query(surface.centres.numpy(), k=1)[1]
    ground_truth_token = vertex_tokens[torch.from_numpy(np.asarray(nearest, dtype=np.int64))]
    ladder = [
        _diagnose_snapshot(
            snapshot,
            ground_truth_token,
            len(object_ids),
            minimum_purity=args.minimum_diagnostic_purity,
            minimum_margin=args.minimum_diagnostic_margin,
        )
        for snapshot in snapshots
    ]

    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_source_only_token_bootstrap",
        coordinate_convention="source_sam_to_surface_tokens_then_isolated_oracle_audit",
        inputs=(
            HashedInput.seal("source_rgb_authority", authority_path),
            HashedInput.seal("label_free_view_selection", selection_path),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
            *tuple(HashedInput.seal(f"sam_manifest_{index}", path) for index, path in enumerate(manifests)),
            *tuple(HashedInput.seal(f"descriptor_manifest_{index}", path) for index, path in enumerate(descriptor_manifests)),
            *((HashedInput.seal("source_association_metric", descriptor_metric_path),) if descriptor_metric_path else ()),
            *tuple(HashedInput.seal(f"source_identity_edges_{index}", path) for index, path in enumerate(identity_edge_manifests)),
            HashedInput.seal("diagnostic_instance_segmentation", segmentation_path),
            HashedInput.seal("diagnostic_instance_aggregation", aggregation_path),
        ),
        source_rgb_opened=args.appearance_weight > 0,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=True,
        metadata={
            "method_stage_oracle_inputs": False,
            "diagnostic_labels_opened_after_method_snapshots": True,
            "query_independent": True,
            "completion_trained": False,
            "deployment_compression_applied": False,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.real_sam_token_association.v1",
        "stage": "source_only_bootstrap_precompletion_diagnostic",
        "method_policy": {
            "token_birth": "unmatched_hierarchy_roots_only",
            "part_proposals_may_reuse_tokens": True,
            "one_to_one_assignment": False,
            "explicit_null": True,
            "minimum_overlap": args.minimum_overlap,
            "null_logit": args.null_logit,
            "temperature": args.temperature,
            "geometry_weight": args.geometry_weight,
            "appearance_descriptor": (
                "official_sam3_masked_vision_feature"
                if descriptor_records is not None
                else "source_mask_rgb_moments_plus_8bin_histogram"
                if args.appearance_weight > 0
                else "none"
            ),
            "appearance_weight": args.appearance_weight,
            "source_trained_descriptor_metric": descriptor_metric_path is not None,
            "association_mode": args.association_mode,
            "batch_size": args.batch_size if args.association_mode == "frozen_batch" else 1,
            "batch_birth_overlap": args.batch_birth_overlap,
            "prototypes_frozen_until_batch_commit": args.association_mode == "frozen_batch",
            "sealed_video_identity_edge_manifest_count": len(identity_edge_manifests),
            "identity_minimum_tracker_to_target_root_iou": args.identity_minimum_tracker_iou,
            "identity_integration": "post_geometry_conflict_free_token_merge",
        },
        "diagnostic_policy": {
            "labels_in_method_path": False,
            "minimum_proposal_purity": args.minimum_diagnostic_purity,
            "minimum_proposal_margin": args.minimum_diagnostic_margin,
        },
        "ladder": ladder,
        "per_view_method_records": view_records,
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--descriptor-manifest", action="append", default=[])
    parser.add_argument("--descriptor-metric", default="")
    parser.add_argument("--identity-edge-manifest", action="append", default=[])
    parser.add_argument("--identity-minimum-tracker-iou", type=float, default=0.50)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--maximum-view-count", type=int, default=64)
    parser.add_argument("--view-count", action="append", type=int, default=[])
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--minimum-overlap", type=float, default=0.20)
    parser.add_argument("--null-logit", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--appearance-weight", type=float, default=0.0)
    parser.add_argument("--association-mode", choices=("online", "frozen_batch"), default="online")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-birth-overlap", type=float, default=0.20)
    parser.add_argument("--minimum-diagnostic-purity", type=float, default=0.70)
    parser.add_argument("--minimum-diagnostic-margin", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--allow-instance-oracle-diagnostic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.view_count:
        args.view_count = [16, 32]
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.association_mode == "frozen_batch" and any(
        count % args.batch_size for count in args.view_count
    ):
        raise ValueError("requested view counts must align with frozen batch boundaries")
    if args.identity_edge_manifest and args.association_mode != "frozen_batch":
        raise ValueError("sealed identity edges currently require frozen batch association")
    report = run(args)
    print(json.dumps(report["ladder"], indent=2))


if __name__ == "__main__":
    main()
