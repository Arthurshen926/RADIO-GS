"""Train a scene-disjoint fragment-pair association diagnostic on ScanNet.

The complete instance labels are supervision only.  Model inputs are rebuilt
from the four sealed source-camera observations and the same scalar pair
contract available for query-free LERF fragments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts.extract_official_crop_summary_teacher import _atomic_torch_save
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.carrier import SurfaceVoxelCarrier
from radio_gs.v4.completion.scannet import camera_from_record, load_scene_cache
from radio_gs.v4.object_memory.fragment_to_object import (
    _constrained_groups,
    _disjoint_same_view_cannot_link,
    _pairwise_appearance,
    _pairwise_geometry,
    _pairwise_visibility_conflict,
)
from radio_gs.v4.object_memory.learned_fragment_affinity import (
    PAIR_FEATURE_LAYOUT,
    FragmentAffinityMLP,
    SetConditionedFragmentAffinity,
    balanced_fragment_affinity_loss,
    fragment_affinity_metrics,
    fragment_pair_features,
    select_fragment_affinity_threshold,
)


SCHEMA = "radio_gs.surface_object_memory_v4.scannet_fragment_affinity.v2"
CHECKPOINT_SCHEMA = "radio_gs.surface_object_memory_v4.fragment_affinity_checkpoint.v1"


def _calibrate_cross_view_appearance(
    raw_appearance: torch.Tensor, view_index: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    count = raw_appearance.shape[0]
    cross_view = view_index[:, None] != view_index[None, :]
    values = raw_appearance[
        cross_view & ~torch.eye(count, dtype=torch.bool)
    ]
    if not values.numel():
        raise ValueError("appearance calibration requires cross-view pairs")
    floor = torch.quantile(values, 0.50)
    ceiling = torch.quantile(values, 0.95)
    calibrated = (
        (raw_appearance - floor) / (ceiling - floor).clamp_min(1e-6)
    ).clamp(0, 1)
    return calibrated, {
        "appearance_calibration_median": float(floor),
        "appearance_calibration_q95": float(ceiling),
    }


def _deterministic_subset(
    elements: torch.Tensor, *, retained_fraction: float, seed: int
) -> torch.Tensor:
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained fragment fraction must lie in (0,1]")
    if retained_fraction == 1 or elements.numel() <= 1:
        return elements
    generator = torch.Generator().manual_seed(int(seed))
    count = max(1, int(round(elements.numel() * retained_fraction)))
    order = torch.randperm(elements.numel(), generator=generator)[:count]
    return elements[order]


def _source_view_fragments(
    *,
    visible: torch.Tensor,
    token_index: torch.Tensor,
    centres: torch.Tensor,
    object_count: int,
    view_id: int,
    noise_mode: str,
    parts_per_object: int,
    retained_fraction: float,
    light_merge_fraction: float,
) -> list[tuple[torch.Tensor, int, str, float]]:
    """Construct deterministic source-only fragments and their oracle labels.

    The oracle instance ids choose supervision and synthetic corruption only;
    they never enter the pair features.  ``sam_like_v1`` retains a full region,
    adds spatially coherent part regions, drops boundary-like samples, and
    lightly contaminates a deterministic subset with a neighbouring object.
    """

    if noise_mode not in {"clean_oracle", "sam_like_v1"}:
        raise ValueError("unsupported fragment noise mode")
    if parts_per_object < 1:
        raise ValueError("parts per object must be positive")
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained fragment fraction must lie in (0,1]")
    if not 0 <= light_merge_fraction < 0.5:
        raise ValueError("light merge fraction must lie in [0,0.5)")

    fragments: list[tuple[torch.Tensor, int, str, float]] = []
    by_object = [visible[token_index[visible] == token] for token in range(object_count)]
    for token_id, whole in enumerate(by_object):
        if not whole.numel():
            continue
        candidates: list[tuple[torch.Tensor, str]] = [(whole, "whole")]
        if noise_mode == "sam_like_v1" and whole.numel() >= 2 * parts_per_object:
            axis = (view_id + token_id) % 3
            ordered = whole[torch.argsort(centres[whole, axis], stable=True)]
            candidates.extend(
                (part, "part")
                for part in torch.tensor_split(ordered, parts_per_object)
                if part.numel()
            )
        for part_id, (elements, kind) in enumerate(candidates):
            if noise_mode == "sam_like_v1":
                elements = _deterministic_subset(
                    elements,
                    retained_fraction=retained_fraction,
                    seed=1_000_003 * view_id + 10_007 * token_id + 101 * part_id,
                )
            mixed = False
            if (
                noise_mode == "sam_like_v1"
                and light_merge_fraction > 0
                and (view_id + token_id + part_id) % 3 == 0
            ):
                foreign = visible[token_index[visible] != token_id]
                if foreign.numel():
                    requested = max(
                        1,
                        int(round(
                            elements.numel()
                            * light_merge_fraction
                            / (1.0 - light_merge_fraction)
                        )),
                    )
                    centroid = centres[elements].mean(0)
                    distance = (centres[foreign] - centroid).square().sum(-1)
                    contaminant = foreign[
                        torch.argsort(distance, stable=True)[:requested]
                    ]
                    elements = torch.unique(torch.cat((elements, contaminant)))
                    mixed = bool(contaminant.numel())
            purity = float((token_index[elements] == token_id).float().mean())
            fragments.append((elements, token_id, "light_merge" if mixed else kind, purity))
    return fragments


def _object_equal_pair_weights(
    first_token: torch.Tensor,
    second_token: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Equalize objects/object-pairs, then equalize every scene during sampling."""

    first_token = torch.as_tensor(first_token, dtype=torch.long)
    second_token = torch.as_tensor(second_token, dtype=torch.long)
    target = torch.as_tensor(target, dtype=torch.bool)
    if first_token.shape != target.shape or second_token.shape != target.shape:
        raise ValueError("pair-token receipts must align with targets")
    weights = torch.zeros(target.shape, dtype=torch.float32)
    for class_mask in (target, ~target):
        ids = torch.where(class_mask)[0]
        if not ids.numel():
            raise ValueError("object-equal weights need both pair classes")
        a = torch.minimum(first_token[ids], second_token[ids])
        b = torch.maximum(first_token[ids], second_token[ids])
        keys = torch.stack((a, b), dim=-1)
        _, inverse, counts = torch.unique(
            keys, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        class_weights = counts[inverse].float().reciprocal()
        weights[ids] = class_weights / class_weights.sum()
    return weights


@torch.no_grad()
def build_scene_pair_dataset(
    payload: dict[str, Any],
    *,
    fragment_noise_mode: str = "clean_oracle",
    parts_per_object: int = 2,
    retained_fraction: float = 0.85,
    light_merge_fraction: float = 0.15,
    pair_scope: str = "cross_view",
) -> dict[str, Any]:
    if pair_scope not in {"cross_view", "all_views"}:
        raise ValueError("fragment pair scope is unsupported")
    centres = torch.as_tensor(payload["centres"], dtype=torch.float32).cpu()
    normals = torch.as_tensor(payload["normals"], dtype=torch.float32).cpu()
    local = torch.as_tensor(payload["local_features"], dtype=torch.float32).cpu()
    token_index = torch.as_tensor(payload["token_index"], dtype=torch.long).cpu()
    object_ids = list(map(int, payload["object_ids"]))
    config = payload["configuration"]
    carrier = SurfaceVoxelCarrier(
        centres,
        float(config["voxel_size"]),
        normals=normals,
        maximum_splat_radius=int(config["maximum_splat_radius"]),
        surface_band_voxels=float(config["surface_band_voxels"]),
        maximum_contributors_per_pixel=int(
            config["maximum_contributors_per_pixel"]
        ),
    )
    cameras = [
        camera_from_record(record) for record in payload["observation_cameras"]
    ]
    kept = {
        (str(record["frame_id"]), int(record["object_id"]))
        for record in payload["mask_dropout_receipt"]["records"]
        if bool(record["kept"])
    }
    view_visibility = torch.zeros(
        len(cameras), carrier.num_elements, dtype=torch.bool
    )
    fragment_positive = []
    fragment_view = []
    fragment_token = []
    fragment_descriptor = []
    fragment_kind = []
    fragment_purity = []
    for view_id, camera in enumerate(cameras):
        visible = torch.unique(carrier.project(camera).element_ids)
        view_visibility[view_id, visible] = True
        retained_token = torch.tensor([
            (str(camera.key), object_id) in kept for object_id in object_ids
        ], dtype=torch.bool)
        visible_token = token_index[visible]
        valid_token = (visible_token >= 0) & (visible_token < len(object_ids))
        retained_mask = torch.zeros(visible.shape, dtype=torch.bool)
        retained_mask[valid_token] = retained_token[visible_token[valid_token]]
        retained_visible = visible[retained_mask]
        for elements, token_id, kind, purity in _source_view_fragments(
            visible=retained_visible,
            token_index=token_index,
            centres=centres,
            object_count=len(object_ids),
            view_id=view_id,
            noise_mode=fragment_noise_mode,
            parts_per_object=parts_per_object,
            retained_fraction=retained_fraction,
            light_merge_fraction=light_merge_fraction,
        ):
            mask = torch.zeros(carrier.num_elements, dtype=torch.float32)
            mask[elements] = 1
            fragment_positive.append(mask)
            fragment_view.append(view_id)
            fragment_token.append(token_id)
            fragment_kind.append(kind)
            fragment_purity.append(purity)
            # Use the fixed projected RADIO slice as an appearance descriptor;
            # RGB availability and normals already affect other contracts.
            fragment_descriptor.append(local[elements, 4:68].mean(0))
    if len(fragment_positive) < 3:
        raise ValueError("scene has too few retained source-view fragments")
    positive = torch.stack(fragment_positive)
    views = torch.tensor(fragment_view, dtype=torch.long)
    tokens = torch.tensor(fragment_token, dtype=torch.long)
    descriptors = torch.stack(fragment_descriptor)[:, None]
    geometry, spatial, _mass = _pairwise_geometry(
        positive, centres, evidence_threshold=0.5
    )
    raw_appearance = _pairwise_appearance(descriptors, positive.shape[0])
    appearance, calibration = _calibrate_cross_view_appearance(
        raw_appearance, views
    )
    conflict = _pairwise_visibility_conflict(
        positive,
        views,
        view_visibility,
        evidence_threshold=0.5,
    )
    pair = fragment_pair_features(geometry, appearance, spatial, conflict)
    pair_mask = (
        views[:, None] != views[None, :]
        if pair_scope == "cross_view"
        else torch.ones((views.numel(), views.numel()), dtype=torch.bool)
    )
    first, second = torch.where(torch.triu(pair_mask, diagonal=1))
    target = tokens[first] == tokens[second]
    if not bool(target.any()) or not bool((~target).any()):
        raise ValueError("scene fragment pairs need positive and negative labels")
    sampling_weight = _object_equal_pair_weights(
        tokens[first], tokens[second], target
    )
    return {
        "features": pair[first, second],
        "pair_feature_matrix": pair,
        "target": target,
        "sampling_weight": sampling_weight,
        "pair_first": first,
        "pair_second": second,
        "fragment_view_index": views,
        "fragment_object_index": tokens,
        "fragment_geometry": geometry,
        "partition_cannot_link_policy": (
            "disjoint_same_view" if pair_scope == "all_views" else "exact_same_view"
        ),
        "audit": {
            "scene_id": str(payload["scene_id"]),
            "source_view_count": len(cameras),
            "fragment_count": int(positive.shape[0]),
            "pair_count": int(target.numel()),
            "positive_pair_count": int(target.sum()),
            "negative_pair_count": int((~target).sum()),
            "object_count": len(object_ids),
            "fragment_noise_mode": fragment_noise_mode,
            "pair_scope": pair_scope,
            "partition_cannot_link_implementation": "nonempty_zero_overlap_v2",
            "same_view_pair_count": int((views[first] == views[second]).sum()),
            "cross_view_pair_count": int((views[first] != views[second]).sum()),
            "whole_fragment_count": fragment_kind.count("whole"),
            "part_fragment_count": fragment_kind.count("part"),
            "light_merge_fragment_count": fragment_kind.count("light_merge"),
            "mean_fragment_purity": sum(fragment_purity) / len(fragment_purity),
            "minimum_fragment_purity": min(fragment_purity),
            "positive_sampling_weight_sum": float(sampling_weight[target].sum()),
            "negative_sampling_weight_sum": float(sampling_weight[~target].sum()),
            **calibration,
        },
    }


@torch.no_grad()
def fragment_partition_metrics(
    pair_probability: torch.Tensor,
    dataset: dict[str, Any],
    *,
    decision_threshold: float,
) -> dict[str, float | int | str]:
    """Measure the deployed global partition, not only independent edges."""

    probability = torch.as_tensor(pair_probability, dtype=torch.float32).cpu()
    first = torch.as_tensor(dataset["pair_first"], dtype=torch.long).cpu()
    second = torch.as_tensor(dataset["pair_second"], dtype=torch.long).cpu()
    views = torch.as_tensor(dataset["fragment_view_index"], dtype=torch.long).cpu()
    objects = torch.as_tensor(dataset["fragment_object_index"], dtype=torch.long).cpu()
    if probability.shape != first.shape or second.shape != first.shape:
        raise ValueError("partition pair probabilities and receipts do not align")
    if views.shape != objects.shape or not views.numel():
        raise ValueError("partition fragment metadata does not align")
    affinity = torch.zeros((views.numel(), views.numel()), dtype=torch.float32)
    affinity[first, second] = probability
    affinity[second, first] = probability
    affinity.fill_diagonal_(1)
    groups = _constrained_groups(
        affinity,
        torch.arange(views.numel()),
        views,
        minimum_affinity=decision_threshold,
        cannot_link=(
            _disjoint_same_view_cannot_link(dataset["fragment_geometry"], views)
            if dataset.get("partition_cannot_link_policy") == "disjoint_same_view"
            else None
        ),
    )
    assigned_group = torch.empty(views.numel(), dtype=torch.long)
    impure_fragments = 0
    pure_groups = 0
    for group_id, members in enumerate(groups):
        ids = torch.as_tensor(members, dtype=torch.long)
        assigned_group[ids] = group_id
        _, counts = torch.unique(objects[ids], return_counts=True)
        impure_fragments += int(ids.numel() - counts.max())
        pure_groups += int(counts.numel() == 1)
    fragmentation = []
    best_recall = []
    best_iou = []
    for object_id in torch.unique(objects, sorted=True):
        ids = torch.where(objects == object_id)[0]
        group_ids, intersection = torch.unique(
            assigned_group[ids], return_counts=True
        )
        group_sizes = torch.tensor([
            len(groups[int(group_id)]) for group_id in group_ids
        ])
        union = ids.numel() + group_sizes - intersection
        fragmentation.append(float(group_ids.numel()))
        best_recall.append(float(intersection.max() / ids.numel()))
        best_iou.append(float((intersection.float() / union).max()))
    return {
        "fragment_count": int(views.numel()),
        "cannot_link_implementation": "nonempty_zero_overlap_v2",
        "oracle_object_count": int(torch.unique(objects).numel()),
        "predicted_hypothesis_count": len(groups),
        "mean_object_fragmentation": sum(fragmentation) / len(fragmentation),
        "object_best_hypothesis_recall": sum(best_recall) / len(best_recall),
        "object_best_hypothesis_iou": sum(best_iou) / len(best_iou),
        "fragment_weighted_merge_impurity": impure_fragments / views.numel(),
        "pure_hypothesis_fraction": pure_groups / len(groups),
    }


def _cohort_scene_ids(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text())
    split = payload.get("split", {})
    training = list(map(str, split.get("training_scene_ids", [])))
    validation = list(map(str, split.get("validation_scene_ids", [])))
    if not training or not validation or set(training) & set(validation):
        raise ValueError("fragment affinity cohort split is invalid")
    if set(training) | set(validation) != set(map(str, payload.get("scene_ids", []))):
        raise ValueError("fragment affinity split does not cover the cohort inventory")
    return training, validation, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": payload.get("schema"),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    datasets: list[dict[str, Any]],
    device: torch.device,
    decision_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    per_scene = {}
    probabilities = []
    targets = []
    for dataset in datasets:
        if getattr(model, "set_conditioned", False):
            logits = model(
                dataset["pair_feature_matrix"].to(device),
                dataset["fragment_view_index"].to(device),
            )[
                dataset["pair_first"].to(device),
                dataset["pair_second"].to(device),
            ]
        else:
            logits = model(dataset["features"].to(device))
        probability = torch.sigmoid(logits).cpu()
        target = dataset["target"]
        scene_id = dataset["audit"]["scene_id"]
        per_scene[scene_id] = {
            **fragment_affinity_metrics(
                probability, target, decision_threshold=decision_threshold
            ),
            **fragment_partition_metrics(
                probability,
                dataset,
                decision_threshold=decision_threshold,
            ),
        }
        probabilities.append(probability)
        targets.append(target)
    pooled = fragment_affinity_metrics(
        torch.cat(probabilities),
        torch.cat(targets),
        decision_threshold=decision_threshold,
    )
    numeric = {
        key: sum(float(value[key]) for value in per_scene.values()) / len(per_scene)
        for key in (
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "average_precision",
        )
    }
    partition_keys = (
        "predicted_hypothesis_count",
        "mean_object_fragmentation",
        "object_best_hypothesis_recall",
        "object_best_hypothesis_iou",
        "fragment_weighted_merge_impurity",
        "pure_hypothesis_fraction",
    )
    partition_macro = {
        key: sum(float(value[key]) for value in per_scene.values()) / len(per_scene)
        for key in partition_keys
    }
    stability = {
        f"{key}_minimum": min(float(value[key]) for value in per_scene.values())
        for key in (
            "object_best_hypothesis_recall",
            "object_best_hypothesis_iou",
            "pure_hypothesis_fraction",
        )
    }
    return per_scene, {
        "scene_macro": numeric,
        "pooled": pooled,
        "partition_scene_macro": partition_macro,
        "partition_heldout_stability": stability,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_training:
        raise PermissionError(
            "fragment affinity uses ScanNet instance labels as training supervision; "
            "pass --allow-instance-oracle-training explicitly"
        )
    if args.step_count <= 0 or args.batch_size < 2 or args.cpu_threads <= 0:
        raise ValueError("fragment affinity training sizes must be positive")
    if args.parts_per_object < 1:
        raise ValueError("parts per object must be positive")
    if not 0 < args.retained_fragment_fraction <= 1:
        raise ValueError("retained fragment fraction must lie in (0,1]")
    if not 0 <= args.light_merge_fraction < 0.5:
        raise ValueError("light merge fraction must lie in [0,0.5)")
    torch.set_num_threads(int(args.cpu_threads))
    training_ids, validation_ids, cohort_receipt = _cohort_scene_ids(
        Path(args.cohort_manifest)
    )
    if set(args.training_scene) != set(training_ids):
        raise ValueError("training scenes differ from the sealed cohort")
    if set(args.validation_scene) != set(validation_ids):
        raise ValueError("validation scenes differ from the sealed cohort")
    cache_by_scene = {}
    cache_receipts = []
    for value in args.scene_cache:
        path = Path(value).resolve(strict=True)
        payload = load_scene_cache(path)
        scene_id = str(payload["scene_id"])
        if scene_id in cache_by_scene:
            raise ValueError("duplicate fragment affinity scene cache")
        cache_by_scene[scene_id] = payload
        cache_receipts.append(
            {"scene_id": scene_id, "path": str(path), "sha256": sha256_file(path)}
        )
    expected = set(training_ids) | set(validation_ids)
    if set(cache_by_scene) != expected:
        raise ValueError("fragment affinity caches differ from the cohort")
    dataset_kwargs = {
        "fragment_noise_mode": str(args.fragment_noise_mode),
        "parts_per_object": int(args.parts_per_object),
        "retained_fraction": float(args.retained_fragment_fraction),
        "light_merge_fraction": float(args.light_merge_fraction),
        "pair_scope": str(args.pair_scope),
    }
    train = [
        build_scene_pair_dataset(cache_by_scene[key], **dataset_kwargs)
        for key in training_ids
    ]
    validation = [
        build_scene_pair_dataset(cache_by_scene[key], **dataset_kwargs)
        for key in validation_ids
    ]
    train_features = torch.cat([item["features"] for item in train])
    train_target = torch.cat([item["target"] for item in train])
    train_sampling_weight = torch.cat([
        item["sampling_weight"] for item in train
    ])
    positive_ids = torch.where(train_target)[0]
    negative_ids = torch.where(~train_target)[0]
    positive_sampling_weight = train_sampling_weight[positive_ids]
    negative_sampling_weight = train_sampling_weight[negative_ids]
    device = torch.device(args.device)
    feature_mode = getattr(args, "feature_mode", "full")
    if feature_mode != "full" and args.model_kind != "pair_mlp":
        raise ValueError("geometry-only ablation currently requires pair_mlp")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed))
    if args.model_kind == "pair_mlp":
        model: torch.nn.Module = FragmentAffinityMLP(
            int(args.hidden_dimension), feature_mode
        ).to(device)
    elif args.model_kind == "set_context_mlp":
        model = SetConditionedFragmentAffinity(
            int(args.hidden_dimension)
        ).to(device)
    else:
        raise ValueError("fragment affinity model kind is unsupported")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    half = max(1, int(args.batch_size) // 2)
    loss_history = []
    model.train()
    for step in range(int(args.step_count)):
        if getattr(model, "set_conditioned", False):
            scene_index = int(torch.randint(
                len(train), (1,), generator=generator
            ))
            dataset = train[scene_index]
            target = dataset["target"]
            scene_positive_ids = torch.where(target)[0]
            scene_negative_ids = torch.where(~target)[0]
            scene_weight = dataset["sampling_weight"]
            positive = scene_positive_ids[torch.multinomial(
                scene_weight[scene_positive_ids],
                half,
                replacement=True,
                generator=generator,
            )]
            negative = scene_negative_ids[torch.multinomial(
                scene_weight[scene_negative_ids],
                half,
                replacement=True,
                generator=generator,
            )]
            indices = torch.cat((positive, negative))
            matrix_logits = model(
                dataset["pair_feature_matrix"].to(device),
                dataset["fragment_view_index"].to(device),
            )
            pair_logits = matrix_logits[
                dataset["pair_first"].to(device),
                dataset["pair_second"].to(device),
            ]
            loss = balanced_fragment_affinity_loss(
                pair_logits[indices.to(device)], target[indices].to(device)
            )
        else:
            positive = positive_ids[torch.multinomial(
                positive_sampling_weight, half, replacement=True, generator=generator
            )]
            negative = negative_ids[torch.multinomial(
                negative_sampling_weight, half, replacement=True, generator=generator
            )]
            indices = torch.cat((positive, negative))
            logits = model(train_features[indices].to(device))
            loss = balanced_fragment_affinity_loss(
                logits, train_target[indices].to(device)
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(int(args.step_count) // 20, 1) == 0:
            loss_history.append({"step": step + 1, "loss": float(loss.detach())})
    implementation_path = Path(__file__).resolve(strict=True)
    primitive_path = Path(
        __file__.replace(
            "training/train_scannet_fragment_affinity.py",
            "object_memory/learned_fragment_affinity.py",
        )
    ).resolve(strict=True)
    model.eval()
    with torch.no_grad():
        validation_probability = torch.cat([
            torch.sigmoid(
                model(
                    item["pair_feature_matrix"].to(device),
                    item["fragment_view_index"].to(device),
                )[
                    item["pair_first"].to(device),
                    item["pair_second"].to(device),
                ]
                if getattr(model, "set_conditioned", False)
                else model(item["features"].to(device))
            ).cpu()
            for item in validation
        ])
    validation_target = torch.cat([item["target"] for item in validation])
    threshold_selection = select_fragment_affinity_threshold(
        validation_probability, validation_target, beta=0.5
    )
    decision_threshold = float(threshold_selection["decision_threshold"])
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "hidden_dimension": int(args.hidden_dimension),
        "model_kind": str(args.model_kind),
        "feature_mode": feature_mode,
        "pair_feature_layout": list(PAIR_FEATURE_LAYOUT),
        "training_scene_ids": training_ids,
        "validation_scene_ids": validation_ids,
        "decision_threshold": decision_threshold,
        "threshold_selection": threshold_selection,
        "fragment_noise_contract": dataset_kwargs,
        "training_sampling": "scene_and_object_equal_with_class_balanced_batches",
        "implementation_sha256": sha256_file(implementation_path),
        "primitive_implementation_sha256": sha256_file(primitive_path),
    }
    checkpoint_path = Path(args.output_checkpoint).resolve()
    # Keep the deployment-domain receipt inside the hash-bound checkpoint,
    # not only in a sidecar report that inference never reads.
    checkpoint.update({
        "appearance_descriptor_contract": "mean_local_projected_radio_F71_slice_4_68",
        "deployment_siglip2_crop_summary_domain_validated": False,
        "real_sam_fragment_noise_modeled": False,
    })
    if checkpoint_path.exists():
        raise FileExistsError(f"fragment affinity checkpoint exists: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(checkpoint, checkpoint_path)
    train_per_scene, train_metrics = _evaluate(
        model, train, device, decision_threshold
    )
    val_per_scene, val_metrics = _evaluate(
        model, validation, device, decision_threshold
    )
    report = {
        "feature_mode": feature_mode,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "schema": SCHEMA,
        "stage": "scene_disjoint_fragment_pair_oracle_diagnostic",
        "complete_instance_ids_are_model_inputs": False,
        "complete_instance_ids_are_supervision_only": True,
        "source_view_fragments_are_oracle_identity_masks": (
            args.fragment_noise_mode == "clean_oracle"
        ),
        "real_sam_fragment_noise_modeled": False,
        "synthetic_sam_like_fragment_noise_modeled": (
            args.fragment_noise_mode == "sam_like_v1"
        ),
        "appearance_descriptor_contract": "mean_local_projected_radio_F71_slice_4_68",
        "deployment_siglip2_crop_summary_domain_validated": False,
        "fragment_noise_contract": dataset_kwargs,
        "pair_feature_layout": list(PAIR_FEATURE_LAYOUT),
        "cohort": cohort_receipt,
        "training_scene_ids": training_ids,
        "validation_scene_ids": validation_ids,
        "training_configuration": {
            "seed": int(args.seed),
            "step_count": int(args.step_count),
            "batch_size": int(args.batch_size),
            "hidden_dimension": int(args.hidden_dimension),
            "model_kind": str(args.model_kind),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "decision_threshold": decision_threshold,
            "threshold_selection": threshold_selection,
            "sampling": "scene_and_object_equal_with_class_balanced_batches",
        },
        "training_loss": loss_history,
        "training_dataset_audit": [item["audit"] for item in train],
        "validation_dataset_audit": [item["audit"] for item in validation],
        "training_per_scene": train_per_scene,
        "training_metrics": train_metrics,
        "validation_per_scene": val_per_scene,
        "validation_metrics": val_metrics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "scene_cache_receipts": cache_receipts,
        "implementation_sha256": sha256_file(implementation_path),
        "primitive_implementation_sha256": sha256_file(primitive_path),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"fragment affinity report exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validation_metrics": val_metrics}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--training-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--step-count", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dimension", type=int, default=32)
    parser.add_argument("--feature-mode", choices=("full", "geometry_only"), default="full")
    parser.add_argument(
        "--model-kind",
        choices=("pair_mlp", "set_context_mlp"),
        default="pair_mlp",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--fragment-noise-mode",
        choices=("clean_oracle", "sam_like_v1"),
        default="clean_oracle",
    )
    parser.add_argument("--parts-per-object", type=int, default=2)
    parser.add_argument("--retained-fragment-fraction", type=float, default=0.85)
    parser.add_argument("--light-merge-fraction", type=float, default=0.15)
    parser.add_argument(
        "--pair-scope",
        choices=("cross_view", "all_views"),
        default="cross_view",
    )
    parser.add_argument("--allow-instance-oracle-training", action="store_true")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
