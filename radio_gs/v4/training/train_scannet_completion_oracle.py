"""Train and evaluate a scene-disjoint supervised object completion oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.carrier import SurfaceVoxelCarrier
from radio_gs.v4.completion import (
    OracleIdentityCompletionMLP,
    PartialObjectMembership,
    build_feature_cosine_similarity,
    build_pair_features,
    build_token_context,
    complete_unknown_only,
    completion_metrics,
)
from radio_gs.v4.completion.scannet import camera_from_record, load_scene_cache
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.contracts.source_split import SourceSplit


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.scannet_completion_oracle.v10"
CHECKPOINT_SCHEMA = "radio_gs.surface_object_memory_v4.completion_oracle_checkpoint.v10"
TOKEN_CARDINALITY_NORMALIZATION = "none_raw_learned_null_logit"
RGB_GEOMETRY_LAYOUT = (
    "source_rgb_r",
    "source_rgb_g",
    "source_rgb_b",
    "source_rgb_available",
    "normal_x",
    "normal_y",
    "normal_z",
)
RADIO_JL64_LAYOUT = tuple(
    f"source_radio_fixed_jl64_{index:02d}" for index in range(64)
)
RGB_RADIO_GEOMETRY_LAYOUT = (
    *RGB_GEOMETRY_LAYOUT[:4],
    *RADIO_JL64_LAYOUT,
    *RGB_GEOMETRY_LAYOUT[-3:],
)
LOCAL_FEATURE_MODES = ("rgb_geometry", "rgb_radio_geometry")
UNKNOWN_SAMPLING_MODES = (
    "token_uniform",
    "token_visibility_stratum_balanced",
)
SCORING_MODES = (
    "mlp",
    "mlp_radio_cosine_residual",
    "availability_dual_mlp",
)
RADIO_FEATURE_START = 4
RADIO_FEATURE_STOP = RADIO_FEATURE_START + len(RADIO_JL64_LAYOUT)
RADIO_ALIGNMENT_CONTROLS = (
    "aligned",
    "shuffled_within_observation_stratum",
)
ALIGNED_RADIO_REFERENCE = {
    "local_feature_mode": "rgb_radio_geometry",
    "unknown_sampling_mode": "token_uniform",
    "scoring_mode": "mlp",
    "radio_alignment_control": "aligned",
    "hidden_dimension": 128,
}


def _physical_scene_family(scene_id: str) -> str:
    parts = str(scene_id).split("_")
    if (
        len(parts) == 2
        and parts[0].startswith("scene")
        and parts[0][len("scene") :].isdigit()
        and parts[1].isdigit()
    ):
        return parts[0]
    return str(scene_id)


def _changed_factors_against_aligned_radio(args: argparse.Namespace) -> list[str]:
    current = {
        "local_feature_mode": args.local_feature_mode,
        "unknown_sampling_mode": args.unknown_sampling_mode,
        "scoring_mode": args.scoring_mode,
        "radio_alignment_control": getattr(args, "radio_alignment_control", "aligned"),
        "hidden_dimension": int(args.hidden_dimension),
    }
    return [
        key for key, reference in ALIGNED_RADIO_REFERENCE.items()
        if current[key] != reference
    ]


def _ablation_scope(changed_factors: list[str]) -> str:
    if not changed_factors:
        return "aligned_radio_reference_arm"
    if len(changed_factors) == 1:
        return f"single_factor:{changed_factors[0]}"
    return "multi_factor:" + ",".join(changed_factors)
SOFT_IOU_COHORTS = (
    "full_3d",
    "unknown_3d",
    "visible_but_unmasked_3d",
    "never_visible_3d",
    "heldout_2d",
)


def _select_local_features(
    payload: dict[str, Any], mode: str
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if mode not in LOCAL_FEATURE_MODES:
        raise ValueError(f"unsupported local feature mode {mode!r}")
    features = torch.as_tensor(payload["local_features"], dtype=torch.float32)
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("completion local features must be a finite [E, F] tensor")
    layout = tuple(payload["configuration"].get("local_feature_layout", ()))
    if features.shape[1] != len(layout):
        raise ValueError("completion local feature tensor/layout dimension mismatch")
    if layout == RGB_GEOMETRY_LAYOUT:
        if mode != "rgb_geometry":
            raise ValueError("rgb_radio_geometry requires the sealed F71 RADIO cache layout")
        return features, RGB_GEOMETRY_LAYOUT
    if layout != RGB_RADIO_GEOMETRY_LAYOUT:
        raise ValueError("completion cache has an unsupported local feature layout")
    if mode == "rgb_geometry":
        selected = torch.cat((features[:, :4], features[:, -3:]), dim=-1)
        return selected, RGB_GEOMETRY_LAYOUT
    return features, RGB_RADIO_GEOMETRY_LAYOUT


def _row_multiset_sha256(values: torch.Tensor) -> str:
    values = torch.as_tensor(values, dtype=torch.float32).contiguous().cpu()
    row_digests = sorted(
        hashlib.sha256(row.numpy().astype("<f4", copy=False).tobytes()).digest()
        for row in values
    )
    return hashlib.sha256(b"".join(row_digests)).hexdigest()


def _shuffle_radio_within_observation_strata(
    local_features: torch.Tensor,
    *,
    scene_id: str,
    membership_observed: torch.Tensor,
    source_visible: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Destroy RADIO/element alignment without changing its row distribution."""

    features = torch.as_tensor(local_features, dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != len(RGB_RADIO_GEOMETRY_LAYOUT):
        raise ValueError("RADIO alignment control requires the sealed F71 layout")
    membership_observed = torch.as_tensor(membership_observed, dtype=torch.bool)
    source_visible = torch.as_tensor(source_visible, dtype=torch.bool)
    if membership_observed.shape != (features.shape[0],) or source_visible.shape != (
        features.shape[0],
    ):
        raise ValueError("RADIO shuffle strata must align with surface elements")
    if bool((membership_observed & ~source_visible).any()):
        raise ValueError("membership-observed RADIO rows must be source-visible")
    result = features.clone()
    radio_before = features[:, RADIO_FEATURE_START:RADIO_FEATURE_STOP]
    mappings: list[tuple[int, int]] = []
    strata_receipts = []
    strata = {
        "membership_observed": membership_observed,
        "visible_but_unmasked": source_visible & ~membership_observed,
    }
    for stratum_name, mask in strata.items():
        indices = torch.where(mask)[0]
        ranked = sorted(
            map(int, indices.tolist()),
            key=lambda element_index: hashlib.sha256(
                "\0".join(
                    (str(scene_id), str(int(seed)), stratum_name, str(element_index))
                ).encode("utf-8")
            ).hexdigest(),
        )
        fixed_points = 0
        stratum_mappings: list[tuple[int, int]] = []
        if len(ranked) > 1:
            source_indices = ranked[-1:] + ranked[:-1]
            destination = torch.tensor(ranked, dtype=torch.long)
            source = torch.tensor(source_indices, dtype=torch.long)
            result[destination, RADIO_FEATURE_START:RADIO_FEATURE_STOP] = radio_before[
                source
            ]
            stratum_mappings = list(zip(ranked, source_indices))
        elif ranked:
            stratum_mappings = [(ranked[0], ranked[0])]
            fixed_points = 1
        mappings.extend(stratum_mappings)
        mapping_bytes = b"".join(
            int(destination).to_bytes(8, "little", signed=False)
            + int(source).to_bytes(8, "little", signed=False)
            for destination, source in stratum_mappings
        )
        stratum_before_multiset = _row_multiset_sha256(radio_before[indices])
        stratum_after_multiset = _row_multiset_sha256(
            result[indices, RADIO_FEATURE_START:RADIO_FEATURE_STOP]
        )
        if stratum_before_multiset != stratum_after_multiset:
            raise RuntimeError(
                f"RADIO alignment control changed the {stratum_name} row multiset"
            )
        moved_count = sum(
            destination != source for destination, source in stratum_mappings
        )
        strata_receipts.append(
            {
                "stratum": stratum_name,
                "row_count": len(ranked),
                "fixed_point_count": fixed_points,
                "moved_row_count": moved_count,
                "moved_fraction": moved_count / len(ranked) if ranked else 0.0,
                "singleton_row_count": int(len(ranked) == 1),
                "singleton_fraction": 1.0 if len(ranked) == 1 else 0.0,
                "radio_row_multiset_sha256_before": stratum_before_multiset,
                "radio_row_multiset_sha256_after": stratum_after_multiset,
                "permutation_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            }
        )
    radio_after = result[:, RADIO_FEATURE_START:RADIO_FEATURE_STOP]
    before_multiset = _row_multiset_sha256(radio_before[source_visible])
    after_multiset = _row_multiset_sha256(radio_after[source_visible])
    if before_multiset != after_multiset:
        raise RuntimeError("RADIO alignment control changed the visible row multiset")
    if not torch.equal(result[:, :RADIO_FEATURE_START], features[:, :RADIO_FEATURE_START]):
        raise RuntimeError("RADIO alignment control changed RGB/availability features")
    if not torch.equal(result[:, RADIO_FEATURE_STOP:], features[:, RADIO_FEATURE_STOP:]):
        raise RuntimeError("RADIO alignment control changed geometry features")
    if not torch.equal(radio_after[~source_visible], radio_before[~source_visible]):
        raise RuntimeError("RADIO alignment control changed unavailable rows")
    receipt = {
        "schema": "radio_gs.surface_object_memory_v4.radio_alignment_control.v1",
        "mode": "shuffled_within_observation_stratum",
        "scene_id": str(scene_id),
        "seed": int(seed),
        "radio_slice": [RADIO_FEATURE_START, RADIO_FEATURE_STOP],
        "target_labels_read_by_permutation": False,
        "visible_radio_row_multiset_sha256_before": before_multiset,
        "visible_radio_row_multiset_sha256_after": after_multiset,
        "strata": strata_receipts,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result, receipt


def _runtime(
    payload: dict[str, Any],
    local_feature_mode: str,
    *,
    radio_alignment_control: str = "aligned",
    radio_alignment_seed: int = 0,
) -> dict[str, Any]:
    centres = torch.as_tensor(payload["centres"], dtype=torch.float32)
    local_features, selected_feature_layout = _select_local_features(
        payload, local_feature_mode
    )
    labels = torch.as_tensor(payload["token_index"], dtype=torch.long)
    has_strata = "source_visible" in payload or "membership_observed" in payload
    if has_strata and not (
        "source_visible" in payload and "membership_observed" in payload
    ):
        raise ValueError("completion cache must provide both source_visible and membership_observed")
    membership_observed = torch.as_tensor(
        payload.get("membership_observed", payload["observed_visible"]), dtype=torch.bool
    )
    if membership_observed.shape != labels.shape:
        raise ValueError("completion membership observation mask must align with elements")
    if not torch.equal(
        membership_observed,
        torch.as_tensor(payload["observed_visible"], dtype=torch.bool),
    ):
        raise ValueError("observed_visible must remain the membership_observed compatibility alias")
    token_count = len(payload["object_ids"])
    partial = PartialObjectMembership.from_oracle_visibility(
        labels, membership_observed, token_count=token_count,
        eligible_elements=payload["completion_valid"],
    )
    unknown_strata = None
    source_visible = None
    if has_strata:
        source_visible = torch.as_tensor(payload["source_visible"], dtype=torch.bool)
        if source_visible.shape != labels.shape:
            raise ValueError("completion source visibility must align with elements")
        for alias in ("appearance_available", "feature_available"):
            if alias in payload and not torch.equal(
                source_visible, torch.as_tensor(payload[alias], dtype=torch.bool)
            ):
                raise ValueError(f"{alias} must agree with source_visible")
        if "mask_supported" in payload and not torch.equal(
            membership_observed,
            torch.as_tensor(payload["mask_supported"], dtype=torch.bool),
        ):
            raise ValueError("mask_supported must agree with membership_observed")
        if bool((membership_observed & ~source_visible).any()):
            raise ValueError("membership-observed elements must be source-visible")
        eligible = partial.eligible_elements
        unknown_strata = {
            "visible_but_unmasked": eligible & source_visible & ~membership_observed,
            "never_visible": eligible & ~source_visible,
        }
        if not torch.equal(
            unknown_strata["visible_but_unmasked"] | unknown_strata["never_visible"],
            partial.unknown.any(-1),
        ):
            raise ValueError("completion visibility strata must partition unknown elements")
    if radio_alignment_control not in RADIO_ALIGNMENT_CONTROLS:
        raise ValueError(f"unsupported RADIO alignment control {radio_alignment_control!r}")
    radio_alignment_receipt = {
        "schema": "radio_gs.surface_object_memory_v4.radio_alignment_control.v1",
        "mode": "aligned",
        "scene_id": str(payload["scene_id"]),
        "seed": int(radio_alignment_seed),
    }
    radio_alignment_receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            radio_alignment_receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if radio_alignment_control != "aligned":
        if local_feature_mode != "rgb_radio_geometry" or source_visible is None:
            raise ValueError("shuffled RADIO control requires sealed F71 visibility strata")
        local_features, radio_alignment_receipt = _shuffle_radio_within_observation_strata(
            local_features,
            scene_id=str(payload["scene_id"]),
            membership_observed=membership_observed,
            source_visible=source_visible,
            seed=radio_alignment_seed,
        )
    config = payload["configuration"]
    carrier = SurfaceVoxelCarrier(
        centres,
        float(config["voxel_size"]),
        normals=payload.get("normals"),
        maximum_splat_radius=int(config["maximum_splat_radius"]),
        surface_band_voxels=float(config["surface_band_voxels"]),
        maximum_contributors_per_pixel=int(config["maximum_contributors_per_pixel"]),
    )
    context = build_token_context(
        centres,
        local_features,
        partial,
        carrier.neighbors().edge_index,
        minimum_scale=float(config["voxel_size"]),
    )
    return {
        "payload": payload,
        "centres": centres,
        "local_features": local_features,
        "labels": labels,
        "partial": partial,
        "carrier": carrier,
        "context": context,
        "minimum_scale": float(config["voxel_size"]),
        "local_feature_mode": local_feature_mode,
        "selected_local_feature_layout": list(selected_feature_layout),
        "unknown_strata": unknown_strata,
        "source_visible": source_visible,
        "radio_alignment_control": radio_alignment_control,
        "radio_alignment_receipt": radio_alignment_receipt,
    }


def _balanced_unknown_indices(
    runtime: dict[str, Any],
    samples_per_token: int,
    generator: torch.Generator,
    *,
    sampling_mode: str = "token_uniform",
) -> torch.Tensor:
    if sampling_mode not in UNKNOWN_SAMPLING_MODES:
        raise ValueError(f"unsupported unknown sampling mode {sampling_mode!r}")
    labels = runtime["labels"]
    unknown = runtime["partial"].unknown.any(-1)
    selected = []

    def sample(candidates: torch.Tensor, count: int) -> torch.Tensor:
        count = min(int(count), int(candidates.numel()))
        if count <= 0:
            return candidates[:0]
        order = torch.randperm(candidates.numel(), generator=generator)[:count]
        return candidates[order]

    def stratum_balanced_sample(cohort: torch.Tensor, budget: int) -> torch.Tensor:
        strata = runtime.get("unknown_strata")
        if strata is None:
            raise ValueError(
                "token_visibility_stratum_balanced requires sealed visibility strata"
            )
        candidates = [
            torch.where(cohort & strata[name])[0]
            for name in ("visible_but_unmasked", "never_visible")
        ]
        active = [index for index, value in enumerate(candidates) if value.numel()]
        if not active:
            return torch.empty(0, dtype=torch.long)
        counts = [0, 0]
        remaining = min(int(budget), sum(int(value.numel()) for value in candidates))
        # Water-fill the fixed budget across non-empty strata.  If one stratum is
        # scarce, retain every example there and reallocate only its unused share.
        while remaining > 0:
            available = [
                index
                for index in active
                if counts[index] < int(candidates[index].numel())
            ]
            if not available:
                break
            share = max(remaining // len(available), 1)
            progressed = 0
            for index in available:
                take = min(
                    share,
                    int(candidates[index].numel()) - counts[index],
                    remaining,
                )
                counts[index] += take
                remaining -= take
                progressed += take
            if progressed <= 0:
                raise RuntimeError("visibility-stratum sampler made no progress")
        parts = [sample(value, counts[index]) for index, value in enumerate(candidates)]
        return torch.cat([part for part in parts if part.numel()])

    for token in range(runtime["partial"].positive.shape[1]):
        cohort = unknown & (labels == token)
        if sampling_mode == "token_uniform":
            candidates = torch.where(cohort)[0]
            if candidates.numel():
                selected.append(sample(candidates, samples_per_token))
        else:
            part = stratum_balanced_sample(cohort, samples_per_token)
            if part.numel():
                selected.append(part)
    null_budget = samples_per_token * min(runtime["partial"].positive.shape[1], 8)
    null_cohort = unknown & (labels < 0)
    if sampling_mode == "token_uniform":
        null_candidates = torch.where(null_cohort)[0]
        if null_candidates.numel():
            selected.append(sample(null_candidates, null_budget))
    else:
        part = stratum_balanced_sample(null_cohort, null_budget)
        if part.numel():
            selected.append(part)
    if not selected:
        raise RuntimeError("training scene contains no unknown positive completion targets")
    return torch.cat(selected)


def _sampling_audit(runtime: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    labels = runtime["labels"][indices]
    canonical_indices = (
        torch.as_tensor(indices, dtype=torch.int64).contiguous().cpu().numpy()
        .astype("<i8", copy=False).tobytes()
    )
    result: dict[str, Any] = {
        "scene_id": runtime["payload"]["scene_id"],
        "selected_element_indices_sha256": hashlib.sha256(
            canonical_indices
        ).hexdigest(),
        "selected_element_indices_dtype": "little_endian_int64",
        "selected_element_indices_order_preserved": True,
        "selected_element_count": int(indices.numel()),
        "selected_object_count": int((labels >= 0).sum()),
        "selected_null_count": int((labels < 0).sum()),
        "unique_element_count": int(indices.unique().numel()),
    }
    strata = runtime.get("unknown_strata")
    if strata is not None:
        for name, mask in strata.items():
            selected = mask[indices]
            result[f"{name}_element_count"] = int(selected.sum())
            result[f"{name}_object_count"] = int((selected & (labels >= 0)).sum())
            result[f"{name}_null_count"] = int((selected & (labels < 0)).sum())
    return result


def _training_examples(
    runtimes: list[dict[str, Any]],
    samples_per_token: int,
    seed: int,
    *,
    sampling_mode: str = "token_uniform",
    scoring_mode: str = "mlp",
) -> tuple[list[dict[str, torch.Tensor]], int, list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(seed)
    examples = []
    sampling_audit = []
    input_dimension = -1
    for runtime in runtimes:
        indices = _balanced_unknown_indices(
            runtime,
            samples_per_token,
            generator,
            sampling_mode=sampling_mode,
        )
        sampling_audit.append(_sampling_audit(runtime, indices))
        features = build_pair_features(
            runtime["centres"], runtime["local_features"], runtime["context"], indices,
            minimum_scale=runtime["minimum_scale"],
        )
        explicit_similarity = None
        if scoring_mode == "mlp_radio_cosine_residual":
            if runtime["local_feature_mode"] != "rgb_radio_geometry":
                raise ValueError("RADIO cosine residual requires rgb_radio_geometry")
            explicit_similarity = build_feature_cosine_similarity(
                runtime["local_features"],
                runtime["context"],
                indices,
                feature_start=RADIO_FEATURE_START,
                feature_stop=RADIO_FEATURE_STOP,
            )
        elif scoring_mode not in ("mlp", "availability_dual_mlp"):
            raise ValueError(f"unsupported completion scoring mode {scoring_mode!r}")
        labels = runtime["labels"][indices].clone()
        labels[labels < 0] = features.shape[1]
        if input_dimension < 0:
            input_dimension = int(features.shape[-1])
        elif features.shape[-1] != input_dimension:
            raise ValueError("all completion scenes must use the same local feature contract")
        example = {"features": features, "labels": labels, "indices": indices}
        if explicit_similarity is not None:
            example["explicit_similarity"] = explicit_similarity
        if scoring_mode == "availability_dual_mlp":
            strata = runtime.get("unknown_strata")
            source_visible = runtime.get("source_visible")
            if strata is None or source_visible is None:
                raise ValueError("dual experts require sealed visibility strata")
            source_available = source_visible[indices]
            if not torch.equal(
                source_available, strata["visible_but_unmasked"][indices]
            ):
                raise RuntimeError("dual-expert training route differs from sealed source visibility")
            example["source_available"] = source_available
        examples.append(example)
    return examples, input_dimension, sampling_audit


def _fit(
    model: OracleIdentityCompletionMLP,
    examples: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    step_count: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)
    python_rng = random.Random(seed + 2)
    model.train()
    loss_history = []
    for step in range(step_count):
        example = examples[python_rng.randrange(len(examples))]
        count = example["labels"].numel()
        indices = torch.randint(count, (min(batch_size, count),), generator=generator)
        pair = example["features"][indices].to(device, non_blocking=True)
        target = example["labels"][indices].to(device, non_blocking=True)
        similarity = example.get("explicit_similarity")
        if similarity is not None:
            similarity = similarity[indices].to(device, non_blocking=True)
        source_available = example.get("source_available")
        if source_available is not None:
            source_available = source_available[indices].to(device, non_blocking=True)
        logits = model.categorical_logits(
            pair,
            explicit_similarity=similarity,
            source_available=source_available,
        )
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(step_count // 20, 1) == 0:
            loss_history.append({"step": step + 1, "cross_entropy": float(loss.detach())})
    return {"loss_history": loss_history, "final_cross_entropy": loss_history[-1]["cross_entropy"]}


@torch.no_grad()
def _predict(
    model: OracleIdentityCompletionMLP,
    runtime: dict[str, Any],
    *,
    device: torch.device,
    element_batch_size: int,
    temperature: float,
    completion_confidence_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    partial = runtime["partial"]
    unknown_indices = torch.where(partial.unknown.any(-1))[0]
    probability = torch.zeros_like(partial.positive, dtype=torch.float32)
    null_probability = torch.zeros(probability.shape[0], dtype=torch.float32)
    model.eval()
    for start in range(0, unknown_indices.numel(), element_batch_size):
        indices = unknown_indices[start : start + element_batch_size]
        pair = build_pair_features(
            runtime["centres"], runtime["local_features"], runtime["context"], indices,
            minimum_scale=runtime["minimum_scale"],
        ).to(device, non_blocking=True)
        similarity = None
        if model.explicit_similarity_residual:
            similarity = build_feature_cosine_similarity(
                runtime["local_features"],
                runtime["context"],
                indices,
                feature_start=RADIO_FEATURE_START,
                feature_stop=RADIO_FEATURE_STOP,
            ).to(device, non_blocking=True)
        source_available = None
        if model.availability_conditioned_experts:
            strata = runtime.get("unknown_strata")
            source_visible = runtime.get("source_visible")
            if strata is None or source_visible is None:
                raise ValueError("dual experts require sealed visibility strata")
            routed = source_visible[indices]
            if not torch.equal(routed, strata["visible_but_unmasked"][indices]):
                raise RuntimeError("dual-expert inference route differs from sealed source visibility")
            source_available = routed.to(device, non_blocking=True)
        categorical = torch.softmax(
            model.categorical_logits(
                pair,
                explicit_similarity=similarity,
                source_available=source_available,
            ) / temperature,
            dim=-1,
        ).cpu()
        probability[indices] = categorical[:, :-1]
        null_probability[indices] = categorical[:, -1]
    return complete_unknown_only(
        partial,
        probability,
        unknown_null_probability=null_probability,
        completion_confidence_cap=completion_confidence_cap,
    )


def _soft_iou_components(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction.reshape(-1, prediction.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    if prediction.shape != target.shape:
        raise ValueError("held-out prediction and mesh target must align")
    intersection = (prediction * target).sum(0)
    union = prediction.sum(0) + target.sum(0) - intersection
    return intersection, union, target.sum(0)


def _soft_iou_token_statistics_from_components(
    intersection: torch.Tensor,
    prediction_mass: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    domain_unit: str,
    cohort_available: bool = True,
) -> dict[str, Any]:
    """Serialize losslessly poolable per-token soft-IoU components.

    The computation is deliberately float64 even though the model posterior is
    float32.  A token is evaluated whenever its union is non-zero, which retains
    false-positive-only tokens instead of silently dropping them.
    """

    components = tuple(
        torch.as_tensor(value, dtype=torch.float64).reshape(-1).cpu()
        for value in (intersection, prediction_mass, target_mass)
    )
    intersection64, prediction_mass64, target_mass64 = components
    if not (
        intersection64.shape == prediction_mass64.shape == target_mass64.shape
    ):
        raise ValueError("soft-IoU sufficient-statistic components must align")
    if not all(bool(torch.isfinite(value).all()) for value in components):
        raise ValueError("soft-IoU sufficient-statistic components must be finite")
    if any(bool((value < 0).any()) for value in components):
        raise ValueError("soft-IoU sufficient-statistic masses must be non-negative")
    union64 = prediction_mass64 + target_mass64 - intersection64
    if bool((union64 < -1e-12).any()):
        raise ValueError("soft-IoU sufficient-statistic union cannot be negative")
    union64 = union64.clamp_min(0)
    evaluated = union64 > 0
    token_statistics = [
        {
            "token_index": int(token_index),
            "intersection": float(intersection64[token_index]),
            "prediction_mass": float(prediction_mass64[token_index]),
            "target_mass": float(target_mass64[token_index]),
            "union": float(union64[token_index]),
            "evaluated": bool(evaluated[token_index]),
        }
        for token_index in range(intersection64.numel())
    ]
    return {
        "schema": "radio_gs.surface_object_memory_v4.soft_iou_sufficient_statistics.v1",
        "numeric_dtype": "float64",
        "domain_unit": str(domain_unit),
        "cohort_available": bool(cohort_available),
        "evaluated_token_policy": "union_positive_including_false_positive_only",
        "token_count": len(token_statistics),
        "token_statistics": token_statistics,
    }


def _soft_iou_token_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    domain_unit: str,
    cohort_available: bool = True,
) -> dict[str, Any]:
    prediction64 = torch.as_tensor(prediction, dtype=torch.float64).cpu()
    target64 = torch.as_tensor(target, dtype=torch.float64).cpu()
    if prediction64.shape != target64.shape or prediction64.ndim != 2:
        raise ValueError("soft-IoU prediction and target must align as [N, K]")
    if mask is None:
        mask64 = torch.ones(prediction64.shape[0], dtype=torch.float64)
    else:
        mask64 = torch.as_tensor(mask, dtype=torch.bool).reshape(-1).cpu().double()
        if mask64.shape != (prediction64.shape[0],):
            raise ValueError("soft-IoU cohort mask must align with the sample domain")
    masked_prediction = prediction64 * mask64[:, None]
    masked_target = target64 * mask64[:, None]
    intersection = (masked_prediction * masked_target).sum(0, dtype=torch.float64)
    prediction_mass = masked_prediction.sum(0, dtype=torch.float64)
    target_mass = masked_target.sum(0, dtype=torch.float64)
    return _soft_iou_token_statistics_from_components(
        intersection,
        prediction_mass,
        target_mass,
        domain_unit=domain_unit,
        cohort_available=cohort_available,
    )


def _element_soft_iou_sufficient_statistics(
    runtime: dict[str, Any], membership: torch.Tensor
) -> dict[str, Any]:
    membership = torch.as_tensor(membership, dtype=torch.float32).cpu()
    labels = torch.as_tensor(runtime["labels"], dtype=torch.long).cpu()
    partial = runtime["partial"]
    if membership.shape != partial.positive.shape or labels.shape != (
        membership.shape[0],
    ):
        raise ValueError("element soft-IoU inputs do not align")
    eligible = torch.as_tensor(partial.eligible_elements, dtype=torch.bool).cpu()
    target = torch.zeros_like(membership)
    object_surface = eligible & (labels >= 0)
    if bool(object_surface.any()):
        if int(labels[object_surface].max()) >= membership.shape[1]:
            raise ValueError("element soft-IoU label exceeds the token domain")
        target[object_surface, labels[object_surface]] = 1.0
    unknown = torch.as_tensor(partial.unknown.any(-1), dtype=torch.bool).cpu() & eligible
    strata = runtime.get("unknown_strata")
    strata_available = strata is not None
    visible_but_unmasked = (
        torch.as_tensor(strata["visible_but_unmasked"], dtype=torch.bool).cpu()
        if strata_available
        else torch.zeros_like(eligible)
    )
    never_visible = (
        torch.as_tensor(strata["never_visible"], dtype=torch.bool).cpu()
        if strata_available
        else torch.zeros_like(eligible)
    )
    cohorts = {
        "full_3d": (eligible, True),
        "unknown_3d": (unknown, True),
        "visible_but_unmasked_3d": (visible_but_unmasked, strata_available),
        "never_visible_3d": (never_visible, strata_available),
    }
    return {
        name: _soft_iou_token_statistics(
            membership,
            target,
            mask=mask,
            domain_unit="surface_element_token",
            cohort_available=available,
        )
        for name, (mask, available) in cohorts.items()
    }


def _render_valid_posterior(
    carrier: SurfaceVoxelCarrier,
    posterior: torch.Tensor,
    valid_elements: torch.Tensor,
    camera: Any,
) -> torch.Tensor:
    projection = carrier.project(camera)
    valid = torch.as_tensor(valid_elements, dtype=torch.bool)[projection.element_ids]
    element_ids = projection.element_ids[valid]
    pixel_ids = projection.pixel_ids[valid]
    weights = projection.weights[valid]
    channels = posterior.shape[1]
    numerator = torch.zeros(projection.num_pixels, channels)
    numerator.index_add_(0, pixel_ids, posterior[element_ids] * weights[:, None])
    denominator = torch.zeros(projection.num_pixels)
    # Invalid carrier elements contribute zero token mass, not zero surface mass.
    # Preserve every projected contributor in the denominator so validity
    # filtering cannot inflate the remaining posterior.
    denominator.index_add_(0, projection.pixel_ids, projection.weights)
    return (numerator / denominator.clamp_min(1e-12)[:, None]).reshape(
        camera.height, camera.width, channels
    )


@torch.no_grad()
def _heldout_2d_metrics(
    runtime: dict[str, Any], membership: torch.Tensor
) -> dict[str, Any]:
    token_count = membership.shape[1]
    valid_elements = runtime["partial"].eligible_elements
    intersection = torch.zeros(token_count)
    union = torch.zeros(token_count)
    target_mass = torch.zeros(token_count)
    stats_intersection = torch.zeros(token_count, dtype=torch.float64)
    stats_prediction_mass = torch.zeros(token_count, dtype=torch.float64)
    stats_target_mass = torch.zeros(token_count, dtype=torch.float64)
    for record, target in zip(
        runtime["payload"]["heldout_cameras"],
        runtime["payload"]["heldout_mesh_target_rasters"],
    ):
        camera = camera_from_record(record)
        rendered_prediction = _render_valid_posterior(
            runtime["carrier"], membership, valid_elements, camera
        )
        current_intersection, current_union, current_target_mass = _soft_iou_components(
            rendered_prediction, torch.as_tensor(target).float()
        )
        intersection += current_intersection
        union += current_union
        target_mass += current_target_mass
        prediction64 = rendered_prediction.reshape(-1, token_count).double()
        target64 = torch.as_tensor(target, dtype=torch.float64).reshape(
            -1, token_count
        ).cpu()
        stats_intersection += (prediction64 * target64).sum(
            0, dtype=torch.float64
        )
        stats_prediction_mass += prediction64.sum(0, dtype=torch.float64)
        stats_target_mass += target64.sum(0, dtype=torch.float64)
    evaluated = union > 0
    values = intersection[evaluated] / union[evaluated].clamp_min(1e-12)
    score = float(values.mean()) if bool(evaluated.any()) else 0.0
    return {
        # Compatibility alias used by the frozen gate; semantics are explicitly
        # the cross-view token aggregation below.
        "heldout_2d_soft_miou": score,
        "heldout_2d_cross_view_token_soft_miou": score,
        "heldout_2d_aggregation": "sum_intersection_union_per_token_across_all_heldout_views",
        "heldout_2d_evaluated_token_count": int(evaluated.sum()),
        "heldout_2d_target_present_token_count": int((target_mass > 0).sum()),
        "heldout_2d_false_positive_only_token_count": int(
            (evaluated & (target_mass == 0)).sum()
        ),
        "heldout_view_count": len(runtime["payload"]["heldout_cameras"]),
        "token_count": token_count,
        "heldout_2d_target_authority": "original_mesh_vertex_instance_raycast",
        "heldout_2d_invalid_contributor_policy": (
            "zero_token_mass_preserve_full_projection_denominator"
        ),
        "heldout_2d_soft_iou_sufficient_statistics": (
            _soft_iou_token_statistics_from_components(
                stats_intersection,
                stats_prediction_mass,
                stats_target_mass,
                domain_unit="heldout_pixel_token",
            )
        ),
    }


@torch.no_grad()
def _evaluate_scene(
    model: OracleIdentityCompletionMLP,
    runtime: dict[str, Any],
    *,
    device: torch.device,
    element_batch_size: int,
    temperature: float,
    completion_confidence_cap: float,
    assignment_threshold: float,
) -> dict[str, Any]:
    learned, null = _predict(
        model, runtime, device=device, element_batch_size=element_batch_size,
        temperature=temperature, completion_confidence_cap=completion_confidence_cap,
    )
    observed_only = runtime["partial"].positive.float()
    baseline_null = 1.0 - observed_only.sum(-1)
    baseline = completion_metrics(
        observed_only, runtime["partial"], runtime["labels"],
        null_probability=baseline_null,
        assignment_threshold=assignment_threshold,
        unknown_strata=runtime.get("unknown_strata"),
    )
    completed = completion_metrics(
        learned, runtime["partial"], runtime["labels"],
        null_probability=null,
        assignment_threshold=assignment_threshold,
        unknown_strata=runtime.get("unknown_strata"),
    )
    baseline_statistics = _element_soft_iou_sufficient_statistics(
        runtime, observed_only
    )
    completed_statistics = _element_soft_iou_sufficient_statistics(runtime, learned)
    baseline.update(_heldout_2d_metrics(runtime, observed_only))
    completed.update(_heldout_2d_metrics(runtime, learned))
    baseline_statistics["heldout_2d"] = baseline.pop(
        "heldout_2d_soft_iou_sufficient_statistics"
    )
    completed_statistics["heldout_2d"] = completed.pop(
        "heldout_2d_soft_iou_sufficient_statistics"
    )
    baseline["soft_iou_sufficient_statistics"] = baseline_statistics
    completed["soft_iou_sufficient_statistics"] = completed_statistics
    oracle_membership = torch.zeros_like(observed_only)
    oracle_object = runtime["partial"].eligible_elements & (runtime["labels"] >= 0)
    oracle_membership[
        oracle_object, runtime["labels"][oracle_object]
    ] = 1.0
    oracle_render_reference = _heldout_2d_metrics(runtime, oracle_membership)
    return {
        "scene_id": runtime["payload"]["scene_id"],
        "element_count": int(runtime["centres"].shape[0]),
        "token_count": int(runtime["partial"].positive.shape[1]),
        "surface_oracle_membership_render_reference": oracle_render_reference,
        "surface_oracle_membership_render_reference_is_strict_ceiling": False,
        "surface_oracle_membership_render_reference_uses_target_membership": True,
        "local_feature_mode": runtime["local_feature_mode"],
        "selected_local_feature_layout": runtime["selected_local_feature_layout"],
        "unknown_strata_available": runtime.get("unknown_strata") is not None,
        "baseline_observed_only": baseline,
        "learned_completion": completed,
        "null_mass": {
            "observed_mean": float(null[runtime["partial"].element_is_observed].mean()),
            "unknown_mean": float(null[runtime["partial"].unknown.any(-1)].mean()),
        },
    }


def _mean_scene_metric(records: list[dict[str, Any]], method: str, key: str) -> float:
    return float(np.mean([record[method][key] for record in records]))


CONFUSION_COUNT_SUFFIXES = (
    "element_count",
    "retained_object_count",
    "retained_set_null_count",
    "predicted_token_count",
    "predicted_null_count",
    "assigned_object_count",
    "correct_token_count",
    "wrong_token_on_object_count",
    "token_on_null_count",
    "null_on_object_count",
    "correct_null_count",
)


def _pooled_categorical_confusion(
    records: list[dict[str, Any]], method: str, prefix: str
) -> dict[str, Any]:
    counts = {
        suffix: int(sum(row[method][f"{prefix}_{suffix}"] for row in records))
        for suffix in CONFUSION_COUNT_SUFFIXES
    }
    if (
        counts["correct_token_count"]
        + counts["wrong_token_on_object_count"]
        + counts["token_on_null_count"]
        + counts["null_on_object_count"]
        + counts["correct_null_count"]
        != counts["element_count"]
    ):
        raise RuntimeError("pooled completion confusion does not partition the cohort")
    predicted_token = counts["predicted_token_count"]
    target_object = counts["retained_object_count"]
    assigned_object = counts["assigned_object_count"]
    target_null = counts["retained_set_null_count"]
    correct_token = counts["correct_token_count"]
    correct_null = counts["correct_null_count"]
    return {
        "counts": counts,
        "metrics": {
            "assignment_precision": (
                correct_token / predicted_token if predicted_token else 0.0
            ),
            "retained_object_coverage": (
                assigned_object / target_object if target_object else 1.0
            ),
            "correct_assignment_recall": (
                correct_token / target_object if target_object else 1.0
            ),
            "assigned_object_top1_accuracy": (
                correct_token / assigned_object if assigned_object else 0.0
            ),
            "retained_set_null_recall": (
                correct_null / target_null if target_null else 1.0
            ),
        },
    }


def _pool_soft_iou_sufficient_statistics(
    records: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    """Pool scene/token records without averaging away domain mass."""

    pooled: dict[str, Any] = {}
    for cohort_name in SOFT_IOU_COHORTS:
        available_statistics = []
        for record in records:
            try:
                statistics = record[method]["soft_iou_sufficient_statistics"][
                    cohort_name
                ]
            except KeyError as error:
                raise KeyError(
                    f"missing {method}/{cohort_name} soft-IoU sufficient statistics"
                ) from error
            if statistics["numeric_dtype"] != "float64":
                raise ValueError("poolable soft-IoU statistics must be float64")
            token_statistics = statistics["token_statistics"]
            if int(statistics["token_count"]) != len(token_statistics):
                raise ValueError("soft-IoU token count does not match its records")
            if [row["token_index"] for row in token_statistics] != list(
                range(len(token_statistics))
            ):
                raise ValueError("soft-IoU token indices must be contiguous and ordered")
            if statistics["cohort_available"]:
                available_statistics.append(statistics)

        domain_units = {
            statistics["domain_unit"] for statistics in available_statistics
        }
        if len(domain_units) > 1:
            raise ValueError("pooled soft-IoU cohorts disagree on their domain unit")
        domain_unit = next(iter(domain_units), (
            "heldout_pixel_token"
            if cohort_name == "heldout_2d"
            else "surface_element_token"
        ))
        token_rows = [
            token
            for statistics in available_statistics
            for token in statistics["token_statistics"]
        ]
        for token in token_rows:
            values = [
                float(token[key])
                for key in (
                    "intersection",
                    "prediction_mass",
                    "target_mass",
                    "union",
                )
            ]
            if not all(np.isfinite(value) and value >= 0 for value in values):
                raise ValueError("pooled soft-IoU masses must be finite and non-negative")
            if not np.isclose(
                values[3], values[1] + values[2] - values[0], atol=1e-9, rtol=1e-12
            ):
                raise ValueError("pooled soft-IoU record has an inconsistent union")
            if bool(token["evaluated"]) != (values[3] > 0):
                raise ValueError("soft-IoU evaluated flag must be equivalent to union > 0")
        evaluated_rows = [token for token in token_rows if token["evaluated"]]
        intersection_sum = float(
            sum(float(token["intersection"]) for token in token_rows)
        )
        prediction_mass_sum = float(
            sum(float(token["prediction_mass"]) for token in token_rows)
        )
        target_mass_sum = float(
            sum(float(token["target_mass"]) for token in token_rows)
        )
        union_sum = float(sum(float(token["union"]) for token in token_rows))
        scene_token_macro = (
            float(
                np.mean(
                    [
                        float(token["intersection"]) / float(token["union"])
                        for token in evaluated_rows
                    ],
                    dtype=np.float64,
                )
            )
            if evaluated_rows
            else 0.0
        )
        pooled[cohort_name] = {
            "numeric_dtype": "float64",
            "domain_unit": domain_unit,
            "scene_token_macro_soft_iou": scene_token_macro,
            "union_summed_element_or_pixel_token_micro_soft_iou": (
                intersection_sum / union_sum if union_sum > 0 else 0.0
            ),
            "sums": {
                "intersection": intersection_sum,
                "prediction_mass": prediction_mass_sum,
                "target_mass": target_mass_sum,
                "union": union_sum,
            },
            "counts": {
                "validation_scene_count": len(records),
                "available_scene_count": len(available_statistics),
                "scene_token_count": len(token_rows),
                "evaluated_scene_token_count": len(evaluated_rows),
                "target_present_scene_token_count": sum(
                    float(token["target_mass"]) > 0 for token in token_rows
                ),
                "false_positive_only_scene_token_count": sum(
                    bool(token["evaluated"]) and float(token["target_mass"]) == 0
                    for token in token_rows
                ),
                "unevaluated_scene_token_count": len(token_rows) - len(evaluated_rows),
            },
        }
    return pooled


def _cohort_manifest_receipt(
    path_value: str | None,
    *,
    split: SourceSplit,
    selected_scene_ids: set[str],
) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get("schema") != (
        "radio_gs.surface_object_memory_v4.scannet_pfir_completion_cohort.v3"
    ):
        raise ValueError("completion cohort manifest is not the strict v3 contract")
    scene_ids = list(map(str, payload.get("scene_ids", ())))
    manifest_split = payload.get("split", {})
    training = list(map(str, manifest_split.get("training_scene_ids", ())))
    validation = list(map(str, manifest_split.get("validation_scene_ids", ())))
    if (
        len(scene_ids) != len(set(scene_ids))
        or set(scene_ids) != selected_scene_ids
        or set(training) != set(split.source_ids)
        or set(validation) != set(split.development_ids)
        or set(training) & set(validation)
    ):
        raise ValueError("completion cohort manifest differs from the requested split")
    if manifest_split.get("physical_family_disjoint") is not True:
        raise ValueError("completion cohort manifest lacks a physical-family-disjoint split")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": str(payload["schema"]),
        "scene_count": len(scene_ids),
        "training_scene_ids": sorted(training),
        "validation_scene_ids": sorted(validation),
        "selection_policy": payload.get("selection", {}).get("policy"),
        "selection_salt": payload.get("selection", {}).get("salt"),
        "validation_selection_salt": manifest_split.get(
            "validation_selection_salt"
        ),
    }


def _aligned_reference_receipt(
    path_value: str | None,
    *,
    args: argparse.Namespace,
    split: SourceSplit,
    selected_payloads: list[dict[str, Any]],
    input_dimension: int,
    model_parameter_count: int,
    selected_feature_layout: tuple[str, ...],
    cohort_manifest: dict[str, Any] | None,
    implementation_sha256: str,
    completion_implementation_sha256: str,
) -> dict[str, Any] | None:
    control = getattr(args, "radio_alignment_control", "aligned")
    if control == "aligned":
        if path_value:
            raise ValueError("an aligned arm must not cite another aligned reference")
        return None
    if not path_value:
        raise ValueError("shuffled RADIO requires --aligned-reference-report")
    path = Path(path_value).resolve(strict=True)
    reference = json.loads(path.read_text())
    if reference.get("schema") != REPORT_SCHEMA:
        raise ValueError("aligned reference report schema differs from this implementation")
    if (
        reference.get("implementation_sha256") != implementation_sha256
        or reference.get("completion_implementation_sha256")
        != completion_implementation_sha256
    ):
        raise ValueError("aligned reference implementation hashes differ")
    if reference.get("radio_alignment_control") != "aligned":
        raise ValueError("RADIO control reference is not an aligned arm")
    expected_split = {
        "training_scene_ids": sorted(split.source_ids),
        "validation_scene_ids": sorted(split.development_ids),
        "overlap": [],
    }
    if reference.get("split") != expected_split:
        raise ValueError("aligned reference uses a different train/validation split")
    current_cache_sha = {
        str(payload["scene_id"]): str(payload["cache_sha256"])
        for payload in selected_payloads
    }
    reference_cache_sha = {
        str(row["scene_id"]): str(row["cache_sha256"])
        for row in reference.get("scene_cache_receipts", ())
    }
    if reference_cache_sha != current_cache_sha:
        raise ValueError("aligned reference uses different sealed scene caches")
    reference_cohort = reference.get("cohort_manifest")
    if (reference_cohort is None) != (cohort_manifest is None) or (
        cohort_manifest is not None
        and reference_cohort.get("sha256") != cohort_manifest["sha256"]
    ):
        raise ValueError("aligned reference uses a different cohort manifest")
    current_configuration = {
        "seed": args.seed,
        "step_count": args.step_count,
        "batch_size": args.batch_size,
        "element_batch_size": args.element_batch_size,
        "samples_per_token": args.samples_per_token,
        "unknown_sampling_mode": args.unknown_sampling_mode,
        "scoring_mode": args.scoring_mode,
        "input_dimension": input_dimension,
        "model_parameter_count": model_parameter_count,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dimension": args.hidden_dimension,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "completion_confidence_cap": args.completion_confidence_cap,
        "assignment_threshold": args.assignment_threshold,
        "minimum_miou_improvement": args.minimum_miou_improvement,
        "minimum_assignment_precision": args.minimum_assignment_precision,
        "minimum_assigned_fraction": args.minimum_assigned_fraction,
        "minimum_null_recall": args.minimum_null_recall,
        "local_feature_mode": args.local_feature_mode,
        "selected_local_feature_layout": list(selected_feature_layout),
        "optimizer": "AdamW",
    }
    reference_configuration = reference.get("training_configuration", {})
    differing = {
        key: (reference_configuration.get(key), value)
        for key, value in current_configuration.items()
        if reference_configuration.get(key) != value
    }
    if differing:
        raise ValueError(
            "shuffled RADIO differs from its aligned reference beyond alignment: "
            + ", ".join(sorted(differing))
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "implementation_sha256": implementation_sha256,
        "completion_implementation_sha256": completion_implementation_sha256,
        "only_changed_factor": "radio_alignment_control",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_instance_oracle_training:
        raise PermissionError("training requires explicit supervised instance-oracle authorization")
    implementation_path = Path(__file__)
    completion_implementation_path = (
        implementation_path.parents[1] / "completion" / "oracle.py"
    )
    implementation_sha256 = sha256_file(implementation_path)
    completion_implementation_sha256 = sha256_file(completion_implementation_path)
    if (
        args.step_count <= 0
        or args.batch_size <= 0
        or args.element_batch_size <= 0
        or args.samples_per_token <= 0
        or args.hidden_dimension <= 0
    ):
        raise ValueError("training sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("completion optimizer parameters are invalid")
    if not 0 <= args.dropout < 1 or args.temperature <= 0:
        raise ValueError("completion dropout/temperature are invalid")
    for name, value in (
        ("completion confidence cap", args.completion_confidence_cap),
        ("assignment threshold", args.assignment_threshold),
        ("minimum assignment precision", args.minimum_assignment_precision),
        ("minimum assigned fraction", args.minimum_assigned_fraction),
        ("minimum null recall", args.minimum_null_recall),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if abs(float(args.assignment_threshold) - 0.5) > 1e-12:
        raise ValueError("the legacy assignment diagnostic is frozen at 0.5")
    if args.local_feature_mode not in LOCAL_FEATURE_MODES:
        raise ValueError("a supported local feature mode is required")
    if args.unknown_sampling_mode not in UNKNOWN_SAMPLING_MODES:
        raise ValueError("a supported unknown sampling mode is required")
    if args.scoring_mode not in SCORING_MODES:
        raise ValueError("a supported completion scoring mode is required")
    radio_alignment_control = getattr(args, "radio_alignment_control", "aligned")
    radio_alignment_seed = int(getattr(args, "radio_alignment_seed", args.seed))
    if radio_alignment_control not in RADIO_ALIGNMENT_CONTROLS:
        raise ValueError("a supported RADIO alignment control is required")
    if radio_alignment_control != "aligned" and (
        args.local_feature_mode != "rgb_radio_geometry"
        or args.scoring_mode != "mlp"
        or args.unknown_sampling_mode != "token_uniform"
    ):
        raise ValueError(
            "shuffled RADIO must isolate alignment under the F71/token-uniform/MLP baseline"
        )
    if (
        args.scoring_mode == "mlp_radio_cosine_residual"
        and args.local_feature_mode != "rgb_radio_geometry"
    ):
        raise ValueError("RADIO cosine residual requires rgb_radio_geometry")
    if (
        args.scoring_mode == "mlp_radio_cosine_residual"
        and args.unknown_sampling_mode != "token_uniform"
    ):
        raise ValueError(
            "the cosine residual must be isolated against token-uniform v5"
        )
    if (
        args.scoring_mode == "availability_dual_mlp"
        and args.unknown_sampling_mode != "token_visibility_stratum_balanced"
    ):
        raise ValueError(
            "dual experts must be isolated against the stratum-balanced v6 sampler"
        )
    if args.minimum_miou_improvement <= 0:
        raise ValueError("minimum mIoU improvement must be positive")
    cache_payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(payload["scene_id"]): payload for payload in cache_payloads}
    if len(by_id) != len(cache_payloads):
        raise ValueError("duplicate completion scene identity/cache")
    split = SourceSplit(
        frozenset(args.training_scene), frozenset(args.validation_scene), frozenset()
    )
    if len(split.source_ids) < 1 or len(split.development_ids) < 1:
        raise ValueError("scene-disjoint completion needs train and validation scenes")
    missing = sorted((split.source_ids | split.development_ids) - by_id.keys())
    if missing:
        raise FileNotFoundError(f"completion scene caches are missing: {missing}")
    training_families = {_physical_scene_family(value) for value in split.source_ids}
    validation_families = {
        _physical_scene_family(value) for value in split.development_ids
    }
    family_overlap = sorted(training_families & validation_families)
    if family_overlap:
        raise ValueError(
            f"completion train/validation physical families overlap: {family_overlap}"
        )
    selected_ids = sorted(split.source_ids | split.development_ids)
    selected_families = [_physical_scene_family(value) for value in selected_ids]
    if len(selected_families) != len(set(selected_families)):
        raise ValueError("completion cohort must be globally physical-family disjoint")
    selected = [by_id[identity] for identity in selected_ids]
    cohort_manifest = _cohort_manifest_receipt(
        getattr(args, "cohort_manifest", None),
        split=split,
        selected_scene_ids=set(by_id) & (split.source_ids | split.development_ids),
    )
    configurations = [payload["configuration"] for payload in selected]
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise ValueError("all scene caches must share one frozen carrier/observation configuration")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA completion training but CUDA is unavailable")
    train_runtime = [
        _runtime(
            by_id[identity],
            args.local_feature_mode,
            radio_alignment_control=radio_alignment_control,
            radio_alignment_seed=radio_alignment_seed,
        )
        for identity in sorted(split.source_ids)
    ]
    validation_runtime = [
        _runtime(
            by_id[identity],
            args.local_feature_mode,
            radio_alignment_control=radio_alignment_control,
            radio_alignment_seed=radio_alignment_seed,
        )
        for identity in sorted(split.development_ids)
    ]
    all_runtimes = train_runtime + validation_runtime
    selected_feature_layout = all_runtimes[0]["selected_local_feature_layout"]
    if any(
        runtime["selected_local_feature_layout"] != selected_feature_layout
        for runtime in all_runtimes[1:]
    ):
        raise ValueError("selected completion feature layouts disagree across caches")
    strata_flags = [runtime.get("unknown_strata") is not None for runtime in all_runtimes]
    if any(strata_flags) and not all(strata_flags):
        raise ValueError("completion caches cannot mix legacy and mask-support protocols")
    strata_available = all(strata_flags)
    examples, input_dimension, sampling_audit = _training_examples(
        train_runtime,
        args.samples_per_token,
        args.seed,
        sampling_mode=args.unknown_sampling_mode,
        scoring_mode=args.scoring_mode,
    )
    model = OracleIdentityCompletionMLP(
        input_dimension,
        args.hidden_dimension,
        args.dropout,
        explicit_similarity_residual=(
            args.scoring_mode == "mlp_radio_cosine_residual"
        ),
        availability_conditioned_experts=(
            args.scoring_mode == "availability_dual_mlp"
        ),
    ).to(device)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    changed_factors = _changed_factors_against_aligned_radio(args)
    ablation_scope = _ablation_scope(changed_factors)
    reference_model_parameter_count = 55042
    capacity_matching = {
        "reference_arm": "F71_radio_aligned_H128_plain_mlp",
        "reference_model_parameter_count": reference_model_parameter_count,
        "arm_model_parameter_count": model_parameter_count,
        "absolute_parameter_count_gap": abs(
            model_parameter_count - reference_model_parameter_count
        ),
        "relative_parameter_count_gap": abs(
            model_parameter_count - reference_model_parameter_count
        ) / reference_model_parameter_count,
        "design": (
            "closest_integer_hidden_dimension_to_reference_parameter_count"
            if args.local_feature_mode == "rgb_geometry"
            and args.scoring_mode == "mlp"
            and args.hidden_dimension == 213
            else "not_a_capacity_matched_rgb_control"
        ),
        "exact_parameter_count_match": (
            model_parameter_count == reference_model_parameter_count
        ),
    }
    aligned_reference = _aligned_reference_receipt(
        getattr(args, "aligned_reference_report", None),
        args=args,
        split=split,
        selected_payloads=selected,
        input_dimension=input_dimension,
        model_parameter_count=model_parameter_count,
        selected_feature_layout=selected_feature_layout,
        cohort_manifest=cohort_manifest,
        implementation_sha256=implementation_sha256,
        completion_implementation_sha256=completion_implementation_sha256,
    )
    training = _fit(
        model, examples, device=device, step_count=args.step_count,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed,
    )
    if (
        sha256_file(implementation_path) != implementation_sha256
        or sha256_file(completion_implementation_path)
        != completion_implementation_sha256
    ):
        raise RuntimeError("completion implementation changed during training")
    scene_cache_receipts = [
        {
            "scene_id": payload["scene_id"],
            "cache_path": payload["cache_path"],
            "cache_sha256": payload["cache_sha256"],
            "sealed_inputs": payload["input_receipt"],
        }
        for payload in selected
    ]
    checkpoint_path = Path(args.output_checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_configuration": {
            "input_dimension": input_dimension,
            "hidden_dimension": args.hidden_dimension,
            "dropout": args.dropout,
            "token_cardinality_normalization": TOKEN_CARDINALITY_NORMALIZATION,
            "local_feature_mode": args.local_feature_mode,
            "source_local_feature_layout": configurations[0]["local_feature_layout"],
            "selected_local_feature_layout": selected_feature_layout,
            "unknown_sampling_mode": args.unknown_sampling_mode,
            "scoring_mode": args.scoring_mode,
            "radio_alignment_control": radio_alignment_control,
            "radio_alignment_seed": radio_alignment_seed,
            "model_parameter_count": model_parameter_count,
            "availability_route": (
                "sealed_source_visible"
                if args.scoring_mode == "availability_dual_mlp"
                else "none"
            ),
            "null_head_sharing": "single_raw_learned_null",
        },
        "training_scene_ids": sorted(split.source_ids),
        "validation_scene_ids": sorted(split.development_ids),
        "scene_disjoint": True,
        "integer_instance_ids_are_model_inputs": False,
        "observed_membership_is_oracle_input": True,
        "unobserved_membership_used_as_target_only": True,
        "full_membership_used_as_target_only": False,
        "token_cardinality_normalization": TOKEN_CARDINALITY_NORMALIZATION,
        "token_cardinality_normalization_applied_in_training_and_inference": False,
        "rejected_log_token_count_null_ablation_restored": True,
        "ablation_scope": ablation_scope,
        "changed_factors_against_aligned_radio_reference": changed_factors,
        "capacity_matching": capacity_matching,
        "local_feature_mode": args.local_feature_mode,
        "source_local_feature_layout": configurations[0]["local_feature_layout"],
        "selected_local_feature_layout": selected_feature_layout,
        "mask_support_strata_available": strata_available,
        "unknown_sampling_mode": args.unknown_sampling_mode,
        "scoring_mode": args.scoring_mode,
        "radio_alignment_control": radio_alignment_control,
        "radio_alignment_seed": radio_alignment_seed,
        "radio_alignment_control_receipts": [
            runtime["radio_alignment_receipt"] for runtime in all_runtimes
        ],
        "aligned_reference_report": aligned_reference,
        "cohort_manifest": cohort_manifest,
        "scene_cache_receipts": scene_cache_receipts,
        "model_parameter_count": model_parameter_count,
        "availability_route": (
            "sealed_source_visible"
            if args.scoring_mode == "availability_dual_mlp"
            else "none"
        ),
        "null_head_sharing": "single_raw_learned_null",
        "threshold_or_temperature_changed_from_v5": False,
        "implementation_sha256": implementation_sha256,
        "completion_implementation_sha256": completion_implementation_sha256,
    }, checkpoint_path)
    per_scene = [
        _evaluate_scene(
            model, runtime, device=device, element_batch_size=args.element_batch_size,
            temperature=args.temperature,
            completion_confidence_cap=args.completion_confidence_cap,
            assignment_threshold=args.assignment_threshold,
        )
        for runtime in validation_runtime
    ]
    if (
        sha256_file(implementation_path) != implementation_sha256
        or sha256_file(completion_implementation_path)
        != completion_implementation_sha256
    ):
        raise RuntimeError("completion implementation changed during evaluation")
    aggregate_keys = [
        "soft_3d_miou",
        "unknown_only_soft_3d_miou",
        "heldout_2d_soft_miou",
        "heldout_2d_cross_view_token_soft_miou",
        "full_object_token_top1_accuracy",
        "full_k_plus_null_categorical_accuracy",
        "token_probability_concentration",
        "target_aware_token_mass_precision",
        "unknown_target_aware_token_mass_precision",
        "unknown_assignment_precision",
        "unknown_retained_object_coverage",
        "unknown_correct_assignment_recall",
        "assigned_unknown_object_top1_accuracy",
        "unknown_retained_set_null_recall",
        "unknown_assignment_precision_at_0p5",
        "unknown_retained_object_coverage_at_0p5",
        "unknown_correct_assignment_recall_at_0p5",
        "assigned_unknown_object_top1_accuracy_at_0p5",
        "known_element_fraction",
        "retained_set_false_positive_mean",
    ]
    if strata_available:
        for stratum in ("visible_but_unmasked", "never_visible"):
            aggregate_keys.extend(
                f"{stratum}_{suffix}"
                for suffix in (
                    "soft_3d_miou",
                    "assignment_precision",
                    "retained_object_coverage",
                    "correct_assignment_recall",
                    "assigned_object_top1_accuracy",
                    "retained_set_null_recall",
                )
            )
    aggregate = {}
    for method in ("baseline_observed_only", "learned_completion"):
        aggregate[method] = {
            key: _mean_scene_metric(per_scene, method, key)
            for key in aggregate_keys
        }
    pooled_soft_iou = {
        method: _pool_soft_iou_sufficient_statistics(per_scene, method)
        for method in ("baseline_observed_only", "learned_completion")
    }
    pooled_unknown = {
        method: _pooled_categorical_confusion(per_scene, method, "unknown")
        for method in ("baseline_observed_only", "learned_completion")
    }
    pooled_strata = None
    if strata_available:
        pooled_strata = {
            stratum: {
                method: _pooled_categorical_confusion(per_scene, method, stratum)
                for method in ("baseline_observed_only", "learned_completion")
            }
            for stratum in ("visible_but_unmasked", "never_visible")
        }
    improvement_3d = (
        aggregate["learned_completion"]["soft_3d_miou"]
        - aggregate["baseline_observed_only"]["soft_3d_miou"]
    )
    improvement_2d = (
        aggregate["learned_completion"]["heldout_2d_soft_miou"]
        - aggregate["baseline_observed_only"]["heldout_2d_soft_miou"]
    )
    gate = {
        "minimum_3d_miou_improvement": args.minimum_miou_improvement,
        "minimum_heldout_2d_miou_improvement": args.minimum_miou_improvement,
        "minimum_unknown_assignment_precision": args.minimum_assignment_precision,
        "minimum_unknown_target_aware_token_mass_precision": args.minimum_assignment_precision,
        "minimum_unknown_retained_object_coverage": args.minimum_assigned_fraction,
        "minimum_unknown_retained_set_null_recall": args.minimum_null_recall,
        "observed_3d_miou_improvement": improvement_3d,
        "observed_heldout_2d_miou_improvement": improvement_2d,
        "directions": {
            "3d_miou": improvement_3d >= args.minimum_miou_improvement,
            "heldout_2d_miou": improvement_2d >= args.minimum_miou_improvement,
            "unknown_assignment_precision": (
                aggregate["learned_completion"]["unknown_assignment_precision"]
                >= args.minimum_assignment_precision
            ),
            "unknown_target_aware_token_mass_precision": (
                aggregate["learned_completion"]["unknown_target_aware_token_mass_precision"]
                >= args.minimum_assignment_precision
            ),
            "unknown_retained_object_coverage": (
                aggregate["learned_completion"]["unknown_retained_object_coverage"]
                >= args.minimum_assigned_fraction
            ),
            "unknown_retained_set_null_recall": (
                aggregate["learned_completion"]["unknown_retained_set_null_recall"]
                >= args.minimum_null_recall
            ),
            "positive_clamp": all(row["learned_completion"]["positive_clamp_max_error"] == 0 for row in per_scene),
            "negative_clamp": all(row["learned_completion"]["negative_clamp_max_error"] == 0 for row in per_scene),
        },
    }
    gate["passes_scene_disjoint_completion_oracle"] = all(gate["directions"].values())
    report = {
        "schema": REPORT_SCHEMA,
        "stage": "scene_disjoint_supervised_completion_oracle",
        "scope": "development pilot; observed-token-conditioned completion with learned null rejection",
        "oracle_identity_diagnostic_only": True,
        "association_isolated": True,
        "training_and_validation_scenes_disjoint": True,
        "integer_instance_ids_are_model_inputs": False,
        "observed_membership_is_oracle_input": True,
        "unobserved_membership_used_as_target_only": True,
        "full_membership_used_as_target_only": False,
        "completion_writes_unknown_only": True,
        "observed_positive_and_negative_are_clamped": True,
        "token_cardinality_normalization": TOKEN_CARDINALITY_NORMALIZATION,
        "token_cardinality_normalization_applied_in_training_and_inference": False,
        "rejected_log_token_count_null_ablation_restored": True,
        "ablation_scope": ablation_scope,
        "changed_factors_against_aligned_radio_reference": changed_factors,
        "capacity_matching": capacity_matching,
        "local_feature_mode": args.local_feature_mode,
        "source_local_feature_layout": configurations[0]["local_feature_layout"],
        "selected_local_feature_layout": selected_feature_layout,
        "mask_support_strata_available": strata_available,
        "unknown_sampling_mode": args.unknown_sampling_mode,
        "scoring_mode": args.scoring_mode,
        "radio_alignment_control": radio_alignment_control,
        "radio_alignment_seed": radio_alignment_seed,
        "radio_alignment_control_receipts": [
            runtime["radio_alignment_receipt"] for runtime in all_runtimes
        ],
        "aligned_reference_report": aligned_reference,
        "cohort_manifest": cohort_manifest,
        "model_parameter_count": model_parameter_count,
        "availability_route": (
            "sealed_source_visible"
            if args.scoring_mode == "availability_dual_mlp"
            else "none"
        ),
        "null_head_sharing": "single_raw_learned_null",
        "primary_assignment_decision": "threshold_free_token_plus_null_argmax",
        "legacy_assignment_threshold_metrics_are_diagnostic_only": True,
        "heldout_2d_metric": "cross_view_token_soft_iou_with_absent_view_false_positives",
        "threshold_or_temperature_changed_from_v5": False,
        "split": {
            "training_scene_ids": sorted(split.source_ids),
            "validation_scene_ids": sorted(split.development_ids),
            "overlap": [],
        },
        "carrier_and_observation_configuration": configurations[0],
        "training_configuration": {
            "seed": args.seed,
            "step_count": args.step_count,
            "batch_size": args.batch_size,
            "element_batch_size": args.element_batch_size,
            "samples_per_token": args.samples_per_token,
            "unknown_sampling_mode": args.unknown_sampling_mode,
            "scoring_mode": args.scoring_mode,
            "radio_alignment_control": radio_alignment_control,
            "radio_alignment_seed": radio_alignment_seed,
            "input_dimension": input_dimension,
            "model_parameter_count": model_parameter_count,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "device": str(device),
            "hidden_dimension": args.hidden_dimension,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "completion_confidence_cap": args.completion_confidence_cap,
            "assignment_threshold": args.assignment_threshold,
            "minimum_miou_improvement": args.minimum_miou_improvement,
            "minimum_assignment_precision": args.minimum_assignment_precision,
            "minimum_assigned_fraction": args.minimum_assigned_fraction,
            "minimum_null_recall": args.minimum_null_recall,
            "local_feature_mode": args.local_feature_mode,
            "source_local_feature_layout": configurations[0]["local_feature_layout"],
            "selected_local_feature_layout": selected_feature_layout,
            "token_probability_concentration_is_diagnostic_only": True,
            "assignment_threshold_is_legacy_diagnostic_only": True,
            "token_cardinality_normalization": TOKEN_CARDINALITY_NORMALIZATION,
            "validation_used_for_model_selection": False,
        },
        "training": {
            **training,
            "sampling_audit": sampling_audit,
        },
        "per_validation_scene": per_scene,
        "scene_macro": aggregate,
        "pooled_soft_iou_sufficient_statistics": pooled_soft_iou,
        "pooled_unknown_categorical_confusion": pooled_unknown,
        "pooled_unknown_strata_categorical_confusion": pooled_strata,
        "gate": gate,
        "external_directional_references_not_directly_comparable": {
            "source_lifted_3d_soft_miou": 0.46016,
            "source_lifted_heldout_2d_soft_miou": 0.46701,
            "reason": "the completion pilot uses a different scene-disjoint cohort and simulated partial observations",
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "scene_cache_receipts": scene_cache_receipts,
        "implementation_sha256": implementation_sha256,
        "completion_implementation_sha256": completion_implementation_sha256,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--training-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument(
        "--local-feature-mode", choices=LOCAL_FEATURE_MODES, required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--samples-per-token", type=int, default=256)
    parser.add_argument(
        "--unknown-sampling-mode",
        choices=UNKNOWN_SAMPLING_MODES,
        required=True,
    )
    parser.add_argument("--scoring-mode", choices=SCORING_MODES, required=True)
    parser.add_argument(
        "--radio-alignment-control",
        choices=RADIO_ALIGNMENT_CONTROLS,
        default="aligned",
    )
    parser.add_argument("--radio-alignment-seed", type=int, default=20260831)
    parser.add_argument("--aligned-reference-report")
    parser.add_argument("--cohort-manifest")
    parser.add_argument("--step-count", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--element-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--completion-confidence-cap", type=float, default=0.95)
    parser.add_argument("--assignment-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-miou-improvement", type=float, default=0.10)
    parser.add_argument("--minimum-assignment-precision", type=float, default=0.94)
    parser.add_argument("--minimum-assigned-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-null-recall", type=float, default=0.90)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-instance-oracle-training", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"scene_macro": report["scene_macro"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
