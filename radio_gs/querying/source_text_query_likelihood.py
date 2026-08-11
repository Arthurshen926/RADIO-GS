"""Source-only text calibration for the shared query-likelihood interface.

The compact field and its text readout are intentionally frozen here.  This
module only turns their *separate* sufficient statistics into examples for
``MonotoneQueryLikelihoodHead``:

* positive class-text affinity;
* canonical-negative affinity;
* the legacy cosine-margin probability, retained as a field prior;
* observation coverage; and
* query-independent field reliability.

No benchmark-specific threshold, target image, LERF query, or evaluation
metric is part of this contract.  Labels are legal only when an immutable
source record declares an official ScanNet training scene.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from .query_likelihood_head import MonotoneQueryLikelihoodHead, QueryLikelihoodInputs


SOURCE_TEXT_SCENE_INPUT_SCHEMA = (
    "radio_gs.source_text_query_likelihood_scene_input.v1"
)
SOURCE_TEXT_TRAINING_SHARD_SCHEMA = (
    "radio_gs.source_text_query_likelihood_training_shard.v1"
)
SOURCE_TEXT_DATASET_MANIFEST_SCHEMA = (
    "radio_gs.source_text_query_likelihood_training_dataset.v1"
)
SOURCE_TEXT_CHECKPOINT_SCHEMA = "radio_gs.source_text_query_likelihood_head.v1"
LEGACY_FIELD_PRIOR_LOGIT_SCALE = 10.0
AFFINITY_CHANNELS = ("positive_class_text", "canonical_negative_text")
REQUIRED_LINEAGE_RECORDS = (
    "descriptor_source",
    "semantic_label_source",
    "class_text_source",
    "canonical_negative_text_source",
    "field_state_source",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes(order="C"))
    else:
        for start in range(0, int(tensor.shape[0]), 4096):
            digest.update(tensor[start : start + 4096].numpy().tobytes(order="C"))
    return digest.hexdigest()


def source_text_likelihood_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_TEXT_TRAINING_SHARD_SCHEMA,
        "schema_version": 1,
        "head": {
            "class": "MonotoneQueryLikelihoodHead",
            "input_class": "QueryLikelihoodInputs",
            "affinity_channel_count": 1,
            "output": "PrimitiveUnaryEvidence(q,c)",
        },
        "input_factorization": {
            "positive_affinity": "(class_text_cosine + 1) / 2",
            "negative_affinity": "(canonical_negative_cosine + 1) / 2",
            "field_prior": (
                "max_scale sigmoid(10 * (class_text_cosine - "
                "max_canonical_negative_cosine_same_scale))"
            ),
            "confidence": "coverage * reliability",
            "affinity_set_statistics": "permutation_invariant_peak_and_mean",
        },
        "training": {
            "partition": "official_scannet_source_train_only",
            "target": (
                "official_source_train_soft_semantic_class_distribution_"
                "without_region_argmax"
            ),
            "objective": (
                "scene_query_macro_confidence_weighted_balanced_binary_cross_entropy"
            ),
            "class_absent_from_scene": "omit_example_without_changing_vocabulary",
            "label_authority_weight": (
                "source_only_official_semantic_member_coverage_loss_weight_only"
            ),
        },
        "forbidden_inputs": [
            "scannet_development_labels",
            "scannet_test_labels",
            "lerf_queries_or_ground_truth",
            "target_rgb_or_mask",
            "benchmark_prediction_or_metric",
            "per_scene_or_per_query_metric_tuning",
        ],
    }


def _require_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file record")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(digest, str):
        raise ValueError(f"{label} must contain path and sha256")
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} sha256 must be a lowercase digest")
    source = Path(path).expanduser().resolve(strict=True)
    if sha256_file(source) != digest:
        raise ValueError(f"{label} changed")
    return {"path": str(source), "sha256": digest}


def _finite_matrix(value: object, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float().contiguous()
    if tensor.ndim != 2 or min(tensor.shape) <= 0:
        raise ValueError(f"{label} must be a non-empty matrix")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains NaN or infinity")
    if bool((tensor.norm(dim=-1) <= 1e-8).any()):
        raise ValueError(f"{label} contains a zero vector")
    return tensor


def _validated_scene_input(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if payload.get("schema") != SOURCE_TEXT_SCENE_INPUT_SCHEMA:
        raise ValueError("unexpected source text scene input schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected source text scene input schema_version")
    if payload.get("partition") != "source_train":
        raise PermissionError("text calibrator accepts source_train scenes only")
    scene_id = payload.get("scene_id")
    physical_space_id = payload.get("physical_space_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be non-empty")
    if not isinstance(physical_space_id, str) or not physical_space_id:
        raise ValueError("physical_space_id must be non-empty")
    safety = payload.get("source_access")
    required_safety = {
        "official_scannet_train_scene": True,
        "source_train_semantic_labels_opened": True,
        "development_labels_opened": False,
        "test_labels_opened": False,
        "lerf_queries_or_ground_truth_opened": False,
        "target_rgb_or_mask_opened": False,
        "benchmark_predictions_or_metrics_opened": False,
        "per_scene_or_per_query_metric_tuning": False,
    }
    if not isinstance(safety, Mapping):
        raise ValueError("source text scene input lacks source_access")
    for key, expected in required_safety.items():
        if safety.get(key) is not expected:
            raise PermissionError(f"source text scene input violates {key}")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(
        REQUIRED_LINEAGE_RECORDS
    ):
        raise ValueError("source text scene lineage is incomplete")
    validated_lineage = {
        key: _require_record(lineage[key], label=key)
        for key in REQUIRED_LINEAGE_RECORDS
    }

    descriptors = torch.as_tensor(payload.get("descriptors")).detach().cpu().float()
    if descriptors.ndim == 2:
        descriptors = descriptors[:, None, :]
    if descriptors.ndim != 3 or min(descriptors.shape) <= 0:
        raise ValueError("descriptors must be [N,D] or [N,S,D]")
    if not bool(torch.isfinite(descriptors).all()) or bool(
        (descriptors.norm(dim=-1) <= 1e-8).any()
    ):
        raise ValueError("descriptors must be finite and nonzero")
    descriptors = descriptors.contiguous()
    rows, _scales, dimension = descriptors.shape
    class_text = _finite_matrix(payload.get("class_text_embeddings"), label="class text")
    negative_text = _finite_matrix(
        payload.get("canonical_negative_text_embeddings"),
        label="canonical negative text",
    )
    if class_text.shape[1] != dimension or negative_text.shape[1] != dimension:
        raise ValueError("descriptor and text dimensions differ")
    class_ids = payload.get("class_ids")
    class_names = payload.get("class_names")
    if (
        not isinstance(class_ids, Sequence)
        or isinstance(class_ids, (str, bytes))
        or not isinstance(class_names, Sequence)
        or isinstance(class_names, (str, bytes))
    ):
        raise ValueError("class ids/names must be ordered sequences")
    ids = [int(value) for value in class_ids]
    names = [str(value) for value in class_names]
    if len(ids) != len(names) or len(ids) != class_text.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("class id/name/text axes differ")
    if any(not name for name in names):
        raise ValueError("class names must be non-empty")
    valid = torch.as_tensor(payload.get("valid")).detach().cpu()
    coverage = torch.as_tensor(payload.get("coverage")).detach().cpu().float().reshape(-1)
    reliability = torch.as_tensor(payload.get("reliability")).detach().cpu().float().reshape(-1)
    training_label_weight = (
        torch.ones(rows, dtype=torch.float32)
        if payload.get("training_label_weight") is None
        else torch.as_tensor(payload.get("training_label_weight"))
        .detach()
        .cpu()
        .float()
        .reshape(-1)
    )
    if valid.shape != (rows,) or valid.dtype != torch.bool:
        raise ValueError("valid must align with descriptor rows")
    soft_target_value = payload.get("semantic_class_distribution")
    if soft_target_value is None:
        labels = (
            torch.as_tensor(payload.get("semantic_label_ids"))
            .detach()
            .cpu()
            .long()
            .reshape(-1)
        )
        if labels.shape != (rows,):
            raise ValueError("semantic labels must align with descriptor rows")
        soft_target = torch.zeros((rows, len(ids)), dtype=torch.float32)
        for class_index, class_id in enumerate(ids):
            soft_target[:, class_index] = labels == int(class_id)
    else:
        soft_target = (
            torch.as_tensor(soft_target_value).detach().cpu().float().contiguous()
        )
        if soft_target.shape != (rows, len(ids)):
            raise ValueError("semantic class distribution must be [N,C]")
        if not bool(torch.isfinite(soft_target).all()) or bool(
            ((soft_target < 0) | (soft_target > 1)).any()
        ):
            raise ValueError("semantic class distribution must be finite [0,1]")
        if bool((soft_target.sum(dim=1) > 1.0 + 1e-6).any()):
            raise ValueError("semantic class distribution row mass exceeds one")
    for name, tensor in (
        ("coverage", coverage),
        ("reliability", reliability),
        ("training_label_weight", training_label_weight),
    ):
        if tensor.shape != (rows,) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be a finite [N] vector")
        if bool(((tensor < 0) | (tensor > 1)).any()):
            raise ValueError(f"{name} must be in [0,1]")
    trainable = valid & (coverage > 0) & (reliability > 0)
    if int(trainable.sum()) < 2:
        raise ValueError("source scene has fewer than two trainable semantic rows")
    present = [
        class_id
        for class_index, class_id in enumerate(ids)
        if float(soft_target[trainable, class_index].sum()) > 0
    ]
    if len(present) < 2:
        raise ValueError("source scene must contain at least two known semantic classes")
    return {
        **payload,
        "scene_id": scene_id,
        "physical_space_id": physical_space_id,
        "lineage": validated_lineage,
        "descriptors": descriptors,
        "class_text_embeddings": class_text,
        "canonical_negative_text_embeddings": negative_text,
        "class_ids": ids,
        "class_names": names,
        "semantic_class_distribution": soft_target,
        "valid": valid,
        "coverage": coverage,
        "reliability": reliability,
        "training_label_weight": training_label_weight,
        "trainable": trainable,
        "present_class_ids": present,
    }


def build_source_text_training_shard(value: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize one immutable-ready source scene into separated affinities."""

    source = _validated_scene_input(value)
    descriptors = F.normalize(source["descriptors"], dim=-1, eps=1e-8)
    class_text = F.normalize(source["class_text_embeddings"], dim=-1, eps=1e-8)
    negative_text = F.normalize(
        source["canonical_negative_text_embeddings"], dim=-1, eps=1e-8
    )
    positive_cosine = torch.einsum("nsd,cd->nsc", descriptors, class_text)
    negative_cosine = torch.einsum("nsd,kd->nsk", descriptors, negative_text)
    per_scale_margin = positive_cosine - negative_cosine.amax(dim=-1, keepdim=True)
    field_prior = torch.sigmoid(
        LEGACY_FIELD_PRIOR_LOGIT_SCALE * per_scale_margin
    ).amax(dim=1)
    positive_affinity = ((positive_cosine + 1.0) * 0.5).clamp(0, 1)
    negative_affinity = ((negative_cosine + 1.0) * 0.5).clamp(0, 1)
    trainable = source["trainable"]
    class_mass = {
        str(class_id): float(
            source["semantic_class_distribution"][trainable, class_index].sum()
        )
        for class_index, class_id in enumerate(source["class_ids"])
    }
    tensors = {
        "positive_affinity": positive_affinity.half().contiguous(),
        "canonical_negative_affinity": negative_affinity.half().contiguous(),
        "field_prior_probability": field_prior.float().contiguous(),
        "semantic_class_distribution": source[
            "semantic_class_distribution"
        ].float().contiguous(),
        "valid": source["valid"].contiguous(),
        "coverage": source["coverage"].float().contiguous(),
        "reliability": source["reliability"].float().contiguous(),
        "training_label_weight": source[
            "training_label_weight"
        ].float().contiguous(),
    }
    return {
        "schema": SOURCE_TEXT_TRAINING_SHARD_SCHEMA,
        "schema_version": 1,
        "contract": source_text_likelihood_contract(),
        "scene_id": source["scene_id"],
        "physical_space_id": source["physical_space_id"],
        "partition": "source_train",
        "class_ids": list(source["class_ids"]),
        "class_names": list(source["class_names"]),
        "present_class_ids": list(source["present_class_ids"]),
        "canonical_negative_count": int(negative_text.shape[0]),
        "scale_count": int(descriptors.shape[1]),
        "descriptor_dimension": int(descriptors.shape[2]),
        "class_positive_mass": class_mass,
        **tensors,
        "lineage": dict(source["lineage"]),
        "source_access": dict(source["source_access"]),
        "channel_sha256": {
            key: _tensor_sha256(tensor) for key, tensor in tensors.items()
        },
    }


def validate_source_text_training_shard(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source text training shard must be a mapping")
    payload = dict(value)
    if payload.get("schema") != SOURCE_TEXT_TRAINING_SHARD_SCHEMA:
        raise ValueError("unexpected source text training shard schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected source text training shard schema_version")
    if payload.get("contract") != source_text_likelihood_contract():
        raise ValueError("source text likelihood contract differs")
    if payload.get("partition") != "source_train":
        raise PermissionError("text likelihood shard is not source_train")
    safety = payload.get("source_access", {})
    if (
        safety.get("official_scannet_train_scene") is not True
        or safety.get("source_train_semantic_labels_opened") is not True
        or safety.get("development_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("lerf_queries_or_ground_truth_opened") is not False
        or safety.get("target_rgb_or_mask_opened") is not False
        or safety.get("benchmark_predictions_or_metrics_opened") is not False
        or safety.get("per_scene_or_per_query_metric_tuning") is not False
    ):
        raise PermissionError("source text shard crosses its training boundary")
    positive = torch.as_tensor(payload.get("positive_affinity")).detach().cpu()
    negative = torch.as_tensor(payload.get("canonical_negative_affinity")).detach().cpu()
    prior = torch.as_tensor(payload.get("field_prior_probability")).detach().cpu().float()
    targets = (
        torch.as_tensor(payload.get("semantic_class_distribution"))
        .detach()
        .cpu()
        .float()
    )
    valid = torch.as_tensor(payload.get("valid")).detach().cpu()
    coverage = torch.as_tensor(payload.get("coverage")).detach().cpu().float().reshape(-1)
    reliability = torch.as_tensor(payload.get("reliability")).detach().cpu().float().reshape(-1)
    training_label_weight = (
        torch.as_tensor(payload.get("training_label_weight"))
        .detach()
        .cpu()
        .float()
        .reshape(-1)
    )
    class_ids = [int(value) for value in payload.get("class_ids", [])]
    class_names = [str(value) for value in payload.get("class_names", [])]
    if positive.ndim != 3 or negative.ndim != 3:
        raise ValueError("source text affinities must be [N,S,C]/[N,S,K]")
    rows, scales, classes = positive.shape
    if (
        negative.shape[0] != rows
        or negative.shape[1] != scales
        or targets.shape != (rows, classes)
        or valid.shape != (rows,)
        or valid.dtype != torch.bool
        or coverage.shape != (rows,)
        or reliability.shape != (rows,)
        or training_label_weight.shape != (rows,)
        or len(class_ids) != classes
        or len(class_names) != classes
    ):
        raise ValueError("source text shard axes differ")
    if bool((targets.sum(dim=1) > 1.0 + 1e-6).any()):
        raise ValueError("source text target row mass exceeds one")
    for name, tensor in (
        ("positive_affinity", positive.float()),
        ("canonical_negative_affinity", negative.float()),
        ("field_prior_probability", prior),
        ("semantic_class_distribution", targets),
        ("coverage", coverage),
        ("reliability", reliability),
        ("training_label_weight", training_label_weight),
    ):
        if not bool(torch.isfinite(tensor).all()) or bool(
            ((tensor < 0) | (tensor > 1)).any()
        ):
            raise ValueError(f"{name} must be finite and in [0,1]")
    channels = payload.get("channel_sha256")
    tensor_map = {
        "positive_affinity": positive,
        "canonical_negative_affinity": negative,
        "field_prior_probability": prior,
        "semantic_class_distribution": targets,
        "valid": valid,
        "coverage": coverage,
        "reliability": reliability,
        "training_label_weight": training_label_weight,
    }
    if not isinstance(channels, Mapping) or set(channels) != set(tensor_map):
        raise ValueError("source text shard channel hashes differ")
    for key, tensor in tensor_map.items():
        if channels.get(key) != _tensor_sha256(tensor):
            raise ValueError(f"source text shard channel changed: {key}")
    for record in REQUIRED_LINEAGE_RECORDS:
        _require_record(payload.get("lineage", {}).get(record), label=record)
    return {
        **payload,
        **tensor_map,
        "class_ids": class_ids,
        "class_names": class_names,
    }


@dataclass(frozen=True)
class SourceTextLikelihoodExample:
    observations: QueryLikelihoodInputs
    target: torch.Tensor
    training_weight: torch.Tensor
    scene_id: str
    class_id: int
    class_name: str


def iter_source_text_likelihood_examples(
    value: Mapping[str, Any],
) -> Iterator[SourceTextLikelihoodExample]:
    payload = validate_source_text_training_shard(value)
    row_weight = (
        payload["valid"].float()
        * payload["training_label_weight"]
        * payload["coverage"]
        * payload["reliability"]
    )
    negative = payload["canonical_negative_affinity"].reshape(
        int(payload["positive_affinity"].shape[0]), -1
    )
    for class_index, (class_id, class_name) in enumerate(
        zip(payload["class_ids"], payload["class_names"])
    ):
        target = payload["semantic_class_distribution"][:, class_index]
        positive_mass = float((row_weight * target).sum())
        negative_mass = float((row_weight * (1.0 - target)).sum())
        if positive_mass <= 0 or negative_mass <= 0:
            continue
        observations = QueryLikelihoodInputs(
            positive_affinity=payload["positive_affinity"][:, :, class_index],
            negative_affinity=negative,
            prior_probability=payload["field_prior_probability"][:, class_index],
            coverage=payload["coverage"],
            reliability=payload["reliability"],
        ).validated()
        yield SourceTextLikelihoodExample(
            observations=observations,
            target=target,
            training_weight=row_weight,
            scene_id=str(payload["scene_id"]),
            class_id=int(class_id),
            class_name=str(class_name),
        )


def confidence_weighted_balanced_bce(
    probability: torch.Tensor,
    target: torch.Tensor,
    training_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q = torch.as_tensor(probability).float().reshape(-1).clamp(1e-6, 1 - 1e-6)
    y = torch.as_tensor(target).float().reshape(-1)
    weight = torch.as_tensor(training_weight).float().reshape(-1)
    if q.shape != y.shape or q.shape != weight.shape:
        raise ValueError("balanced source text tensors must align")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("training_weight must be finite and non-negative")
    if not bool(torch.isfinite(y).all()) or bool(((y < 0) | (y > 1)).any()):
        raise ValueError("source text target must be finite and in [0,1]")
    positive_weight = weight * y
    negative_weight = weight * (1 - y)
    positive_total = positive_weight.sum()
    negative_total = negative_weight.sum()
    if float(positive_total) <= 0 or float(negative_total) <= 0:
        raise ValueError("balanced source text loss requires both classes")
    positive_bce = -(positive_weight * q.log()).sum() / positive_total
    negative_bce = -(negative_weight * torch.log1p(-q)).sum() / negative_total
    loss = 0.5 * (positive_bce + negative_bce)
    return loss, {
        "balanced_bce": loss,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "positive_weight": positive_total,
        "negative_weight": negative_total,
    }


def initialize_source_text_head(
    head: MonotoneQueryLikelihoodHead,
    *,
    affinity_weight: float = 0.05,
    prior_weight: float = 1.0,
) -> None:
    """Initialize near the frozen legacy prior without changing head defaults."""

    if head.affinity_channel_count != 1:
        raise ValueError("source text calibrator requires one affinity channel")
    if affinity_weight <= 0 or prior_weight <= 0:
        raise ValueError("source text initialization weights must be positive")

    def inverse_softplus(value: float) -> float:
        return math.log(math.expm1(float(value)))

    with torch.no_grad():
        head.bias.zero_()
        head.raw_positive_weights.fill_(inverse_softplus(affinity_weight))
        head.raw_negative_weights.fill_(inverse_softplus(affinity_weight))
        head.raw_prior_weight.fill_(inverse_softplus(prior_weight))


__all__ = [
    "AFFINITY_CHANNELS",
    "LEGACY_FIELD_PRIOR_LOGIT_SCALE",
    "REQUIRED_LINEAGE_RECORDS",
    "SOURCE_TEXT_CHECKPOINT_SCHEMA",
    "SOURCE_TEXT_DATASET_MANIFEST_SCHEMA",
    "SOURCE_TEXT_SCENE_INPUT_SCHEMA",
    "SOURCE_TEXT_TRAINING_SHARD_SCHEMA",
    "SourceTextLikelihoodExample",
    "build_source_text_training_shard",
    "confidence_weighted_balanced_bce",
    "initialize_source_text_head",
    "iter_source_text_likelihood_examples",
    "sha256_file",
    "source_text_likelihood_contract",
    "validate_source_text_training_shard",
]
