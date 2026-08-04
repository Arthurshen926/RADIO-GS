"""Fail-closed query-independent relation-feature caches for graph diffusion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256


ARTIFACT_TYPE = "query_independent_cradio_dino_v3_primitive_pca40_relation"
SCHEMA_VERSION = 1
TENSOR_KEYS = {
    "global_rows",
    "relation_features",
    "feature_mean",
    "feature_std",
    "pca_mean",
    "pca_components",
    "pca_singular_values",
}
PAYLOAD_KEYS = {
    "schema_version",
    "artifact_type",
    "scene_id",
    "num_global_rows",
    "source_feature_sha256",
    "source_xyz_sha256",
    "tensors",
    "tensor_sha256",
    "tensor_bundle_sha256",
    "metadata",
}


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    digest = str(value)
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


@dataclass(frozen=True)
class QueryDiffusionRelationCache:
    global_rows: torch.Tensor
    relation_features: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    pca_mean: torch.Tensor
    pca_components: torch.Tensor
    pca_singular_values: torch.Tensor
    num_global_rows: int
    source_feature_sha256: str
    source_xyz_sha256: str
    metadata: Mapping[str, object]

    @property
    def num_nodes(self) -> int:
        return int(self.global_rows.numel())

    @property
    def relation_dimension(self) -> int:
        return int(self.relation_features.shape[1])


def validate_query_diffusion_relation_payload(
    payload: object,
    *,
    expected_scene_id: str | None = None,
    expected_global_rows: torch.Tensor | None = None,
    expected_num_global_rows: int | None = None,
    expected_source_feature_sha256: str = "",
    expected_source_xyz_sha256: str = "",
    expected_registration_sha256: str = "",
    expected_capability_sidecar_sha256: str = "",
    expected_field_checkpoint_sha256: str = "",
) -> QueryDiffusionRelationCache:
    """Validate provenance, safety declarations, schema, and every tensor."""

    if not isinstance(payload, Mapping) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("query-diffusion relation cache schema differs")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
        or payload["artifact_type"] != ARTIFACT_TYPE
    ):
        raise ValueError("unsupported query-diffusion relation cache")
    scene_id = str(payload["scene_id"])
    if expected_scene_id is not None and scene_id != str(expected_scene_id):
        raise ValueError("query-diffusion relation scene differs")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("query-diffusion relation metadata is malformed")
    safety = {
        "query_independent": True,
        "labels_opened": False,
        "target_rgb_opened": False,
        "target_masks_opened": False,
        "target_metrics_opened": False,
        "native_ludvig_dinov2_pca40_exact": False,
    }
    if any(metadata.get(key) is not value for key, value in safety.items()):
        raise ValueError("query-diffusion relation safety boundary differs")
    if metadata.get("relation_source") != "official_C_RADIOv4_dino_v3_7b_primitive_rows":
        raise ValueError("query-diffusion relation source differs")
    if metadata.get("transform") != "standardize_PCA40_singular_value_weighted":
        raise ValueError("query-diffusion relation transform differs")
    if int(metadata.get("pca_components", -1)) != 40:
        raise ValueError("query-diffusion relation dimension differs")
    for key, expected in (
        ("experiment_registration_sha256", expected_registration_sha256),
        ("capability_sidecar_sha256", expected_capability_sidecar_sha256),
        ("field_checkpoint_sha256", expected_field_checkpoint_sha256),
    ):
        actual = str(metadata.get(key, ""))
        if not _is_sha256(actual) or (expected and actual != expected):
            raise ValueError(f"query-diffusion relation {key} differs")
    source_feature_sha256 = str(payload["source_feature_sha256"])
    source_xyz_sha256 = str(payload["source_xyz_sha256"])
    if not _is_sha256(source_feature_sha256) or not _is_sha256(source_xyz_sha256):
        raise ValueError("query-diffusion relation source digest is malformed")
    if expected_source_feature_sha256 and source_feature_sha256 != expected_source_feature_sha256:
        raise ValueError("query-diffusion relation source feature differs")
    if expected_source_xyz_sha256 and source_xyz_sha256 != expected_source_xyz_sha256:
        raise ValueError("query-diffusion relation source geometry differs")

    tensors = payload["tensors"]
    if not isinstance(tensors, Mapping) or set(tensors) != TENSOR_KEYS:
        raise ValueError("query-diffusion relation tensor schema differs")
    rows = tensors["global_rows"]
    relation = tensors["relation_features"]
    if (
        not torch.is_tensor(rows)
        or rows.device.type != "cpu"
        or rows.dtype != torch.int64
        or rows.ndim != 1
        or not rows.is_contiguous()
        or rows.unique().numel() != rows.numel()
    ):
        raise ValueError("query-diffusion relation global rows are malformed")
    count = int(rows.numel())
    if count == 0 or bool((rows < 0).any()):
        raise ValueError("query-diffusion relation global rows are empty or negative")
    num_global_rows = int(payload["num_global_rows"])
    if num_global_rows <= int(rows.max()):
        raise ValueError("query-diffusion relation global row domain differs")
    if expected_num_global_rows is not None and num_global_rows != int(expected_num_global_rows):
        raise ValueError("query-diffusion relation global row count differs")
    if expected_global_rows is not None and not torch.equal(
        rows, torch.as_tensor(expected_global_rows).long().cpu()
    ):
        raise ValueError("query-diffusion relation valid rows differ")
    if (
        not torch.is_tensor(relation)
        or relation.device.type != "cpu"
        or relation.dtype != torch.float32
        or tuple(relation.shape) != (count, 40)
        or not relation.is_contiguous()
        or not bool(torch.isfinite(relation).all())
    ):
        raise ValueError("query-diffusion relation features are malformed")
    input_dimension = int(metadata.get("source_dimension", -1))
    for name, shape in (
        ("feature_mean", (input_dimension,)),
        ("feature_std", (input_dimension,)),
        ("pca_mean", (input_dimension,)),
        ("pca_components", (40, input_dimension)),
        ("pca_singular_values", (40,)),
    ):
        value = tensors[name]
        if (
            input_dimension <= 0
            or not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"query-diffusion relation tensor {name} is malformed")
    if not bool((tensors["feature_std"] > 0).all()):
        raise ValueError("query-diffusion relation standard deviation is not positive")
    if not bool((tensors["pca_singular_values"] > 0).all()):
        raise ValueError("query-diffusion relation singular values are not positive")
    digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    if (
        payload["tensor_sha256"] != digests
        or payload["tensor_bundle_sha256"] != canonical_json_sha256(digests)
    ):
        raise ValueError("query-diffusion relation tensor digest differs")
    return QueryDiffusionRelationCache(
        global_rows=rows,
        relation_features=relation,
        feature_mean=tensors["feature_mean"],
        feature_std=tensors["feature_std"],
        pca_mean=tensors["pca_mean"],
        pca_components=tensors["pca_components"],
        pca_singular_values=tensors["pca_singular_values"],
        num_global_rows=num_global_rows,
        source_feature_sha256=source_feature_sha256,
        source_xyz_sha256=source_xyz_sha256,
        metadata=dict(metadata),
    )


def load_query_diffusion_relation_cache(
    path: str | Path, **expected: object
) -> QueryDiffusionRelationCache:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return validate_query_diffusion_relation_payload(payload, **expected)
