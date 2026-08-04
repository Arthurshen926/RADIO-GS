#!/usr/bin/env python3
"""Build a query-independent PCA40 relation cache from official C-RADIO rows.

This adapts the released LUDVIG standardize/PCA40/singular-value-weighting
sequence to canonical C-RADIO DINOv3 *primitive* rows.  It is deliberately and
permanently labelled non-native: LUDVIG fits native DINOv2 image patch tokens
before uplift, whereas this cache fits already-uplifted C-RADIO rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.prompt_responsibility_cache import sha256_file, tensor_sha256
from radio_gs.interfaces.query_diffusion_relation_cache import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    canonical_json_sha256,
    validate_query_diffusion_relation_payload,
)


REGISTRATION_SHA256 = (
    "4728cfc6cfc5028bcb4a8d966afb7914ca2a24e2a753b5494407c8a346c2bc4c"
)
SOURCE_ADAPTOR = "dino_v3_7b.feature_projection"
SOURCE_DIMENSION = 4096
PCA_COMPONENTS = 40
PCA_SUBSAMPLE = 500_000
PCA_SEED = 0


def chunked_tensor_sha256(value: torch.Tensor, *, row_chunk_size: int = 4096) -> str:
    """Hash a large row-major CPU tensor without one full-size bytes copy."""

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    digest = hashlib.sha256()
    for start in range(0, tensor.shape[0], int(row_chunk_size)):
        chunk = tensor[start : start + int(row_chunk_size)].contiguous().numpy()
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def float_rows_sha256(value: torch.Tensor, *, row_chunk_size: int = 65536) -> str:
    """Match the geometry authority's raw little-endian float32 row digest."""

    rows = torch.as_tensor(value).detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    for start in range(0, rows.shape[0], int(row_chunk_size)):
        array = rows[start : start + int(row_chunk_size)].numpy().astype(
            "<f4", copy=False
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def fit_ludvig_inspired_pca_relation(
    source_features: torch.Tensor,
    *,
    n_components: int = PCA_COMPONENTS,
    pca_subsample: int = PCA_SUBSAMPLE,
    seed: int = PCA_SEED,
    projection_chunk_size: int = 8192,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Apply the registered standardize/PCA/eigenvalue-weight sequence on CPU."""

    from sklearn import __version__ as sklearn_version
    from sklearn.decomposition import PCA

    source = torch.as_tensor(source_features).detach()
    if source.device.type != "cpu" or source.ndim != 2:
        raise ValueError("source features must be a CPU matrix")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("source features contain NaN or infinity")
    count, dimension = map(int, source.shape)
    components = int(n_components)
    if components <= 0 or components > min(count, dimension):
        raise ValueError("PCA component count is invalid")
    if int(pca_subsample) < components or int(projection_chunk_size) <= 0:
        raise ValueError("PCA subsample or projection chunk size is invalid")

    # LUDVIG performs these two reductions in torch before calling sklearn.
    # correction=1 is explicit here so a future torch default cannot drift.
    standardized = source.float().contiguous()
    # float16 capability rows already require and produce a distinct float32
    # allocation.  Clone only when a float32 caller aliases the source; this
    # avoids an unnecessary second multi-gigabyte copy on real caches.
    if standardized.data_ptr() == source.data_ptr():
        standardized = standardized.clone()
    feature_mean = standardized.mean(dim=0)
    feature_std = standardized.std(dim=0, correction=1)
    if not bool(torch.isfinite(feature_mean).all()) or not bool(
        torch.isfinite(feature_std).all()
    ):
        raise ValueError("source standardization statistics are not finite")
    if not bool((feature_std > 0).all()):
        raise ValueError("source has a zero-variance relation dimension")
    standardized.sub_(feature_mean).div_(feature_std)

    # The release seeds NumPy globally and leaves PCA.random_state=None.  Keep
    # that exact state progression: optional np.random.choice happens before
    # randomized PCA consumes the same global RNG.
    np.random.seed(int(seed))
    standardized_numpy = standardized.numpy()
    sampled_indices: np.ndarray | None = None
    if count > int(pca_subsample):
        sampled_indices = np.random.choice(
            np.arange(count), int(pca_subsample), replace=False
        )
        pca_on = standardized_numpy[sampled_indices]
    else:
        pca_on = standardized_numpy
    shares_source = bool(np.shares_memory(pca_on, standardized_numpy))
    # copy=False is a memory-bounded numerical equivalent of the release's
    # default copy=True.  sklearn centers pca_on in place; the branch below
    # accounts for whether pca_on aliases the full standardized matrix.
    pca = PCA(
        n_components=components,
        copy=False,
        svd_solver="auto",
        random_state=None,
    )
    pca.fit(pca_on)
    pca_mean = torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)).contiguous()
    pca_components = torch.from_numpy(
        np.asarray(pca.components_, dtype=np.float32)
    ).contiguous()
    pca_singular_values = torch.from_numpy(
        np.asarray(pca.singular_values_, dtype=np.float32)
    ).contiguous()
    if not bool(torch.isfinite(pca_components).all()) or not bool(
        (pca_singular_values > 0).all()
    ):
        raise ValueError("PCA fit produced invalid transform parameters")

    relation = torch.empty((count, components), dtype=torch.float32)
    for start in range(0, count, int(projection_chunk_size)):
        stop = min(start + int(projection_chunk_size), count)
        block = standardized[start:stop]
        # PCA(copy=False) already centered the aliased full matrix.  A sampled
        # advanced-index fit does not alias, so center full rows explicitly.
        if not shares_source:
            block = block - pca_mean
        relation[start:stop] = (block @ pca_components.T) * pca_singular_values
    if not bool(torch.isfinite(relation).all()):
        raise ValueError("PCA relation features contain NaN or infinity")

    tensors = {
        "relation_features": relation.contiguous(),
        "feature_mean": feature_mean.float().contiguous(),
        "feature_std": feature_std.float().contiguous(),
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "pca_singular_values": pca_singular_values,
    }
    diagnostics = {
        "sklearn_version": str(sklearn_version),
        "pca_solver_resolved": str(getattr(pca, "_fit_svd_solver", "unknown")),
        "pca_fit_rows": int(len(pca_on)),
        "pca_subsample_applied": sampled_indices is not None,
        "pca_seed": int(seed),
        "pca_copy": False,
        "pca_copy_false_numeric_role": "memory_only_no_method_change",
        "standardization_std_correction": 1,
        "singular_value_weighting": True,
    }
    return tensors, diagnostics


def _atomic_torch_save(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    registration = Path(args.experiment_registration).resolve()
    registration_sha256 = sha256_file(registration)
    if registration_sha256 != REGISTRATION_SHA256:
        raise ValueError("PCA40 experiment registration differs")
    capability_path = Path(args.capability_cache).resolve()
    sidecar_path = Path(str(capability_path) + ".json")
    sidecar_sha256 = sha256_file(sidecar_path)
    bank = load_canonical_capability_bank(capability_path)
    signature = bank.signatures.get("appearance")
    if (
        signature is None
        or signature.adaptor_name != SOURCE_ADAPTOR
        or signature.adaptor_output_dim != SOURCE_DIMENSION
        or signature.token_type != "primitive"
        or signature.normalization != "l2"
    ):
        raise ValueError("capability appearance signature is not registered C-RADIO DINO")
    source = bank.valid_feature_banks()["appearance"]
    if source.shape != (int(bank.valid.sum()), SOURCE_DIMENSION):
        raise ValueError("capability valid appearance rows differ")
    source_feature_sha256 = chunked_tensor_sha256(
        source, row_chunk_size=int(args.hash_row_chunk_size)
    )
    source_xyz_sha256 = float_rows_sha256(bank.xyz)
    global_rows = bank.global_rows.long().contiguous()
    fit_tensors, diagnostics = fit_ludvig_inspired_pca_relation(
        source,
        n_components=PCA_COMPONENTS,
        pca_subsample=PCA_SUBSAMPLE,
        seed=PCA_SEED,
        projection_chunk_size=int(args.projection_chunk_size),
    )
    del source
    tensors = {"global_rows": global_rows, **fit_tensors}
    digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    metadata = {
        "relation_source": "official_C_RADIOv4_dino_v3_7b_primitive_rows",
        "source_capability_path": str(capability_path),
        "capability_sidecar_sha256": sidecar_sha256,
        "field_checkpoint_sha256": str(bank.metadata["field_checkpoint_sha256"]),
        "source_adaptor": SOURCE_ADAPTOR,
        "source_dimension": SOURCE_DIMENSION,
        "source_token_type": "primitive",
        "source_normalization": "l2",
        "source_storage_dtype": str(bank.appearance.dtype),
        "transform": "standardize_PCA40_singular_value_weighted",
        "standardization": "torch_float32_mean_sample_std_correction_1",
        "pca_components": PCA_COMPONENTS,
        "pca_subsample": PCA_SUBSAMPLE,
        "pca_seed": PCA_SEED,
        "eigval_weighting": "sklearn_PCA_singular_values",
        "graph_input_normalization": "per_row_l2_eps_1e-8",
        "experiment_registration_path": str(registration),
        "experiment_registration_sha256": registration_sha256,
        "query_independent": True,
        "labels_opened": False,
        "target_rgb_opened": False,
        "target_masks_opened": False,
        "target_metrics_opened": False,
        "native_ludvig_dinov2_pca40_exact": False,
        "compatibility_boundary": (
            "C_RADIO_DINOv3_primitive_PCA40_adaptation_not_native_LUDVIG_DINOv2_tokens"
        ),
        **diagnostics,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": str(args.scene_id),
        "num_global_rows": int(bank.num_gaussians),
        "source_feature_sha256": source_feature_sha256,
        "source_xyz_sha256": source_xyz_sha256,
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": canonical_json_sha256(digests),
        "metadata": metadata,
    }
    validate_query_diffusion_relation_payload(
        payload,
        expected_scene_id=str(args.scene_id),
        expected_global_rows=global_rows,
        expected_num_global_rows=bank.num_gaussians,
        expected_source_feature_sha256=source_feature_sha256,
        expected_source_xyz_sha256=source_xyz_sha256,
        expected_registration_sha256=registration_sha256,
        expected_capability_sidecar_sha256=sidecar_sha256,
        expected_field_checkpoint_sha256=str(bank.metadata["field_checkpoint_sha256"]),
    )
    output = Path(args.output).resolve()
    _atomic_torch_save(payload, output)
    output_sha256 = sha256_file(output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    validate_query_diffusion_relation_payload(
        reloaded,
        expected_scene_id=str(args.scene_id),
        expected_global_rows=global_rows,
        expected_num_global_rows=bank.num_gaussians,
        expected_source_feature_sha256=source_feature_sha256,
        expected_source_xyz_sha256=source_xyz_sha256,
        expected_registration_sha256=registration_sha256,
        expected_capability_sidecar_sha256=sidecar_sha256,
        expected_field_checkpoint_sha256=str(bank.metadata["field_checkpoint_sha256"]),
    )
    if sha256_file(output) != output_sha256:
        raise ValueError("PCA40 relation cache changed across frozen reload")
    receipt = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": str(args.scene_id),
        "output": str(output),
        "output_sha256": output_sha256,
        "tensor_bundle_sha256": payload["tensor_bundle_sha256"],
        "relation_feature_sha256": digests["relation_features"],
        "source_feature_sha256": source_feature_sha256,
        "source_xyz_sha256": source_xyz_sha256,
        "num_global_rows": int(bank.num_gaussians),
        "valid_rows": int(global_rows.numel()),
        "relation_dimension": PCA_COMPONENTS,
        "metadata": metadata,
    }
    _atomic_json_save(receipt, Path(str(output) + ".json"))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection-chunk-size", type=int, default=8192)
    parser.add_argument("--hash-row-chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
