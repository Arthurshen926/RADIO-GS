#!/usr/bin/env python3
"""Train one compact, query-independent canonical RADIO field from MPR targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from radio_gs.field import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    CanonicalGaussianField,
    FeatureSpaceSignature,
    fit_affine_basis,
    load_canonical_field_checkpoint,
    validate_observation_contract_metadata,
)
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from radio_gs.training.canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
)
from radio_gs.training.primitive_consensus import PrimitiveConsensus
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import sha256_file


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _consensus_from_cache(
    cache: dict,
    *,
    preserve_target_dtype: bool = False,
) -> PrimitiveConsensus:
    targets = torch.as_tensor(cache["features"]).cpu()
    if not preserve_target_dtype:
        targets = targets.float()
    valid = torch.as_tensor(cache["valid"]).bool().cpu()
    counts = torch.as_tensor(cache["view_counts"]).long().cpu()
    reliability = cache.get("reliability")
    if reliability is None:
        maximum = max(1, int(counts.max()) if counts.numel() else 1)
        reliability = torch.stack(
            [counts.float() / maximum, valid.float(), valid.float()], dim=-1
        )
    else:
        reliability = torch.as_tensor(reliability).float().cpu()
    return PrimitiveConsensus(
        targets=targets,
        valid=valid,
        observation_count=counts,
        reliability=reliability,
        per_view_agreement=torch.empty(0, targets.shape[0]),
    )


def _load_capability_mpr_target(
    path: str | Path,
    *,
    expected_space: str,
    raw_cache: dict,
    raw_metadata: dict,
    radio_checkpoint_sha256: str,
    expected_cache_sha256: str = "",
    expected_feature_output_bundle_sha256: str = "",
) -> tuple[PrimitiveConsensus, dict]:
    """Load an official-adaptor-before-MPR target with strict provenance."""

    cache, cache_sha256, cache_path = load_mpr_cache(
        path,
        expected_sha256=str(expected_cache_sha256) or None,
        expected_feature_space=expected_space,
        require_reliability=True,
        require_formal_safety=True,
    )
    metadata = dict(cache.get("metadata", {}))
    raw_contract = raw_metadata.get("observation_lifting_contract", {})
    raw_contract_name = (
        str(raw_contract.get("name", CANONICAL_OBSERVATION_CONTRACT_NAME))
        if isinstance(raw_contract, dict)
        else CANONICAL_OBSERVATION_CONTRACT_NAME
    )
    validate_observation_contract_metadata(
        metadata,
        require_declaration="observation_lifting_contract" in raw_metadata,
        contract_name=raw_contract_name,
    )
    if str(metadata.get("feature_space", "")) != str(expected_space):
        raise ValueError(f"expected a {expected_space} MPR cache")
    safety = {
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "capability_projection_before_mpr": True,
        "custom_adaptor_head": False,
    }
    for key, expected in safety.items():
        if metadata.get(key) is not expected:
            raise ValueError(f"{expected_space} MPR violates safety contract: {key}")
    if str(metadata.get("official_adaptor_checkpoint_sha256", "")) != str(
        radio_checkpoint_sha256
    ):
        raise ValueError(f"{expected_space} MPR uses another RADIO checkpoint")
    capability_map_source = str(metadata.get("capability_map_source", "project_raw"))
    if capability_map_source not in {"project_raw", "official_extracted"}:
        raise ValueError(
            f"{expected_space} MPR has an unsupported capability map source "
            f"{capability_map_source!r}"
        )
    if capability_map_source == "official_extracted":
        if (
            str(
                metadata.get(
                    "official_adaptor_checkpoint_provenance",
                    "",
                )
            )
            != "explicit_file_sha256"
        ):
            raise ValueError(
                f"{expected_space} direct official MPR is not bound to the "
                "extraction checkpoint SHA256"
            )
        required_native_provenance = {
            "capability_native_map_manifest",
            "capability_native_map_manifest_sha256",
            "capability_native_map_radio_checkpoint_load_contract",
            "capability_adaptor_execution",
        }
        missing_native_provenance = sorted(
            key
            for key in required_native_provenance
            if not str(metadata.get(key, ""))
        )
        if missing_native_provenance:
            raise ValueError(
                f"{expected_space} direct official MPR lacks native-map provenance: "
                f"{missing_native_provenance}"
            )
        if (
            str(metadata.get("capability_adaptor_execution", ""))
            != "official_c_radio_runtime_adaptor_output"
        ):
            raise ValueError(
                f"{expected_space} direct official MPR did not use the official "
                "C-RADIO runtime adaptor output"
            )
        if (
            metadata.get(
                "capability_native_map_radio_checkpoint_load_contract"
            )
            != "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
        ):
            raise ValueError(
                f"{expected_space} direct official MPR used an unrestricted "
                "RADIO checkpoint loader"
            )
        expected_bundle = str(expected_feature_output_bundle_sha256 or "")
        if (
            not expected_bundle
            or metadata.get("feature_output_bundle_sha256") != expected_bundle
            or metadata.get("capability_native_map_output_bundle_sha256")
            != expected_bundle
        ):
            raise ValueError(
                f"{expected_space} MPR belongs to another feature output bundle"
            )

    raw_xyz = torch.as_tensor(raw_cache["xyz"]).float().cpu()
    target_xyz = torch.as_tensor(cache.get("xyz")).float().cpu()
    if target_xyz.shape != raw_xyz.shape or _sha256_tensor_rows(
        target_xyz
    ) != _sha256_tensor_rows(raw_xyz):
        raise ValueError(f"{expected_space} MPR geometry does not align with raw MPR")
    raw_valid = torch.as_tensor(raw_cache["valid"]).bool().cpu()
    target_valid = torch.as_tensor(cache.get("valid")).bool().cpu()
    raw_counts = torch.as_tensor(raw_cache["view_counts"]).long().cpu()
    target_counts = torch.as_tensor(cache.get("view_counts")).long().cpu()
    if not torch.equal(target_valid, raw_valid) or not torch.equal(
        target_counts, raw_counts
    ):
        raise ValueError(
            f"{expected_space} MPR must use the exact raw-MPR observation support"
        )
    responsibility_sha256 = str(
        metadata.get("registration_responsibility_cache_sha256", "")
    )
    if str(raw_metadata.get("aggregation_mode", "")) == "raster_gaussian_top1":
        raw_responsibility_sha256 = str(
            raw_metadata.get("registration_responsibility_cache_sha256", "")
        )
        if (
            not raw_responsibility_sha256
            or not bool(raw_metadata.get("shared_registration_responsibility", False))
            or not bool(metadata.get("shared_registration_responsibility", False))
            or responsibility_sha256 != raw_responsibility_sha256
        ):
            raise ValueError(
                f"{expected_space} MPR must reuse the exact raw-MPR "
                "registration responsibility sidecar"
            )
    policy_keys = (
        "config",
        "checkpoint",
        "selected_frame_indices",
        "excluded_frame_ids",
        "aggregation_mode",
        "registration_weight_mode",
        "raster_view_fusion",
        "raster_topk",
        "depth_tolerance",
        "relative_depth_tolerance",
        "alpha_threshold",
        "normalize_each_view",
    )
    mismatched = [
        key for key in policy_keys if metadata.get(key) != raw_metadata.get(key)
    ]
    if mismatched:
        raise ValueError(
            f"{expected_space} MPR policy differs from raw MPR: {mismatched}"
        )
    consensus = _consensus_from_cache(cache, preserve_target_dtype=True)
    return consensus, {
        "path": str(cache_path.resolve()),
        "sha256": cache_sha256,
        "feature_space": expected_space,
        "feature_dim": int(consensus.targets.shape[1]),
        "projection_order": "official_adaptor_then_geometry_matched_mpr",
        "official_adaptor_name": metadata.get("official_adaptor_name"),
        "official_adaptor_checkpoint_sha256": metadata.get(
            "official_adaptor_checkpoint_sha256"
        ),
        "official_adaptor_checkpoint_provenance": metadata.get(
            "official_adaptor_checkpoint_provenance", ""
        ),
        "capability_map_source": capability_map_source,
        "capability_native_map_manifest": metadata.get(
            "capability_native_map_manifest", ""
        ),
        "capability_native_map_manifest_sha256": metadata.get(
            "capability_native_map_manifest_sha256", ""
        ),
        "feature_output_bundle_sha256": metadata.get(
            "feature_output_bundle_sha256", ""
        ),
        "capability_native_map_output_bundle_sha256": metadata.get(
            "capability_native_map_output_bundle_sha256", ""
        ),
        "capability_native_map_radio_checkpoint_load_contract": metadata.get(
            "capability_native_map_radio_checkpoint_load_contract", ""
        ),
        "capability_native_map_grid": metadata.get(
            "capability_native_map_grid", []
        ),
        "capability_adaptor_execution": metadata.get(
            "capability_adaptor_execution", ""
        ),
        "selected_frame_indices": metadata.get("selected_frame_indices", []),
        "registration_responsibility_cache_sha256": responsibility_sha256,
        "uses_query_or_benchmark_supervision": False,
    }


@torch.no_grad()
def _reconstruction_metrics(
    field: CanonicalGaussianField,
    consensus: PrimitiveConsensus,
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    cosines: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, rows.numel(), batch_size):
        batch = rows[start : start + batch_size]
        predicted = field.radio_features(batch.to(device)).float().cpu()
        target = consensus.targets[batch].float()
        cosines.append(F.cosine_similarity(predicted, target, dim=-1, eps=1e-8))
        rmses.append((predicted - target).square().mean(dim=-1).sqrt())
    cosine = torch.cat(cosines)
    rmse = torch.cat(rmses)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(cosine.quantile(0.05)),
        "mean_rmse": float(rmse.mean()),
    }


@torch.no_grad()
def _capability_reconstruction_metrics(
    field: CanonicalGaussianField,
    official_views: FrozenRadioViews,
    targets: dict[str, PrimitiveConsensus],
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    values: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
    device = field.local_codes.device
    for start in range(0, rows.numel(), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        radio = field.radio_features(batch.to(device)).float()
        for name, consensus in targets.items():
            valid = consensus.valid[batch]
            if not bool(valid.any()):
                continue
            projected = (
                official_views.project_dino_primitives(radio)
                if name == "dino_v3"
                else official_views.project_sam3_primitives(radio)
            )
            target = consensus.targets[batch].to(device).float()
            values[name].append(
                F.cosine_similarity(
                    projected[valid.to(device)],
                    target[valid.to(device)],
                    dim=-1,
                    eps=1e-8,
                ).cpu()
            )
    report: dict[str, float] = {}
    for name, parts in values.items():
        if not parts:
            report[f"{name}_target_mean_cosine"] = 0.0
            report[f"{name}_target_p05_cosine"] = 0.0
            continue
        cosine = torch.cat(parts)
        report[f"{name}_target_mean_cosine"] = float(cosine.mean())
        report[f"{name}_target_p05_cosine"] = float(torch.quantile(cosine, 0.05))
    return report


@torch.no_grad()
def _cross_basis_projection(local_decoder, output_decoder) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local PCA coordinates into the higher-rank output PCA coordinates."""

    scale_ratio = local_decoder.scale / output_decoder.scale
    matrix = (
        local_decoder.basis.transpose(0, 1) * scale_ratio[None]
    ) @ output_decoder.basis
    bias = (
        (local_decoder.mean - output_decoder.mean) / output_decoder.scale
    ) @ output_decoder.basis
    return matrix.transpose(0, 1).contiguous(), bias.contiguous()


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    observation_contract_mode = str(
        getattr(args, "observation_contract", "unchecked")
    )
    strict_contract_modes = {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }
    cache, mpr_cache_sha256, mpr_cache_path = load_mpr_cache(
        args.mpr_cache,
        expected_sha256=(
            str(getattr(args, "expected_mpr_cache_sha256", "")) or None
        ),
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=observation_contract_mode in strict_contract_modes,
    )
    metadata = dict(cache.get("metadata", {}))
    if observation_contract_mode != "unchecked":
        validate_observation_contract_metadata(
            metadata,
            require_declaration=observation_contract_mode in strict_contract_modes,
            contract_name=(
                observation_contract_mode
                if observation_contract_mode in strict_contract_modes
                else None
            ),
        )
    if metadata.get("benchmark_masks_opened", False) or metadata.get("text_queries_opened", False):
        raise ValueError("MPR training cache is contaminated by benchmark queries or masks")
    if str(metadata.get("feature_space", "radio")) != "radio":
        raise ValueError("canonical main field must reconstruct raw RADIO, not a query head")
    consensus = _consensus_from_cache(cache)
    radio_hash = sha256_file(args.radio_checkpoint)
    expected_radio_hash = str(
        getattr(args, "expected_radio_checkpoint_sha256", "")
    )
    if expected_radio_hash and radio_hash != expected_radio_hash:
        raise ValueError("RADIO checkpoint differs from caller authority")
    expected_feature_bundle_sha256 = str(
        getattr(args, "expected_feature_output_bundle_sha256", "")
    )
    if observation_contract_mode in strict_contract_modes and (
        not expected_feature_bundle_sha256
        or metadata.get("feature_output_bundle_sha256")
        != expected_feature_bundle_sha256
    ):
        raise ValueError("raw MPR belongs to another feature output bundle")
    capability_targets: dict[str, PrimitiveConsensus] = {}
    capability_target_provenance: dict[str, dict] = {}
    for name, path in (
        ("dino_v3", args.dino_mpr_cache),
        ("sam3", args.sam3_mpr_cache),
    ):
        if not str(path).strip():
            continue
        target, provenance = _load_capability_mpr_target(
            path,
            expected_space=name,
            raw_cache=cache,
            raw_metadata=metadata,
            radio_checkpoint_sha256=radio_hash,
            expected_cache_sha256=str(
                getattr(args, f"expected_{name}_mpr_cache_sha256", "")
            ),
            expected_feature_output_bundle_sha256=(
                expected_feature_bundle_sha256
            ),
        )
        capability_targets[name] = target
        capability_target_provenance[name] = provenance
    if capability_targets and not args.official_capability_loss:
        raise ValueError(
            "auxiliary capability MPR targets require --official-capability-loss"
        )
    primitive_positions = torch.as_tensor(cache["xyz"]).float().cpu()
    valid_rows = torch.where(consensus.valid)[0]
    signature = FeatureSpaceSignature(
        radio_version=args.radio_version,
        radio_checkpoint_sha256=radio_hash,
        raw_feature_dim=consensus.targets.shape[1],
        adaptor_name="backbone",
        token_type="primitive",
        normalization=(
            "radio_direction_unit"
            if bool(metadata.get("normalize_each_view", False))
            else "radio_raw_full"
        ),
        crop_policy="training_views_depth_alpha_checked_mpr",
        # The field stores exactly the declared MPR RADIO semantics.  Semantic alignment is a
        # separately selected, frozen capability view and is never part of the
        # field checkpoint contract.
        semantic_alignment="none",
    )
    initial_field_provenance: dict = {}
    if str(args.initial_field_checkpoint).strip():
        initial_path = Path(args.initial_field_checkpoint)
        field, initial_payload = load_canonical_field_checkpoint(
            initial_path,
            map_location="cpu",
            expected_sha256=(
                str(
                    getattr(
                        args,
                        "expected_initial_field_checkpoint_sha256",
                        "",
                    )
                )
                or None
            ),
        )
        if initial_payload.get("benchmark_masks_opened", False) or initial_payload.get(
            "text_queries_opened", False
        ):
            raise ValueError("initial field used benchmark masks or text queries")
        if field.num_gaussians != consensus.targets.shape[0]:
            raise ValueError("initial field Gaussian count differs from the MPR cache")
        if field.decoder.feature_dim != consensus.targets.shape[1]:
            raise ValueError("initial field RADIO dimension differs from the MPR cache")
        signature.assert_compatible(
            field.signature, allow_field_checkpoint_difference=True
        )
        expected_geometry = str(
            cache.get("geometry_fingerprint", {}).get("xyz_sha256", "")
        )
        actual_geometry = str(
            initial_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
        )
        if not expected_geometry or actual_geometry != expected_geometry:
            raise ValueError("initial field geometry differs from the MPR cache")
        if field.reliability.shape != consensus.reliability.shape:
            raise ValueError("initial field reliability shape differs from the MPR cache")
        # Registration support is part of the current raw target contract.
        # Updating this fixed buffer affects control and treatment identically;
        # no learned state or query signal is introduced.
        with torch.no_grad():
            field.reliability.copy_(consensus.reliability)
        field = field.to(device)
        basis_fit_report = dict(initial_payload.get("basis_fit_report", {}))
        initial_field_provenance = {
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
            "source_final_metrics": initial_payload.get("final_metrics", {}),
            "source_capability_target_mode": initial_payload.get(
                "capability_target_mode", "legacy_or_unspecified"
            ),
            "source_training_epochs": len(initial_payload.get("history", [])),
            "architecture_reused_exactly": True,
            "learned_state_reinitialized": False,
            "fixed_reliability_refreshed_from_current_raw_mpr": True,
        }
    else:
        if valid_rows.numel() < int(args.coefficient_dim):
            raise ValueError("too few valid primitive targets for the requested basis")
        decoder, fit_report = fit_affine_basis(
            consensus.targets[valid_rows],
            int(args.coefficient_dim),
            standardize=not args.no_standardize,
            max_samples=int(args.pca_samples),
            seed=int(args.seed),
            trainable_basis=not args.freeze_basis,
        )
        basis_fit_report = asdict(fit_report)
        local_dim = (
            int(args.local_dim)
            if int(args.local_dim) > 0
            else int(args.coefficient_dim)
        )
        use_fusion = bool(args.primitive_fusion)
        if (
            local_dim != int(args.coefficient_dim)
            or int(args.spatial_coarse_dim) > 0
        ) and not use_fusion:
            raise ValueError("compact local/spatial codes require --primitive-fusion")
        spatial_hash = None
        if int(args.spatial_coarse_dim) > 0:
            spatial_hash = {
                "output_dim": int(args.spatial_coarse_dim),
                "num_levels": int(args.hash_levels),
                "features_per_level": int(args.hash_features_per_level),
                "log2_hashmap_size": int(args.hash_log2_size),
                "base_resolution": int(args.hash_base_resolution),
                "max_resolution": int(args.hash_max_resolution),
                "hidden_dim": int(args.hash_hidden_dim),
            }
        field = CanonicalGaussianField(
            num_gaussians=consensus.targets.shape[0],
            decoder=decoder,
            signature=signature,
            local_dim=local_dim,
            coarse_dim=int(args.spatial_coarse_dim),
            primitive_positions=(
                primitive_positions if spatial_hash is not None else None
            ),
            spatial_hash=spatial_hash,
            reliability=consensus.reliability,
            fusion_reliability=bool(args.fusion_reliability),
            hidden_dim=int(args.hidden_dim),
            fusion_residual_blocks=int(
                getattr(args, "fusion_residual_blocks", 0)
            ),
            use_fusion=use_fusion,
        ).to(device)
        with torch.no_grad():
            if local_dim == int(args.coefficient_dim):
                encoded = decoder.encode(consensus.targets.to(device))
            else:
                local_decoder, _local_fit_report = fit_affine_basis(
                    consensus.targets[valid_rows],
                    local_dim,
                    standardize=not args.no_standardize,
                    max_samples=int(args.pca_samples),
                    seed=int(args.seed),
                    trainable_basis=False,
                )
                encoded = local_decoder.encode(consensus.targets).to(device)
                if field.fusion is None:
                    raise RuntimeError("local compression requires primitive fusion")
                weight, bias = _cross_basis_projection(local_decoder, decoder.cpu())
                decoder.to(device)
                field.fusion.initialize_base_projection(weight, bias)
            field.local_codes.copy_(encoded)

    official_views = None
    if args.official_capability_loss:
        official_views = FrozenRadioViews.from_radio_checkpoint(
            args.radio_checkpoint,
            expected_sha256=radio_hash,
        ).to(device)
    loss_config = CanonicalFieldLossConfig(
        mpr_weight=float(args.mpr_weight),
        dino_weight=float(args.dino_weight if official_views is not None else 0.0),
        sam3_weight=float(args.sam3_weight if official_views is not None else 0.0),
        relation_weight=0.0,
        coefficient_weight=float(args.coefficient_weight),
        basis_orthogonality_weight=float(args.basis_orthogonality_weight),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    order = valid_rows[torch.randperm(valid_rows.numel(), generator=generator)]
    validation_count = max(1, int(round(order.numel() * float(args.validation_fraction))))
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:]
    if training_rows.numel() == 0:
        training_rows = validation_rows
    optimizer = torch.optim.AdamW(
        [parameter for parameter in field.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        epoch_order = training_rows[
            torch.randperm(training_rows.numel(), generator=generator)
        ]
        totals: list[float] = []
        field.train()
        for start in range(0, epoch_order.numel(), int(args.batch_size)):
            rows = epoch_order[start : start + int(args.batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _stats = canonical_primitive_loss(
                field,
                consensus,
                rows,
                official_views=official_views,
                capability_targets=capability_targets,
                config=loss_config,
            )
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
        field.eval()
        validation = _reconstruction_metrics(
            field, consensus, validation_rows, int(args.eval_batch_size)
        )
        if official_views is not None and capability_targets:
            validation.update(
                _capability_reconstruction_metrics(
                    field,
                    official_views,
                    capability_targets,
                    validation_rows,
                    int(args.eval_batch_size),
                )
            )
        record = {
            "epoch": epoch + 1,
            "loss": sum(totals) / max(1, len(totals)),
            **validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if (
            epoch + 1 >= int(args.min_epochs)
            and validation["mean_cosine"] >= float(args.target_cosine)
        ):
            break

    field.eval()
    final_metrics = _reconstruction_metrics(
        field, consensus, valid_rows, int(args.eval_batch_size)
    )
    final_capability_metrics = (
        _capability_reconstruction_metrics(
            field,
            official_views,
            capability_targets,
            valid_rows,
            int(args.eval_batch_size),
        )
        if official_views is not None and capability_targets
        else {}
    )
    field.cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture = {
        "num_gaussians": field.num_gaussians,
        "feature_dim": field.decoder.feature_dim,
        "coefficient_dim": field.decoder.coefficient_dim,
        "local_dim": field.local_codes.shape[1],
        "coarse_dim": field.coarse_dim,
        "spatial_hash": (
            field.spatial_encoder.architecture()
            if field.spatial_encoder is not None
            else None
        ),
        "position_storage": "normalized_fp16" if field.coarse_dim else "none",
        "fusion_reliability": field.fusion_reliability,
        "hidden_dim": (
            int(field.fusion.network[0].out_features)
            if field.fusion is not None
            else int(args.hidden_dim)
        ),
        "fusion_residual_blocks": int(field.fusion_residual_blocks),
        "use_fusion": field.fusion is not None,
        "trainable_basis": bool(field.decoder.basis.requires_grad),
        "trainable_statistics": bool(
            field.decoder.mean.requires_grad or field.decoder.scale.requires_grad
        ),
    }
    training_config = {
        key: value
        for key, value in vars(args).items()
    }
    training_config_sha256 = hashlib.sha256(
        json.dumps(
            training_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "architecture": architecture,
        "feature_signature": field.signature.to_dict(),
        "state_dict": field.state_dict(),
        "reliability": consensus.reliability.half(),
        "geometry_fingerprint": cache.get("geometry_fingerprint", {}),
        "mpr_cache": str(mpr_cache_path),
        "mpr_cache_sha256": mpr_cache_sha256,
        "mpr_cache_metadata": metadata,
        "feature_output_bundle_sha256": expected_feature_bundle_sha256,
        "basis_fit_report": basis_fit_report,
        "initial_field_checkpoint": initial_field_provenance,
        "loss_config": asdict(loss_config),
        "capability_target_mode": (
            "official_adaptor_then_geometry_matched_mpr"
            if capability_targets
            else "adaptor_of_raw_mpr_target"
            if official_views is not None
            else "none"
        ),
        "capability_mpr_targets": capability_target_provenance,
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "history": history,
        "final_metrics": final_metrics,
        "final_capability_metrics": final_capability_metrics,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "output": str(output),
        "num_gaussians": field.num_gaussians,
        "valid_gaussians": int(valid_rows.numel()),
        "coefficient_dim": field.decoder.coefficient_dim,
        "local_dim": field.local_codes.shape[1],
        "coarse_dim": field.coarse_dim,
        "basis_fit": basis_fit_report,
        "initial_field_checkpoint": initial_field_provenance,
        "final_metrics": final_metrics,
        "final_capability_metrics": final_capability_metrics,
        "capability_target_mode": payload["capability_target_mode"],
        "capability_mpr_targets": capability_target_provenance,
        "mpr_cache_sha256": mpr_cache_sha256,
        "feature_output_bundle_sha256": expected_feature_bundle_sha256,
        "training_config_sha256": training_config_sha256,
        "feature_signature": field.signature.to_dict(),
        "xyz_sha256": _sha256_tensor_rows(torch.as_tensor(cache["xyz"])),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    temporary_report = report_path.with_suffix(
        report_path.suffix + ".tmp"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument(
        "--expected-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for the raw RADIO MPR cache.",
    )
    parser.add_argument(
        "--observation-contract",
        choices=[
            CANONICAL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
            "compatible-legacy",
            "unchecked",
        ],
        default=CANONICAL_OBSERVATION_CONTRACT_NAME,
        help="Require the shared dataset-independent MPR contract for new fields.",
    )
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument(
        "--expected-radio-checkpoint-sha256",
        default="",
        help="Caller-trusted SHA-256 for the official RADIO checkpoint.",
    )
    parser.add_argument(
        "--expected-feature-output-bundle-sha256",
        default="",
        help="Caller-trusted SHA-256 of the extracted feature output bundle.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--initial-field-checkpoint",
        default="",
        help=(
            "Continue from an exactly geometry/signature-compatible canonical "
            "field. Its architecture and learned state are reused unchanged; "
            "architecture initialization flags are ignored."
        ),
    )
    parser.add_argument(
        "--expected-initial-field-checkpoint-sha256",
        default="",
        help="Caller-trusted SHA-256 for --initial-field-checkpoint.",
    )
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coefficient-dim", type=int, default=256)
    parser.add_argument(
        "--local-dim",
        type=int,
        default=0,
        help="Per-Gaussian code dimension; 0 uses coefficient-dim.",
    )
    parser.add_argument("--spatial-coarse-dim", type=int, default=0)
    parser.add_argument("--hash-levels", type=int, default=8)
    parser.add_argument("--hash-features-per-level", type=int, default=2)
    parser.add_argument("--hash-log2-size", type=int, default=15)
    parser.add_argument("--hash-base-resolution", type=int, default=8)
    parser.add_argument("--hash-max-resolution", type=int, default=512)
    parser.add_argument("--hash-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--fusion-reliability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose fixed MPR observation reliability to primitive fusion.",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument(
        "--fusion-residual-blocks",
        type=int,
        default=0,
        help=(
            "Optional token-wise coefficient residual depth after primitive "
            "local/coarse/reliability fusion; zero preserves schema-v1 behavior."
        ),
    )
    parser.add_argument(
        "--primitive-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional local/coarse/reliability residual fusion; stage-1 main is direct local coefficients.",
    )
    parser.add_argument("--pca-samples", type=int, default=50000)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--freeze-basis", action="store_true")
    parser.add_argument("--official-capability-loss", action="store_true")
    parser.add_argument(
        "--dino-mpr-cache",
        default="",
        help=(
            "Optional query-free target built by applying the official DINOv3 "
            "spatial adaptor to each 2-D teacher view before matched MPR."
        ),
    )
    parser.add_argument(
        "--expected-dino-v3-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for --dino-mpr-cache.",
    )
    parser.add_argument(
        "--sam3-mpr-cache",
        default="",
        help=(
            "Optional query-free target built by applying the official SAM3 "
            "spatial adaptor to each 2-D teacher view before matched MPR."
        ),
    )
    parser.add_argument(
        "--expected-sam3-mpr-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for --sam3-mpr-cache.",
    )
    parser.add_argument("--mpr-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=0.20)
    parser.add_argument("--sam3-weight", type=float, default=0.20)
    parser.add_argument("--coefficient-weight", type=float, default=1e-5)
    parser.add_argument("--basis-orthogonality-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum optimization epochs before the raw-MPR early-stop rule applies.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--target-cosine", type=float, default=0.985)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.fusion_residual_blocks < 0:
        parser.error("--fusion-residual-blocks cannot be negative")
    if args.fusion_residual_blocks and not args.primitive_fusion:
        parser.error("--fusion-residual-blocks requires --primitive-fusion")
    if args.min_epochs <= 0 or args.min_epochs > args.epochs:
        parser.error("--min-epochs must lie in [1, --epochs]")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
