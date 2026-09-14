"""Target-free deployment adapter for the retained ScanNet completion arm.

The learned models are scene-agnostic, but their training runtime used oracle
identities to *simulate* source observations.  This legacy adapter replaces
that simulation with real source-only SAM token assignments.  Its frozen
categorical contract cannot represent overlapping soft evidence, so ambiguous
rows remain unknown instead of being silently hardened to an argmax winner.
No benchmark labels, target RGB, or text query enters the completion path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.completion.oracle import (
    OracleIdentityCompletionMLP,
    PartialObjectMembership,
    build_token_context,
)
from radio_gs.v4.completion.spatial_slots import TokenSpatialSupportSlots
from radio_gs.v4.training.diagnose_scannet_knn_mass_spatial_slots import (
    _load_weights_only,
)
from radio_gs.v4.training.diagnose_scannet_learned_mass_calibration import (
    build_mass_features,
)
from radio_gs.v4.training.diagnose_scannet_oracle_mass_projection import (
    oracle_mass_project,
)
from radio_gs.v4.training.diagnose_scannet_ridge_mass_spatial_slots import (
    predict_log_correction,
)
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    _clamp_contract,
    _frozen_unary_probabilities,
    _load_frozen_unary_model,
    _posterior_to_membership,
)
from radio_gs.v4.training.train_scannet_completion_oracle import (
    CHECKPOINT_SCHEMA as COMPLETION_ORACLE_CHECKPOINT_SCHEMA,
    REPORT_SCHEMA as COMPLETION_ORACLE_REPORT_SCHEMA,
    _predict as _predict_completion_oracle,
)
from radio_gs.v4.training.train_scannet_spatial_slots import (
    _full_posterior,
    build_observed_pca_geometry,
)


def _camera_record(camera: Any) -> dict[str, Any]:
    return {
        "key": str(camera.key),
        "intrinsic": torch.as_tensor(camera.intrinsic).cpu(),
        "camera_to_world": torch.as_tensor(camera.camera_to_world).cpu(),
        "height": int(camera.height),
        "width": int(camera.width),
    }


def build_real_token_runtime(
    *,
    carrier: Any,
    local_features: torch.Tensor,
    source_visible: torch.Tensor,
    observed_membership: torch.Tensor,
    observation_cameras: list[Any],
    view_token_ids: list[torch.Tensor],
    observed_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert real source-token evidence to the frozen completion contract."""

    membership = torch.as_tensor(observed_membership, dtype=torch.float32).cpu()
    features = torch.as_tensor(local_features, dtype=torch.float32).cpu()
    visible = torch.as_tensor(source_visible, dtype=torch.bool).cpu()
    if membership.ndim != 2 or membership.shape[1] == 0:
        raise ValueError("real observed membership must have shape [E,K], K>0")
    if features.shape != (membership.shape[0], 71):
        raise ValueError("LERF completion requires the sealed F71 local layout")
    if visible.shape != (membership.shape[0],):
        raise ValueError("source visibility must align with carrier elements")
    if len(observation_cameras) != len(view_token_ids) or not observation_cameras:
        raise ValueError("source cameras and proposal-token assignments must align")
    if not 0 <= observed_threshold < 1:
        raise ValueError("observed threshold must lie in [0,1)")
    if not torch.isfinite(membership).all() or bool((membership < 0).any()):
        raise ValueError("real observed membership must be finite and non-negative")

    above_threshold = membership > float(observed_threshold)
    support_count = above_threshold.sum(-1)
    ambiguous_observed = support_count > 1
    membership_observed = support_count == 1
    if bool((membership_observed & ~visible).any()):
        raise ValueError("categorical mask evidence appears outside source visibility")
    # The categorical ScanNet checkpoint has no legal soft/top-2 input.  Use
    # argmax only where exactly one token is supported; conflicting rows stay
    # unknown for every token and are left to the future fragment-set model.
    best_token = membership.argmax(-1)
    if not bool(membership_observed.any()):
        raise RuntimeError("real token adapter received no observed carrier support")
    observed_label_full = torch.full((membership.shape[0],), -1, dtype=torch.long)
    observed_label_full[membership_observed] = best_token[membership_observed]
    token_mass_full = torch.bincount(
        observed_label_full[membership_observed], minlength=membership.shape[1]
    )
    active_token_ids = torch.where(token_mass_full > 0)[0]
    if active_token_ids.numel() == 0:
        raise RuntimeError("real association produced no categorical token seeds")
    full_to_active = torch.full((membership.shape[1],), -1, dtype=torch.long)
    full_to_active[active_token_ids] = torch.arange(active_token_ids.numel())
    observed_label = torch.full_like(observed_label_full, -1)
    observed_label[membership_observed] = full_to_active[
        observed_label_full[membership_observed]
    ]
    partial = PartialObjectMembership.from_oracle_visibility(
        observed_label,
        membership_observed,
        token_count=int(active_token_ids.numel()),
        eligible_elements=torch.ones(membership.shape[0], dtype=torch.bool),
    )
    token_mass = partial.positive.sum(0)
    if bool((token_mass <= 0).any()):
        raise AssertionError("active-token compaction retained an empty token")

    frame_keys = [str(camera.key) for camera in observation_cameras]
    records = []
    seen_token = torch.zeros(active_token_ids.numel(), dtype=torch.bool)
    for camera, token_ids in zip(observation_cameras, view_token_ids):
        assigned = torch.as_tensor(token_ids, dtype=torch.long).cpu()
        if assigned.ndim != 1:
            raise ValueError("per-view token assignments must be vectors")
        for token_id, original_token_id in enumerate(active_token_ids.tolist()):
            kept = bool((assigned == original_token_id).any())
            seen_token[token_id] |= kept
            records.append(
                {
                    "frame_id": str(camera.key),
                    "object_id": original_token_id,
                    "kept": kept,
                }
            )
    if not bool(seen_token.all()):
        raise RuntimeError("a real token has no source proposal assignment receipt")

    edge_index = carrier.neighbors().edge_index
    context = build_token_context(
        carrier.centres,
        features,
        partial,
        edge_index,
        minimum_scale=float(carrier.voxel_size),
    )
    payload = {
        # Preserve the original association ids as object ids while using a
        # compact token axis internally.  The caller expands completion back
        # to the full semantic-token axis after inference.
        "object_ids": active_token_ids.tolist(),
        "observation_cameras": [_camera_record(camera) for camera in observation_cameras],
        "mask_dropout_receipt": {"records": records},
    }
    runtime = {
        "payload": payload,
        "centres": carrier.centres.float().cpu(),
        "local_features": features,
        # This is an observed-source label simulation only.  It is never a
        # complete membership target and unknown rows remain -1.
        "labels": observed_label,
        "partial": partial,
        "carrier": carrier,
        "context": context,
        "minimum_scale": float(carrier.voxel_size),
        "source_visible": visible,
        "edge_index": edge_index,
        "active_token_ids": active_token_ids,
        "input_token_count": int(membership.shape[1]),
    }
    inactive_token_ids = torch.where(token_mass_full == 0)[0]
    audit = {
        "input_token_count": int(membership.shape[1]),
        "active_completion_token_count": int(active_token_ids.numel()),
        "inactive_unseeded_token_count": int(inactive_token_ids.numel()),
        "inactive_unseeded_token_ids": inactive_token_ids.tolist(),
        "categorical_observed_element_count": int(membership_observed.sum()),
        "categorical_observed_element_fraction": float(membership_observed.float().mean()),
        "ambiguous_observed_element_count": int(ambiguous_observed.sum()),
        "ambiguous_observed_element_fraction": float(ambiguous_observed.float().mean()),
        "ambiguous_observed_policy": "retain_unknown_for_legacy_categorical_completion",
        # Kept as a deprecated compatibility field for old report readers.
        "overlap_discarded_element_count": int(ambiguous_observed.sum()),
        "minimum_observed_token_mass": int(token_mass.min()),
        "maximum_observed_token_mass": int(token_mass.max()),
        "source_view_count": len(frame_keys),
        "complete_target_labels_read": False,
        "heldout_rgb_read": False,
        "query_read": False,
    }
    return runtime, audit


@torch.no_grad()
def apply_noise_matched_completion_candidate(
    runtime: dict[str, Any],
    *,
    report_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    element_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the scene-disjoint coherent-fragment completion checkpoint.

    The checkpoint was trained with oracle identities only to simulate noisy
    source observations.  At deployment the integer identities are replaced
    by the source-only object-hypothesis axis in ``runtime``.
    """

    if element_batch_size <= 0:
        raise ValueError("completion inference element batch size must be positive")
    report_file = Path(report_path).resolve(strict=True)
    checkpoint_file = Path(checkpoint_path).resolve(strict=True)
    report = json.loads(report_file.read_text())
    checkpoint_sha256 = sha256_file(checkpoint_file)
    if report.get("schema") != COMPLETION_ORACLE_REPORT_SCHEMA:
        raise ValueError("completion report schema differs")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("completion report and checkpoint are not bound")
    checkpoint = _load_weights_only(checkpoint_file)
    if checkpoint.get("schema") != COMPLETION_ORACLE_CHECKPOINT_SCHEMA:
        raise ValueError("completion checkpoint schema differs")
    if (
        checkpoint.get("training_observation_noise_mode")
        != "coherent_fragment_set"
    ):
        raise ValueError("LERF deployment requires coherent-fragment matched training")
    configuration = checkpoint.get("model_configuration", {})
    scoring_mode = str(configuration.get("scoring_mode"))
    if (
        configuration.get("local_feature_mode") != "rgb_radio_geometry"
        or scoring_mode not in {"mlp", "independent_bernoulli_mlp"}
        or configuration.get("radio_alignment_control") != "aligned"
    ):
        raise ValueError("LERF deployment requires an aligned F71 shared-MLP arm")
    local_features = torch.as_tensor(runtime.get("local_features"))
    if local_features.ndim != 2 or local_features.shape[1] != 71:
        raise ValueError("LERF deployment runtime must provide sealed F71 features")
    model = OracleIdentityCompletionMLP(
        int(configuration["input_dimension"]),
        hidden_dimension=int(configuration["hidden_dimension"]),
        dropout=float(configuration["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        configuration["model_parameter_count"]
    ):
        raise ValueError("completion checkpoint parameter count differs")
    training_configuration = report.get("training_configuration", {})
    membership, null = _predict_completion_oracle(
        model,
        runtime,
        device=device,
        element_batch_size=element_batch_size,
        temperature=float(training_configuration["temperature"]),
        completion_confidence_cap=float(
            training_configuration["completion_confidence_cap"]
        ),
        scoring_mode=scoring_mode,
    )
    return membership, null, {
        "method": "scene_disjoint_coherent_fragment_noise_completion_mlp",
        "report": str(report_file),
        "report_sha256": sha256_file(report_file),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha256,
        "training_observation_noise_mode": checkpoint[
            "training_observation_noise_mode"
        ],
        "validation_observation_noise_mode": checkpoint[
            "validation_observation_noise_mode"
        ],
        "scoring_mode": scoring_mode,
        "training_scene_count": len(checkpoint["training_scene_ids"]),
        "validation_scene_count": len(checkpoint["validation_scene_ids"]),
        "target_membership_read": False,
        "heldout_rgb_read": False,
        "query_read": False,
    }


@torch.no_grad()
def apply_scannet_spatial_mass_candidate(
    runtime: dict[str, Any],
    *,
    base_report_path: str | Path,
    base_checkpoint_path: str | Path,
    slot_checkpoint_path: str | Path,
    mass_report_path: str | Path,
    device: torch.device,
    unary_element_batch_size: int,
    inference_element_batch_size: int,
    projection_iteration_count: int = 256,
    projection_damping: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the frozen unary, bias-free slots, and selected ridge mass head."""

    base_report = json.loads(Path(base_report_path).resolve(strict=True).read_text())
    mass_report = json.loads(Path(mass_report_path).resolve(strict=True).read_text())
    if mass_report.get("feature_mode") != "source_view_coverage_f71":
        raise ValueError("LERF adapter requires the selected source-view/F71 mass report")
    if mass_report.get("validation_used_for_selection") is not False:
        raise ValueError("mass report must be selected without validation access")
    base_checkpoint = _load_weights_only(Path(base_checkpoint_path).resolve(strict=True))
    slot_checkpoint = _load_weights_only(Path(slot_checkpoint_path).resolve(strict=True))
    if slot_checkpoint.get("mode") != "spatial_only":
        raise ValueError("LERF adapter requires the frozen bias-free spatial checkpoint")

    frozen_model = _load_frozen_unary_model(base_checkpoint, device=device)
    temperature = float(base_report["training_configuration"]["temperature"])
    confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )
    unary = _frozen_unary_probabilities(
        frozen_model,
        runtime,
        device=device,
        element_batch_size=unary_element_batch_size,
        temperature=temperature,
    )

    slot_configuration = slot_checkpoint["model_configuration"]
    slot_model = TokenSpatialSupportSlots(
        input_dimension=int(slot_configuration["input_dimension"]),
        hidden_dimension=int(slot_configuration["hidden_dimension"]),
        dropout=float(slot_configuration["dropout"]),
        use_token_bias=bool(slot_configuration["use_token_bias"]),
    ).to(device)
    slot_model.load_state_dict(slot_checkpoint["model_state_dict"], strict=True)
    slot_model.eval().requires_grad_(False)
    slot_source = build_mass_features(runtime, unary, feature_mode="summary_f71")
    slot_record = {
        "runtime": runtime,
        "unary": unary,
        "token_features": slot_source["features"],
        "pca_geometry": build_observed_pca_geometry(runtime),
    }
    slot_posterior, slot_audit = _full_posterior(
        slot_model,
        slot_record,
        device=device,
        element_batch_size=inference_element_batch_size,
    )

    mass_source = build_mass_features(
        runtime, unary, feature_mode="source_view_coverage_f71"
    )
    feature_mean = torch.tensor(mass_report["feature_mean"], dtype=torch.float32)
    feature_scale = torch.tensor(mass_report["feature_scale"], dtype=torch.float32)
    coefficient = torch.tensor(mass_report["coefficient"], dtype=torch.float32)
    if mass_source["features"].shape[1] != feature_mean.numel():
        raise RuntimeError("LERF mass features differ from the selected ridge report")
    correction = predict_log_correction(
        mass_source["features"],
        coefficient,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    blend = float(mass_report["selection"]["selected_blend"])
    frozen_mass = mass_source["frozen_mass"].clamp_min(mass_source["observed_mass"])
    predicted_mass = (
        frozen_mass * torch.exp(blend * correction)
    ).clamp_min(mass_source["observed_mass"])
    clamp_mask, clamp_probabilities = _clamp_contract(runtime)
    combined, projection_audit = oracle_mass_project(
        slot_posterior.to(device),
        clamp_mask.to(device),
        clamp_probabilities.to(device),
        predicted_mass.to(device),
        iteration_count=projection_iteration_count,
        damping=projection_damping,
    )
    membership, null = _posterior_to_membership(
        combined.cpu(), runtime, completion_confidence_cap=confidence_cap
    )
    audit = {
        "method": "frozen_unary_bias_free_spatial_slots_source_view_f71_ridge_mass",
        "token_count": int(membership.shape[1]),
        "unknown_element_count": int(runtime["partial"].unknown.any(-1).sum()),
        "predicted_mass_minimum": float(predicted_mass.min()),
        "predicted_mass_mean": float(predicted_mass.mean()),
        "predicted_mass_maximum": float(predicted_mass.max()),
        "observed_mass_mean": float(mass_source["observed_mass"].mean()),
        "frozen_mass_mean": float(frozen_mass.mean()),
        "completed_nonzero_element_fraction": float((membership.max(-1).values > 0).float().mean()),
        "mean_null_probability": float(null.mean()),
        "slot": slot_audit,
        "mass_projection": projection_audit,
        "mass_feature_mode": mass_report["feature_mode"],
        "mass_selected_ridge": float(mass_report["selection"]["selected_ridge"]),
        "mass_selected_blend": blend,
        "target_membership_read": False,
        "heldout_rgb_read": False,
        "query_read": False,
    }
    return membership.cpu(), audit
