#!/usr/bin/env python3
"""One-scene, instance-disjoint source-to-held-out-target unary pilot.

This is intentionally a development pilot, not a benchmark evaluator.  It
uses official projected ScanNet instance IDs from one clean nonbenchmark scene
to construct source prompts and target-view supervision.  Target RGB is never
read.  The target masks are used only by the loss and metrics after source-only
primitive features have been built.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.querying.registered_evidence_to_unary import (
    RegisteredEvidenceFeatures,
    RegisteredEvidenceToUnaryV1,
    build_registered_evidence_features,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_payload,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
    write_torch_noclobber,
)


SCENE_ID = "scene0001_00"
SEED = 260806
MIN_PIXELS = 32
MIN_VIEWS = 4


@dataclass(frozen=True)
class SparseView:
    frame_id: int
    gaussian_ids: torch.Tensor
    pixel_ids: torch.Tensor
    weights: torch.Tensor
    pixel_mass: torch.Tensor
    instance_image: torch.Tensor


@dataclass(frozen=True)
class PromptExample:
    instance_id: int
    split: str
    mode: str
    source_frame: int
    target_frames: tuple[int, ...]
    features: RegisteredEvidenceFeatures


def _hash_text(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def instance_split(instance_id: int) -> str:
    digest = _hash_text("registered_evidence_to_unary_v1", SCENE_ID, instance_id)
    return "validation" if int(digest[:16], 16) % 5 == 0 else "train"


def select_source_frame(instance_id: int, frames: Iterable[int]) -> int:
    values = list(frames)
    if not values:
        raise ValueError("source frame candidates cannot be empty")
    return min(values, key=lambda frame: _hash_text("source", SCENE_ID, instance_id, frame))


def _load_sparse_view(
    *,
    view_path: Path,
    frame_id: int,
    label_archive: zipfile.ZipFile,
    expected_view_sha256: str,
    height: int,
    width: int,
    num_gaussians: int,
) -> SparseView:
    payload, _, _ = load_torch_payload(
        view_path,
        expected_sha256=expected_view_sha256,
        map_location="cpu",
        label=f"exact responsibility view {frame_id}",
    )
    if (
        payload.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_view.v1"
        or int(payload.get("frame_index", -1)) != int(frame_id)
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or int(payload.get("num_pixels", -1)) != int(height * width)
    ):
        raise ValueError(f"exact responsibility view differs: {view_path}")
    gids = torch.as_tensor(payload["gaussian_ids"]).long().contiguous()
    pids = torch.as_tensor(payload["pixel_ids"]).long().contiguous()
    weights = torch.as_tensor(payload["base_weights"]).float().contiguous()
    if gids.shape != pids.shape or gids.shape != weights.shape:
        raise ValueError("exact responsibility sparse columns differ")
    pixel_mass = torch.zeros(height * width, dtype=torch.float32)
    pixel_mass.index_add_(0, pids, weights)
    label_name = f"instance-filt/{frame_id}.png"
    try:
        label_bytes = label_archive.read(label_name)
    except KeyError as error:
        raise FileNotFoundError(
            f"missing official instance projection in sealed zip: {label_name}"
        ) from error
    image = Image.open(io.BytesIO(label_bytes)).resize(
        (width, height), Image.Resampling.NEAREST
    )
    instance_image = torch.from_numpy(np.asarray(image, dtype=np.int64).copy()).reshape(-1)
    return SparseView(frame_id, gids, pids, weights, pixel_mass, instance_image)


def _adjoint_mass(view: SparseView, pixel_mask: torch.Tensor, rows: int) -> torch.Tensor:
    mask = torch.as_tensor(pixel_mask).reshape(-1).float()
    if mask.shape != view.pixel_mass.shape:
        raise ValueError("prompt mask differs from exact responsibility grid")
    result = torch.zeros(rows, dtype=torch.float32)
    result.index_add_(0, view.gaussian_ids, view.weights * mask[view.pixel_ids])
    return result


def _deterministic_subset(
    mask: torch.Tensor,
    *,
    count: int,
    namespace: tuple[object, ...],
) -> torch.Tensor:
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1).tolist()
    indices.sort(key=lambda index: _hash_text(*namespace, index))
    chosen = indices[: min(int(count), len(indices))]
    result = torch.zeros_like(mask.reshape(-1), dtype=torch.bool)
    if chosen:
        result[torch.tensor(chosen, dtype=torch.long)] = True
    return result


def prompt_masks(
    *,
    view: SparseView,
    instance_id: int,
    mode: str,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_full = view.instance_image == int(instance_id)
    supported = view.pixel_mass > 0
    if mode == "full_mask":
        return positive_full & supported, (~positive_full) & supported
    if mode != "scribble":
        raise ValueError(f"unknown prompt mode: {mode}")
    positive = _deterministic_subset(
        positive_full & supported,
        count=16,
        namespace=("positive_scribble", SCENE_ID, instance_id, view.frame_id),
    )
    image = positive_full.reshape(1, 1, height, width).float()
    dilated = F.max_pool2d(image, kernel_size=11, stride=1, padding=5).bool().reshape(-1)
    ring = dilated & ~positive_full & supported
    negative = _deterministic_subset(
        ring,
        count=32,
        namespace=("negative_scribble", SCENE_ID, instance_id, view.frame_id),
    )
    return positive, negative


def _prototype_margin(
    capability: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
) -> torch.Tensor:
    rows = int(capability.shape[0])
    if positive_weight.shape != (rows,) or negative_weight.shape != (rows,):
        raise ValueError("prototype weights differ from capability rows")
    if float(positive_weight.sum()) <= 0 or float(negative_weight.sum()) <= 0:
        raise ValueError("source prompt lacks positive or negative capability mass")
    positive = F.normalize(positive_weight @ capability, dim=0)
    negative = F.normalize(negative_weight @ capability, dim=0)
    return capability @ (positive - negative)


def _copy_features_to(
    value: RegisteredEvidenceFeatures, device: torch.device
) -> RegisteredEvidenceFeatures:
    return RegisteredEvidenceFeatures(
        values=value.values.to(device),
        analytic_probability=value.analytic_probability.to(device),
        registered_probability=value.registered_probability.to(device),
        labeled_coverage=value.labeled_coverage.to(device),
        capability_valid=value.capability_valid.to(device),
    )


def build_examples(
    *,
    views: dict[int, SparseView],
    eligible: dict[int, list[int]],
    capability_bank: dict,
    factorized_state: dict,
    device: torch.device,
    height: int,
    width: int,
) -> list[PromptExample]:
    valid = torch.as_tensor(capability_bank["valid"]).bool()
    global_rows = torch.as_tensor(factorized_state["global_rows"]).long()
    if not torch.equal(torch.nonzero(valid, as_tuple=False).reshape(-1), global_rows):
        raise ValueError("capability and factorized-state compact rows differ")
    rows = int(valid.numel())
    dino = F.normalize(capability_bank["appearance_dino_v3"].to(device).float(), dim=1)
    sam = F.normalize(capability_bank["boundary_sam3"].to(device).float(), dim=1)
    compact_rows = global_rows.to(device)
    valid_device = valid.to(device)

    def full_state(name: str, *, boolean: bool = False) -> torch.Tensor:
        compact = torch.as_tensor(factorized_state[name]).to(device)
        result = torch.zeros(rows, dtype=torch.bool if boolean else torch.float32, device=device)
        result[compact_rows] = compact.bool() if boolean else compact.float()
        return result

    dispersion = full_state("directional_dispersion")
    amplitude_std = full_state("log_amplitude_std")
    evidence = full_state("observation_evidence")
    purity = full_state("visibility_purity_value")
    purity_known = full_state("visibility_purity_known", boolean=True)
    examples: list[PromptExample] = []
    for instance_id, frame_ids in sorted(eligible.items()):
        source_frame = select_source_frame(instance_id, frame_ids)
        target_frames = tuple(frame for frame in frame_ids if frame != source_frame)
        source = views[source_frame]
        for mode in ("full_mask", "scribble"):
            positive_mask, negative_mask = prompt_masks(
                view=source,
                instance_id=instance_id,
                mode=mode,
                height=height,
                width=width,
            )
            positive_mass = _adjoint_mass(source, positive_mask, rows).to(device)
            negative_mass = _adjoint_mass(source, negative_mask, rows).to(device)
            visible_mass = _adjoint_mass(
                source, positive_mask | negative_mask if mode == "full_mask" else source.pixel_mass > 0, rows
            ).to(device)
            positive_compact = positive_mass[compact_rows]
            negative_compact = negative_mass[compact_rows]
            dino_margin = torch.zeros(rows, device=device)
            sam_margin = torch.zeros(rows, device=device)
            dino_margin[compact_rows] = _prototype_margin(
                dino, positive_compact, negative_compact
            )
            sam_margin[compact_rows] = _prototype_margin(
                sam, positive_compact, negative_compact
            )
            features = build_registered_evidence_features(
                foreground_mass=positive_mass,
                background_mass=negative_mass,
                visible_mass=visible_mass,
                dino_margin=dino_margin,
                sam_margin=sam_margin,
                directional_dispersion=dispersion,
                log_amplitude_std=amplitude_std,
                observation_evidence=evidence,
                visibility_purity_value=purity,
                visibility_purity_known=purity_known,
                capability_valid=valid_device,
                source_view_support=(visible_mass > 0).float(),
            )
            examples.append(
                PromptExample(
                    instance_id=instance_id,
                    split=instance_split(instance_id),
                    mode=mode,
                    source_frame=source_frame,
                    target_frames=target_frames,
                    features=features,
                )
            )
    del dino, sam
    return examples


def _subset_features(
    features: RegisteredEvidenceFeatures, rows: torch.Tensor
) -> RegisteredEvidenceFeatures:
    return RegisteredEvidenceFeatures(
        values=features.values[rows],
        analytic_probability=features.analytic_probability[rows],
        registered_probability=features.registered_probability[rows],
        labeled_coverage=features.labeled_coverage[rows],
        capability_valid=features.capability_valid[rows],
    )


def _render_unique(
    primitive_values: torch.Tensor,
    view: SparseView,
    unique_rows: torch.Tensor,
    inverse: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pids = view.pixel_ids.to(device)
    weights = view.weights.to(device)
    numerator = torch.zeros_like(view.pixel_mass, device=device)
    numerator.index_add_(0, pids, weights * primitive_values[inverse])
    mass = view.pixel_mass.to(device)
    supported = mass > 0
    probability = torch.zeros_like(mass)
    probability[supported] = numerator[supported] / mass[supported]
    return probability, supported


def _view_rows(view: SparseView, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.unique(view.gaussian_ids.to(device), sorted=True, return_inverse=True)


def _balanced_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("target view must contain positive and negative supported pixels")
    losses = F.binary_cross_entropy(
        probability.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
    )
    return 0.5 * losses[positive].mean() + 0.5 * losses[negative].mean()


def _binary_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    score = scores.detach().double().cpu().reshape(-1)
    label = labels.detach().bool().cpu().reshape(-1)
    positives = int(label.sum())
    negatives = int((~label).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("binary metrics require both classes")
    order = torch.argsort(score, descending=True, stable=True)
    sorted_label = label[order].double()
    tp = sorted_label.cumsum(0)
    fp = (1.0 - sorted_label).cumsum(0)
    precision_curve = tp / (tp + fp)
    recall_curve = tp / positives
    ap = float((precision_curve * sorted_label).sum() / positives)
    tpr = torch.cat((torch.zeros(1), recall_curve, torch.ones(1)))
    fpr = torch.cat((torch.zeros(1), fp / negatives, torch.ones(1)))
    auroc = float(torch.trapz(tpr, fpr))
    union = positives + torch.arange(1, score.numel() + 1, dtype=torch.float64) - tp
    oracle_iou = float((tp / union.clamp_min(1)).max())
    prediction = score >= 0.5
    fixed_tp = int((prediction & label).sum())
    fixed_fp = int((prediction & ~label).sum())
    fixed_fn = int((~prediction & label).sum())
    iou = fixed_tp / max(1, fixed_tp + fixed_fp + fixed_fn)
    precision = fixed_tp / max(1, fixed_tp + fixed_fp)
    recall = fixed_tp / max(1, fixed_tp + fixed_fn)
    area_ratio = int(prediction.sum()) / positives
    bce = float(
        F.binary_cross_entropy(score.float().clamp(1e-6, 1 - 1e-6), label.float())
    )
    return {
        "average_precision": ap,
        "auroc": auroc,
        "iou_at_0_5": iou,
        "oracle_iou": oracle_iou,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "area_ratio": area_ratio,
        "bce": bce,
        "pixels": int(score.numel()),
        "positive_pixels": positives,
    }


@torch.no_grad()
def evaluate(
    *,
    head: RegisteredEvidenceToUnaryV1,
    examples: list[PromptExample],
    views: dict[int, SparseView],
    device: torch.device,
) -> dict:
    head.eval()
    records: list[dict] = []
    for example in examples:
        candidate_scores: list[torch.Tensor] = []
        analytic_scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for frame in example.target_frames:
            view = views[frame]
            unique, inverse = _view_rows(view, device)
            subset = _subset_features(example.features, unique)
            output = head(subset)
            prediction, supported = _render_unique(
                output.foreground_probability, view, unique, inverse, device
            )
            analytic, _ = _render_unique(
                subset.analytic_probability, view, unique, inverse, device
            )
            target = (view.instance_image.to(device) == example.instance_id)
            keep = supported
            candidate_scores.append(prediction[keep].cpu())
            analytic_scores.append(analytic[keep].cpu())
            labels.append(target[keep].cpu())
        label = torch.cat(labels)
        candidate = _binary_metrics(torch.cat(candidate_scores), label)
        analytic = _binary_metrics(torch.cat(analytic_scores), label)
        records.append(
            {
                "instance_id": example.instance_id,
                "mode": example.mode,
                "source_frame": example.source_frame,
                "target_view_count": len(example.target_frames),
                "candidate": candidate,
                "analytic": analytic,
                "delta": {
                    key: candidate[key] - analytic[key]
                    for key in (
                        "average_precision",
                        "auroc",
                        "iou_at_0_5",
                        "oracle_iou",
                        "precision_at_0_5",
                        "recall_at_0_5",
                        "area_ratio",
                        "bce",
                    )
                },
            }
        )

    def macro(path: str, mode: str | None = None) -> dict[str, float]:
        selected = [row for row in records if mode is None or row["mode"] == mode]
        keys = selected[0][path]
        return {
            key: float(sum(float(row[path][key]) for row in selected) / len(selected))
            for key in keys
            if key not in {"pixels", "positive_pixels"}
        }

    return {
        "records": records,
        "macro": {
            "all": {"candidate": macro("candidate"), "analytic": macro("analytic")},
            "full_mask": {
                "candidate": macro("candidate", "full_mask"),
                "analytic": macro("analytic", "full_mask"),
            },
            "scribble": {
                "candidate": macro("candidate", "scribble"),
                "analytic": macro("analytic", "scribble"),
            },
        },
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _validate_execution_authority(args: argparse.Namespace) -> tuple[dict, bytes]:
    """Bind every formal input before labels or tensors are deserialized."""

    authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="prompt-unary execution authority",
    )
    if (
        authority.get("schema")
        != "radio_gs.registered_evidence_to_unary.clean_scannet_execution_authority.v1"
        or authority.get("schema_version") != 1
        or authority.get("scene_id") != SCENE_ID
        or authority.get("formal_restart_from_epoch") != 0
        or authority.get("method_configuration_frozen_unchanged_after_abort") is not True
    ):
        raise ValueError("prompt-unary execution authority contract differs")
    records = authority.get("inputs")
    if not isinstance(records, dict):
        raise ValueError("prompt-unary execution input records differ")
    bindings = {
        "preregistration": args.preregistration,
        "responsibility_manifest": args.responsibility_manifest,
        "label_zip": args.label_zip,
        "capability_bank": args.capability_bank,
        "factorized_state": args.factorized_state,
        "implementation": Path(__file__).resolve(),
    }
    expected_args = {
        "preregistration": args.expected_preregistration_sha256,
        "responsibility_manifest": args.expected_responsibility_manifest_sha256,
        "label_zip": args.expected_label_zip_sha256,
        "capability_bank": args.expected_capability_bank_sha256,
        "factorized_state": args.expected_factorized_state_sha256,
    }
    for label, path in bindings.items():
        record = records.get(label)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"execution authority {label} record differs")
        resolved = Path(path).expanduser().resolve()
        if Path(str(record["path"])).expanduser().resolve() != resolved:
            raise ValueError(f"execution authority {label} path differs")
        expected = str(record["sha256"])
        if label in expected_args and str(expected_args[label]) != expected:
            raise ValueError(f"CLI and execution authority {label} SHA differ")
        if sha256_file(resolved) != expected:
            raise ValueError(f"execution authority {label} SHA differs")
    if (
        authority.get("source_access", {}).get(
            "clean_source_training_instance_labels_opened"
        )
        is not True
        or authority.get("source_access", {}).get(
            "clean_target_view_labels_enter_model_inputs"
        )
        is not False
        or any(
            authority.get("source_access", {}).get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "benchmark_queries_opened",
                "benchmark_labels_opened",
                "benchmark_metrics_opened",
            )
        )
    ):
        raise ValueError("prompt-unary execution source-access contract differs")
    if Path(args.output).exists() or Path(args.output).is_symlink():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    checkpoint = Path(args.output).with_suffix(".pth")
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    label_bytes, _, _ = stable_descriptor_load(
        args.label_zip,
        lambda handle: handle.read(),
        expected_sha256=args.expected_label_zip_sha256,
        label="official instance zip",
    )
    authority["verified_authority_path"] = str(authority_path)
    authority["verified_authority_sha256"] = authority_sha
    return authority, label_bytes


def _validate_cross_asset_authority(
    *,
    manifest: dict,
    manifest_sha256: str,
    capability: dict,
    state: dict,
) -> dict:
    if not torch.equal(capability.get("xyz"), state.get("xyz")) or not torch.equal(
        capability.get("valid"), state.get("valid")
    ):
        raise ValueError("capability and factorized-state primitive rows differ")
    cap_metadata = capability.get("metadata", {})
    state_metadata = state.get("metadata", {})
    cap_row_authority = cap_metadata.get("primitive_row_authority", {})
    geometry = state_metadata.get("geometry_fingerprint")
    cap_geometry = cap_metadata.get("mpr_geometry_fingerprint")
    if (
        cap_metadata.get("registration_responsibility_cache_sha256")
        != manifest_sha256
        or state_metadata.get("registration_responsibility_cache_sha256")
        != manifest_sha256
        or cap_geometry != geometry
        or cap_row_authority.get("num_global_rows") != int(capability["xyz"].shape[0])
        or cap_row_authority.get("num_active_rows") != int(capability["valid"].sum())
        or int(geometry.get("num_gaussians", -1)) != int(capability["xyz"].shape[0])
        or manifest.get("metadata", {}).get("registration_weight_mode")
        != "exact_front_to_back_marginal_responsibility"
    ):
        raise ValueError("responsibility/capability/state authority chain differs")
    return {
        "num_global_rows": int(capability["xyz"].shape[0]),
        "num_active_rows": int(capability["valid"].sum()),
        "geometry_fingerprint": geometry,
        "primitive_row_authority": cap_row_authority,
        "registration_responsibility_cache_sha256": manifest_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responsibility-manifest", type=Path, required=True)
    parser.add_argument("--expected-responsibility-manifest-sha256", required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--expected-label-zip-sha256", required=True)
    parser.add_argument("--capability-bank", type=Path, required=True)
    parser.add_argument("--expected-capability-bank-sha256", required=True)
    parser.add_argument("--factorized-state", type=Path, required=True)
    parser.add_argument("--expected-factorized-state-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    execution_authority, label_bytes = _validate_execution_authority(args)
    manifest, manifest_sha, _ = load_json_object(
        args.responsibility_manifest,
        expected_sha256=args.expected_responsibility_manifest_sha256,
        label="exact responsibility manifest",
    )
    height = int(manifest["metadata"]["feature_height"])
    width = int(manifest["metadata"]["feature_width"])
    frame_ids = [int(value) for value in manifest["frame_indices"]]
    views_root = Path(str(args.responsibility_manifest) + ".views")
    records = manifest.get("views")
    if not isinstance(records, list) or len(records) != len(frame_ids):
        raise ValueError("exact responsibility view records differ")
    first, _, _ = load_torch_payload(
        views_root / "view_00000.pt",
        expected_sha256=str(records[0]["sha256"]),
        map_location="cpu",
        label="first exact responsibility view",
    )
    num_gaussians = int(first["num_gaussians"])
    with zipfile.ZipFile(io.BytesIO(label_bytes), "r") as label_archive:
        views = {
            frame: _load_sparse_view(
                view_path=views_root / f"view_{index:05d}.pt",
                frame_id=frame,
                label_archive=label_archive,
                expected_view_sha256=str(records[index]["sha256"]),
                height=height,
                width=width,
                num_gaussians=num_gaussians,
            )
            for index, frame in enumerate(frame_ids)
        }
    occurrences: dict[int, list[int]] = {}
    for frame, view in views.items():
        ids, counts = torch.unique(view.instance_image, return_counts=True)
        for instance_id, count in zip(ids.tolist(), counts.tolist()):
            if instance_id > 0 and count >= MIN_PIXELS:
                occurrences.setdefault(int(instance_id), []).append(frame)
    eligible = {
        instance_id: frames
        for instance_id, frames in occurrences.items()
        if len(frames) >= MIN_VIEWS
    }
    train_ids = sorted(i for i in eligible if instance_split(i) == "train")
    validation_ids = sorted(i for i in eligible if instance_split(i) == "validation")
    if len(validation_ids) < 3:
        raise RuntimeError("frozen instance split is underpowered")
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("train and validation instances overlap")

    capability, _, _ = load_torch_payload(
        args.capability_bank,
        expected_sha256=args.expected_capability_bank_sha256,
        map_location="cpu",
        label="capability bank",
    )
    state, _, _ = load_torch_payload(
        args.factorized_state,
        expected_sha256=args.expected_factorized_state_sha256,
        map_location="cpu",
        label="factorized primitive state",
    )
    cross_asset_authority = _validate_cross_asset_authority(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        capability=capability,
        state=state,
    )
    prompt_count = 2 * len(eligible)
    feature_storage_bytes = prompt_count * num_gaussians * (13 * 4 + 3 * 4 + 1)
    capability_fp32_bytes = int(
        4
        * (
            capability["appearance_dino_v3"].numel()
            + capability["boundary_sam3"].numel()
        )
    )
    maximum_estimated_working_bytes = (
        2 * capability_fp32_bytes + feature_storage_bytes + 1024**3
    )
    if device.type == "cuda":
        total_device_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        if maximum_estimated_working_bytes > int(0.7 * total_device_bytes):
            raise RuntimeError("prompt full-row feature memory audit exceeds 70% of GPU")
    else:
        total_device_bytes = 0
    examples = build_examples(
        views=views,
        eligible=eligible,
        capability_bank=capability,
        factorized_state=state,
        device=device,
        height=height,
        width=width,
    )
    del capability, state
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [
        example for example in examples if example.split == "validation"
    ]

    head = RegisteredEvidenceToUnaryV1(hidden_dim=32, max_delta_logit=4.0).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline = evaluate(
        head=head, examples=validation_examples, views=views, device=device
    )
    best = baseline
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    history: list[dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        losses: list[float] = []
        ordered = sorted(
            train_examples,
            key=lambda example: _hash_text("train_order", epoch, example.instance_id, example.mode),
        )
        for example in ordered:
            target_frame = example.target_frames[(epoch - 1) % len(example.target_frames)]
            view = views[target_frame]
            unique, inverse = _view_rows(view, device)
            subset = _subset_features(example.features, unique)
            output = head(subset)
            prediction, supported = _render_unique(
                output.foreground_probability, view, unique, inverse, device
            )
            target = (view.instance_image.to(device) == example.instance_id).float()
            keep = supported
            data_loss = _balanced_bce(prediction[keep], target[keep])
            residual_loss = output.bounded_logit_residual.square().mean()
            rendered_confidence, _ = _render_unique(
                output.confidence, view, unique, inverse, device
            )
            confidence_target = 1.0 - (prediction.detach() - target).abs()
            confidence_loss = F.mse_loss(
                rendered_confidence[keep], confidence_target[keep]
            )
            loss = data_loss + 0.05 * residual_loss + 0.02 * confidence_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(
            head=head, examples=validation_examples, views=views, device=device
        )
        metrics = validation["macro"]["all"]["candidate"]
        best_metrics = best["macro"]["all"]["candidate"]
        selection = (metrics["average_precision"], -metrics["bce"], -epoch)
        best_selection = (
            best_metrics["average_precision"],
            -best_metrics["bce"],
            -best_epoch,
        )
        if selection > best_selection:
            best = validation
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean(losses),
                "validation_ap": metrics["average_precision"],
                "validation_iou_at_0_5": metrics["iou_at_0_5"],
                "validation_bce": metrics["bce"],
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    head.load_state_dict(best_state)
    final = evaluate(head=head, examples=validation_examples, views=views, device=device)
    candidate = final["macro"]["all"]["candidate"]
    analytic = final["macro"]["all"]["analytic"]
    deltas = {
        key: candidate[key] - analytic[key]
        for key in candidate
        if key in analytic
    }
    worst_ap_delta = min(row["delta"]["average_precision"] for row in final["records"])
    promoted = (
        deltas["average_precision"] > 0
        and deltas["iou_at_0_5"] > 0
        and deltas["precision_at_0_5"] >= -0.01
        and 0.8 <= candidate["area_ratio"] <= 1.25
        and worst_ap_delta >= -0.05
    )
    result = {
        "schema": "radio_gs.registered_evidence_to_unary.clean_scannet_pilot_result.v1",
        "schema_version": 1,
        "scene_id": SCENE_ID,
        "method": "RegisteredEvidenceToUnaryV1",
        "graph": "off",
        "connected_selection": "off",
        "best_epoch": best_epoch,
        "eligible_instance_ids": sorted(eligible),
        "train_instance_ids": train_ids,
        "validation_instance_ids": validation_ids,
        "instance_disjoint": not bool(set(train_ids) & set(validation_ids)),
        "source_target_view_disjoint": all(
            example.source_frame not in example.target_frames for example in examples
        ),
        "metrics": final,
        "macro_delta_candidate_minus_analytic": deltas,
        "worst_validation_prompt_ap_delta": worst_ap_delta,
        "promotion_gate_passed": promoted,
        "decision": "eligible_for_cross_scene_clean_confirmation" if promoted else "stop_v1_before_benchmarks",
        "history": history,
        "authority": {
            "execution_authority": {
                "path": execution_authority["verified_authority_path"],
                "sha256": execution_authority["verified_authority_sha256"],
            },
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "sha256": sha256_file(args.preregistration),
            },
            "official_instance_zip": {
                "path": str(args.label_zip.resolve()),
                "sha256": sha256_file(args.label_zip),
            },
            "responsibility_manifest": {
                "path": str(args.responsibility_manifest.resolve()),
                "sha256": sha256_file(args.responsibility_manifest),
            },
            "capability_bank_sha256": sha256_file(args.capability_bank),
            "factorized_state_sha256": sha256_file(args.factorized_state),
        },
        "cross_asset_authority": cross_asset_authority,
        "memory_audit": {
            "num_prompts": prompt_count,
            "num_global_rows": num_gaussians,
            "feature_storage_bytes": feature_storage_bytes,
            "capability_fp32_bytes": capability_fp32_bytes,
            "maximum_estimated_working_bytes": maximum_estimated_working_bytes,
            "device_total_bytes": total_device_bytes,
            "ceiling_fraction_of_device": 0.7,
        },
        "source_access": {
            "clean_source_training_instance_labels_opened": True,
            "clean_heldout_target_view_instance_labels_used_only_for_loss_and_metrics": True,
            "clean_target_view_labels_enter_model_inputs": False,
            "clean_target_view_rgb_opened_for_prompt_head": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_target_labels_opened": False,
            "benchmark_metrics_opened": False,
            "target_rgb_used_at_inference": False,
            "per_scene_tuning": False,
        },
    }
    write_frozen_json(args.output, result)
    checkpoint = args.output.with_suffix(".pth")
    write_torch_noclobber(
        checkpoint,
        {
            "schema": "radio_gs.registered_evidence_to_unary.checkpoint.v1",
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "train_instance_ids": train_ids,
            "validation_instance_ids": validation_ids,
            "result_sha256": sha256_file(args.output),
        },
    )
    print(json.dumps({"output": str(args.output), "promoted": promoted, "best_epoch": best_epoch}))


if __name__ == "__main__":
    main()
