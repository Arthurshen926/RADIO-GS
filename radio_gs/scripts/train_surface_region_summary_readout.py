#!/usr/bin/env python3
"""Train the global query-free 3-D surface-region RADIO summary readout."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SEPARATE_CONTEXT_POOLING,
    SurfaceRegionSummaryReadoutV2,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.utils.immutable_artifacts import (
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _paths(raw: str) -> list[Path]:
    result = []
    for value in str(raw).replace(",", " ").split():
        matches = (
            [Path(path) for path in sorted(glob.glob(value))]
            if any(c in value for c in "*?[")
            else [Path(value)]
        )
        result.extend(matches)
    if not result or any(not path.is_file() for path in result):
        raise FileNotFoundError("surface-region cache list is empty or missing")
    return result


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _seed_training(
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Seed model initialization, augmentation, and data order coherently."""

    value = int(seed)
    if value < 0:
        raise ValueError("training seed must be non-negative")
    torch.manual_seed(value)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(value)
    return torch.Generator().manual_seed(value)


def _load(paths: list[Path], expected_role: str) -> tuple[dict, dict]:
    keys = (
        "radio_features", "geometry", "token_mask", "reliability",
        "official_summary_tokens", "official_crop_summaries", "teacher_mask",
        "anchor_index",
    )
    parts = {key: [] for key in keys}; scenes = set(); hashes = []; contracts = []; contract_specs = []
    teacher_region_specs = []
    radio_checkpoint_hashes = []
    region_ids: set[str] = set()
    row_scenes: list[str] = []
    row_scenes_complete = True
    excluded_spaces: set[str] | None = None
    exclusion_files: dict[str, str] = {}
    for path in paths:
        payload, _, _ = load_torch_mapping(
            path,
            map_location="cpu",
            label="SurfaceRegion training cache",
        )
        metadata = payload.get("metadata", {})
        if metadata.get("schema_version") != 3 or metadata.get("split_role") != expected_role:
            raise ValueError(f"{path} has wrong 3-D cache schema/split")
        contract = SurfaceRegionContractV2(
            **{
                **metadata["region_contract"],
                "radii_m": tuple(metadata["region_contract"]["radii_m"]),
            }
        )
        contract.assert_compatible(metadata)
        contracts.append(contract.digest)
        contract_specs.append(contract.to_dict())
        teacher_semantics = metadata.get(
            "teacher_region_semantics",
            "selected_core_and_context_extent_legacy",
        )
        if teacher_semantics == (
            "fixed_core_geodesic_support_without_input_context_v1"
        ):
            target_protocol = metadata.get(
                "teacher_target_protocol",
                {},
            )
            protocol_digest = hashlib.sha256(
                json.dumps(
                    target_protocol,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                metadata.get("teacher_target_source")
                not in {"fresh_official_runtime", "exact_cache_replay"}
                or metadata.get("teacher_regions_saturated") != 0
                or metadata.get("complete_scene_regions") is not True
                or metadata.get("teacher_target_schema_version") != 1
                or metadata.get("teacher_crop_protocol")
                != (
                    "core_support_defined_unmasked_bbox_min24_"
                    "context_pad0_v1"
                )
                or metadata.get("teacher_target_protocol_sha256")
                != protocol_digest
            ):
                raise ValueError(
                    f"{path} has an incomplete fixed-core teacher protocol"
                )
            cache_records = metadata.get("region_records", [])
            if len(cache_records) != len(payload["radio_features"]):
                raise ValueError(
                    f"{path} has misaligned fixed-core region records"
                )
            for record in cache_records:
                region_id = str(record.get("region_id", ""))
                scene_id = str(record.get("scene", ""))
                if (
                    not region_id
                    or region_id in region_ids
                    or not scene_id
                    or scene_id not in metadata.get("scene_names", [])
                ):
                    raise ValueError(
                        "surface-region caches contain duplicate/invalid "
                        "region IDs or scene bindings"
                    )
                region_ids.add(region_id)
                row_scenes.append(scene_id)
        else:
            cache_scenes = [str(value) for value in metadata.get("scene_names", [])]
            if len(cache_scenes) == 1:
                row_scenes.extend(
                    [cache_scenes[0]] * len(payload["radio_features"])
                )
            else:
                row_scenes_complete = False
        teacher_region_specs.append(
            {
                "semantics": teacher_semantics,
                "contract": metadata.get("teacher_region_contract"),
                "contract_sha256": metadata.get(
                    "teacher_region_contract_sha256",
                    "",
                ),
                "target_source": metadata.get(
                    "teacher_target_source",
                    "legacy_in_cache",
                ),
                "target_protocol_sha256": metadata.get(
                    "teacher_target_protocol_sha256",
                    "",
                ),
            }
        )
        radio_checkpoint_sha256 = str(
            metadata.get("radio_checkpoint_sha256", "")
        )
        if not radio_checkpoint_sha256:
            raise ValueError(f"{path} lacks RADIO checkpoint provenance")
        radio_checkpoint_hashes.append(radio_checkpoint_sha256)
        if any(metadata.get(key, True) for key in (
            "uses_benchmark_scenes", "uses_benchmark_test_vocabulary",
            "annotations_opened", "labels_opened", "instances_opened", "text_opened",
        )):
            raise ValueError(f"{path} violates the query-free scene-disjoint contract")
        scenes.update(str(value) for value in metadata["scene_names"])
        hashes.append(str(metadata["split_file_sha256"]))
        cache_exclusions = {
            str(value) for value in metadata.get("excluded_physical_spaces", [])
        }
        if excluded_spaces is None:
            excluded_spaces = cache_exclusions
        elif cache_exclusions != excluded_spaces:
            raise ValueError("surface-region cache exclusion contracts differ")
        if not bool(metadata.get("physical_space_disjoint", True)):
            raise ValueError(f"{path} does not certify physical-space disjointness")
        for record in metadata.get("exclusion_files", []):
            resolved = str(record["path"])
            digest = str(record["sha256"])
            previous = exclusion_files.setdefault(resolved, digest)
            if previous != digest:
                raise ValueError("surface-region exclusion file hashes differ")
        for key in keys:
            parts[key].append(torch.as_tensor(payload[key]))
    merged = {key: torch.cat(value, dim=0) for key, value in parts.items()}
    if row_scenes_complete:
        if len(row_scenes) != len(merged["radio_features"]):
            raise ValueError("surface-region row/scene bindings are misaligned")
        merged["scene_ids"] = row_scenes
    if len(set(contracts)) != 1:
        raise ValueError("surface-region cache contracts differ")
    if any(spec != contract_specs[0] for spec in contract_specs[1:]):
        raise ValueError("surface-region cache contract specifications differ")
    if any(
        spec != teacher_region_specs[0]
        for spec in teacher_region_specs[1:]
    ):
        raise ValueError("surface-region teacher contracts differ")
    if len(set(radio_checkpoint_hashes)) != 1:
        raise ValueError("surface-region RADIO checkpoints differ")
    merged_meta = {"scenes": sorted(scenes), "split_hashes": sorted(set(hashes)),
                   "cache_paths": [str(path.resolve()) for path in paths],
                   "region_contract_sha256": contracts[0]}
    merged_meta["region_contract"] = contract_specs[0]
    merged_meta["teacher_region"] = teacher_region_specs[0]
    merged_meta["radio_checkpoint_sha256"] = (
        radio_checkpoint_hashes[0]
    )
    merged_meta["excluded_physical_spaces"] = sorted(excluded_spaces or set())
    merged_meta["exclusion_files"] = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(exclusion_files.items())
    ]
    merged_meta["physical_space_disjoint"] = True
    return merged, merged_meta


def _targets(data: dict, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = data["official_summary_tokens"][rows].float()
    descriptors = F.normalize(data["official_crop_summaries"][rows].float(), dim=-1)
    mask = data["teacher_mask"][rows].bool()
    # The medoid is selected in the official SigLIP2 descriptor space, not
    # by averaging or comparing backbone summary tokens.
    normalized = descriptors
    similarity = torch.einsum("bvd,bwd->bvw", normalized, normalized)
    similarity = similarity.masked_fill(~mask[:, None, :], 0.0)
    medoid = similarity.sum(-1).masked_fill(~mask, -1e9).argmax(-1)
    batch = torch.arange(len(rows))
    target_token = tokens[batch, medoid]
    weights = mask.float() / mask.sum(1, keepdim=True)
    target_descriptor = F.normalize(
        (descriptors * weights[..., None]).sum(1), dim=-1, eps=1e-8
    )
    return target_token, target_descriptor, descriptors, mask


def inject_tangent_direction_noise(
    features: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    angle_degrees: float,
) -> torch.Tensor:
    """Apply isotropic canonical-reconstruction noise to unit RADIO directions."""

    values = F.normalize(torch.as_tensor(features).float(), dim=-1, eps=1e-8)
    mask = torch.as_tensor(token_mask, device=values.device).bool()
    if angle_degrees <= 0:
        return values * mask[..., None]
    tangent = torch.randn_like(values)
    tangent = tangent - (tangent * values).sum(-1, keepdim=True) * values
    tangent = F.normalize(tangent, dim=-1, eps=1e-8)
    # Half-normal angular noise matches a non-negative reconstruction error;
    # clipping avoids rare, unphysical augmentation outliers.
    angle = torch.randn(values.shape[:-1], device=values.device).abs().clamp_max(2.0)
    angle = angle * (float(angle_degrees) * torch.pi / 180.0)
    result = values * angle.cos()[..., None] + tangent * angle.sin()[..., None]
    return F.normalize(result, dim=-1, eps=1e-8) * mask[..., None]


@torch.no_grad()
def _evaluate(model, head, data, device, batch_size: int) -> dict:
    token_cos, descriptor_cos, multiview_cos = [], [], []
    for start in range(0, len(data["radio_features"]), int(batch_size)):
        rows = torch.arange(start, min(start + int(batch_size), len(data["radio_features"])))
        token, descriptor, all_descriptors, teacher_mask = _targets(data, rows)
        predicted = model(
            data["radio_features"][rows].to(device), data["geometry"][rows].to(device),
            anchor_index=data["anchor_index"][rows].to(device),
            token_mask=data["token_mask"][rows].to(device),
            reliability=data["reliability"][rows].to(device),
        )
        projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
        token_cos.extend(F.cosine_similarity(predicted.cpu(), token, dim=-1).tolist())
        descriptor_cos.extend(F.cosine_similarity(projected.cpu(), descriptor, dim=-1).tolist())
        pair = torch.einsum("bd,bvd->bv", projected.cpu(), all_descriptors)
        multiview_cos.extend(pair[teacher_mask].tolist())
    return {
        "summary_token_cosine": sum(token_cos) / len(token_cos),
        "mean_descriptor_cosine": sum(descriptor_cos) / len(descriptor_cos),
        "all_view_descriptor_cosine": sum(multiview_cos) / len(multiview_cos),
    }


def train(args: argparse.Namespace) -> dict:
    train_data, train_meta = _load(_paths(args.train_caches), "train")
    val_data, val_meta = _load(_paths(args.validation_caches), "validation")
    overlap = set(train_meta["scenes"]) & set(val_meta["scenes"])
    if overlap:
        raise ValueError(f"train/validation scene leakage: {sorted(overlap)}")
    if train_meta["region_contract_sha256"] != val_meta["region_contract_sha256"]:
        raise ValueError("train/validation region contracts differ")
    if (
        train_meta["excluded_physical_spaces"]
        != val_meta["excluded_physical_spaces"]
    ):
        raise ValueError("train/validation benchmark exclusion contracts differ")
    if train_meta["teacher_region"] != val_meta["teacher_region"]:
        raise ValueError("train/validation teacher protocols differ")
    if (
        train_meta["radio_checkpoint_sha256"]
        != val_meta["radio_checkpoint_sha256"]
    ):
        raise ValueError("train/validation RADIO checkpoints differ")
    if (
        _sha256_file(Path(args.radio_checkpoint))
        != train_meta["radio_checkpoint_sha256"]
    ):
        raise ValueError(
            "training RADIO checkpoint differs from cache provenance"
    )
    device = torch.device(args.device)
    generator = _seed_training(int(args.seed), device=device)
    model = SurfaceRegionSummaryReadoutV2(
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode=str(
            getattr(args, "reliability_attention_mode", "log_prior")
        ),
        context_pooling_mode=str(
            getattr(args, "context_pooling_mode", JOINT_CONTEXT_POOLING)
        ),
    ).to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device).eval()
    for parameter in head.parameters(): parameter.requires_grad_(False)
    model.eval()
    baseline = _evaluate(model, head, val_data, device, int(args.batch_size))
    baseline_score = 0.5 * (
        baseline["mean_descriptor_cosine"] + baseline["all_view_descriptor_cosine"]
    )
    print(json.dumps({"untrained_baseline": baseline, "selection_score": baseline_score}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate),
                                  weight_decay=float(args.weight_decay))
    best_score, best_epoch, best_state, history, stale = -1.0, 0, None, [], 0
    for epoch in range(int(args.epochs)):
        order = torch.randperm(len(train_data["radio_features"]), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start:start + int(args.batch_size)]
            target_token, target_descriptor, all_descriptors, teacher_mask = _targets(train_data, rows)
            token_mask = train_data["token_mask"][rows].to(device)
            radio_features = inject_tangent_direction_noise(
                train_data["radio_features"][rows].to(device), token_mask,
                angle_degrees=float(args.canonical_noise_degrees),
            )
            predicted = model(
                radio_features,
                train_data["geometry"][rows].to(device),
                anchor_index=train_data["anchor_index"][rows].to(device),
                token_mask=token_mask,
                reliability=train_data["reliability"][rows].to(device),
            )
            projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1)
            target_token, target_descriptor = target_token.to(device), target_descriptor.to(device)
            token_loss = (1 - F.cosine_similarity(predicted, target_token, dim=-1)).mean()
            all_descriptors = all_descriptors.to(device)
            teacher_mask = teacher_mask.to(device)
            all_view_cosine = torch.einsum("bd,bvd->bv", projected, all_descriptors)
            descriptor_loss = (1 - all_view_cosine)[teacher_mask].mean()
            teacher_rel = target_descriptor @ target_descriptor.T
            predicted_rel = projected @ projected.T
            relation_loss = F.smooth_l1_loss(predicted_rel, teacher_rel)
            loss = (float(args.token_weight) * token_loss + descriptor_loss
                    + float(args.relation_weight) * relation_loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); metrics = _evaluate(model, head, val_data, device, int(args.batch_size))
        score = 0.5 * (metrics["mean_descriptor_cosine"] + metrics["all_view_descriptor_cosine"])
        record = {"epoch": epoch + 1, "loss": sum(losses) / len(losses),
                  "selection_score": score, **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if int(args.patience) and stale >= int(args.patience): break
    assert best_state is not None
    model.load_state_dict(best_state)
    architecture = model.architecture(train_meta["region_contract_sha256"])
    provenance = {
        "training_scope": "global_cross_scene_3d_surface_v2", "frozen": True,
        "uses_benchmark_scenes": False, "uses_benchmark_test_vocabulary": False,
        "train": train_meta, "validation": val_meta,
        "scene_disjoint": True, "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "region_contract_sha256": train_meta["region_contract_sha256"],
        "region_contract": train_meta["region_contract"],
        "canonical_direction_noise_degrees": float(args.canonical_noise_degrees),
        "canonical_noise_calibration": str(args.canonical_noise_calibration),
        "random_seed_contract": {
            "seed": int(args.seed),
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        },
    }
    payload = {"schema_version": 3, "architecture": architecture,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
               "provenance": provenance, "history": history, "best_epoch": best_epoch,
               "best_selection_score": best_score, "untrained_baseline": baseline,
               "untrained_baseline_score": baseline_score, "training_config": vars(args)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {"output": str(output.resolve()), "checkpoint_sha256": digest,
              "architecture": architecture, "best_epoch": best_epoch,
              "best_selection_score": best_score,
              "untrained_baseline": baseline,
              "selection_score_delta": best_score - baseline_score,
              "validation": _evaluate(model.to(device), head, val_data, device, int(args.batch_size)),
              "train_scenes": len(train_meta["scenes"]),
              "validation_scenes": len(val_meta["scenes"]), "scene_overlap": []}
    report_path = output.with_suffix(output.suffix + ".json")
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument(
        "--reliability-attention-mode",
        choices=("log_prior", "input_only"),
        default="log_prior",
        help=(
            "Keep the frozen multiplicative confidence prior or use reliability "
            "only through the geometry input to avoid train/inference prior shift."
        ),
    )
    parser.add_argument(
        "--context-pooling-mode",
        choices=(JOINT_CONTEXT_POOLING, SEPARATE_CONTEXT_POOLING),
        default=JOINT_CONTEXT_POOLING,
        help=(
            "Keep legacy joint attention or preserve the core base while "
            "pooling context as a separately normalized conditioning stream."
        ),
    )
    parser.add_argument("--canonical-noise-degrees", type=float, default=0.0)
    parser.add_argument(
        "--canonical-noise-calibration", default="",
        help="Frozen-field angular-residual audit used to set the augmentation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args(); print(json.dumps(train(args), indent=2))


if __name__ == "__main__": main()
