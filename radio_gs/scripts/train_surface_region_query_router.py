#!/usr/bin/env python3
"""Train a generic canonical-negative router over a frozen region codebook."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_query_router import (
    SurfaceRegionQueryRouterV1,
)
from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryResidualCodebookV1,
)
from radio_gs.losses.surface_region_codebook_loss import (
    scene_listwise_and_hard_negative_loss,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.train_surface_region_codebook_v3 import (
    _complete_scene_batches,
    _gradient_norm,
    _load_text_bank,
    _ranking_metrics,
)
from radio_gs.scripts.train_surface_region_residual_codebook import (
    _assert_checkpoint_training_contract,
    _head_descriptors,
)
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load as load_surface_caches,
    _paths as cache_paths,
    _seed_training,
)
from radio_gs.utils.immutable_artifacts import (
    load_sha_bound_project_checkpoint_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _load_generic_negatives(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    payload, digest, source = load_sha_bound_project_checkpoint_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="generic canonical-negative text bank",
    )
    expected_queries = ["object", "things", "stuff", "texture"]
    embeddings = payload.get("embeddings")
    if (
        payload.get("text_encoder") != "siglip2"
        or payload.get("text_canonicalization")
        not in {None, "official_c_radio_siglip2_g"}
        or payload.get("queries") != expected_queries
        or payload.get("prompt_templates") != ["{query}"]
        or not isinstance(embeddings, torch.Tensor)
        or tuple(embeddings.shape) != (4, 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("generic canonical-negative bank contract differs")
    return F.normalize(embeddings.float(), dim=-1, eps=1e-8), {
        "path": str(source),
        "sha256": digest,
        "queries": expected_queries,
        "benchmark_category_vocabulary": False,
        "text_canonicalization_metadata_present": (
            "text_canonicalization" in payload
        ),
        "text_canonicalization_authority": (
            "explicit_legacy_frozen_cache_allowance"
            if "text_canonicalization" not in payload
            else "cache_metadata"
        ),
    }


@torch.no_grad()
def _materialize_codebook(
    codebook: SurfaceRegionSummaryResidualCodebookV1,
    head: SigLIP2SummaryHead,
    data: Mapping[str, object],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    codebook.eval()
    tokens: list[torch.Tensor] = []
    descriptors: list[torch.Tensor] = []
    exact_fallback = True
    total = len(torch.as_tensor(data["radio_features"]))
    for start in range(0, total, batch_size):
        rows = torch.arange(start, min(start + batch_size, total))
        features = torch.as_tensor(data["radio_features"])[rows].to(device).float()
        geometry = torch.as_tensor(data["geometry"])[rows].to(device).float()
        anchor = torch.as_tensor(data["anchor_index"])[rows].to(device)
        token_mask = torch.as_tensor(data["token_mask"])[rows].to(device)
        reliability = torch.as_tensor(data["reliability"])[rows].to(device).float()
        output = codebook.forward_codebook(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=token_mask,
            reliability=reliability,
        )
        direct = codebook.base(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=token_mask,
            reliability=reliability,
        )
        exact_fallback = exact_fallback and torch.equal(
            output.canonical_token, direct
        ) and torch.equal(output.slot_tokens[:, 0], direct)
        _, projected = _head_descriptors(
            head, output.canonical_token, output.slot_tokens[:, 1:]
        )
        tokens.append(output.slot_tokens.cpu())
        descriptors.append(projected.cpu())
    if not exact_fallback:
        raise RuntimeError("frozen residual codebook lost its exact V2 fallback")
    return {
        "slot_tokens": torch.cat(tokens),
        "slot_descriptors": torch.cat(descriptors),
    }


def _canonical_negative_slot_scores(
    descriptors: torch.Tensor,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    values = F.normalize(torch.as_tensor(descriptors).float(), dim=-1, eps=1e-8)
    positive = F.normalize(
        torch.as_tensor(positive_text, device=values.device).float(),
        dim=-1,
        eps=1e-8,
    )
    negative = F.normalize(
        torch.as_tensor(negative_text, device=values.device).float(),
        dim=-1,
        eps=1e-8,
    )
    positive_cosine = torch.einsum("bkd,qd->bkq", values, positive)
    hardest_negative = torch.einsum(
        "bkd,nd->bkn", values, negative
    ).amax(dim=-1, keepdim=True)
    return torch.sigmoid(
        float(logit_scale) * (positive_cosine - hardest_negative)
    )


def _teacher_response(
    data: Mapping[str, object],
    rows: torch.Tensor,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
    *,
    logit_scale: float,
) -> torch.Tensor:
    descriptors = F.normalize(
        torch.as_tensor(data["official_crop_summaries"])[rows].to(device).float(),
        dim=-1,
        eps=1e-8,
    )
    mask = torch.as_tensor(data["teacher_mask"])[rows].to(device).bool()
    scores = _canonical_negative_slot_scores(
        descriptors,
        positive_text,
        negative_text,
        logit_scale=logit_scale,
    )
    return scores.masked_fill(
        ~mask[..., None], torch.finfo(scores.dtype).min
    ).amax(dim=1)


def _response_metrics(
    student: torch.Tensor,
    teacher: torch.Tensor,
    scene_ids: Sequence[str],
) -> dict[str, float]:
    response_profile = F.cosine_similarity(student, teacher, dim=-1)
    metrics = {
        "text_response_smooth_l1": float(F.smooth_l1_loss(student, teacher)),
        "response_profile_cosine_mean": float(response_profile.mean()),
        "response_profile_cosine_p05": float(
            torch.quantile(response_profile, 0.05)
        ),
    }
    metrics.update(_ranking_metrics(student, teacher, scene_ids))
    return metrics


def _loss_terms(
    router: SurfaceRegionQueryRouterV1,
    materialized: Mapping[str, torch.Tensor],
    data: Mapping[str, object],
    rows: torch.Tensor,
    scene_ids: Sequence[str],
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
    *,
    logit_scale: float,
) -> dict[str, torch.Tensor]:
    descriptors = materialized["slot_descriptors"][rows].to(device).float()
    tokens = materialized["slot_tokens"][rows].to(device).float()
    output = router(
        descriptors,
        tokens,
        positive_text,
        negative_text,
        logit_scale=logit_scale,
    )
    target = _teacher_response(
        data,
        rows,
        positive_text,
        negative_text,
        device,
        logit_scale=logit_scale,
    )
    independent = F.smooth_l1_loss(output.response, target)
    listwise, hard_negative = scene_listwise_and_hard_negative_loss(
        output.response, target, scene_ids
    )
    return {
        "main": independent,
        "independent": independent,
        "listwise": listwise,
        "hard_negative": hard_negative,
        "residual_gate_mean": output.residual_gate.mean(),
    }


@torch.no_grad()
def _evaluate(
    router: SurfaceRegionQueryRouterV1,
    materialized: Mapping[str, torch.Tensor],
    data: Mapping[str, object],
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    logit_scale: float,
) -> tuple[dict[str, float], dict[str, float]]:
    router.eval()
    candidate_parts: list[torch.Tensor] = []
    control_parts: list[torch.Tensor] = []
    teacher_parts: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    total = len(materialized["slot_descriptors"])
    for start in range(0, total, batch_size):
        rows = torch.arange(start, min(start + batch_size, total))
        descriptors = materialized["slot_descriptors"][rows].to(device).float()
        tokens = materialized["slot_tokens"][rows].to(device).float()
        output = router(
            descriptors,
            tokens,
            positive_text,
            negative_text,
            logit_scale=logit_scale,
        )
        candidate_parts.append(output.response.cpu())
        control_parts.append(output.slot_scores[:, 0].cpu())
        teacher_parts.append(
            _teacher_response(
                data,
                rows,
                positive_text,
                negative_text,
                device,
                logit_scale=logit_scale,
            ).cpu()
        )
        gates.append(output.residual_gate.cpu())
        weights.append(output.slot_weights.cpu())
    candidate = torch.cat(candidate_parts)
    control = torch.cat(control_parts)
    teacher = torch.cat(teacher_parts)
    gate = torch.cat(gates)
    slot_weights = torch.cat(weights)
    scenes = list(data["scene_ids"])
    candidate_metrics = _response_metrics(candidate, teacher, scenes)
    candidate_metrics.update(
        {
            "residual_gate_mean": float(gate.mean()),
            "residual_gate_p95": float(torch.quantile(gate, 0.95)),
            "residual_gate_positive_fraction": float((gate > 0).float().mean()),
            "mean_residual_attention": float(slot_weights[:, 1:].sum(1).mean()),
            "response_delta_from_control_abs_mean": float(
                (candidate - control).abs().mean()
            ),
        }
    )
    control_metrics = _response_metrics(control, teacher, scenes)
    return candidate_metrics, control_metrics


def _generic_gate(
    candidate: Mapping[str, float],
    control: Mapping[str, float],
) -> dict[str, object]:
    checks = {
        "text_response_smooth_l1": candidate["text_response_smooth_l1"]
        < control["text_response_smooth_l1"],
        "support_top1_agreement": candidate["support_top1_agreement"]
        >= control["support_top1_agreement"],
        "ranking_spearman_p05": candidate["ranking_spearman_p05"]
        - control["ranking_spearman_p05"]
        >= -0.002,
        "response_profile_cosine_p05": candidate["response_profile_cosine_p05"]
        - control["response_profile_cosine_p05"]
        >= -0.002,
        "mean_residual_attention": candidate["mean_residual_attention"] >= 0.02,
    }
    return {
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "overall_pass": all(checks.values()),
        "deltas": {
            key: candidate[key] - control[key]
            for key in (
                "text_response_smooth_l1",
                "support_top1_agreement",
                "ranking_spearman_p05",
                "response_profile_cosine_p05",
            )
        },
    }


def _selection_feasible(
    candidate: Mapping[str, float],
    control: Mapping[str, float],
) -> bool:
    return (
        candidate["text_response_smooth_l1"] < control["text_response_smooth_l1"]
        and candidate["support_top1_agreement"]
        >= control["support_top1_agreement"]
        and candidate["ranking_spearman_p05"]
        - control["ranking_spearman_p05"]
        >= -0.002
        and candidate["response_profile_cosine_p05"]
        - control["response_profile_cosine_p05"]
        >= -0.002
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    train_paths = cache_paths(args.train_caches)
    validation_paths = cache_paths(args.validation_caches)
    train_data, train_meta = load_surface_caches(train_paths, "train")
    validation_data, validation_meta = load_surface_caches(
        validation_paths, "validation"
    )
    overlap = set(train_meta["scenes"]) & set(validation_meta["scenes"])
    if overlap:
        raise ValueError(f"query-router scene leakage: {sorted(overlap)}")
    for field in (
        "region_contract_sha256",
        "teacher_region",
        "radio_checkpoint_sha256",
        "excluded_physical_spaces",
    ):
        if train_meta[field] != validation_meta[field]:
            raise ValueError(f"query-router train/validation {field} differs")
    radio_path = Path(args.radio_checkpoint)
    if sha256_file(radio_path) != train_meta["radio_checkpoint_sha256"]:
        raise ValueError("RADIO checkpoint differs from cache provenance")
    codebook_path = Path(args.codebook)
    codebook_sha256 = sha256_file(codebook_path)
    if codebook_sha256 != args.codebook_sha256:
        raise ValueError("frozen residual codebook SHA-256 differs")
    codebook, codebook_payload = SurfaceRegionSummaryResidualCodebookV1.from_checkpoint(
        codebook_path, map_location="cpu"
    )
    _assert_checkpoint_training_contract(
        codebook_payload,
        expected_contract_sha256=str(train_meta["region_contract_sha256"]),
        expected_radio_sha256=str(train_meta["radio_checkpoint_sha256"]),
        label="frozen residual codebook",
    )
    if (
        codebook_payload.get("provenance", {}).get("uses_benchmark_scenes")
        is not False
        or codebook_payload.get("provenance", {}).get(
            "uses_benchmark_test_vocabulary"
        )
        is not False
    ):
        raise ValueError("frozen codebook provenance is benchmark contaminated")
    fit_text, fit_record = _load_text_bank(
        Path(args.fit_text_bank),
        expected_sha256=args.fit_text_bank_sha256,
        expected_split="fit",
    )
    validation_text, validation_text_record = _load_text_bank(
        Path(args.validation_text_bank),
        expected_sha256=args.validation_text_bank_sha256,
        expected_split="dev",
    )
    negative_text, negative_record = _load_generic_negatives(
        Path(args.negative_text_bank),
        expected_sha256=args.negative_text_bank_sha256,
    )
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"query-router output already exists: {output}")
    device = torch.device(args.device)
    generator = _seed_training(int(args.seed), device=device)
    codebook = codebook.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(radio_path).to(device).eval()
    head.requires_grad_(False)
    train_materialized = _materialize_codebook(
        codebook,
        head,
        train_data,
        device,
        batch_size=int(args.materialize_batch_size),
    )
    validation_materialized = _materialize_codebook(
        codebook,
        head,
        validation_data,
        device,
        batch_size=int(args.materialize_batch_size),
    )
    del codebook, head
    router = SurfaceRegionQueryRouterV1(
        hidden_dim=int(args.hidden_dim),
        codebook_sha256=codebook_sha256,
    ).to(device)
    fit_text = fit_text.to(device)
    validation_text = validation_text.to(device)
    negative_text = negative_text.to(device)
    parameters = tuple(router.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    initial, control_metrics = _evaluate(
        router,
        validation_materialized,
        validation_data,
        validation_text,
        negative_text,
        device,
        batch_size=int(args.eval_batch_size),
        logit_scale=float(args.logit_scale),
    )
    if initial != {
        **control_metrics,
        "residual_gate_mean": 0.0,
        "residual_gate_p95": 0.0,
        "residual_gate_positive_fraction": 0.0,
        "mean_residual_attention": 0.0,
        "response_delta_from_control_abs_mean": 0.0,
    }:
        raise RuntimeError("zero router does not exactly reproduce V2 response metrics")
    print(
        json.dumps(
            {"untrained": initial, "frozen_v2_control": control_metrics},
            sort_keys=True,
        ),
        flush=True,
    )
    calibration: dict[str, float] | None = None
    history: list[dict[str, object]] = []
    feasible_state: dict[str, torch.Tensor] | None = None
    feasible_metrics: dict[str, float] | None = None
    feasible_epoch = 0
    feasible_key: tuple[float, float] | None = None
    last_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        router.train()
        batches = _complete_scene_batches(
            list(train_data["scene_ids"]),
            target_rows=int(args.batch_size),
            generator=generator,
        )
        losses: list[float] = []
        component_sums: dict[str, float] = {}
        for rows in batches:
            scenes = [str(train_data["scene_ids"][int(row)]) for row in rows]
            terms = _loss_terms(
                router,
                train_materialized,
                train_data,
                rows,
                scenes,
                fit_text,
                negative_text,
                device,
                logit_scale=float(args.logit_scale),
            )
            if calibration is None:
                main_norm = _gradient_norm(
                    terms["main"], parameters, retain_graph=True
                )
                calibration = {
                    "main_gradient_l2": main_norm,
                    "auxiliary_gradient_ratio_each": float(
                        args.auxiliary_gradient_ratio_each
                    ),
                }
                for name in ("listwise", "hard_negative"):
                    norm = _gradient_norm(
                        terms[name], parameters, retain_graph=True
                    )
                    calibration[f"{name}_gradient_l2"] = norm
                    calibration[f"{name}_lambda"] = (
                        float(args.auxiliary_gradient_ratio_each)
                        * main_norm
                        / norm
                    )
                print(json.dumps({"gradient_calibration": calibration}), flush=True)
            loss = (
                terms["main"]
                + calibration["listwise_lambda"] * terms["listwise"]
                + calibration["hard_negative_lambda"] * terms["hard_negative"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in terms.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    value.detach().cpu()
                )
        metrics, recomputed_control = _evaluate(
            router,
            validation_materialized,
            validation_data,
            validation_text,
            negative_text,
            device,
            batch_size=int(args.eval_batch_size),
            logit_scale=float(args.logit_scale),
        )
        if recomputed_control != control_metrics:
            raise RuntimeError("frozen V2 control changed during router training")
        feasible = _selection_feasible(metrics, control_metrics)
        record: dict[str, object] = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "selection_feasible": feasible,
            "train_components": {
                key: value / len(batches) for key, value in component_sums.items()
            },
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        last_state = copy.deepcopy(router.state_dict())
        if feasible:
            key = (
                -metrics["text_response_smooth_l1"],
                metrics["support_top1_agreement"],
            )
            if feasible_key is None or key > feasible_key:
                feasible_key = key
                feasible_epoch = epoch
                feasible_state = copy.deepcopy(router.state_dict())
                feasible_metrics = dict(metrics)
                stale = 0
            else:
                stale += 1
            if int(args.patience) > 0 and stale >= int(args.patience):
                break
    if last_state is None or calibration is None:
        raise RuntimeError("query-router training did not complete")
    selection_status = (
        "control_referenced_feasible_epoch_selected"
        if feasible_state is not None
        else "no_feasible_epoch_diagnostic_final_state_only"
    )
    selected_state = feasible_state if feasible_state is not None else last_state
    selected_epoch = feasible_epoch if feasible_state is not None else int(history[-1]["epoch"])
    router.load_state_dict(selected_state)
    final, final_control = _evaluate(
        router,
        validation_materialized,
        validation_data,
        validation_text,
        negative_text,
        device,
        batch_size=int(args.eval_batch_size),
        logit_scale=float(args.logit_scale),
    )
    if final_control != control_metrics:
        raise RuntimeError("final V2 control differs")
    if feasible_metrics is not None and final != feasible_metrics:
        raise RuntimeError("selected router metrics are not reproducible")
    gate = _generic_gate(final, control_metrics)
    provenance = {
        "training_scope": "global_scene_disjoint_canonical_negative_query_router_v1",
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "scene_disjoint": True,
        "query_split_disjoint": True,
        "train": train_meta,
        "validation": validation_meta,
        "fit_text_bank": fit_record,
        "validation_text_bank": validation_text_record,
        "generic_negative_text_bank": negative_record,
        "frozen_codebook": {
            "path": str(codebook_path.resolve()),
            "sha256": codebook_sha256,
        },
        "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "score_contract": "canonical_negative_bernoulli_query_first",
        "logit_scale": float(args.logit_scale),
    }
    payload = {
        "schema_version": 1,
        "architecture": router.architecture(),
        "state_dict": {
            key: value.detach().cpu() for key, value in selected_state.items()
        },
        "provenance": provenance,
        "history": history,
        "selected_epoch": selected_epoch,
        "selection_status": selection_status,
        "untrained_baseline": initial,
        "frozen_v2_control": control_metrics,
        "validation": final,
        "generic_gate": gate,
        "gradient_calibration": calibration,
        "training_config": {
            "seed": int(args.seed),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "logit_scale": float(args.logit_scale),
            "auxiliary_gradient_ratio_each": float(
                args.auxiliary_gradient_ratio_each
            ),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema_version": 1,
        "status": "complete" if gate["overall_pass"] else "generic_gate_failed",
        "output": str(output),
        "checkpoint_sha256": sha256_file(output),
        "selected_epoch": selected_epoch,
        "selection_status": selection_status,
        "validation": final,
        "frozen_v2_control": control_metrics,
        "generic_gate": gate,
        "gradient_calibration": calibration,
        "scene_overlap": [],
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--fit-text-bank-sha256", required=True)
    parser.add_argument("--validation-text-bank", required=True)
    parser.add_argument("--validation-text-bank-sha256", required=True)
    parser.add_argument("--negative-text-bank", required=True)
    parser.add_argument("--negative-text-bank-sha256", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codebook-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--materialize-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--logit-scale", type=float, default=10.0)
    parser.add_argument(
        "--auxiliary-gradient-ratio-each", type=float, default=0.25
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
