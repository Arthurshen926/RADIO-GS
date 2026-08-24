#!/usr/bin/env python3
"""Joint LERF cross-scene query-view -> target-view extent training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.interfaces.query_packet import QueryPacket
from radio_gs.models.query_native_gaussian_memory import (
    FixedCosineQueryProjection, GaussianGeometry, LowRankSceneCanonicalizer, ModalityQueryAdapter,
    QueryNativeGaussianPosteriorDecoder,
)
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _iou_from_scores, _load_mapping, _proposal_soft_support, _proposal_support,
    _sample_without_replacement, _similarity_scores_for_proposal,
    compose_membership_query_features, visible_membership_target,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def _load_scene(seed_model_path: str, seed_sha: str, episode_path: str, episode_sha: str, threshold: float, text_specs: dict[str, Any] | None = None) -> dict[str, Any]:
    seed, seed_record = _load_mapping(seed_model_path, seed_sha, "prior source-only scene manifest")
    inputs = seed["metadata"]["inputs"]
    loaded: dict[str, Any] = {}
    records: dict[str, Any] = {"seed_manifest": seed_record}
    for key in ("membership", "language_teacher", "query_cache", "universal_field"):
        loaded[key], records[key] = _load_mapping(inputs[key]["path"], inputs[key]["sha256"], key)
    episodes, records["episodes"] = _load_mapping(episode_path, episode_sha, "compiled object episodes")
    if episodes.get("metadata", {}).get("unlisted_semantics") != "unknown_excluded_from_loss":
        raise ValueError("joint LERF episode semantics differs")
    field_path = Path(inputs["field"]["path"])
    field, field_payload, _ = load_factorized_canonical_field_checkpoint(
        field_path, map_location="cpu", expected_sha256=inputs["field"]["sha256"]
    )
    membership = loaded["membership"]
    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    count = int(membership["num_proposals"])
    hard = weights >= threshold
    hard_support = _proposal_support(rows[hard], proposals[hard], count)
    soft_support, soft_values = _proposal_soft_support(rows, proposals, weights, count)
    teacher = loaded["language_teacher"]
    semantic = F.normalize(
        .75 * F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
        + .25 * F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1), dim=-1,
    )
    query = compose_membership_query_features(semantic, None)
    cache = loaded["query_cache"]
    baseline = F.normalize(torch.as_tensor(cache.get("features", cache.get("summary_features"))).float(), dim=-1)
    xyz = torch.as_tensor(cache["xyz"]).float().contiguous()
    xyz_sha = hashlib.sha256(xyz.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
    if xyz_sha != field_payload["geometry_fingerprint"]["xyz_sha256"]:
        raise ValueError("joint LERF geometry differs")
    reliability = torch.as_tensor(loaded["universal_field"]["reliability"]).float()
    with torch.inference_mode():
        latent = field.query_memory(representation="coefficients").cpu().float().contiguous()
    eq = torch.as_tensor(episodes["episode_query_proposal"]).long()
    et = torch.as_tensor(episodes["episode_target_proposal"]).long()
    ev = torch.as_tensor(episodes["episode_target_view"]).long()
    eo = torch.as_tensor(episodes["episode_object_id"]).long()
    offsets = torch.as_tensor(episodes["negative_proposal_offsets"]).long()
    negative_proposals = torch.as_tensor(episodes["negative_proposals"]).long()
    if not torch.equal(ev, views[et]) or offsets.numel() != eq.numel() + 1:
        raise ValueError("joint LERF compiled episode domain differs")
    negatives: list[torch.Tensor] = []
    for index in range(eq.numel()):
        values: set[int] = set()
        positive = set(hard_support[int(et[index])].tolist())
        for proposal in negative_proposals[offsets[index]:offsets[index + 1]].tolist():
            values.update(set(hard_support[int(proposal)].tolist()) - positive)
        negatives.append(torch.tensor(sorted(values), dtype=torch.long))
    generic_text: dict[str, dict[str, torch.Tensor]] = {}
    for split, record in (text_specs or {}).items():
        authority, records[f"generic_text_{split}"] = _load_mapping(
            record["path"], record["sha256"], f"generic text {split} extent authority",
        )
        if (
            authority.get("metadata", {}).get("benchmark_vocabulary_opened") is not False
            or authority.get("metadata", {}).get("text_bank_split") != split
            or not torch.equal(torch.as_tensor(authority["episode_query_proposal"]).long(), eq)
            or not torch.equal(torch.as_tensor(authority["episode_target_proposal"]).long(), et)
        ):
            raise ValueError("joint LERF generic-text authority differs")
        generic_text[split] = {
            "embedding": torch.as_tensor(authority["selected_text_embedding"]).float(),
            "eligible": torch.as_tensor(authority["eligible"]).bool(),
            "confidence": torch.as_tensor(authority["shared_margin"]).float(),
        }
    return {
        "scene": str(seed["scene"]), "records": records, "field_record": inputs["field"],
        "latent": latent, "reliability": reliability, "xyz": xyz, "baseline": baseline,
        "semantic": semantic, "query": query, "views": views, "observed": torch.as_tensor(membership["view_observed"]).bool(),
        "hard_support": hard_support, "soft_support": soft_support, "soft_values": soft_values,
        "episode_query": eq, "episode_target": et, "episode_view": ev, "episode_object": eo,
        "negative_support": negatives, "generic_text": generic_text,
    }


def _split(data: dict[str, Any], stride: int, validation_residue: int, heldout_residue: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compiled same-object episodes may legitimately be positive-only; missing
    # explicit negatives remain unknown and do not invalidate the episode.
    valid = torch.ones(data["episode_query"].numel(), dtype=torch.bool)
    validation = (data["episode_view"] % stride == validation_residue) & valid
    heldout = (data["episode_view"] % stride == heldout_residue) & valid
    training = (~validation) & (~heldout) & valid
    if int(training.sum()) < 2 or not bool(validation.any()) or not bool(heldout.any()):
        raise ValueError(f"joint LERF split differs for {data['scene']}")
    return training, validation, heldout


def _sample(data: dict[str, Any], sample: int, generator: torch.Generator, positive_cap: int, negative_cap: int) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    query = int(data["episode_query"][sample]); target_proposal = int(data["episode_target"][sample])
    positive = data["soft_support"][target_proposal]; target = data["soft_values"][target_proposal]
    if positive.numel() > positive_cap:
        order = torch.randperm(positive.numel(), generator=generator)[:positive_cap]
        positive, target = positive[order], target[order]
    negative = _sample_without_replacement(data["negative_support"][sample], negative_cap, generator)
    return query, target_proposal, torch.cat((positive, negative)), torch.cat((target, torch.zeros(negative.numel())))


@torch.inference_mode()
def _device_cache(data: dict[str, Any], scene_index: int, canonicalizer: LowRankSceneCanonicalizer, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "latent": canonicalizer(data["latent"].to(device), scene_index),
        "reliability": data["reliability"].to(device), "xyz": data["xyz"].to(device),
        "baseline": data["baseline"].to(device), "semantic": data["semantic"].to(device),
    }


@torch.inference_mode()
def _score(data: dict[str, Any], scene_index: int, sample: int, adapter: ModalityQueryAdapter, decoder: QueryNativeGaussianPosteriorDecoder, canonicalizer: LowRankSceneCanonicalizer, device: torch.device, cache: dict[str, torch.Tensor] | None = None, text_split: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = int(data["episode_query"][sample]); target = int(data["episode_target"][sample])
    visible = torch.where(data["observed"][int(data["views"][target])])[0]
    cached = cache if cache is not None else _device_cache(data, scene_index, canonicalizer, device)
    latent = cached["latent"][visible]
    identity_query = (
        data["generic_text"][text_split]["embedding"][sample].to(device)
        if text_split is not None else cached["semantic"][query]
    )
    query_value = (
        identity_query[None] if text_split is not None
        else data["query"][query:query + 1].to(device)
    )
    prior = cached["baseline"][visible] @ identity_query
    token = adapter(query_value)
    logits, _ = decoder(latent, cached["reliability"][visible], QueryPacket(token, "text" if text_split else "image"), identity_prior=prior, geometry=GaussianGeometry(cached["xyz"][visible]))
    truth = visible_membership_target(visible, data["hard_support"][target], num_rows=data["latent"].shape[0])
    primitive = (cached["baseline"][visible] @ identity_query).cpu()
    return torch.sigmoid(logits).cpu(), primitive, truth


def _global_threshold(per_scene: list[list[tuple[torch.Tensor, torch.Tensor]]], candidates: int) -> tuple[float, float]:
    values = torch.cat([score for scene in per_scene for score, _ in scene])
    thresholds = torch.linspace(float(values.min()), float(values.max()), candidates)
    objectives = []
    for threshold in thresholds:
        scene_ious = [torch.tensor([_iou_from_scores(score, truth, float(threshold)) for score, truth in scene]).mean() for scene in per_scene]
        objectives.append(float(torch.stack(scene_ious).mean()))
    best = int(torch.tensor(objectives).argmax())
    return float(thresholds[best]), objectives[best]


def _calibrate_selective_blend(
    posterior: list[list[tuple[torch.Tensor, torch.Tensor]]],
    primitive: list[list[tuple[torch.Tensor, torch.Tensor]]],
    posterior_threshold: float,
    primitive_threshold: float,
    candidates: int,
    noninferiority_tolerance: float,
) -> dict[str, Any]:
    """Select how strongly a typed posterior may depart from its identity prior."""
    posterior_scale = float(torch.cat([score for scene in posterior for score, _ in scene]).std().clamp_min(1e-6))
    primitive_scale = float(torch.cat([score for scene in primitive for score, _ in scene]).std().clamp_min(1e-6))
    best_key: tuple[float, float, float] | None = None; best: dict[str, Any] | None = None
    for weight in torch.linspace(0.0, 1.0, candidates).tolist():
        scene_ious=[]; primitive_ious=[]
        for posterior_scene, primitive_scene in zip(posterior, primitive):
            blended=[]; baseline=[]
            for (score,truth),(base,base_truth) in zip(posterior_scene,primitive_scene):
                if not torch.equal(truth,base_truth): raise ValueError("selective blend truth domain differs")
                signed = weight * (score-posterior_threshold)/posterior_scale + (1.0-weight) * (base-primitive_threshold)/primitive_scale
                blended.append(_iou_from_scores(signed,truth,0.0)); baseline.append(_iou_from_scores(base,truth,primitive_threshold))
            scene_ious.append(float(torch.tensor(blended).mean())); primitive_ious.append(float(torch.tensor(baseline).mean()))
        deltas=[value-base for value,base in zip(scene_ious,primitive_ious)]
        key=(float(min(deltas)>=-noninferiority_tolerance),sum(deltas)/len(deltas),-weight)
        if best_key is None or key>best_key:
            best_key=key; best={"weight":weight,"posterior_scale":posterior_scale,"primitive_scale":primitive_scale,"scene_deltas":deltas,"scene_macro_delta":sum(deltas)/len(deltas)}
    if best is None: raise RuntimeError("selective text blend calibration failed")
    return best


def _blend_score(score: torch.Tensor, primitive: torch.Tensor, calibration: dict[str, Any], posterior_threshold: float, primitive_threshold: float) -> torch.Tensor:
    weight=float(calibration["weight"])
    return weight*(score-posterior_threshold)/float(calibration["posterior_scale"])+(1.0-weight)*(primitive-primitive_threshold)/float(calibration["primitive_scale"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = json.loads(Path(args.scene_specs).read_text())
    datasets = [_load_scene(x["seed_model"], x["seed_model_sha256"], x["episodes"], x["episodes_sha256"], args.evaluation_membership_threshold, x.get("generic_text")) for x in specs]
    splits = [_split(data, args.holdout_stride, args.validation_residue, args.heldout_residue) for data in datasets]
    latent_dim = datasets[0]["latent"].shape[1]; query_input_dim = datasets[0]["query"].shape[1]
    if any(data["latent"].shape[1] != latent_dim or data["query"].shape[1] != query_input_dim for data in datasets):
        raise ValueError("joint LERF representation dimensions differ")
    device = torch.device(args.device); torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    adapter = (FixedCosineQueryProjection(query_input_dim,args.query_dim,args.projection_seed) if args.fixed_query_projection else ModalityQueryAdapter(query_input_dim, args.query_dim)).to(device)
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=latent_dim, query_dim=args.query_dim,
        hidden_dim=args.hidden_dim, topk_anchors=args.topk_anchors,
    ).to(device)
    canonicalizer = LowRankSceneCanonicalizer(len(datasets), latent_dim, args.scene_canonicalizer_rank).to(device)
    parameters = list(adapter.parameters()) + list(decoder.parameters()) + list(canonicalizer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    object_buckets: list[dict[int, torch.Tensor]] = []
    text_buckets: list[dict[int, torch.Tensor]] = []
    for data, (training, _, _) in zip(datasets, splits):
        buckets = {int(obj): torch.where(training & (data["episode_object"] == obj))[0] for obj in torch.unique(data["episode_object"][training]).tolist()}
        object_buckets.append({key: value for key, value in buckets.items() if value.numel()})
        if data["generic_text"]:
            eligible = data["generic_text"]["fit"]["eligible"]
            text = {int(obj): torch.where(training & eligible & (data["episode_object"] == obj))[0] for obj in torch.unique(data["episode_object"][training & eligible]).tolist()}
            text_buckets.append({key: value for key, value in text.items() if value.numel()})
        else:
            text_buckets.append({})
    best_loss = float("inf"); best_state = None
    best_selection_key: tuple[float, float, float, float] | None = None
    best_selection_report: dict[str, Any] | None = None
    for step in range(args.steps):
        scene_index = step % len(datasets); data = datasets[scene_index]
        use_text = bool(text_buckets[scene_index]) and (step // len(datasets)) % 2 == 1
        active_buckets = text_buckets[scene_index] if use_text else object_buckets[scene_index]
        keys = sorted(active_buckets); object_id = keys[(step // (len(datasets) * 2)) % len(keys)]
        bucket = active_buckets[object_id]
        sample = int(bucket[torch.randint(bucket.numel(), (), generator=generator)])
        query, _, rows, target = _sample(data, sample, generator, args.positive_cap, args.negative_cap)
        local = canonicalizer(data["latent"][rows].to(device), scene_index)
        identity_query = data["generic_text"]["fit"]["embedding"][sample].to(device) if use_text else data["semantic"][query].to(device)
        query_value = identity_query[None] if use_text else data["query"][query:query + 1].to(device)
        prior = data["baseline"][rows].to(device) @ identity_query
        token = adapter(query_value)
        logits, identity = decoder(local, data["reliability"][rows].to(device), QueryPacket(token, "text" if use_text else "image"), identity_prior=prior, geometry=GaussianGeometry(data["xyz"][rows].to(device)))
        target_device = target.to(device); probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target_device)
        dice = 1 - (2 * (probability * target_device).sum() + 1) / (probability.sum() + target_device.sum() + 1)
        brier = F.mse_loss(probability, target_device)
        pos = target_device >= args.evaluation_membership_threshold; neg = target_device == 0
        rank = F.relu(args.identity_margin - identity[pos].mean() + identity[neg].mean()) if bool(pos.any() and neg.any()) else torch.zeros((), device=device)
        alignment = torch.zeros((), device=device)
        if use_text:
            image_token = adapter(data["query"][query:query + 1].to(device))
            alignment = 1.0 - F.cosine_similarity(token, image_token.detach()).mean()
        loss = bce + args.dice_weight * dice + args.brier_weight * brier + args.rank_weight * rank + args.cross_modal_alignment_weight * alignment
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 5); optimizer.step()
        if (step + 1) % args.validation_interval == 0 or step + 1 == args.steps:
            adapter.eval(); decoder.eval(); canonicalizer.eval(); losses = []
            with torch.no_grad():
                for si, (scene, (_, validation, _)) in enumerate(zip(datasets, splits)):
                    local_losses = []
                    for sample_index in torch.where(validation)[0].tolist():
                        q, _, r, t = _sample(scene, sample_index, generator, args.positive_cap, args.negative_cap)
                        lat = canonicalizer(scene["latent"][r].to(device), si)
                        prior = scene["baseline"][r].to(device) @ scene["semantic"][q].to(device)
                        token = adapter(scene["query"][q:q + 1].to(device))
                        logits, _ = decoder(lat, scene["reliability"][r].to(device), QueryPacket(token, "image"), identity_prior=prior, geometry=GaussianGeometry(scene["xyz"][r].to(device)))
                        local_losses.append(F.binary_cross_entropy_with_logits(logits, t.to(device)))
                    losses.append(torch.stack(local_losses).mean())
            validation_loss = float(torch.stack(losses).mean())
            posterior_sets=[]; primitive_sets=[]; text_posterior_sets=[]; text_primitive_sets=[]
            for si,(scene,(_, validation, _)) in enumerate(zip(datasets,splits)):
                scene_cache = _device_cache(scene, si, canonicalizer, device)
                pset=[]; bset=[]
                for sample_index in torch.where(validation)[0].tolist():
                    score, primitive, truth = _score(
                        scene, si, sample_index, adapter, decoder, canonicalizer, device, scene_cache,
                    )
                    pset.append((score,truth)); bset.append((primitive,truth))
                text_pset=[]; text_bset=[]
                if scene["generic_text"]:
                    text_validation = validation & scene["generic_text"]["dev"]["eligible"]
                    for sample_index in torch.where(text_validation)[0].tolist():
                        score, primitive, truth = _score(
                            scene, si, sample_index, adapter, decoder, canonicalizer,
                            device, scene_cache, text_split="dev",
                        )
                        text_pset.append((score,truth)); text_bset.append((primitive,truth))
                posterior_sets.append(pset); primitive_sets.append(bset)
                text_posterior_sets.append(text_pset); text_primitive_sets.append(text_bset)
                del scene_cache
                if device.type == "cuda": torch.cuda.empty_cache()
            selection_threshold, selection_macro = _global_threshold(posterior_sets,args.threshold_candidates)
            selection_primitive_threshold, selection_primitive_macro = _global_threshold(primitive_sets,args.threshold_candidates)
            selection_text_threshold, selection_text_macro = _global_threshold(text_posterior_sets,args.threshold_candidates)
            selection_text_primitive_threshold, selection_text_primitive_macro = _global_threshold(text_primitive_sets,args.threshold_candidates)
            selection_text_blend = _calibrate_selective_blend(
                text_posterior_sets,text_primitive_sets,selection_text_threshold,
                selection_text_primitive_threshold,args.blend_candidates,
                args.scene_noninferiority_tolerance,
            )
            scene_deltas=[]
            text_scene_deltas=[]
            for pset,bset,text_pset,text_bset in zip(posterior_sets,primitive_sets,text_posterior_sets,text_primitive_sets):
                posterior_iou=float(torch.tensor([_iou_from_scores(score,truth,selection_threshold) for score,truth in pset]).mean())
                primitive_iou=float(torch.tensor([_iou_from_scores(score,truth,selection_primitive_threshold) for score,truth in bset]).mean())
                scene_deltas.append(posterior_iou-primitive_iou)
                text_posterior_iou=float(torch.tensor([_iou_from_scores(_blend_score(score,base,selection_text_blend,selection_text_threshold,selection_text_primitive_threshold),truth,0.0) for (score,truth),(base,_) in zip(text_pset,text_bset)]).mean())
                text_primitive_iou=float(torch.tensor([_iou_from_scores(score,truth,selection_text_primitive_threshold) for score,truth in text_bset]).mean())
                text_scene_deltas.append(text_posterior_iou-text_primitive_iou)
            all_deltas=scene_deltas+text_scene_deltas
            selection_key=(float(min(all_deltas)>=-args.scene_noninferiority_tolerance),min(all_deltas),min(selection_macro-selection_primitive_macro,selection_text_macro-selection_text_primitive_macro),-validation_loss)
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key=selection_key
                best_loss = validation_loss
                best_selection_report={"step":step+1,"threshold":selection_threshold,"text_threshold":selection_text_threshold,"text_selective_blend":selection_text_blend,"scene_deltas":scene_deltas,"text_scene_deltas":text_scene_deltas,"scene_macro_delta":selection_macro-selection_primitive_macro,"text_scene_macro_delta":selection_text_blend["scene_macro_delta"],"scene_macro_bce":validation_loss}
                best_state = {"adapter": {k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}, "decoder": {k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}, "canonicalizer": {k:v.detach().cpu().clone() for k,v in canonicalizer.state_dict().items()}}
            adapter.train(); decoder.train(); canonicalizer.train()
    if best_state is None: raise RuntimeError("joint LERF optimization did not complete")
    adapter.load_state_dict(best_state["adapter"]); decoder.load_state_dict(best_state["decoder"]); canonicalizer.load_state_dict(best_state["canonicalizer"])
    adapter.eval(); decoder.eval(); canonicalizer.eval()
    posterior_validation=[]; primitive_validation=[]; text_posterior_validation=[]; text_primitive_validation=[]
    for si,(data,(_,validation,_)) in enumerate(zip(datasets,splits)):
        scene_cache = _device_cache(data, si, canonicalizer, device)
        p=[]; b=[]
        for sample in torch.where(validation)[0].tolist():
            score, primitive, truth = _score(data,si,sample,adapter,decoder,canonicalizer,device,scene_cache); p.append((score,truth)); b.append((primitive,truth))
        text_p=[]; text_b=[]
        if data["generic_text"]:
            text_validation = validation & data["generic_text"]["dev"]["eligible"]
            for sample in torch.where(text_validation)[0].tolist():
                score, primitive, truth = _score(data,si,sample,adapter,decoder,canonicalizer,device,scene_cache,text_split="dev"); text_p.append((score,truth)); text_b.append((primitive,truth))
        posterior_validation.append(p); primitive_validation.append(b)
        text_posterior_validation.append(text_p); text_primitive_validation.append(text_b)
        del scene_cache
        if device.type == "cuda": torch.cuda.empty_cache()
    threshold, validation_iou = _global_threshold(posterior_validation,args.threshold_candidates)
    primitive_threshold, primitive_validation_iou = _global_threshold(primitive_validation,args.threshold_candidates)
    text_threshold, text_validation_iou = _global_threshold(text_posterior_validation,args.threshold_candidates)
    text_primitive_threshold, text_primitive_validation_iou = _global_threshold(text_primitive_validation,args.threshold_candidates)
    text_selective_blend = _calibrate_selective_blend(
        text_posterior_validation,text_primitive_validation,text_threshold,
        text_primitive_threshold,args.blend_candidates,args.scene_noninferiority_tolerance,
    )
    results={}; scene_pass=[]; text_audit_results={}; text_audit_pass=[]
    for si,(data,(_,_,heldout)) in enumerate(zip(datasets,splits)):
        scene_cache = _device_cache(data, si, canonicalizer, device)
        posterior=[]; primitive=[]; briers=[]; purities=[]; coverages=[]
        for sample in torch.where(heldout)[0].tolist():
            score, base, truth = _score(data,si,sample,adapter,decoder,canonicalizer,device,scene_cache)
            posterior.append(_iou_from_scores(score,truth,threshold)); primitive.append(_iou_from_scores(base,truth,primitive_threshold))
            briers.append(float(F.mse_loss(score,truth)))
            predicted=score>=threshold; positive=truth>0.5
            purities.append(float((positive & predicted).sum()/predicted.sum().clamp_min(1)))
            coverages.append(float((positive & predicted).sum()/positive.sum().clamp_min(1)))
        post=float(torch.tensor(posterior).mean()); base=float(torch.tensor(primitive).mean())
        passed=post >= base-args.scene_noninferiority_tolerance
        scene_pass.append(passed); results[data["scene"]]={"episodes":len(posterior),"primitive_iou":base,"posterior_iou":post,"delta":post-base,"brier":sum(briers)/len(briers),"purity":sum(purities)/len(purities),"coverage":sum(coverages)/len(coverages),"noninferior":passed}
        if data["generic_text"]:
            text_audit = heldout & data["generic_text"]["audit"]["eligible"]
            text_posterior=[]; text_primitive=[]
            for sample in torch.where(text_audit)[0].tolist():
                score, primitive_score, truth = _score(data,si,sample,adapter,decoder,canonicalizer,device,scene_cache,text_split="audit")
                text_posterior.append(_iou_from_scores(_blend_score(score,primitive_score,text_selective_blend,text_threshold,text_primitive_threshold),truth,0.0)); text_primitive.append(_iou_from_scores(primitive_score,truth,text_primitive_threshold))
            if not text_posterior: raise ValueError(f"{data['scene']}: generic-text audit split is empty")
            text_post=float(torch.tensor(text_posterior).mean()); text_base=float(torch.tensor(text_primitive).mean()); text_ok=text_post>=text_base-args.scene_noninferiority_tolerance
            text_audit_pass.append(text_ok); text_audit_results[data["scene"]]={"episodes":len(text_posterior),"primitive_iou":text_base,"posterior_iou":text_post,"delta":text_post-text_base,"noninferior":text_ok}
        del scene_cache
        if device.type == "cuda": torch.cuda.empty_cache()
    macro_post=sum(x["posterior_iou"] for x in results.values())/len(results); macro_base=sum(x["primitive_iou"] for x in results.values())/len(results)
    text_macro_delta=(sum(x["delta"] for x in text_audit_results.values())/len(text_audit_results)) if text_audit_results else 0.0
    passed=all(scene_pass) and macro_post>=macro_base+args.minimum_macro_gain and (not text_audit_results or (all(text_audit_pass) and text_macro_delta>=args.minimum_text_macro_gain))
    output=Path(args.output).expanduser().resolve()
    write_torch_noclobber(output,{"schema":"radio_gs.lerf_joint_query_native_extent.v1","schema_version":1,"adapter_state_dict":best_state["adapter"],"decoder_state_dict":best_state["decoder"],"scene_canonicalizer_state_dict":best_state["canonicalizer"],"threshold":threshold,"text_threshold":text_threshold,"text_primitive_threshold":text_primitive_threshold,"text_selective_blend":text_selective_blend,"metadata":{"source_only":True,"field_frozen":True,"per_gaussian_parameters_added":False,"memory_representation":"coefficients","shared_cross_scene_decoder":True,"scene_canonicalizer_rank":args.scene_canonicalizer_rank,"fixed_query_projection":bool(args.fixed_query_projection),"projection_seed":args.projection_seed,"generic_text_extent_authority":bool(text_audit_results),"generic_text_bank_protocol":"fit_train_dev_checkpoint_audit_gate","typed_posterior_calibration":"image_text_disjoint_thresholds_and_dev_selective_blend","scene_balanced":True,"object_track_balanced":True,"unknown_excluded_from_loss":True,"scene_specs":specs}})
    report={"status":"source_gate_pass" if passed else "source_gate_fail","best_validation_scene_macro_bce":best_loss,"checkpoint_selection":best_selection_report,"validation":{"posterior_threshold":threshold,"posterior_scene_macro_iou":validation_iou,"primitive_threshold":primitive_threshold,"primitive_scene_macro_iou":primitive_validation_iou,"text_posterior_threshold":text_threshold,"text_posterior_scene_macro_iou":text_validation_iou,"text_primitive_threshold":text_primitive_threshold,"text_primitive_scene_macro_iou":text_primitive_validation_iou,"text_selective_blend":text_selective_blend},"source_heldout":results,"source_heldout_scene_macro":{"primitive_iou":macro_base,"posterior_iou":macro_post,"delta":macro_post-macro_base},"generic_text_audit":text_audit_results,"generic_text_audit_scene_macro_delta":text_macro_delta,"gate":{"all_scenes_noninferior":all(scene_pass),"generic_text_all_scenes_noninferior":all(text_audit_pass) if text_audit_results else None,"minimum_macro_gain":args.minimum_macro_gain,"minimum_text_macro_gain":args.minimum_text_macro_gain,"passed":passed},"output":file_record(output)}
    write_frozen_json(output.with_suffix(output.suffix+".json"),report); return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--scene-specs",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=3600); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--query-dim",type=int,default=128); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--topk-anchors",type=int,default=6); p.add_argument("--scene-canonicalizer-rank",type=int,default=8)
    p.add_argument("--positive-cap",type=int,default=1024); p.add_argument("--negative-cap",type=int,default=2048); p.add_argument("--holdout-stride",type=int,default=4); p.add_argument("--heldout-residue",type=int,default=3); p.add_argument("--validation-residue",type=int,default=2); p.add_argument("--validation-interval",type=int,default=60); p.add_argument("--threshold-candidates",type=int,default=64); p.add_argument("--blend-candidates",type=int,default=21)
    p.add_argument("--fixed-query-projection",action="store_true"); p.add_argument("--projection-seed",type=int,default=20260824); p.add_argument("--cross-modal-alignment-weight",type=float,default=.25); p.add_argument("--evaluation-membership-threshold",type=float,default=.5); p.add_argument("--dice-weight",type=float,default=.5); p.add_argument("--brier-weight",type=float,default=.25); p.add_argument("--rank-weight",type=float,default=.25); p.add_argument("--identity-margin",type=float,default=.05); p.add_argument("--scene-noninferiority-tolerance",type=float,default=.0); p.add_argument("--minimum-macro-gain",type=float,default=.01); p.add_argument("--minimum-text-macro-gain",type=float,default=.01); p.add_argument("--seed",type=int,default=20260824)
    print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))


if __name__=="__main__": main()
