#!/usr/bin/env python3
"""Train token-free shared LERF extent and run oracle/text AnchorPacket diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.models.query_native_gaussian_memory import AnchorConditionedExtentDecoder, AnchorPacket, GaussianGeometry, LowRankSceneCanonicalizer
from radio_gs.querying.object_track_extent_authority import compile_object_track_extent_authority
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _iou_from_scores, _load_mapping, _sample_without_replacement, visible_membership_target
from radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder import _global_threshold, _load_scene, _split
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def _radius_key(radius: float) -> str: return str(float(radius)).replace(".","p")


def _load(args: argparse.Namespace) -> tuple[list[dict[str,Any]],list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],list[dict[str,Any]]]:
    specs=json.loads(Path(args.scene_specs).read_text()); datasets=[]; records=[]
    for spec in specs:
        text_authority=spec.get("instance_text",spec.get("generic_text"))
        cache_path=spec.get("instance_anchor_cache",spec["anchor_cache"]);cache_sha=spec.get("instance_anchor_cache_sha256",spec["anchor_cache_sha256"])
        data=_load_scene(spec["seed_model"],spec["seed_model_sha256"],spec["episodes"],spec["episodes_sha256"],args.evaluation_membership_threshold,text_authority)
        cache,record=_load_mapping(cache_path,cache_sha,"peak-local AnchorPacket cache")
        if cache.get("scene")!=data["scene"] or not torch.equal(torch.as_tensor(cache["episode_query_proposal"]).long(),data["episode_query"]): raise ValueError("AnchorPacket cache episode domain differs")
        data["anchor_cache"]=cache
        data["track_authority"]=compile_object_track_extent_authority(
            data["episode_object"],data["episode_query"],data["episode_target"],
            data["soft_support"],data["soft_values"],data["negative_support"],
        )
        datasets.append(data);records.append({"scene":data["scene"],"anchor_cache":record,"text_authority":"instance_text" if spec.get("instance_text") else "generic_text"})
    splits=[_track_split(data) if getattr(args,"track_complete",False) else _split(data,args.holdout_stride,args.validation_residue,args.heldout_residue) for data in datasets]
    return datasets,splits,records


def _track_split(data: dict[str,Any]) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Hold out complete physical tracks instead of leaking tracks across views."""
    objects=torch.unique(data["episode_object"],sorted=True)
    if objects.numel()<3: raise ValueError(f"track-complete split needs three objects for {data['scene']}")
    assignment={int(value):position%3 for position,value in enumerate(objects.tolist())}
    residue=torch.tensor([assignment[int(value)] for value in data["episode_object"].tolist()])
    return residue==0,residue==1,residue==2


def _cached_anchor(data: dict[str,Any], sample: int, modality: str, split: str | None, radius: float) -> tuple[torch.Tensor,int,float]:
    suffix=_radius_key(radius);prefix="image" if modality=="image" else f"text_{split}"
    return (torch.as_tensor(data["anchor_cache"][f"{prefix}_anchor_rows_r{suffix}"][sample]).long(),int(data["anchor_cache"][f"{prefix}_peak_rows_r{suffix}"][sample]),float(data["anchor_cache"][f"{prefix}_local_radius_r{suffix}"][sample]))


def _identity_query(data: dict[str,Any], sample: int, modality: str, split: str | None) -> torch.Tensor:
    return data["semantic"][int(data["episode_query"][sample])] if modality=="image" else data["generic_text"][str(split)]["embedding"][sample]


def _batch(data: dict[str,Any], sample: int, modality: str, split: str | None, radius: float, generator: torch.Generator, positive_cap: int, negative_cap: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,int,float]:
    if data.get("track_complete",False):
        authority=data["track_authority"][int(data["episode_object"][sample])]
        positive=authority.positive_rows;positive_value=authority.positive_probability
        negative=authority.explicit_negative_rows
    else:
        target_proposal=int(data["episode_target"][sample]);positive=data["soft_support"][target_proposal];positive_value=data["soft_values"][target_proposal]
        negative=data["negative_support"][sample]
    if positive.numel()>positive_cap:
        order=torch.randperm(positive.numel(),generator=generator)[:positive_cap];positive=positive[order];positive_value=positive_value[order]
    negative=_sample_without_replacement(negative,negative_cap,generator)
    anchors,peak,local_radius=_cached_anchor(data,sample,modality,split,radius); rows=torch.unique(torch.cat((positive,negative,anchors,torch.tensor([peak]))),sorted=True)
    target=torch.zeros(rows.numel());known=torch.zeros(rows.numel(),dtype=torch.bool)
    positive_position=torch.searchsorted(rows,positive);target[positive_position]=positive_value;known[positive_position]=True
    if negative.numel(): known[torch.searchsorted(rows,negative)]=True
    anchor_position=torch.searchsorted(rows,anchors);peak_position=int(torch.searchsorted(rows,torch.tensor(peak)))
    return rows,target,known,anchor_position,peak_position,local_radius


def _authority(xyz: torch.Tensor, peak: int, local_radius: float, multiplier: float) -> torch.Tensor:
    return (torch.linalg.vector_norm(xyz-xyz[peak],dim=1)<=float(local_radius)*float(multiplier)).float()


def _calibrated_identity(identity: torch.Tensor,args: argparse.Namespace) -> torch.Tensor:
    """Put cosine retrieval into a decision-preserving logit gauge."""
    if not args.decision_calibrated_identity: return identity
    if args.identity_scale<=0: raise ValueError("identity calibration scale must be positive")
    return (identity-float(args.identity_center))/float(args.identity_scale)


def _forward_batch(data: dict[str,Any], scene_index: int, sample: int, modality: str, split: str | None, radius: float, decoder: AnchorConditionedExtentDecoder, canonicalizer: LowRankSceneCanonicalizer, device: torch.device, generator: torch.Generator, args: argparse.Namespace) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,bool]:
    rows,target,known,anchor_position,peak_position,local_radius=_batch(data,sample,modality,split,radius,generator,args.positive_cap,args.negative_cap)
    latent=canonicalizer(data["latent"][rows].to(device),scene_index);reliability=data["reliability"][rows].to(device);xyz=data["xyz"][rows].to(device)
    identity=_calibrated_identity(data["baseline"][rows].to(device)@_identity_query(data,sample,modality,split).to(device),args)
    packet=AnchorPacket(anchor_position.to(device),identity[anchor_position].detach(),peak_position,local_radius)
    logits=decoder(latent,reliability,identity,packet,GaussianGeometry(xyz),_authority(xyz,peak_position,local_radius,args.authority_radius_multiplier))
    if data.get("track_complete",False): has_negative=bool(data["track_authority"][int(data["episode_object"][sample])].explicit_negative_rows.numel())
    else: has_negative=bool(data["negative_support"][sample].numel())
    return logits[known.to(device)],target[known].to(device),identity[known.to(device)],has_negative


@torch.inference_mode()
def _score(data: dict[str,Any], scene_index: int, sample: int, modality: str, split: str | None, anchor_modality: str, radius: float, decoder: AnchorConditionedExtentDecoder, canonicalizer: LowRankSceneCanonicalizer, device: torch.device, args: argparse.Namespace, cache: dict[str,torch.Tensor] | None = None) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,float]:
    target=int(data["episode_target"][sample]);visible=torch.where(data["observed"][int(data["views"][target])])[0]
    if data.get("track_complete",False): visible=torch.where(data["observed"].any(0))[0]
    anchors_global,peak_global,local_radius=_cached_anchor(data,sample,anchor_modality,split if anchor_modality=="text" else None,radius)
    if data.get("track_complete",False):
        query=_identity_query(data,sample,modality,split).to(device);scores=[];bases=[];outside_sum=0.0;outside_count=0
        for begin in range(0,visible.numel(),args.evaluation_chunk_rows):
            output_rows=visible[begin:begin+args.evaluation_chunk_rows]
            rows=torch.unique(torch.cat((output_rows,anchors_global,torch.tensor([peak_global]))),sorted=True)
            latent=canonicalizer(data["latent"][rows].to(device),scene_index);reliability=data["reliability"][rows].to(device);xyz=data["xyz"][rows].to(device);baseline=data["baseline"][rows].to(device)
            identity=_calibrated_identity(baseline@query,args);anchor_position=torch.searchsorted(rows,anchors_global).to(device);peak_position=int(torch.searchsorted(rows,torch.tensor(peak_global)))
            packet=AnchorPacket(anchor_position,identity[anchor_position].detach(),peak_position,local_radius);authority=_authority(xyz,peak_position,local_radius,args.authority_radius_multiplier)
            logits=decoder(latent,reliability,identity,packet,GaussianGeometry(xyz),authority);positions=torch.searchsorted(rows,output_rows).to(device)
            scores.append(torch.sigmoid(logits[positions]).cpu());bases.append(identity[positions].cpu());outside_sum+=float(torch.sigmoid(logits[positions])[authority[positions]==0].sum());outside_count+=int((authority[positions]==0).sum())
        track=data["track_authority"][int(data["episode_object"][sample])];support=track.positive_rows[track.positive_probability>=args.evaluation_membership_threshold]
        truth=visible_membership_target(visible,support,num_rows=data["latent"].shape[0])
        return torch.cat(scores),torch.cat(bases),truth,outside_sum/max(1,outside_count)
    local=visible.to(device)
    if cache is None:
        latent=canonicalizer(data["latent"][visible].to(device),scene_index);reliability=data["reliability"][visible].to(device);xyz=data["xyz"][visible].to(device);baseline=data["baseline"][visible].to(device)
    else:
        latent=cache["latent"][local];reliability=cache["reliability"][local];xyz=cache["xyz"][local];baseline=cache["baseline"][local]
    identity=_calibrated_identity(baseline@_identity_query(data,sample,modality,split).to(device),args)
    anchors=torch.searchsorted(visible,anchors_global);peak=int(torch.searchsorted(visible,torch.tensor(peak_global)))
    if bool((anchors>=visible.numel()).any()) or not torch.equal(visible[anchors],anchors_global) or int(visible[peak])!=peak_global: raise ValueError("AnchorPacket visible domain differs")
    packet=AnchorPacket(anchors.to(device),identity[anchors.to(device)].detach(),peak,local_radius);authority=_authority(xyz,peak,local_radius,args.authority_radius_multiplier)
    logits=decoder(latent,reliability,identity,packet,GaussianGeometry(xyz),authority)
    if data.get("track_complete",False):
        track=data["track_authority"][int(data["episode_object"][sample])]
        support=track.positive_rows[track.positive_probability>=args.evaluation_membership_threshold]
    else: support=data["hard_support"][target]
    truth=visible_membership_target(visible,support,num_rows=data["latent"].shape[0])
    outside=float(torch.sigmoid(logits)[authority==0].mean()) if bool((authority==0).any()) else 0.0
    return torch.sigmoid(logits).cpu(),identity.cpu(),truth,outside


@torch.inference_mode()
def _sets(datasets,splits,which,modality,split,anchor_modality,radius,decoder,canonicalizer,device,args):
    result=[];primitive=[];outside=[]
    for si,(data,parts) in enumerate(zip(datasets,splits)):
        cache=None if data.get("track_complete",False) else {"latent":canonicalizer(data["latent"].to(device),si),"reliability":data["reliability"].to(device),"xyz":data["xyz"].to(device),"baseline":data["baseline"].to(device)}
        indices=parts[which]
        if modality=="text": indices=indices&data["generic_text"][split]["eligible"]
        if data.get("track_complete",False) and args.evaluation_queries_per_track>0:
            retained=torch.zeros_like(indices)
            for object_id in torch.unique(data["episode_object"][indices],sorted=True).tolist():
                candidates=torch.where(indices&(data["episode_object"]==int(object_id)))[0]
                retained[candidates[:args.evaluation_queries_per_track]]=True
            indices=retained
        scene=[];primitive_scene=[]
        for sample in torch.where(indices)[0].tolist():
            score,base,truth,mass=_score(data,si,sample,modality,split,anchor_modality,radius,decoder,canonicalizer,device,args,cache);scene.append((score,truth));primitive_scene.append((base,truth));outside.append(mass)
        result.append(scene);primitive.append(primitive_scene)
        del cache
        if device.type=="cuda": torch.cuda.empty_cache()
    return result,primitive,outside


def _evaluate(candidate,baseline,threshold,baseline_threshold):
    scenes={};passes=[]
    for candidate_scene,baseline_scene in zip(candidate,baseline):
        c=float(torch.tensor([_iou_from_scores(score,truth,threshold) for score,truth in candidate_scene]).mean());b=float(torch.tensor([_iou_from_scores(score,truth,baseline_threshold) for score,truth in baseline_scene]).mean());scenes[len(scenes)]={"posterior_iou":c,"primitive_iou":b,"delta":c-b};passes.append(c>=b)
    return scenes,sum(value["delta"] for value in scenes.values())/len(scenes),all(passes)


def run(args: argparse.Namespace) -> dict[str,Any]:
    datasets,splits,records=_load(args);device=torch.device(args.device);torch.manual_seed(args.seed);generator=torch.Generator().manual_seed(args.seed+1)
    for data in datasets: data["track_complete"]=bool(args.track_complete)
    latent_dim=datasets[0]["latent"].shape[1];decoder=AnchorConditionedExtentDecoder(latent_dim=latent_dim,key_dim=args.key_dim,hidden_dim=args.hidden_dim,gauge_normalize_identity=args.gauge_normalized_identity,use_identity_conditioning=not args.extent_without_identity_map).to(device);canonicalizer=LowRankSceneCanonicalizer(len(datasets),latent_dim,args.scene_canonicalizer_rank).to(device)
    parameters=list(decoder.parameters())+list(canonicalizer.parameters());optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=args.weight_decay)
    buckets=[]
    text_buckets=[]
    for data,(training,_,_) in zip(datasets,splits):
        local={int(obj):torch.where(training&(data["episode_object"]==obj))[0] for obj in torch.unique(data["episode_object"][training]).tolist()};buckets.append({k:v for k,v in local.items() if v.numel()})
        eligible=training&data["generic_text"]["fit"]["eligible"]
        text_local={int(obj):torch.where(eligible&(data["episode_object"]==obj))[0] for obj in torch.unique(data["episode_object"][eligible]).tolist()};text_buckets.append({k:v for k,v in text_local.items() if v.numel()})
    best_key=None;best_state=None;best_selection=None;history=[]
    for step in range(args.steps):
        si=step%len(datasets);keys=sorted(buckets[si]);obj=keys[(step//len(datasets))%len(keys)];pool=buckets[si][obj];sample=int(pool[torch.randint(pool.numel(),(),generator=generator)])
        logits,target,identity,has_negative=_forward_batch(datasets[si],si,sample,"image",None,args.radius_fraction,decoder,canonicalizer,device,generator,args)
        probability=torch.sigmoid(logits);bce=F.binary_cross_entropy_with_logits(logits,target);dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1);loss=(1.0 if has_negative else args.positive_only_weight)*(bce+args.dice_weight*dice+args.brier_weight*F.mse_loss(probability,target))
        if args.text_training_weight>0 and text_buckets[si]:
            text_keys=sorted(text_buckets[si]);text_object=text_keys[(step//len(datasets))%len(text_keys)];text_pool=text_buckets[si][text_object];text_sample=int(text_pool[torch.randint(text_pool.numel(),(),generator=generator)])
            text_logits,text_target,_,text_has_negative=_forward_batch(datasets[si],si,text_sample,"text","fit",args.radius_fraction,decoder,canonicalizer,device,generator,args)
            text_probability=torch.sigmoid(text_logits);text_bce=F.binary_cross_entropy_with_logits(text_logits,text_target);text_dice=1-(2*(text_probability*text_target).sum()+1)/(text_probability.sum()+text_target.sum()+1);text_loss=(1.0 if text_has_negative else args.positive_only_weight)*(text_bce+args.dice_weight*text_dice+args.brier_weight*F.mse_loss(text_probability,text_target));loss=loss+args.text_training_weight*text_loss
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5);optimizer.step()
        if (step+1)%args.validation_interval==0 or step+1==args.steps:
            validation=[]
            with torch.no_grad():
                for sj,(data,(_,valid,_)) in enumerate(zip(datasets,splits)):
                    local=[]
                    for index in torch.where(valid)[0].tolist():
                        l,t,_,_= _forward_batch(data,sj,index,"image",None,args.radius_fraction,decoder,canonicalizer,device,generator,args);local.append(F.binary_cross_entropy_with_logits(l,t))
                    validation.append(torch.stack(local).mean())
            value=float(torch.stack(validation).mean())
            selection,selection_base,_=_sets(datasets,splits,1,"image",None,"image",args.radius_fraction,decoder,canonicalizer,device,args)
            if args.fixed_decision_thresholds:
                selection_threshold,selection_base_threshold=.5,0.0
                selection_scenes,selection_delta,selection_noninferior=_evaluate(selection,selection_base,selection_threshold,selection_base_threshold)
                selection_macro=sum(value["posterior_iou"] for value in selection_scenes.values())/len(selection_scenes);selection_base_macro=sum(value["primitive_iou"] for value in selection_scenes.values())/len(selection_scenes)
            else:
                selection_threshold,selection_macro=_global_threshold(selection,args.threshold_candidates);selection_base_threshold,selection_base_macro=_global_threshold(selection_base,args.threshold_candidates)
                selection_scenes,selection_delta,selection_noninferior=_evaluate(selection,selection_base,selection_threshold,selection_base_threshold)
            minimum_scene_delta=min(x["delta"] for x in selection_scenes.values())
            key=(float(selection_noninferior),minimum_scene_delta,selection_delta,-value);history.append({"step":step+1,"validation_bce":value,"scene_macro_delta":selection_delta,"minimum_scene_delta":minimum_scene_delta,"all_scenes_noninferior":selection_noninferior})
            if best_key is None or key>best_key:
                best_key=key;best_selection={"step":step+1,"scene_macro_delta":selection_delta,"minimum_scene_delta":minimum_scene_delta,"all_scenes_noninferior":selection_noninferior,"threshold":selection_threshold,"primitive_threshold":selection_base_threshold,"posterior_macro_iou":selection_macro,"primitive_macro_iou":selection_base_macro,"validation_bce":value};best_state={"decoder":{k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()},"canonicalizer":{k:v.detach().cpu().clone() for k,v in canonicalizer.state_dict().items()}}
    if best_state is None: raise RuntimeError("anchor-conditioned training did not complete")
    decoder.load_state_dict(best_state["decoder"]);canonicalizer.load_state_dict(best_state["canonicalizer"]);decoder.eval();canonicalizer.eval()
    image_threshold=float(best_selection["threshold"]);image_base_threshold=float(best_selection["primitive_threshold"]);image_dev_iou=float(best_selection["posterior_macro_iou"]);image_base_iou=float(best_selection["primitive_macro_iou"])
    text_modes={};text_dev_thresholds={}
    if not args.skip_text_diagnostics:
        for mode in ("oracle_image_anchor","text_local_anchor"):
            anchor_modality="image" if mode.startswith("oracle") else "text";dev,base,_=_sets(datasets,splits,1,"text","dev",anchor_modality,args.radius_fraction,decoder,canonicalizer,device,args);text_dev_thresholds[mode]=(_global_threshold(dev,args.threshold_candidates)[0],_global_threshold(base,args.threshold_candidates)[0])
    image_audit,image_audit_base,image_out=_sets(datasets,splits,2,"image",None,"image",args.radius_fraction,decoder,canonicalizer,device,args);image_result,image_delta,image_pass=_evaluate(image_audit,image_audit_base,image_threshold,image_base_threshold)
    for mode,(threshold,base_threshold) in text_dev_thresholds.items():
        anchor_modality="image" if mode.startswith("oracle") else "text";audit,base,outside=_sets(datasets,splits,2,"text","audit",anchor_modality,args.radius_fraction,decoder,canonicalizer,device,args);values,delta,passed=_evaluate(audit,base,threshold,base_threshold);text_modes[mode]={"scenes":values,"macro_delta":delta,"all_scenes_noninferior":passed,"outside_authority_probability":sum(outside)/len(outside)}
    passed=bool(best_selection["all_scenes_noninferior"]) and image_pass and image_delta>=args.minimum_macro_gain
    output=Path(args.output).resolve();write_torch_noclobber(output,{"schema":"radio_gs.anchor_conditioned_lerf_extent.v1","schema_version":1,"decoder_state_dict":best_state["decoder"],"scene_canonicalizer_state_dict":best_state["canonicalizer"],"image_threshold":image_threshold,"image_primitive_threshold":image_base_threshold,"metadata":{"token_free_extent":True,"anchor_packet_interface":True,"peak_local":True,"authority_gated_replay":True,"decision_calibrated_identity":bool(args.decision_calibrated_identity),"fixed_decision_thresholds":bool(args.fixed_decision_thresholds),"identity_center":float(args.identity_center),"identity_scale":float(args.identity_scale),"track_complete_supervision":bool(args.track_complete),"track_disjoint_split":bool(args.track_complete),"gauge_normalized_identity":bool(args.gauge_normalized_identity),"extent_uses_row_identity_map":not bool(args.extent_without_identity_map),"text_training_weight":float(args.text_training_weight),"positive_only_weight":args.positive_only_weight,"source_only":True,"field_frozen":True,"scene_records":records}})
    report={"status":"source_image_gate_pass" if passed else "source_image_gate_fail","checkpoint_selection":best_selection,"image_audit":{"scenes":image_result,"macro_delta":image_delta,"all_scenes_noninferior":image_pass,"outside_authority_probability":sum(image_out)/len(image_out)},"text_anchor_diagnostics":text_modes,"validation":{"image_posterior_iou":image_dev_iou,"image_primitive_iou":image_base_iou},"history":history,"output":file_record(output)};write_frozen_json(output.with_suffix(output.suffix+".json"),report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene-specs",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--steps",type=int,default=1200);p.add_argument("--validation-interval",type=int,default=120);p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--key-dim",type=int,default=128);p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--scene-canonicalizer-rank",type=int,default=4);p.add_argument("--radius-fraction",type=float,default=.02);p.add_argument("--authority-radius-multiplier",type=float,default=4.0);p.add_argument("--decision-calibrated-identity",action="store_true");p.add_argument("--fixed-decision-thresholds",action="store_true");p.add_argument("--identity-center",type=float,default=.6685559153556824);p.add_argument("--identity-scale",type=float,default=.05);p.add_argument("--positive-cap",type=int,default=1024);p.add_argument("--negative-cap",type=int,default=2048);p.add_argument("--positive-only-weight",type=float,default=.5);p.add_argument("--text-training-weight",type=float,default=0.0);p.add_argument("--dice-weight",type=float,default=.5);p.add_argument("--brier-weight",type=float,default=.25);p.add_argument("--threshold-candidates",type=int,default=32);p.add_argument("--minimum-macro-gain",type=float,default=.01);p.add_argument("--evaluation-membership-threshold",type=float,default=.5);p.add_argument("--holdout-stride",type=int,default=4);p.add_argument("--heldout-residue",type=int,default=3);p.add_argument("--validation-residue",type=int,default=2);p.add_argument("--track-complete",action="store_true");p.add_argument("--gauge-normalized-identity",action="store_true");p.add_argument("--extent-without-identity-map",action="store_true");p.add_argument("--skip-text-diagnostics",action="store_true");p.add_argument("--evaluation-queries-per-track",type=int,default=2);p.add_argument("--evaluation-chunk-rows",type=int,default=32768);p.add_argument("--seed",type=int,default=20260825);print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))
if __name__=="__main__":main()
