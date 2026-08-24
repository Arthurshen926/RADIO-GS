#!/usr/bin/env python3
"""Train source-only text retrieval to match image anchors, never target masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.models.query_native_gaussian_memory import (
    AnchorConditionedExtentDecoder,
    GaussianGeometry,
    LowRankSceneCanonicalizer,
    TextAnchorIdentityAdapter,
    compile_peak_local_anchor_packet,
)
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _iou_from_scores,
    _load_mapping,
    visible_membership_target,
)
from radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder import (
    _global_threshold,
    _load_scene,
    _split,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def _load(args: argparse.Namespace):
    specs=json.loads(Path(args.scene_specs).read_text());datasets=[];records=[]
    for spec in specs:
        text=spec.get("instance_text")
        if not text: raise ValueError("instance text authority is required")
        data=_load_scene(spec["seed_model"],spec["seed_model_sha256"],spec["episodes"],spec["episodes_sha256"],args.evaluation_membership_threshold,text)
        cache,cache_record=_load_mapping(spec["instance_anchor_cache"],spec["instance_anchor_cache_sha256"],"instance AnchorPacket cache")
        data["anchor_cache"]=cache;datasets.append(data);records.append({"scene":data["scene"],"anchor_cache":cache_record})
    checkpoint,checkpoint_record=_load_mapping(args.extent_checkpoint,args.expected_extent_sha256,"frozen anchor-conditioned extent")
    return datasets,[_split(x,args.holdout_stride,args.validation_residue,args.heldout_residue) for x in datasets],records,checkpoint,checkpoint_record


def _training_rows(data:dict[str,Any],sample:int,args:argparse.Namespace,generator:torch.Generator)->torch.Tensor:
    target=int(data["episode_target"][sample]);view=int(data["views"][target]);visible=torch.where(data["observed"][view])[0]
    positive=data["hard_support"][target];positive=positive[torch.isin(positive,visible)]
    negative=data["negative_support"][sample];negative=negative[torch.isin(negative,visible)]
    suffix=str(float(args.radius_fraction)).replace(".","p")
    image=torch.as_tensor(data["anchor_cache"][f"image_anchor_rows_r{suffix}"][sample]).long()
    text=torch.as_tensor(data["anchor_cache"][f"text_fit_anchor_rows_r{suffix}"][sample]).long()
    count=min(args.random_rows,visible.numel());random=visible[torch.randperm(visible.numel(),generator=generator)[:count]]
    return torch.unique(torch.cat((positive,negative,image,text,random)),sorted=True)


def _anchor_loss(data:dict[str,Any],sample:int,adapter:TextAnchorIdentityAdapter,device:torch.device,args:argparse.Namespace,generator:torch.Generator)->torch.Tensor:
    rows=_training_rows(data,sample,args,generator);baseline=data["baseline"][rows].to(device)
    image_query=data["semantic"][int(data["episode_query"][sample])].to(device)
    text_query=data["generic_text"]["fit"]["embedding"][sample].to(device)
    teacher=(baseline@F.normalize(image_query,dim=0)).detach();student=baseline@adapter(text_query[None])[0]
    temperature=float(args.distribution_temperature)
    distribution=F.kl_div(F.log_softmax(student/temperature,dim=0),F.softmax(teacher/temperature,dim=0),reduction="batchmean")*(temperature**2)*rows.numel()
    peak=F.cross_entropy((student/temperature)[None],teacher.argmax()[None])
    target=int(data["episode_target"][sample]);positive=torch.isin(rows,data["hard_support"][target]);negative=torch.isin(rows,data["negative_support"][sample])
    ranking=student.new_zeros(())
    if bool(positive.any()) and bool(negative.any()):
        ranking=F.softplus((student[negative].max()-student[positive].topk(min(args.positive_topk,int(positive.sum()))).values.mean()+args.ranking_margin)/temperature)*temperature
    return distribution+args.peak_weight*peak+args.ranking_weight*ranking


@torch.inference_mode()
def _score(data,scene_index,sample,text_split,adapter,decoder,canonicalizer,device,args,cache):
    target=int(data["episode_target"][sample]);visible=torch.where(data["observed"][int(data["views"][target])])[0];local=visible.to(device)
    raw_query=F.normalize(data["generic_text"][text_split]["embedding"][sample].to(device),dim=0);adapted_query=adapter(raw_query[None])[0]
    identity=cache["baseline"][local]@raw_query;retrieval_identity=cache["baseline"][local]@adapted_query
    xyz=cache["xyz"][local];packet=compile_peak_local_anchor_packet(retrieval_identity,xyz,args.topk,args.radius_fraction);raw_packet=compile_peak_local_anchor_packet(identity,xyz,args.topk,args.radius_fraction);fallback=False
    if args.anchor_agreement_multiplier>0 and float(torch.linalg.vector_norm(xyz[packet.peak_row]-xyz[raw_packet.peak_row]))>raw_packet.local_radius*args.anchor_agreement_multiplier: packet=raw_packet;fallback=True
    authority=(torch.linalg.vector_norm(xyz-xyz[packet.peak_row],dim=1)<=packet.local_radius*args.authority_radius_multiplier).float()
    logits=decoder(cache["latent"][local],cache["reliability"][local],identity,packet,GaussianGeometry(xyz),authority)
    truth=visible_membership_target(visible,data["hard_support"][target],num_rows=data["latent"].shape[0])
    return torch.sigmoid(logits).cpu(),identity.cpu(),truth,float(truth[packet.rows.cpu()].float().mean()),bool(truth[packet.peak_row]),fallback


def _sets(datasets,splits,which,split,adapter,decoder,canonicalizer,device,args):
    posterior=[];primitive=[];purity=[];peaks=[];fallbacks=[]
    for si,(data,parts) in enumerate(zip(datasets,splits)):
        cache={"latent":canonicalizer(data["latent"].to(device),si),"reliability":data["reliability"].to(device),"xyz":data["xyz"].to(device),"baseline":data["baseline"].to(device)}
        selected=parts[which]&data["generic_text"][split]["eligible"];local=[];base=[]
        for sample in torch.where(selected)[0].tolist():
            score,identity,truth,pure,peak,fallback=_score(data,si,sample,split,adapter,decoder,canonicalizer,device,args,cache);local.append((score,truth));base.append((identity,truth));purity.append(pure);peaks.append(peak);fallbacks.append(fallback)
        posterior.append(local);primitive.append(base);del cache
        if device.type=="cuda": torch.cuda.empty_cache()
    return posterior,primitive,purity,peaks,fallbacks


def _evaluate(candidate,baseline,threshold,base_threshold):
    scenes={};passed=[]
    for candidate_scene,baseline_scene in zip(candidate,baseline):
        c=float(torch.tensor([_iou_from_scores(s,t,threshold) for s,t in candidate_scene]).mean());b=float(torch.tensor([_iou_from_scores(s,t,base_threshold) for s,t in baseline_scene]).mean());scenes[len(scenes)]={"posterior_iou":c,"identity_iou":b,"delta":c-b};passed.append(c>=b)
    return scenes,sum(x["delta"] for x in scenes.values())/len(scenes),all(passed)


def run(args:argparse.Namespace)->dict[str,Any]:
    datasets,splits,records,checkpoint,checkpoint_record=_load(args);device=torch.device(args.device);torch.manual_seed(args.seed);generator=torch.Generator().manual_seed(args.seed+1)
    latent_dim=datasets[0]["latent"].shape[1];decoder=AnchorConditionedExtentDecoder(latent_dim=latent_dim,key_dim=args.key_dim,hidden_dim=args.hidden_dim).to(device);canonicalizer=LowRankSceneCanonicalizer(len(datasets),latent_dim,args.scene_canonicalizer_rank).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"]);canonicalizer.load_state_dict(checkpoint["scene_canonicalizer_state_dict"]);decoder.eval();canonicalizer.eval()
    for p in decoder.parameters(): p.requires_grad_(False)
    for p in canonicalizer.parameters(): p.requires_grad_(False)
    adapter=TextAnchorIdentityAdapter(datasets[0]["baseline"].shape[1],args.rank).to(device)
    initial_record=None
    if args.initial_adapter:
        initial,initial_record=_load_mapping(args.initial_adapter,args.expected_initial_adapter_sha256,"initial text anchor adapter");adapter.load_state_dict(initial["adapter_state_dict"])
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    buckets=[]
    for data,(training,_,_) in zip(datasets,splits): buckets.append(torch.where(training&data["generic_text"]["fit"]["eligible"])[0])
    if any(not x.numel() for x in buckets): raise ValueError("text-to-anchor training bucket is empty")
    history=[]
    for step in range(args.steps):
        si=step%len(datasets);pool=buckets[si];sample=int(pool[torch.randint(pool.numel(),(),generator=generator)]);loss=_anchor_loss(datasets[si],sample,adapter,device,args,generator)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),5);optimizer.step()
        if (step+1)%args.log_interval==0: history.append({"step":step+1,"loss":float(loss.detach())})
    adapter.eval();dev,dev_base,_,_,_=_sets(datasets,splits,1,"dev",adapter,decoder,canonicalizer,device,args);threshold,dev_iou=_global_threshold(dev,args.threshold_candidates);base_threshold,base_iou=_global_threshold(dev_base,args.threshold_candidates)
    audit,audit_base,purity,peaks,fallbacks=_sets(datasets,splits,2,"audit",adapter,decoder,canonicalizer,device,args);scenes,delta,noninferior=_evaluate(audit,audit_base,threshold,base_threshold);passed=noninferior and delta>=args.minimum_macro_gain
    output=Path(args.output).resolve();write_torch_noclobber(output,{"schema":"radio_gs.lerf_text_anchor_identity_adapter.v1","schema_version":1,"adapter_state_dict":adapter.state_dict(),"metadata":{"source_only":True,"benchmark_vocabulary_opened":False,"benchmark_images_opened":False,"benchmark_masks_opened":False,"target_masks_opened":False,"training_objective":"gaussian_identity_distribution_peak_and_sibling_ranking","adapted_identity_role":"anchor_compilation_only","posterior_identity_role":"frozen_raw_text_replay","anchor_agreement_multiplier":args.anchor_agreement_multiplier,"initial_adapter":initial_record,"extent_checkpoint":checkpoint_record,"scene_records":records}})
    report={"status":"source_text_anchor_gate_pass" if passed else "source_text_anchor_gate_fail","validation":{"posterior_iou":dev_iou,"identity_iou":base_iou,"threshold":threshold,"identity_threshold":base_threshold},"audit":{"scenes":scenes,"macro_delta":delta,"all_scenes_noninferior":noninferior,"anchor_purity":sum(purity)/len(purity),"peak_accuracy":sum(peaks)/len(peaks),"raw_anchor_fallback_fraction":sum(fallbacks)/len(fallbacks)},"history":history,"output":file_record(output)};write_frozen_json(output.with_suffix(output.suffix+".json"),report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene-specs",required=True);p.add_argument("--extent-checkpoint",required=True);p.add_argument("--expected-extent-sha256",required=True);p.add_argument("--initial-adapter",default="");p.add_argument("--expected-initial-adapter-sha256",default="");p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--steps",type=int,default=800);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--rank",type=int,default=32);p.add_argument("--key-dim",type=int,default=128);p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--scene-canonicalizer-rank",type=int,default=4);p.add_argument("--topk",type=int,default=6);p.add_argument("--radius-fraction",type=float,default=.04);p.add_argument("--authority-radius-multiplier",type=float,default=4.0);p.add_argument("--anchor-agreement-multiplier",type=float,default=0.0);p.add_argument("--random-rows",type=int,default=2048);p.add_argument("--positive-topk",type=int,default=32);p.add_argument("--distribution-temperature",type=float,default=.07);p.add_argument("--ranking-margin",type=float,default=.05);p.add_argument("--peak-weight",type=float,default=.25);p.add_argument("--ranking-weight",type=float,default=.5);p.add_argument("--threshold-candidates",type=int,default=32);p.add_argument("--minimum-macro-gain",type=float,default=.01);p.add_argument("--evaluation-membership-threshold",type=float,default=.5);p.add_argument("--holdout-stride",type=int,default=4);p.add_argument("--heldout-residue",type=int,default=3);p.add_argument("--validation-residue",type=int,default=2);p.add_argument("--log-interval",type=int,default=100);p.add_argument("--seed",type=int,default=20260825);print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))


if __name__=="__main__": main()
