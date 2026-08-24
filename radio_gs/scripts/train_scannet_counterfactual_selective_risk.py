#!/usr/bin/env python3
"""Train one ScanNet source-scene-LOSO frozen-candidate risk estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.query_native_gaussian_memory import CounterfactualSelectiveRiskEstimator, LowRankSceneCanonicalizer
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber

SPLITS=("19","15","10")


def _load(path: Path) -> dict[str, Any]:
    record=file_record(path); labels,_=_load_mapping(str(path),record["sha256"],"counterfactual risk labels")
    score_record=labels["metadata"]["inputs"]["scores"]
    scores,_=_load_mapping(score_record["path"],score_record["sha256"],"frozen score identity")
    inputs=scores["metadata"]["inputs"]
    field,_,_=load_factorized_canonical_field_checkpoint(inputs["field"]["path"],map_location="cpu",expected_sha256=inputs["field"]["sha256"])
    with torch.inference_mode(): latent=field.query_memory(representation="coefficients").cpu().float().contiguous()
    universal,_=_load_mapping(inputs["universal_field"]["path"],inputs["universal_field"]["sha256"],"Universal Field reliability")
    reliability=torch.as_tensor(universal["reliability"]).float()
    if latent.shape[0]!=labels["xyz"].shape[0] or reliability.shape!=(latent.shape[0],5): raise ValueError("risk-estimator Gaussian domain differs")
    return {"scene":labels["scene"],"latent":latent,"reliability":reliability,"labels":labels,"record":record}


def _examples(data: dict[str,Any], *, validation: bool, stride: int, residue: int, decisive_only: bool = True) -> list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]:
    rows=torch.arange(data["latent"].shape[0]); selected=(rows%stride==residue)==validation
    output=[]
    for split in SPLITS:
        label=torch.as_tensor(data["labels"][f"labels_split_{split}"]).long(); decisive=label!=2
        index=torch.where(selected & (decisive if decisive_only else torch.ones_like(decisive)))[0]
        if index.numel(): output.append((index,torch.as_tensor(data["labels"][f"features_split_{split}"]).float(),label))
    return output


@torch.inference_mode()
def _evaluate(model,canonicalizer,data,scene_index,device,stride,residue,threshold):
    benefit=harm=adopted=total=0.; count=0
    for index,features,label in _examples(data,validation=True,stride=stride,residue=residue):
        for start in range(0,index.numel(),32768):
            rows=index[start:start+32768]; latent=canonicalizer(data["latent"][rows].to(device),scene_index)
            logits=model(latent,data["reliability"][rows].to(device),features[rows].to(device)); probability=F.softmax(logits,1).cpu()
            adopt=(probability[:,0]>=threshold) & (probability[:,0]>probability[:,1]) & (probability[:,0]>probability[:,2])
            weight=data["labels"]["significance"][rows].float(); truth=label[rows]
            benefit+=float(weight[adopt & (truth==0)].sum()); harm+=float(weight[adopt & (truth==1)].sum()); total+=float(weight.sum()); adopted+=float(weight[adopt].sum()); count+=int(adopt.sum())
    return {"beneficial_weight":benefit,"harmful_weight":harm,"adopted_weight":adopted,"total_weight":total,"adopted_rows":count,"harmful_fraction":harm/max(benefit+harm,1e-12),"net_benefit":benefit-harm}


def run(args):
    root=Path(args.label_root); paths=sorted(root.glob("*.pt")); datasets=[_load(p) for p in paths]
    heldout_index=next((i for i,x in enumerate(datasets) if x["scene"]==args.heldout_scene),None)
    if heldout_index is None: raise ValueError("heldout scene absent")
    train_indices=[i for i in range(len(datasets)) if i!=heldout_index]; device=torch.device(args.device); torch.manual_seed(args.seed)
    model=CounterfactualSelectiveRiskEstimator(hidden_dim=args.hidden_dim).to(device)
    canonicalizer=LowRankSceneCanonicalizer(len(datasets),512,args.scene_canonicalizer_rank).to(device)
    parameters=list(model.parameters())+list(canonicalizer.parameters()); optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=args.weight_decay)
    pools=[_examples(datasets[i],validation=False,stride=args.validation_stride,residue=args.validation_residue,decisive_only=False) for i in train_indices]
    generator=torch.Generator().manual_seed(args.seed+1); best=float("inf"); state=None
    class_weight=torch.tensor([args.beneficial_weight,args.harmful_weight,args.neutral_auxiliary_weight],device=device)
    for step in range(args.steps):
        position=step%len(train_indices); scene_index=train_indices[position]; data=datasets[scene_index]; choices=pools[position]; index,features,label=choices[(step//len(train_indices))%len(choices)]
        desired_class=(step//(len(train_indices)*len(choices)))%3
        eligible=index[label[index]==desired_class]
        if not eligible.numel(): eligible=index
        rows=eligible[torch.randint(eligible.numel(),(min(args.batch_size,eligible.numel()),),generator=generator)]
        logits=model(canonicalizer(data["latent"][rows].to(device),scene_index),data["reliability"][rows].to(device),features[rows].to(device))
        loss=F.cross_entropy(logits,label[rows].to(device),weight=class_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step+1)%args.validation_interval==0 or step+1==args.steps:
            losses=[]; model.eval(); canonicalizer.eval()
            with torch.no_grad():
                for si in train_indices:
                    local=[]
                    for idx,feat,lab in _examples(datasets[si],validation=True,stride=args.validation_stride,residue=args.validation_residue,decisive_only=False):
                        rows=idx[::max(1,idx.numel()//4096)][:4096]
                        logits=model(canonicalizer(datasets[si]["latent"][rows].to(device),si),datasets[si]["reliability"][rows].to(device),feat[rows].to(device))
                        local.append(F.cross_entropy(logits,lab[rows].to(device),weight=class_weight))
                    losses.append(torch.stack(local).mean())
            value=float(torch.stack(losses).mean())
            if value<best: best=value; state={"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"canonicalizer":{k:v.detach().cpu().clone() for k,v in canonicalizer.state_dict().items()}}
            model.train(); canonicalizer.train()
    model.load_state_dict(state["model"]); canonicalizer.load_state_dict(state["canonicalizer"]); model.eval(); canonicalizer.eval()
    candidates=torch.linspace(.34,.9,args.threshold_candidates); selected_threshold=None
    for threshold in candidates:
        metrics=[_evaluate(model,canonicalizer,datasets[i],i,device,args.validation_stride,args.validation_residue,float(threshold)) for i in train_indices]
        harmful=sum(x["harmful_weight"] for x in metrics); benefit=sum(x["beneficial_weight"] for x in metrics)
        if benefit>harmful and harmful/max(benefit+harmful,1e-12)<=args.maximum_harmful_fraction and sum(x["adopted_rows"] for x in metrics)>0:
            selected_threshold=float(threshold); break
    if selected_threshold is None: selected_threshold=1.01
    heldout=_evaluate(model,canonicalizer,datasets[heldout_index],heldout_index,device,args.validation_stride,args.validation_residue,selected_threshold)
    passed=selected_threshold<=1 and heldout["adopted_rows"]>0 and heldout["net_benefit"]>0 and heldout["harmful_fraction"]<=args.maximum_harmful_fraction
    output=Path(args.output).resolve(); write_torch_noclobber(output,{"schema":"radio_gs.scannet_counterfactual_selective_risk_loso.v1","schema_version":1,"heldout_scene":args.heldout_scene,"model_state_dict":state["model"],"scene_canonicalizer_state_dict":state["canonicalizer"],"adoption_threshold":selected_threshold,"metadata":{"score_decoder_frozen":True,"source_only":True,"neutral_not_forced_negative":True,"selection_objective":"harmful_weighted_mass_then_nonempty_net_benefit","class_indexed_parameters":False,"label_inputs":[x["record"] for x in datasets]}})
    report={"status":"source_loso_pass" if passed else "source_loso_fail","heldout_scene":args.heldout_scene,"best_training_scene_macro_validation_loss":best,"adoption_threshold":selected_threshold,"heldout":heldout,"gate":{"maximum_harmful_fraction":args.maximum_harmful_fraction,"nonempty_adoption":heldout["adopted_rows"]>0,"positive_net_benefit":heldout["net_benefit"]>0,"passed":passed},"output":file_record(output)}; write_frozen_json(output.with_suffix(output.suffix+".json"),report); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--label-root",required=True); p.add_argument("--heldout-scene",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--steps",type=int,default=1200); p.add_argument("--batch-size",type=int,default=4096); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--scene-canonicalizer-rank",type=int,default=8); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--beneficial-weight",type=float,default=1); p.add_argument("--harmful-weight",type=float,default=4); p.add_argument("--neutral-auxiliary-weight",type=float,default=.1); p.add_argument("--validation-stride",type=int,default=5); p.add_argument("--validation-residue",type=int,default=4); p.add_argument("--validation-interval",type=int,default=100); p.add_argument("--threshold-candidates",type=int,default=29); p.add_argument("--maximum-harmful-fraction",type=float,default=.25); p.add_argument("--seed",type=int,default=20260824); print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))
if __name__=="__main__": main()
