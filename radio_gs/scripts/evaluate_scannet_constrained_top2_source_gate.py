#!/usr/bin/env python3
"""Run the single fixed ScanNet top-2 source-only decision experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json

SCENES=("scene0000_00","scene0062_00","scene0070_00","scene0097_00","scene0140_00","scene0347_00","scene0400_00","scene0590_00")
SPLITS=("19","15","10")


def constrained_top2(baseline:torch.Tensor,candidate:torch.Tensor,alpha:float)->torch.Tensor:
    """Move only the baseline/candidate top-1 pairwise margin."""
    if baseline.shape!=candidate.shape or baseline.ndim!=2: raise ValueError("constrained top2 score domain differs")
    base_class=baseline.argmax(1);candidate_class=candidate.argmax(1);rows=torch.arange(baseline.shape[0]);output=baseline.clone();different=base_class!=candidate_class
    if bool(different.any()):
        selected=rows[different];b=base_class[different];c=candidate_class[different]
        base_margin=baseline[selected,c]-baseline[selected,b];candidate_margin=candidate[selected,c]-candidate[selected,b];shift=.5*float(alpha)*(candidate_margin-base_margin)
        output[selected,c]+=shift;output[selected,b]-=shift
    return output


def _accuracy(prediction,truth,weights,mask):
    denominator=weights[mask].sum()
    return float((weights[mask]*(prediction[mask]==truth[mask]).float()).sum()/denominator) if float(denominator)>0 else None


def run(args:argparse.Namespace)->dict:
    root=Path(args.score_root).resolve();metric_root=Path(args.metric_root).resolve();scenes={};all_noninferior=[];total_recovery=0.;total_harm=0.;total_changed=0
    for scene in SCENES:
        score_path=root/scene/"source_distilled_scores.pt";metric_path=metric_root/f"{scene}.pt";scores,sr=_load_mapping(score_path,file_record(score_path)["sha256"],"frozen ScanNet score triplet");metric,mr=_load_mapping(metric_path,file_record(metric_path)["sha256"],"query-independent metric weights")
        if scores.get("metadata",{}).get("contains_frozen_counterfactual_triplet") is not True or metric.get("metadata",{}).get("query_independent") is not True: raise ValueError("ScanNet top2 source contract differs")
        weights=torch.as_tensor(metric["significance"]).float();valid=torch.as_tensor(scores["valid"]).bool()&torch.as_tensor(scores["direct_observed"]).bool();scene_report={"inputs":{"scores":sr,"metric_weights":mr},"splits":{}}
        for split in SPLITS:
            baseline=torch.as_tensor(scores[f"baseline_scores_split_{split}"]).float();raw=torch.as_tensor(scores[f"raw_candidate_scores_split_{split}"]).float();teacher=torch.as_tensor(scores[f"teacher_scores_split_{split}"]).float();candidate=constrained_top2(baseline,raw,args.alpha)
            teacher_top2=teacher.topk(2,dim=1).values;confident=valid&((teacher_top2[:,0]-teacher_top2[:,1])>=args.minimum_teacher_margin);truth=teacher.argmax(1);base=baseline.argmax(1);new=candidate.argmax(1);heldout=(truth%args.query_holdout_modulus)==args.query_holdout_residue;changed=confident&(new!=base);recovery=changed&(base!=truth)&(new==truth);harm=changed&(base==truth)&(new!=truth)
            recovery_mass=float(weights[recovery].sum());harm_mass=float(weights[harm].sum());total_recovery+=recovery_mass;total_harm+=harm_mass;total_changed+=int(changed.sum())
            base_all=_accuracy(base,truth,weights,confident);new_all=_accuracy(new,truth,weights,confident);base_hold=_accuracy(base,truth,weights,confident&heldout);new_hold=_accuracy(new,truth,weights,confident&heldout);noninferior=(base_hold is None or new_hold>=base_hold) and (base_all is None or new_all>=base_all);all_noninferior.append(noninferior)
            scene_report["splits"][split]={"baseline_weighted_accuracy":base_all,"candidate_weighted_accuracy":new_all,"heldout_baseline_weighted_accuracy":base_hold,"heldout_candidate_weighted_accuracy":new_hold,"changed_rows":int(changed.sum()),"recovery_mass":recovery_mass,"harm_mass":harm_mass,"noninferior":noninferior}
        scenes[scene]=scene_report
    passed=all(all_noninferior) and total_changed>0 and total_recovery>total_harm
    return {"schema":"radio_gs.scannet_constrained_top2_source_gate.v1","status":"source_gate_pass" if passed else "source_gate_fail","source_only":True,"benchmark_labels_opened":False,"benchmark_masks_opened":False,"single_fixed_candidate":True,"candidate":{"operation":"baseline plus pairwise margin transfer only on baseline/raw-candidate top1 coordinates","alpha":args.alpha},"query_holdout":{"modulus":args.query_holdout_modulus,"residue":args.query_holdout_residue},"scenes":scenes,"aggregate":{"all_scene_split_noninferior":all(all_noninferior),"changed_rows":total_changed,"recovery_mass":total_recovery,"harm_mass":total_harm,"net_recovery_mass":total_recovery-total_harm,"passed":passed}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--score-root",required=True);p.add_argument("--metric-root",required=True);p.add_argument("--output",required=True);p.add_argument("--alpha",type=float,default=1.0);p.add_argument("--minimum-teacher-margin",type=float,default=.02);p.add_argument("--query-holdout-modulus",type=int,default=4);p.add_argument("--query-holdout-residue",type=int,default=0);args=p.parse_args();report=run(args);write_frozen_json(Path(args.output).resolve(),report);print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":main()
