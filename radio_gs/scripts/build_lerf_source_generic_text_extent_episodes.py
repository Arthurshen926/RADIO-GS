#!/usr/bin/env python3
"""Compile benchmark-disjoint generic-text cross-view extent episodes."""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def compile_authority(episodes: dict[str, Any], teacher: dict[str, Any], bank: dict[str, Any], topk: int, minimum_margin: float) -> dict[str, Any]:
    query=torch.as_tensor(episodes["episode_query_proposal"]).long(); target=torch.as_tensor(episodes["episode_target_proposal"]).long()
    descriptor=F.normalize(torch.as_tensor(teacher["descriptors"]).float(),dim=-1)
    text=F.normalize(torch.as_tensor(bank["embeddings"]).float(),dim=-1)
    if descriptor.shape[1]!=text.shape[1] or int(max(query.max(),target.max()))>=descriptor.shape[0]: raise ValueError("generic-text episode descriptor domain differs")
    query_score=descriptor[query]@text.T; target_score=descriptor[target]@text.T
    shared=torch.minimum(query_score,target_score); best=shared.topk(2,dim=1); selected=best.indices[:,0]; margin=best.values[:,0]-best.values[:,1]
    k=min(int(topk),text.shape[0]); qtop=query_score.topk(k,dim=1).indices; ttop=target_score.topk(k,dim=1).indices
    in_query=(qtop==selected[:,None]).any(1); in_target=(ttop==selected[:,None]).any(1)
    eligible=in_query & in_target & (margin>=float(minimum_margin))
    if not bool(eligible.any()): raise ValueError("generic-text compiler produced no eligible episodes")
    names=[str(bank["queries"][int(i)]) for i in selected]
    return {"schema":"radio_gs.lerf_source_generic_text_extent_episodes.v1","schema_version":1,"episode_query_proposal":query,"episode_target_proposal":target,"episode_target_view":torch.as_tensor(episodes["episode_target_view"]).long(),"episode_object_id":torch.as_tensor(episodes["episode_object_id"]).long(),"selected_text_index":selected,"selected_text_embedding":text[selected].half().contiguous(),"selected_text_query":names,"shared_similarity":best.values[:,0].float(),"shared_margin":margin.float(),"eligible":eligible,"metadata":{"source_only":True,"benchmark_vocabulary_opened":False,"benchmark_images_opened":False,"benchmark_masks_opened":False,"evaluation_rgb_opened":False,"text_bank_split":str(bank.get("split")),"pairing":"maximum_minimum_cross_view_crop_text_cosine","eligibility":"selected_token_in_both_view_topk_and_shared_margin","topk":k,"minimum_margin":float(minimum_margin),"eligible_count":int(eligible.sum()),"episode_count":int(query.numel())}}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--episodes",required=True); p.add_argument("--expected-episodes-sha256",required=True); p.add_argument("--language-teacher",required=True); p.add_argument("--expected-language-teacher-sha256",required=True); p.add_argument("--text-bank",required=True); p.add_argument("--expected-text-bank-sha256",required=True); p.add_argument("--output",required=True); p.add_argument("--topk",type=int,default=10); p.add_argument("--minimum-margin",type=float,default=.001); args=p.parse_args()
    episodes,er=_load_mapping(args.episodes,args.expected_episodes_sha256,"cross-view object episodes"); teacher,tr=_load_mapping(args.language_teacher,args.expected_language_teacher_sha256,"source crop language teacher"); bank,br=_load_mapping(args.text_bank,args.expected_text_bank_sha256,"target-blind generic text bank")
    if bank.get("benchmark_vocabulary_opened") is not False or bank.get("uses_benchmark_vocabulary_for_construction") is not False: raise ValueError("generic text bank information contract differs")
    payload=compile_authority(episodes,teacher,bank,args.topk,args.minimum_margin); payload["metadata"]["inputs"]={"episodes":er,"language_teacher":tr,"text_bank":br}; output=Path(args.output).resolve(); write_torch_noclobber(output,payload); report={"status":"complete","split":payload["metadata"]["text_bank_split"],"episodes":payload["metadata"]["episode_count"],"eligible":payload["metadata"]["eligible_count"],"output":file_record(output)}; write_frozen_json(output.with_suffix(output.suffix+".json"),report); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
