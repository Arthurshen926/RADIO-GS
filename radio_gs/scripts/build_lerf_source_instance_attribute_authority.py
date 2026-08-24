#!/usr/bin/env python3
"""Bind compositional attribute text to source objects with sibling contrast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def compile_authority(episodes,teacher,bank,topk,minimum_margin,sibling_weight):
    query=torch.as_tensor(episodes["episode_query_proposal"]).long();target=torch.as_tensor(episodes["episode_target_proposal"]).long();objects=torch.as_tensor(episodes["episode_object_id"]).long();descriptor=F.normalize(torch.as_tensor(teacher["descriptors"]).float(),dim=-1);text=F.normalize(torch.as_tensor(bank["embeddings"]).float(),dim=-1)
    qscore=descriptor[query]@text.T;tscore=descriptor[target]@text.T;offset=torch.as_tensor(episodes["negative_proposal_offsets"]).long();negative=torch.as_tensor(episodes["negative_proposals"]).long();selected=torch.empty(query.numel(),dtype=torch.long);margin=torch.empty(query.numel());has_sibling=torch.zeros(query.numel(),dtype=torch.bool);sibling_max=torch.zeros_like(qscore)
    # One description is selected for the physical track, not independently
    # for each view pair.  The minimum across confirmed views enforces
    # cross-view persistence; explicit different-instance proposals enforce
    # sibling discriminability.
    for object_id in torch.unique(objects,sorted=True).tolist():
        episode_rows=torch.where(objects==int(object_id))[0];proposals=torch.unique(torch.cat((query[episode_rows],target[episode_rows])))
        persistent=(descriptor[proposals]@text.T).min(0).values
        siblings=[]
        for index in episode_rows.tolist():
            values=negative[offset[index]:offset[index+1]]
            if values.numel(): siblings.append(values)
        if siblings:
            sibling_proposals=torch.unique(torch.cat(siblings));sibling=(descriptor[sibling_proposals]@text.T).max(0).values;has_sibling[episode_rows]=True
        else: sibling=torch.zeros_like(persistent)
        objective=persistent-float(sibling_weight)*sibling;best=objective.topk(2);choice=best.indices[0];selected[episode_rows]=choice;margin[episode_rows]=best.values[0]-best.values[1];sibling_max[episode_rows]=sibling
    shared=torch.minimum(qscore,tscore);k=min(int(topk),text.shape[0]);qtop=qscore.topk(k,dim=1).indices;ttop=tscore.topk(k,dim=1).indices;eligible=(qtop==selected[:,None]).any(1)&(ttop==selected[:,None]).any(1)&(margin>=float(minimum_margin))
    return {"schema":"radio_gs.lerf_source_instance_attribute_authority.v2","schema_version":2,"episode_query_proposal":query,"episode_target_proposal":target,"episode_target_view":torch.as_tensor(episodes["episode_target_view"]).long(),"episode_object_id":objects,"selected_text_index":selected,"selected_text_embedding":text[selected].half().contiguous(),"selected_text_query":[str(bank["queries"][int(i)]) for i in selected],"shared_similarity":shared.gather(1,selected[:,None]).squeeze(1),"sibling_similarity":sibling_max.gather(1,selected[:,None]).squeeze(1),"shared_margin":margin,"has_explicit_sibling":has_sibling,"eligible":eligible,"metadata":{"source_only":True,"benchmark_vocabulary_opened":False,"benchmark_images_opened":False,"benchmark_masks_opened":False,"evaluation_rgb_opened":False,"text_bank_split":str(bank["split"]),"description_type":"color_material_shape_or_adjective_plus_noun","pairing":"track_minimum_cross_view_similarity_minus_explicit_sibling_similarity","track_consistent_description":True,"sibling_weight":float(sibling_weight),"topk":k,"minimum_margin":float(minimum_margin),"eligible_count":int(eligible.sum()),"episode_count":int(query.numel())}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--episodes",required=True);p.add_argument("--expected-episodes-sha256",required=True);p.add_argument("--language-teacher",required=True);p.add_argument("--expected-language-teacher-sha256",required=True);p.add_argument("--text-bank",required=True);p.add_argument("--expected-text-bank-sha256",required=True);p.add_argument("--output",required=True);p.add_argument("--topk",type=int,default=20);p.add_argument("--minimum-margin",type=float,default=.001);p.add_argument("--sibling-weight",type=float,default=.5);args=p.parse_args();episodes,er=_load_mapping(args.episodes,args.expected_episodes_sha256,"cross-view object episodes");teacher,tr=_load_mapping(args.language_teacher,args.expected_language_teacher_sha256,"source crop teacher");bank,br=_load_mapping(args.text_bank,args.expected_text_bank_sha256,"attribute text bank")
    if bank.get("benchmark_vocabulary_opened") is not False or bank.get("uses_benchmark_vocabulary_for_construction") is not False: raise ValueError("attribute bank contract differs")
    payload=compile_authority(episodes,teacher,bank,args.topk,args.minimum_margin,args.sibling_weight);payload["metadata"]["inputs"]={"episodes":er,"language_teacher":tr,"text_bank":br};output=Path(args.output).resolve();write_torch_noclobber(output,payload);report={"status":"complete","split":payload["metadata"]["text_bank_split"],"episodes":payload["metadata"]["episode_count"],"eligible":payload["metadata"]["eligible_count"],"with_sibling":int(payload["has_explicit_sibling"].sum()),"output":file_record(output)};write_frozen_json(output.with_suffix(output.suffix+".json"),report);print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
