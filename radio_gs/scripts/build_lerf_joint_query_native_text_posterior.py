#!/usr/bin/env python3
"""Apply a passed joint LERF extent decoder to benchmark text queries."""

from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.interfaces.query_packet import QueryPacket
from radio_gs.models.query_native_gaussian_memory import FixedCosineQueryProjection, GaussianGeometry, LowRankSceneCanonicalizer, ModalityQueryAdapter, QueryNativeGaussianPosteriorDecoder
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def run(args):
    model,model_record=_load_mapping(args.model,args.expected_model_sha256,"passed joint LERF decoder")
    report=json.loads(Path(args.model+".json").read_text())
    if report.get("status")!="source_gate_pass": raise ValueError("joint decoder did not pass source gate")
    seed,seed_record=_load_mapping(args.scene_manifest,args.expected_scene_manifest_sha256,"scene input manifest")
    inputs=seed["metadata"]["inputs"]
    field,_,_=load_factorized_canonical_field_checkpoint(inputs["field"]["path"],map_location="cpu",expected_sha256=inputs["field"]["sha256"])
    universal,_=_load_mapping(inputs["universal_field"]["path"],inputs["universal_field"]["sha256"],"Universal Field")
    cache,_=_load_mapping(inputs["query_cache"]["path"],inputs["query_cache"]["sha256"],"primitive query cache")
    text,text_record=_load_mapping(args.text_embeddings,args.expected_text_embeddings_sha256,"official LERF text embeddings")
    names=json.loads(args.query_names); lookup={name:i for i,name in enumerate(text["queries"])}
    if any(name not in lookup for name in names): raise ValueError("LERF text query absent from official cache")
    queries=F.normalize(torch.as_tensor(text["embeddings"])[[lookup[name] for name in names]].float(),dim=-1)
    baseline=F.normalize(torch.as_tensor(cache.get("features",cache.get("summary_features"))).float(),dim=-1)
    xyz=torch.as_tensor(cache["xyz"]).float(); reliability=torch.as_tensor(universal["reliability"]).float()
    with torch.inference_mode(): latent=field.query_memory(representation="coefficients").cpu().float()
    device=torch.device(args.device); fixed=bool(model.get("metadata",{}).get("fixed_query_projection",False)); adapter=(FixedCosineQueryProjection(queries.shape[1],args.query_dim,int(model.get("metadata",{}).get("projection_seed",20260824))) if fixed else ModalityQueryAdapter(queries.shape[1],args.query_dim)).to(device); decoder=QueryNativeGaussianPosteriorDecoder(latent_dim=latent.shape[1],query_dim=args.query_dim,hidden_dim=args.hidden_dim,topk_anchors=args.topk_anchors).to(device); canonicalizer=LowRankSceneCanonicalizer(len(model["scene_canonicalizer_state_dict"]["scene_code.weight"]),latent.shape[1],args.scene_canonicalizer_rank).to(device)
    adapter.load_state_dict(model["adapter_state_dict"]); decoder.load_state_dict(model["decoder_state_dict"]); canonicalizer.load_state_dict(model["scene_canonicalizer_state_dict"]); adapter.eval(); decoder.eval(); canonicalizer.eval()
    latent_device=latent.to(device); reliability_device=reliability.to(device); xyz_device=xyz.to(device); baseline_device=baseline.to(device); scores=[]; identities=[]
    with torch.inference_mode():
        if args.scene_index>=0: latent_device=canonicalizer(latent_device,args.scene_index)
        for query in queries:
            token=adapter(query[None].to(device)); prior=baseline_device@query.to(device)
            logits,identity=decoder(latent_device,reliability_device,QueryPacket(token,"text"),identity_prior=prior,geometry=GaussianGeometry(xyz_device))
            scores.append(torch.sigmoid(logits).cpu()); identities.append(torch.sigmoid(identity).cpu())
    output=Path(args.output).resolve(); write_torch_noclobber(output,{"schema":"radio_gs.lerf_query_native_text_object_posterior.v1","schema_version":1,"scene":args.scene,"query_scores":torch.stack(scores,1),"identity_query_scores":torch.stack(identities,1),"valid":torch.ones(latent.shape[0],dtype=torch.bool),"xyz":xyz,"metadata":{"query_names":names,"query_family":"text_object_extent","typed_posterior":"object_aware_universal_field_v2_text_object_posterior_joint_query_native_v1","localization_authority":"field_siglip2_relevancy_identity","segmentation_authority":"joint_cross_view_object_extent_decoder","separate_identity_localization":True,"persistent_second_semantic_field":False,"benchmark_images_opened":False,"benchmark_masks_opened":False,"evaluation_rgb_opened":False,"query_text_opened_at_readout":True,"source_gate":report,"joint_model":model_record,"scene_manifest":seed_record,"text_embeddings":text_record,"scene_canonicalizer_index":args.scene_index}})
    result={"status":"complete","scene":args.scene,"queries":len(names),"output":file_record(output)}; write_frozen_json(output.with_suffix(output.suffix+".json"),result); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--scene",required=True); p.add_argument("--scene-index",type=int,required=True); p.add_argument("--scene-manifest",required=True); p.add_argument("--expected-scene-manifest-sha256",required=True); p.add_argument("--model",required=True); p.add_argument("--expected-model-sha256",required=True); p.add_argument("--text-embeddings",required=True); p.add_argument("--expected-text-embeddings-sha256",required=True); p.add_argument("--query-names",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--query-dim",type=int,default=128); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--topk-anchors",type=int,default=6); p.add_argument("--scene-canonicalizer-rank",type=int,default=4); print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))
if __name__=="__main__": main()
