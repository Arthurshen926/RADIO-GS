#!/usr/bin/env python3
"""Apply passed text-to-anchor retrieval and shared extent to LERF queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.query_native_gaussian_memory import (
    AnchorConditionedExtentDecoder,
    GaussianGeometry,
    LowRankSceneCanonicalizer,
    TextAnchorIdentityAdapter,
    compile_bounded_spatial_authority,
    compile_peak_local_anchor_packet,
)
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def _validate_source_gate(extent, extent_report, adapter_report):
    """Fail closed unless training, calibration, and both source gates agree."""
    if extent_report.get("status") != "source_image_gate_pass":
        raise ValueError("LERF anchor extent source image gate did not pass")
    if adapter_report.get("status") != "source_text_anchor_gate_pass":
        raise ValueError("LERF text anchor source gate did not pass")
    selection = extent_report.get("checkpoint_selection", {})
    audit = extent_report.get("image_audit", {})
    if selection.get("all_scenes_noninferior") is not True:
        raise ValueError("LERF extent checkpoint failed source validation noninferiority")
    if audit.get("all_scenes_noninferior") is not True:
        raise ValueError("LERF extent checkpoint failed source audit noninferiority")
    metadata = extent.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("extent checkpoint metadata differs")
    calibrated = metadata.get("decision_calibrated_identity") is True
    fixed = metadata.get("fixed_decision_thresholds") is True
    if calibrated != fixed:
        raise ValueError("LERF identity gauge and fixed decision threshold differ")
    if not calibrated:
        raise ValueError("formal LERF anchor posterior requires calibrated identity logits")
    scale = float(metadata.get("identity_scale", 0.0))
    if scale <= 0:
        raise ValueError("extent identity calibration scale differs")
    return metadata


def _validated_query_validity(cache, rows):
    valid = torch.as_tensor(cache.get("valid")).bool()
    if valid.shape != (int(rows),):
        raise ValueError("primitive query validity row domain differs")
    return valid


def run(args):
    extent,extent_record=_load_mapping(args.extent_model,args.expected_extent_sha256,"passed anchor extent");extent_report=json.loads(Path(args.extent_model+".json").read_text())
    adapter_payload,adapter_record=_load_mapping(args.text_adapter,args.expected_text_adapter_sha256,"passed text anchor adapter");adapter_report=json.loads(Path(args.text_adapter+".json").read_text())
    extent_metadata=_validate_source_gate(extent,extent_report,adapter_report)
    seed,seed_record=_load_mapping(args.scene_manifest,args.expected_scene_manifest_sha256,"scene input manifest");inputs=seed["metadata"]["inputs"]
    field,_,_=load_factorized_canonical_field_checkpoint(inputs["field"]["path"],map_location="cpu",expected_sha256=inputs["field"]["sha256"]);universal,_=_load_mapping(inputs["universal_field"]["path"],inputs["universal_field"]["sha256"],"Universal Field");cache,_=_load_mapping(inputs["query_cache"]["path"],inputs["query_cache"]["sha256"],"primitive query cache");text,text_record=_load_mapping(args.text_embeddings,args.expected_text_embeddings_sha256,"official LERF text embeddings")
    names=json.loads(args.query_names);lookup={name:i for i,name in enumerate(text["queries"])}
    if any(name not in lookup for name in names): raise ValueError("LERF text query absent from official cache")
    queries=F.normalize(torch.as_tensor(text["embeddings"])[[lookup[x] for x in names]].float(),dim=-1);baseline=F.normalize(torch.as_tensor(cache.get("features",cache.get("summary_features"))).float(),dim=-1);xyz=torch.as_tensor(cache["xyz"]).float();reliability=torch.as_tensor(universal["reliability"]).float();valid=_validated_query_validity(cache,baseline.shape[0])
    with torch.inference_mode():latent=field.query_memory(representation="coefficients").cpu().float()
    device=torch.device(args.device);adapter=TextAnchorIdentityAdapter(queries.shape[1],args.adapter_rank).to(device);decoder=AnchorConditionedExtentDecoder(latent.shape[1],reliability.shape[1],args.key_dim,args.hidden_dim,gauge_normalize_identity=bool(extent_metadata.get("gauge_normalized_identity",False)),use_identity_conditioning=bool(extent_metadata.get("extent_uses_row_identity_map",True))).to(device);canonicalizer=LowRankSceneCanonicalizer(len(extent["scene_canonicalizer_state_dict"]["scene_code.weight"]),latent.shape[1],args.scene_canonicalizer_rank).to(device)
    adapter.load_state_dict(adapter_payload["adapter_state_dict"]);decoder.load_state_dict(extent["decoder_state_dict"]);canonicalizer.load_state_dict(extent["scene_canonicalizer_state_dict"]);adapter.eval();decoder.eval();canonicalizer.eval();latent=latent.to(device);reliability=reliability.to(device);xyz_device=xyz.to(device);baseline=baseline.to(device)
    if args.scene_index>=0: latent=canonicalizer(latent,args.scene_index)
    elif not args.allow_unseen_identity_canonicalizer: raise ValueError("unseen scene lacks a source-gated canonicalizer path")
    posteriors=[];identities=[];anchor_rows=[];peak_rows=[];local_radii=[]
    with torch.inference_mode():
        adapted=adapter(queries.to(device))
        for raw_query,adapted_query in zip(queries.to(device),adapted):
            raw_identity=baseline@raw_query;retrieval_identity=baseline@adapted_query;packet=compile_peak_local_anchor_packet(retrieval_identity,xyz_device,args.topk,args.radius_fraction);raw_packet=compile_peak_local_anchor_packet(raw_identity,xyz_device,args.topk,args.radius_fraction)
            if args.anchor_agreement_multiplier>0 and float(torch.linalg.vector_norm(xyz_device[packet.peak_row]-xyz_device[raw_packet.peak_row]))>raw_packet.local_radius*args.anchor_agreement_multiplier: packet=raw_packet
            identity=raw_identity
            if bool(extent_metadata.get("decision_calibrated_identity",False)):
                scale=float(extent_metadata.get("identity_scale",0.0))
                if scale<=0: raise ValueError("extent identity calibration scale differs")
                identity=(identity-float(extent_metadata.get("identity_center",0.0)))/scale
            authority=compile_bounded_spatial_authority(xyz_device,packet.peak_row,packet.local_radius,args.authority_radius_multiplier,args.maximum_authority_fraction)&valid.to(device);logits=decoder(latent,reliability,identity,packet,GaussianGeometry(xyz_device),authority)
            probability=torch.sigmoid(logits);identity_probability=torch.sigmoid(identity);probability[~valid.to(device)]=0;identity_probability[~valid.to(device)]=0
            posteriors.append(probability.cpu());identities.append(identity_probability.cpu());anchor_rows.append(packet.rows.cpu());peak_rows.append(packet.peak_row);local_radii.append(packet.local_radius)
    score_threshold=.5
    output=Path(args.output).resolve();write_torch_noclobber(output,{"schema":"radio_gs.lerf_anchor_conditioned_text_posterior.v1","schema_version":1,"scene":args.scene,"query_scores":torch.stack(posteriors,1),"identity_query_scores":torch.stack(identities,1),"anchor_rows":torch.stack(anchor_rows),"peak_rows":torch.tensor(peak_rows),"local_radii":torch.tensor(local_radii),"valid":valid,"xyz":xyz,"metadata":{"query_names":names,"query_family":"text_object_extent","typed_posterior":"object_aware_universal_field_v2_text_object_posterior_anchor_conditioned_v1","score_threshold":score_threshold,"modality_specific_identity":True,"token_free_shared_extent":True,"peak_local_anchor":True,"authority_gated_replay":True,"maximum_authority_fraction":args.maximum_authority_fraction,"validity_authority":"primitive_query_cache_valid","invalid_row_policy":"zero_score_and_excluded_by_valid_mask","separate_identity_localization":True,"localization_authority":"field_siglip2_relevancy_identity","persistent_second_semantic_field":False,"benchmark_images_opened":False,"benchmark_masks_opened":False,"evaluation_rgb_opened":False,"query_text_opened_at_readout":True,"extent_model":extent_record,"text_adapter":adapter_record,"scene_manifest":seed_record,"text_embeddings":text_record,"scene_canonicalizer_index":args.scene_index,"unseen_identity_canonicalizer_explicitly_allowed":bool(args.allow_unseen_identity_canonicalizer)}})
    result={"status":"complete","scene":args.scene,"queries":len(names),"output":file_record(output)};write_frozen_json(output.with_suffix(output.suffix+".json"),result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene",required=True);p.add_argument("--scene-index",type=int,required=True);p.add_argument("--scene-manifest",required=True);p.add_argument("--expected-scene-manifest-sha256",required=True);p.add_argument("--extent-model",required=True);p.add_argument("--expected-extent-sha256",required=True);p.add_argument("--text-adapter",required=True);p.add_argument("--expected-text-adapter-sha256",required=True);p.add_argument("--text-embeddings",required=True);p.add_argument("--expected-text-embeddings-sha256",required=True);p.add_argument("--query-names",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--adapter-rank",type=int,default=32);p.add_argument("--key-dim",type=int,default=128);p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--scene-canonicalizer-rank",type=int,default=4);p.add_argument("--topk",type=int,default=6);p.add_argument("--radius-fraction",type=float,default=.04);p.add_argument("--authority-radius-multiplier",type=float,default=4.0);p.add_argument("--maximum-authority-fraction",type=float,default=.2);p.add_argument("--anchor-agreement-multiplier",type=float,default=0.0);p.add_argument("--allow-unseen-identity-canonicalizer",action="store_true");print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))


if __name__=="__main__":main()
