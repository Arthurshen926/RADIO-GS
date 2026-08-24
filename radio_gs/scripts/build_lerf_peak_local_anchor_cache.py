#!/usr/bin/env python3
"""Materialize source-only full-view image/text peak-local AnchorPackets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.models.query_native_gaussian_memory import compile_peak_local_anchor_packet
from radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder import _load_scene
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber


def run(args: argparse.Namespace) -> dict:
    specs=json.loads(Path(args.scene_specs).read_text()); spec=next((value for value in specs if value["scene"]==args.scene),None)
    if spec is None: raise ValueError(f"scene {args.scene} absent from specs")
    text_authority=spec.get("instance_text",spec.get("generic_text"))
    data=_load_scene(spec["seed_model"],spec["seed_model_sha256"],spec["episodes"],spec["episodes_sha256"],args.evaluation_membership_threshold,text_authority)
    device=torch.device(args.device); baseline=data["baseline"].to(device); xyz=data["xyz"].to(device)
    radii=[float(value) for value in args.radius_fractions.split(",")]; episode_count=data["episode_query"].numel(); payload={"schema":"radio_gs.lerf_peak_local_anchor_cache.v1","schema_version":1,"scene":args.scene,"episode_query_proposal":data["episode_query"],"episode_target_proposal":data["episode_target"],"episode_target_view":data["episode_view"]}
    for radius in radii:
        image_rows=[]; image_peaks=[]; image_radius=[]
        text_rows={split:[] for split in data["generic_text"]}; text_peaks={split:[] for split in data["generic_text"]}; text_radius={split:[] for split in data["generic_text"]}
        for sample in range(episode_count):
            target=int(data["episode_target"][sample]); visible=torch.where(data["observed"][int(data["views"][target])])[0].to(device); local_xyz=xyz[visible]
            query=int(data["episode_query"][sample]); identity=baseline[visible]@data["semantic"][query].to(device); packet=compile_peak_local_anchor_packet(identity,local_xyz,args.topk,radius)
            image_rows.append(visible[packet.rows].cpu()); image_peaks.append(int(visible[packet.peak_row])); image_radius.append(packet.local_radius)
            for split in data["generic_text"]:
                text_identity=baseline[visible]@data["generic_text"][split]["embedding"][sample].to(device); text_packet=compile_peak_local_anchor_packet(text_identity,local_xyz,args.topk,radius)
                text_rows[split].append(visible[text_packet.rows].cpu()); text_peaks[split].append(int(visible[text_packet.peak_row])); text_radius[split].append(text_packet.local_radius)
        suffix=str(radius).replace(".","p")
        payload[f"image_anchor_rows_r{suffix}"]=torch.stack(image_rows); payload[f"image_peak_rows_r{suffix}"]=torch.tensor(image_peaks); payload[f"image_local_radius_r{suffix}"]=torch.tensor(image_radius)
        for split in data["generic_text"]:
            payload[f"text_{split}_anchor_rows_r{suffix}"]=torch.stack(text_rows[split]); payload[f"text_{split}_peak_rows_r{suffix}"]=torch.tensor(text_peaks[split]); payload[f"text_{split}_local_radius_r{suffix}"]=torch.tensor(text_radius[split])
    payload["metadata"]={"source_only":True,"benchmark_images_opened":False,"benchmark_masks_opened":False,"benchmark_vocabulary_opened":False,"evaluation_rgb_opened":False,"text_authority":"instance_text" if spec.get("instance_text") else "generic_text","topk":args.topk,"radius_fractions":radii,"compiler":"global_identity_peak_then_local_3d_topk","inputs":{"seed_model":{"path":spec["seed_model"],"sha256":spec["seed_model_sha256"]},"episodes":{"path":spec["episodes"],"sha256":spec["episodes_sha256"]}}}
    output=Path(args.output).resolve();write_torch_noclobber(output,payload);report={"status":"complete","scene":args.scene,"episodes":episode_count,"output":file_record(output)};write_frozen_json(output.with_suffix(output.suffix+".json"),report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene-specs",required=True);p.add_argument("--scene",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--topk",type=int,default=6);p.add_argument("--radius-fractions",default="0.01,0.02,0.04");p.add_argument("--evaluation-membership-threshold",type=float,default=.5);print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))
if __name__=="__main__":main()
