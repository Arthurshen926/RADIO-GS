"""Analytic fixed-scale support bounds; no optimized anchors, scales or cutoffs."""
import argparse
import gc
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from scripts.research.attribute_query_field_architecture import BASE,OUT,COHORT,read,write
from radio_gs.query_field.storage import load_field


def envelope(field,ids,weights):
    ids=ids.to(field.xyz.device);weights=weights.to(field.xyz.device).float()
    keep=field.observed[ids]&(weights>0);ids=ids[keep];weights=weights[keep]
    centre=(field.xyz[ids]*weights[:,None]).sum(0)/weights.sum()
    maximum_affinity=float(field.log_temperature.exp().clamp(1,50)*(1-field.affinity_bias))
    distance=((field.xyz-centre)/(field.default_scale*field.world_units_per_metre)).square().sum(-1)
    upper=(maximum_affinity-.5*distance).sigmoid().masked_fill(~field.observed,0)
    radius=field.default_scale*math.sqrt(max(0,2*maximum_affinity))
    return upper,radius,centre


@torch.no_grad()
def uqis(cohort,device):
    package=Path(cohort['uqis_package']);roster=read(package/'evaluator/targets.json')
    for job in cohort['uqis']:
        out=OUT/'scope_uqis'/f"{job['name']}.json"
        if out.exists():continue
        field,_=load_field(job['field'],device);domain=next(d for d in roster['scene_domains'] if d['scene_id']==job['name'])
        xyz=np.load(package/'method'/domain['mesh_xyz_path']);ids=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        _,index=cKDTree(field.xyz.cpu().numpy()).query(xyz,workers=4)
        _,gindex=cKDTree(xyz).query(field.xyz.cpu().numpy(),workers=4)
        observed=field.observed.cpu().numpy()[index];rows=[]
        for target in [t for t in roster['targets'] if t['scene_id']==job['name']]:
            truth=ids==target['instance_id'];target_scale=float(2*np.linalg.norm(xyz[truth].std(0)))
            for modality,qid in target['queries'].items():
                saved=torch.load(BASE/'predictions/uqis/gaussians'/f'{qid}.pt',map_location='cpu',weights_only=True)
                row=dict(scene=job['name'],query_id=qid,modality=modality,target_scale_m=target_scale,
                         default_scale_m=field.default_scale,target_to_default_scale=target_scale/field.default_scale)
                candidates=saved['candidates'];weights=saved['candidate_weights']
                if not len(candidates):
                    row['supported']=False;rows.append(row);continue
                if modality in {'text','image'}:
                    candidates=candidates[weights.argmax()].reshape(1);weights=torch.ones(1)
                upper,radius,centre=envelope(field,candidates,weights)
                if modality=='point_3d':
                    # The deployed exact-element prompt explicitly clamps its anchor.
                    upper[candidates.to(device)]=1
                assert float((saved['membership'].to(device)-upper).max())<=1e-6
                mask=upper.cpu().numpy()[index]>.5
                selected=saved['selected'].numpy()[index]
                keep=field.observed.cpu()[candidates]&(weights>0)
                target_support=torch.from_numpy(ids[gindex[candidates.numpy()]]==target['instance_id'])
                row.update(supported=True,envelope_radius_m=radius,
                    effective_target_support_mass=float(weights[keep&target_support].sum()/weights[keep].sum()),
                    target_unobserved=int((truth&~observed).sum()),
                    target_observed_outside_envelope=int((truth&observed&~mask).sum()),
                    target_observed_inside_envelope_missed=int((truth&observed&mask&~selected).sum()),
                    target_selected=int((truth&selected).sum()),target_elements=int(truth.sum()),
                    envelope_target_recall=float(mask[truth].mean()))
                assert sum(row[k] for k in ['target_unobserved','target_observed_outside_envelope',
                    'target_observed_inside_envelope_missed','target_selected'])==row['target_elements']
                rows.append(row)
        write(out,dict(scene=job['name'],queries=rows,interpretation='Fixed temperature/bias/scale bound from cosine <=1; no scale optimization'))
        print('scope uqis',job['name'],flush=True);del field;gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def nvos(cohort,device):
    import cv2
    from radio_gs.query_field.public_nvos import load_strokes,retained_response
    from radio_gs.query_field.adapters import scribble_packet
    from radio_gs.query_field.geometry import projection
    for job in cohort['nvos']:
        out=OUT/'scope_nvos'/f"{job['name']}.json"
        if out.exists():continue
        config=read(job['prediction_config']);q=config['jobs'][0];field,_=load_field(job['field'],device)
        fixture=torch.load(q['camera_fixture'],map_location='cpu',weights_only=True)
        ref=SimpleNamespace(**fixture['reference']);target_camera=SimpleNamespace(**fixture['target'])
        masks,_=load_strokes(q['positive'],q['negative'],config['projection_hw'])
        geometry=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
                                get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
        _,op,_,_=projection(geometry,ref,*config['projection_hw'],field.world_units_per_metre)
        packet,_=scribble_packet(op,masks[0],masks[1]);upper,radius,centre=envelope(field,packet.positive.indices,packet.positive.weights)
        saved=torch.load(BASE/'predictions/nvos'/job['name']/job['name']/'predictions.pt',map_location='cpu',weights_only=True)
        assert float((saved['membership'].to(device)-upper).max())<=1e-6
        response=retained_response(field,upper,target_camera,*config['output_hw'],field.world_units_per_metre).cpu().numpy()
        scoring=read(job['scoring_config']);raw=np.asarray(Image.open(scoring['gt_mask']))
        truth=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
        response=cv2.resize(response,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
        actual=cv2.resize(saved['retained_response'].numpy(),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
        assert float((actual-response).max())<=1e-5
        normalized=torch.sparse.mm(op,upper[:,None])[:,0].cpu()
        prompt=masks[0].flatten()
        write(out,dict(scene=job['name'],default_scale_m=field.default_scale,envelope_radius_m=radius,
            current_target_recall=float((actual[truth]>=.5).mean()),
            optimistic_response_target_recall=float((response[truth]>=.5).mean()),
            optimistic_positive_prompt_mean=float((normalized*prompt).sum()/prompt.sum()),
            interpretation='Analytic cosine<=1 member upper bound through unchanged retained response; recall bound only, not IoU or a model result'))
        print('scope nvos',job['name'],flush=True);del field;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('family',choices=['uqis','nvos']);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4);globals()[args.family](read(COHORT),args.device)
