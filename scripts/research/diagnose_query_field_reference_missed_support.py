"""Post-score missing-pixel evidence attribution; never a selection oracle.

Full-scene target alpha contributions identify available reference evidence on
rows projecting into missed pixels, not true per-Gaussian instance membership.
"""
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.query_field.geometry import projection
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.reference_scope import reference_member_evidence
from scripts.research.run_query_field_reference_member_readout import ROOT,design,read,write,receipt,check


def main():
    d=design();read(ROOT/'scoring_complete.json');cases={}
    names=['outside_member_domain','already_selected','unselected_reference_foreground','unselected_reference_background','unselected_reference_unseen']
    for j in d['jobs']:
        cfg=read(j['configuration']['path']);job=next(x for x in cfg['jobs'] if x['name'].endswith('__admission'))
        scoring=read(j['scoring']['path']);field,_=load_field(job['field'],'cuda')
        fixture=torch.load(job['camera_fixture'],weights_only=True);h,w=cfg['projection_hw'];scale=fixture['world_units_per_metre']
        geometry=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
            get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
        _,operator,_,_=projection(geometry,SimpleNamespace(**fixture['reference']),h,w,scale)
        raw=np.asarray(Image.open(job['reference_mask']));q=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
        q=F.interpolate(torch.from_numpy(q.copy()).float()[None,None],(h,w),mode='area')[0,0]
        ratio,_,_=reference_member_evidence(operator,field.observed,q,torch.ones(len(field.xyz),device='cuda',dtype=torch.bool))
        observed=torch.sparse.mm(operator,field.observed.to(operator.dtype)[:,None]).flatten()
        weight=torch.where(observed>.5,observed.clamp_min(1e-8).reciprocal(),0.)
        mass=torch.sparse.mm(operator.transpose(0,1),weight[:,None]).flatten()
        p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
        payload=torch.load(p/'predictions.pt',map_location='cpu',weights_only=True);selected=payload['selected'].cuda()
        group=torch.where(mass==0,4,torch.where(ratio>=.5,2,3))
        group=torch.where(selected,1,group);group=torch.where(field.member_available,group,0)
        raw,_,_,_=projection(geometry,SimpleNamespace(**fixture['target']),h,w,scale)
        pixels,gids=raw.indices()
        values=torch.zeros(h*w*5,device='cuda').scatter_add_(0,pixels*5+group[gids],raw.values()).reshape(h,w,5)
        maps=F.interpolate(values.permute(2,0,1)[None],cfg['output_hw'],mode='bilinear',align_corners=False)[0].cpu()
        # Target labels are used only in this post-scoring attribution.
        check(dict(path=scoring['gt_mask'],sha256=read(ROOT/'predictions'/j['name']/'evaluation.json')['gt_sha256']))
        image=np.asarray(Image.open(scoring['gt_mask']));gt=np.any(image>127,axis=2) if image.ndim==3 else image>127
        size=(gt.shape[1],gt.shape[0]);pred=cv2.resize(payload['response'].numpy(),size,interpolation=cv2.INTER_LINEAR)>cfg['alpha_threshold']
        missing=gt & ~pred
        contribution=np.array([cv2.resize(x.numpy(),size,interpolation=cv2.INTER_LINEAR)[missing].sum(dtype=np.float64) for x in maps])
        total=float(contribution.sum())
        cases[j['name']]=dict(missed_pixels=int(missing.sum()),target_pixels=int(gt.sum()),
            alpha_mass=dict(zip(names,contribution.tolist())),alpha_share=dict(zip(names,(contribution/max(total,1e-12)).tolist())),
            mean_full_scene_alpha_in_missed_pixels=total/max(int(missing.sum()),1),
            whole_field_group_counts=dict(zip(names,torch.bincount(group,minlength=5).cpu().tolist())))
        print(j['name'],cases[j['name']],flush=True)
        del field,geometry,operator,raw,values,maps
    report=dict(complete=True,cases=cases,post_scoring_only=True,target_labels_used_for_diagnosis=True,code=receipt(__file__),
        interpretation='full-scene contribution association at actual FN pixels; no Gaussian identity labels, counterfactual IoU or proof of occlusion cause',
        exact_zero_is_not_all_occlusion=True,reference_mask_is_teacher_input_not_GT=True)
    write(ROOT/'diagnostics/missed_support.json',report)
    write('docs/experiments/2026-10-10-reference-member-readout-missed-support.json',report)

if __name__=='__main__':torch.set_num_threads(4);main()
