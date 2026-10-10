"""Post-scoring additive FP attribution, not an inference or oracle selector.

Group contributions are measured inside the baseline's actual selected-subset
renderer; removing a group changes transmittance, so contribution shares are not
counterfactual recovery estimates. GT is read only after prediction/scoring.
"""
import argparse
from types import SimpleNamespace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.query_field.geometry import projection
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_reference_member_evidence import ROOT,design,read,write,receipt,check


def main(device):
    d=design();read(ROOT/'scoring_complete.json');rows={}
    names=['reference_unseen','reference_visible_kept','reference_visible_rejected']
    for j in d['jobs']:
        cfg=read(j['configuration']['path']);job=cfg['jobs'][0];scoring=read(j['scoring']['path'])
        field,_=load_field(job['field'],device);fixture=torch.load(job['camera_fixture'],weights_only=True)
        h,w=cfg['projection_hw'];scale=fixture['world_units_per_metre']
        geometry=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
            get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
        _,operator,_,_=projection(geometry,SimpleNamespace(**fixture['reference']),h,w,scale)
        observed=torch.sparse.mm(operator,field.observed.to(operator.dtype)[:,None]).flatten()
        weight=torch.where(observed>.5,observed.clamp_min(1e-8).reciprocal(),0.)
        mass=torch.sparse.mm(operator.transpose(0,1),weight[:,None]).flatten()
        payloads={}
        for arm in d['arms']:
            p=ROOT/'predictions'/j['name']/(j['name']+'__'+arm)
            check(dict(path=str(p/'predictions.pt'),sha256=read(p/'prediction.json')['predictions_sha256']))
            payloads[arm]=torch.load(p/'predictions.pt',map_location='cpu',weights_only=True)
        old,new=(payloads[a] for a in d['arms']);selected=old['selected'].to(device)
        unseen=mass[selected]==0;kept=new['selected'].to(device)[selected]
        group=torch.where(unseen,0,torch.where(kept,1,2))
        xyz=field.xyz[selected]
        subset=SimpleNamespace(xyz=xyz,get_xyz=lambda:xyz,get_rotation=lambda:field.quaternions[selected],
            get_scaling=lambda:field.scales[selected],get_opacity=lambda:field.opacities[selected])
        raw,_,_,_=projection(subset,SimpleNamespace(**fixture['target']),h,w,scale)
        pixels,gids=raw.indices();flat=pixels*3+group[gids]
        contributions=torch.zeros(h*w*3,device=device).scatter_add_(0,flat,raw.values()).reshape(h,w,3)
        projected=F.interpolate(contributions.permute(2,0,1)[None],cfg['output_hw'],mode='bilinear',align_corners=False)[0].cpu()
        error=float((projected.sum(0)-old['response']).abs().max());assert error<1e-5
        # Only now open labels for this diagnostic. They never enter a decoder.
        check(dict(path=scoring['gt_mask'],sha256=read(ROOT/'predictions'/j['name']/'evaluation.json')['gt_sha256']))
        image=np.asarray(Image.open(scoring['gt_mask']));gt=np.any(image>127,axis=2) if image.ndim==3 else image>127
        size=(gt.shape[1],gt.shape[0]);before=cv2.resize(old['response'].numpy(),size,interpolation=cv2.INTER_LINEAR)>cfg['alpha_threshold']
        after=cv2.resize(new['response'].numpy(),size,interpolation=cv2.INTER_LINEAR)>cfg['alpha_threshold']
        groups=np.stack([cv2.resize(x.numpy(),size,interpolation=cv2.INTER_LINEAR) for x in projected])
        shares={}
        for label,mask in [('false_positive',before & ~gt),('true_positive',before & gt)]:
            values=groups[:,mask].sum(1,dtype=np.float64);total=float(values.sum())
            shares[label]=dict(alpha_mass=dict(zip(names,values.tolist())),alpha_share=dict(zip(names,(values/max(total,1e-12)).tolist())),pixels=int(mask.sum()))
        rows[j['name']]=dict(group_members=dict(zip(names,torch.bincount(group,minlength=3).cpu().tolist())),
            baseline_reconstruction_max_error=error,contribution_attribution=shares,
            actual_error_changes=dict(false_positives_removed=int((before & ~after & ~gt).sum()),
                true_positives_lost=int((before & ~after & gt).sum()),false_negatives_recovered=int((~before & after & gt).sum()),
                false_positives_added=int((~before & after & ~gt).sum())),
            contribution_shares_are_not_counterfactual_recovery=True)
        print(j['name'],rows[j['name']],flush=True)
        del field,operator,raw,geometry,subset,contributions
    report=dict(complete=True,cases=rows,post_scoring_only=True,source_access=False,target_labels_used_for_diagnosis=True,
        whole_five_task=False,code=receipt(__file__),scope='additive alpha attribution under original selected geometry; not identity labels for Gaussian rows or a counterfactual upper bound')
    write(ROOT/'diagnostics/member_attribution.json',report)
    write('docs/experiments/2026-10-10-reference-member-evidence-attribution.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4);main(a.device)
