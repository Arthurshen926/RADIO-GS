"""Diagnose fixed-geometry label conflicts without fitting or changing a field."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch
from radio_gs.training.mixture_feasibility import mixture_error_floor
from radio_gs.training.text_task_aggregation import canonical_text_tasks
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    receipt=json.loads((a.root/'episodes.json').read_text())
    if sha256_file(a.root/'episodes.pt')!=receipt['sha256']:raise ValueError('episodes identity')
    data=torch.load(a.root/'episodes.pt',weights_only=False)
    if sha256_file(a.root/'teachers.json')!=data['teacher_manifest_sha256']:raise ValueError('teacher identity')
    info=json.loads((a.root/'teachers.json').read_text())
    if sha256_file(a.root/'responsibility.json')!=info['authority_sha256']:raise ValueError('authority identity')
    auth=json.loads((a.root/'responsibility.json').read_text())
    n=auth['num_gaussians'];masks=data['masks'].flatten(1).to(a.device)
    observed=data['seen'][[info['frames'].index(f) for f in info['train_frames']]].any(0).to(a.device)
    matrices=[];supports=[]
    for r in auth['views']:
        path=a.root/r['relative_path']
        if sha256_file(path)!=r['sha256']:raise ValueError('view identity')
        hit=torch.load(path,map_location=a.device,weights_only=False)
        pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
        full=torch.zeros(masks.shape[1],device=a.device).scatter_add_(0,pid,weight)
        weight=weight*observed[gid]
        mass=torch.zeros_like(full).scatter_add_(0,pid,weight)
        supports.append(mass/full.clamp_min(1e-8)>.5)
        matrices.append(torch.sparse_coo_tensor(torch.stack([gid,pid]),weight/mass[pid].clamp_min(1e-8),(n,len(full))).coalesce())
    groups=defaultdict(list)
    for index,e in canonical_text_tasks(list(enumerate(data['episodes']))):
        groups[(e['split'],e['modality'],e.get('text_index',e['query_region']))].append((index,e))
    rows=[]
    for (split,modality,query),episodes in groups.items():
        aggregate=torch.zeros((n,2),device=a.device);individual=0.
        for index,e in episodes:
            pos=masks[e.get('positive_regions',[e['target_region']])].any(0)
            neg=masks[e['negative_regions']].any(0) if e['negative_regions'] else torch.zeros_like(pos)
            known=(pos^neg)&supports[e['target_view']]
            labels=torch.stack([pos&known,neg&known],dim=1).float()
            mass=torch.sparse.mm(matrices[e['target_view']],labels)
            individual+=mixture_error_floor(mass[:,0],mass[:,1])['minimum_absolute_error_sum']
            aggregate+=mass
        result=mixture_error_floor(aggregate[:,0],aggregate[:,1])
        result.update(split=split,modality=modality,query_index=query,target_views=len(episodes),
                      independent_view_error_sum=individual,
                      additional_cross_view_error_sum=max(0.,result['minimum_absolute_error_sum']-individual))
        rows.append(result)
    summary={}
    for split in ('train','validation'):
        summary[split]={}
        for modality in ('text','image','prompt'):
            selected=[r for r in rows if r['split']==split and r['modality']==modality]
            mass=sum(r['label_contribution_mass'] for r in selected)
            summary[split][modality]={'queries':len(selected),'minimum_pixel_weighted_mae':sum(r['minimum_absolute_error_sum'] for r in selected)/mass if mass else None,'additional_cross_view_mae':sum(r['additional_cross_view_error_sum'] for r in selected)/mass if mass else None}
    a.output.write_text(json.dumps({'schema':'query_mixture_feasibility.v1','episode_sha256':receipt['sha256'],'authority_sha256':info['authority_sha256'],'benchmark_ground_truth':False,'support':'training-visible geometry; conditional on >0.5 mass','interpretation':'exact L1 relaxation with independent scalar per element per query; not IoU ceiling; no field updates','summary':summary,'queries':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
