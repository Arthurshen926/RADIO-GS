"""Fixed-field NVOS factorial and label-free region reachability predictions.

Labels are loaded only by the existing scorer after all eight prediction groups.
Coverage arms are diagnostic optimistic supersets, never deployed candidates.
"""
import argparse
import copy
import gc
from pathlib import Path
import subprocess
from types import SimpleNamespace
from dataclasses import replace
import torch
from torch.nn import functional as F
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261009/extent_stages_v1')
PREVIOUS=Path('output/query_field_20261009/positive_rays_v1')
ARMS=['local','no_region','no_negative','unary','all_regions','compatible_regions','admitted_regions']
PROTOCOL=Path('docs/experiments/2026-10-09-extent-stages-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);previous=read(PREVIOUS/'design.json');jobs=[]
    for j in previous['jobs']:
        check(j['configuration']);cfg=copy.deepcopy(read(j['configuration']['path']))
        base=next(x for x in cfg['jobs'] if x['query_program']=='positive_ray_local')
        cfg['jobs']=[dict(base,name=j['name']+'__'+a,query_program='extent_diagnostic_'+a) for a in ARMS]
        cfg['prediction_policy']='factorial attribution plus optimistic reachability, no promotion'
        path=ROOT/'configs'/(j['name']+'.json');write(path,cfg)
        scoring=read(j['scoring']['path']);scoring['prediction_config']=str(path)
        sp=ROOT/'configs'/(j['name']+'_score.json');write(sp,scoring)
        jobs.append(dict(name=j['name'],configuration=receipt(path),scoring=receipt(sp),field=j['field'],manifest=j['manifest'],inputs=j['inputs']))
    assert len(jobs)==8
    write(ROOT/'design.json',dict(jobs=jobs,arms=ARMS,primary='attribution_not_model_selection',
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol=receipt(PROTOCOL),code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)],
        same_old_source_frames=120,training_steps=0,new_scene_state_bytes=0,labels_used_by_prediction=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol']]:check(r)
    return d


@torch.no_grad()
def worker(name):
    from radio_gs.query_field.storage import load_field,sha256
    from radio_gs.query_field.public_nvos import load_strokes,retained_response
    from radio_gs.query_field.geometry import projection,selected_alpha
    from radio_gs.query_field.adapters import scribble_packet,raster_support,raster_extent_scale
    from radio_gs.query_field.positive_rays import query
    from radio_gs.query_field.contracts import Support
    d=design();j=next(x for x in d['jobs'] if x['name']==name)
    for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
    cfg=read(j['configuration']['path']);job=cfg['jobs'][0]
    field,_=load_field(job['field'],'cuda');fixture=torch.load(job['camera_fixture'],weights_only=True)
    ref=SimpleNamespace(**fixture['reference']);target=SimpleNamespace(**fixture['target'])
    h,w=cfg['projection_hw'];world=fixture['world_units_per_metre']
    masks,_=load_strokes(job['positive'],job['negative'],(h,w))
    geom=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    _,op,_,_=projection(geom,ref,h,w,world)
    scale=None if job.get('scope','default')=='default' else raster_extent_scale(op,masks[0],field.xyz,world,field.default_scale)
    packet,_=scribble_packet(op,masks[0],masks[1],scale=scale,preserve_rays=field.support_aggregation=='expected_energy',negative_rays=field.negative_rule=='competitive_rays')
    p,_=raster_support(op,masks[0],preserve_rays=True);packet=replace(packet,positive=p)
    evidence=field.evidence;results={}
    for arm,region,negative in [('local',True,True),('no_region',False,True),('no_negative',True,False),('unary',False,False)]:
        field.evidence=evidence if region else None
        result=query(field,packet if negative else replace(packet,negative=None))
        results[arm]=(result.selected,result.membership_scores)
    field.evidence=evidence
    saved=torch.load(PREVIOUS/'predictions'/name/(name+'__positive_ray_local')/'predictions.pt',weights_only=True)
    error=float((results['local'][1].cpu()-saved['membership']).abs().max())
    if error>1e-6 or not torch.equal(results['local'][0].cpu(),saved['selected']):raise ValueError('fixed local parent changed')
    valid=field.member_available[p.indices]&(p.weights>0)
    support=Support(p.indices[valid],p.weights[valid],p.components[valid]);scale=field.default_scale if scale is None else scale
    coefficient=evidence.coefficients(support,scale);ratio=evidence.scales/scale
    gates={'all_regions':torch.ones_like(coefficient,dtype=torch.bool),'compatible_regions':(ratio>=.5)&(ratio<=2),'admitted_regions':coefficient>0}
    B=evidence.matrix('positive')
    for arm,gate in gates.items():
        mask=(torch.sparse.mm(B,gate.to(B.dtype)[:,None]).flatten()>0)&field.member_available
        results[arm]=(mask,mask.float())
    assert bool((~results['admitted_regions'][0]|results['compatible_regions'][0]).all())
    assert bool((~results['compatible_regions'][0]|results['all_regions'][0]).all())
    root=ROOT/'predictions'/name;root.mkdir(parents=True,exist_ok=False);records=[]
    for job in cfg['jobs']:
        arm=job['name'].split('__')[-1];selected,membership=results[arm]
        out=root/job['name'];out.mkdir()
        alpha=selected_alpha(field,selected,target,h,w,world)
        response=F.interpolate(alpha.reshape(1,1,h,w),cfg['output_hw'],mode='bilinear',align_corners=False)[0,0].cpu()
        rh,rw=cfg['output_hw'];retained=retained_response(field,membership,target,rh,rw,world).cpu()
        torch.save(dict(selected=selected.cpu(),membership=membership.cpu(),response=response,retained_response=retained),out/'predictions.pt')
        report=dict(configuration=cfg,job=job,predictions_sha256=sha256(out/'predictions.pt'),target_labels_opened=False,
            diagnostic_upper_bound=arm in gates,source_rgb_access=False,baseline_max_error=error,selected=int(selected.sum()))
        write(out/'prediction.json',report);records.append(receipt(out/'prediction.json'))
    positive,negative=evidence.masses(support,scale)
    write(root/'stages.json',dict(records=records,regions=len(coefficient),compatible=int(gates['compatible_regions'].sum()),admitted=int(gates['admitted_regions'].sum()),
        positive_mass_rows=int((positive>0).sum()),positive_dominates_rows=int((positive>negative).sum()),baseline_max_error=error,
        region_mass_quantiles=torch.quantile(coefficient,torch.tensor([0.,.5,.9,1.],device=coefficient.device)).tolist(),
        target_labels_opened=False,coverage_not_object_correctness=True))
    print('predicted',name,flush=True)


def run(gpus):
    d=design()
    lanes(d['jobs'],gpus,lambda j,gpu:execute(['-m','scripts.research.diagnose_query_field_extent_stages','worker','--name',j['name']],ROOT/'logs'/(j['name']+'.log'),gpu))
    records=[]
    for j in d['jobs']:
        for job in read(j['configuration']['path'])['jobs']:
            p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    assert len(records)==56;write(ROOT/'predictions_complete.json',dict(records=records))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name']);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    summarize()


def summarize():
    d=design();read(ROOT/'scoring_complete.json');per={a:{} for a in ARMS}
    for j in d['jobs']:
        for r in read(ROOT/'predictions'/j['name']/'evaluation.json')['results']:
            per[r['job'].split('__')[-1]][j['name']]=dict(selection_iou=r['foreground_iou'],response_iou=r['retained_response']['foreground_iou'],precision=r['precision'],recall=r['recall'])
    from scripts.research.summarize_query_field_relations import paired
    macro={a:{k:sum(x[k] for x in per[a].values())/8 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a:{k:paired({s:v[k] for s,v in per['local'].items()},{s:v[k] for s,v in per[a].items()},'NVOS '+k) for k in macro[a]} for a in ARMS[1:4]}
    r=dict(complete=True,per_case=per,macro=macro,comparisons=comparisons,implementation_commit=d['implementation_commit'],
        interpretation='coverage arms are optimistic geometric supersets, not learned instance predictions',model_promotion=False,whole_five_task=False,
        stages={j['name']:read(ROOT/'predictions'/j['name']/'stages.json') for j in d['jobs']})
    write(ROOT/'summary.json',r);write('docs/experiments/2026-10-09-extent-stages-results.json',r);print(macro,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run','summarize']);p.add_argument('--name');p.add_argument('--gpus',default='2,3');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.name)
    elif a.stage=='run':run(list(map(int,a.gpus.split(','))))
    else:summarize()
