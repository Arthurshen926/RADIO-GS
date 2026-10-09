"""Exhaustive single-region capacity/admission diagnostic; never a deployed oracle."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import torch
from scripts.research.run_query_field_relations import read, write, receipt, check, lanes, execute
ROOT=Path('output/query_field_20261009/region_candidates_v1')
PREVIOUS=Path('output/query_field_20261009/extent_stages_v1')
PROTOCOL=Path('docs/experiments/2026-10-09-region-candidates-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    d=read(PREVIOUS/'design.json')
    write(ROOT/'design.json',dict(jobs=d['jobs'],code=[receipt(__file__)]+[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))],protocol=receipt(PROTOCOL),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),labels_used_in_generation=False,diagnostic_only=True))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol']]:check(r)
    return d


@torch.no_grad()
def worker(name):
    from radio_gs.query_field.storage import load_field
    from radio_gs.query_field.public_nvos import load_strokes
    from radio_gs.query_field.geometry import projection, selected_alpha
    from radio_gs.query_field.adapters import raster_support
    from radio_gs.query_field.contracts import Support
    j=next(j for j in design()['jobs'] if j['name']==name)
    for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
    cfg=read(j['configuration']['path']);job=cfg['jobs'][0]
    assert job['scope']=='default'
    field,_=load_field(job['field'],'cuda');f=torch.load(job['camera_fixture'],weights_only=True)
    ref=SimpleNamespace(**f['reference']);target=SimpleNamespace(**f['target']);world=f['world_units_per_metre'];h,w=cfg['projection_hw']
    masks,_=load_strokes(job['positive'],job['negative'],(h,w))
    geom=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    _,op,_,_=projection(geom,ref,h,w,world)
    p,_=raster_support(op,masks[0],preserve_rays=True)
    valid=field.member_available[p.indices]&(p.weights>0)
    p=Support(p.indices[valid],p.weights[valid],p.components[valid])
    e=field.evidence;c=e.coefficients(p,field.default_scale);ratio=e.scales/field.default_scale
    # Transpose CSR indexes each regional Gaussian subset without dense N x R state.
    b=e.matrix('positive').transpose(0,1).to_sparse_csr()
    pointer=b.crow_indices().cpu().tolist();indices=b.col_indices()
    root=ROOT/'predictions'/name;root.mkdir(parents=True,exist_ok=False)
    arrays=np.lib.format.open_memmap(root/'alpha.npy',mode='w+',dtype=np.float32,shape=(len(c),h,w))
    for r in range(len(c)):
        selected=torch.zeros(len(field.xyz),dtype=torch.bool,device='cuda')
        selected[indices[pointer[r]:pointer[r+1]]]=True;selected &=field.member_available
        arrays[r]=selected_alpha(field,selected,target,h,w,world).reshape(h,w).cpu().numpy()
        if r%200==0:print(name,r,len(c),flush=True)
    arrays.flush();del arrays
    controls={}
    for arm in ['local','all_regions','compatible_regions','admitted_regions']:
        path=PREVIOUS/'predictions'/name/(name+'__'+arm)/'predictions.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        controls[arm]=selected_alpha(field,saved['selected'].cuda(),target,h,w,world).reshape(h,w).cpu().numpy()
    np.savez(root/'controls.npz',**controls)
    write(root/'metadata.json',dict(coefficient=c.cpu().tolist(),compatible=((ratio>=.5)&(ratio<=2)).cpu().tolist(),views=e.views.cpu().tolist(),hw=[h,w],threshold=cfg['alpha_threshold'],target_labels_opened=False,regions=len(c)))
    write(root/'receipt.json',dict(files=[receipt(root/p) for p in ['alpha.npy','controls.npz','metadata.json']]))


def metrics(pred,target):
    tp=np.count_nonzero(pred&target,axis=(-2,-1));area=np.count_nonzero(pred,axis=(-2,-1));total=int(target.sum())
    return dict(iou=tp/np.maximum(area+total-tp,1),precision=tp/np.maximum(area,1),recall=tp/max(total,1))


def score():
    import cv2
    from PIL import Image
    d=design();read(ROOT/'predictions_complete.json');results={}
    for j in d['jobs']:
        root=ROOT/'predictions'/j['name'];meta=read(root/'metadata.json')
        for r in read(root/'receipt.json')['files']:check(r)
        check(j['scoring']);gt=read(j['scoring']['path'])['gt_mask'];raw=np.asarray(Image.open(gt));target=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
        h,w=meta['hw'];target=cv2.resize(target.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        a=np.load(root/'alpha.npy',mmap_mode='r');stats=metrics(a>meta['threshold'],target)
        c=np.array(meta['coefficient']);gates=dict(all=np.ones(len(c),bool),compatible=np.array(meta['compatible']),admitted=c>0)
        choices={}
        for name,gate in gates.items():
            ids=np.flatnonzero(gate)
            if len(ids):
                best=int(ids[np.argmax(stats['iou'][ids])]);choices[name]=dict(region=best,**{k:float(v[best]) for k,v in stats.items()},coefficient=float(c[best]),count=len(ids))
            else:choices[name]=None
        top=int(c.argmax());choices['coefficient_top']=dict(region=top,**{k:float(v[top]) for k,v in stats.items()},coefficient=float(c[top])) if c[top]>0 else None
        controls={k:{m:float(v) for m,v in metrics(a>meta['threshold'],target).items()} for k,a in np.load(root/'controls.npz').items()}
        results[j['name']]=dict(choices=choices,controls=controls,gt=receipt(gt),regions=len(c))
    report=dict(complete=True,diagnostic_only=True,benchmark_metrics=False,model_promotion=False,per_case=results,macro_iou={k:sum(r['choices'][k]['iou'] if r['choices'][k] else 0 for r in results.values())/len(results) for k in ['all','compatible','admitted','coefficient_top']},interpretation='Single-region oracle at 192x256; GT nearest-resized. Not a union oracle, whole-model bound, or deployable selector.')
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-09-region-candidates-results.json',report)
    print(report['macro_iou'],flush=True)


def run(gpus):
    d=design();lanes(d['jobs'],gpus,lambda j,g:execute(['-m','scripts.research.diagnose_query_field_region_candidates','worker','--name',j['name']],ROOT/'logs'/(j['name']+'.log'),g))
    records=[]
    for j in d['jobs']:
        p=ROOT/'predictions'/j['name']/'receipt.json'
        for r in read(p)['files']:check(r)
        records.append(receipt(p))
    assert len(records)==8;write(ROOT/'predictions_complete.json',dict(records=records));score()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run']);p.add_argument('--name');p.add_argument('--gpus',default='2,3');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.name)
    else:run(list(map(int,a.gpus.split(','))))
