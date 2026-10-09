"""Label-assisted, offline regional-union capacity diagnostic; never inference."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.diagnose_query_field_region_candidates import metrics
ROOT=Path('output/query_field_20261009/region_composition_v1')
PREVIOUS=Path('output/query_field_20261009/region_candidates_v1')
BARRIER=Path('output/query_field_20261009/scope_conditioning_v1/predictions_complete.json')
PROTOCOL=Path('docs/experiments/2026-10-09-region-composition-protocol.md')


def greedy_union(masks,target):
    """Strict IoU ascent in 2D union space; deterministic lowest-index ties.

    This is a feasible greedy construction, not a global union optimum.
    Caller must never use target labels in a deployed inference path.
    """
    masks=np.asarray(masks,dtype=bool)
    target=target.astype(bool);current=np.zeros_like(target);selected=[];trace=[]
    intersection=0;area=0;target_area=int(target.sum())
    if not target_area:raise ValueError('diagnostic needs nonempty target')
    while True:
        tp=np.count_nonzero(masks & (target & ~current)[None],axis=(-2,-1))
        extra=np.count_nonzero(masks & ~current[None],axis=(-2,-1))
        union=area+extra+target_area-intersection-tp
        score=(intersection+tp)/np.maximum(union,1)
        index=int(score.argmax());old=intersection/max(area+target_area-intersection,1)
        if score[index]<=old:break
        assert index not in selected
        selected.append(index);current |= masks[index]
        intersection=int((current&target).sum());area=int(current.sum())
        trace.append(float(score[index]))
    return selected,current,trace


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);d=read(PREVIOUS/'design.json')
    write(ROOT/'design.json',dict(jobs=d['jobs'],protocol=receipt(PROTOCOL),code=[receipt(__file__),receipt('scripts/research/diagnose_query_field_region_candidates.py'),receipt('scripts/research/run_query_field_relations.py')]+[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))],implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),label_assisted_diagnostic=True,model_promotion=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol']]:check(r)
    return d


@torch.no_grad()
def worker(name):
    import cv2
    from PIL import Image
    from radio_gs.query_field.storage import load_field
    from radio_gs.query_field.geometry import selected_alpha
    barrier=read(BARRIER)
    assert len(barrier['records'])==24
    j=next(j for j in design()['jobs'] if j['name']==name)
    for r in [j['configuration'],j['scoring'],j['field'],j['manifest']]+j['inputs']:check(r)
    prior=PREVIOUS/'predictions'/name
    for r in read(prior/'receipt.json')['files']:check(r)
    cfg=read(j['configuration']['path']);job=cfg['jobs'][0];meta=read(prior/'metadata.json');h,w=meta['hw']
    gt=read(j['scoring']['path'])['gt_mask'];raw=np.asarray(Image.open(gt));target=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
    target=cv2.resize(target.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    masks=np.load(prior/'alpha.npy',mmap_mode='r')>meta['threshold']
    ids,proxy,trace=greedy_union(masks,target)
    assert not trace or trace[0]==float(metrics(masks,target)['iou'].max())
    field,_=load_field(job['field'],'cuda');fixture=torch.load(job['camera_fixture'],weights_only=True)
    b=field.evidence.matrix('positive');coeff=torch.zeros(b.shape[1],device='cuda');coeff[ids]=1
    selected=(torch.sparse.mm(b,coeff[:,None]).flatten()>0)&field.member_available
    alpha=selected_alpha(field,selected,SimpleNamespace(**fixture['target']),h,w,fixture['world_units_per_metre']).reshape(h,w).cpu().numpy()
    actual=alpha>meta['threshold']
    root=ROOT/'diagnostics'/name;root.mkdir(parents=True,exist_ok=False)
    torch.save(dict(selected=selected.cpu(),alpha=torch.from_numpy(alpha),proxy=torch.from_numpy(proxy)),root/'diagnostic.pt')
    prior_result=read(PREVIOUS/'summary.json')['per_case'][name]
    assert receipt(gt)['sha256']==prior_result['gt']['sha256']
    report=dict(case=name,selected_regions=ids,region_count=len(ids),total_regions=len(masks),trace=trace,
        single_region_oracle=prior_result['choices']['all']['iou'],
        proxy={k:float(v) for k,v in metrics(proxy,target).items()},actual={k:float(v) for k,v in metrics(actual,target).items()},
        label_assisted=True,gt=receipt(gt),benchmark_metrics=False,global_optimum=False,
        alpha_monotonicity_violations=int((proxy&~actual).sum()),payload=receipt(root/'diagnostic.pt'))
    write(root/'report.json',report);print(name,report['actual'],flush=True)


def run(gpus):
    d=design();read(BARRIER)
    lanes(d['jobs'],gpus,lambda j,g:execute(['-m','scripts.research.diagnose_query_field_region_composition','worker','--name',j['name']],ROOT/'logs'/(j['name']+'.log'),g))
    per={j['name']:read(ROOT/'diagnostics'/j['name']/'report.json') for j in d['jobs']}
    for r in per.values():check(r['payload']);check(r['gt'])
    report=dict(complete=True,label_assisted_diagnostic=True,benchmark_metrics=False,model_promotion=False,global_optimum=False,per_case=per,
        macro={kind:{m:sum(v[kind][m] for v in per.values())/8 for m in ['iou','precision','recall']} for kind in ['proxy','actual']},
        scope_prediction_barrier=receipt(BARRIER),implementation_commit=d['implementation_commit'])
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-09-region-composition-results.json',report);print(report['macro'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run']);p.add_argument('--name');p.add_argument('--gpus',default='2,3');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.name)
    else:run(list(map(int,a.gpus.split(','))))
