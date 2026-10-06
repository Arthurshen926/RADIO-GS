"""LABEL-USING diagnostic only: replace predicted precision/recall by measured values.

Never a method result or deployment config. Measures the retained scalar-quality
fusion's sensitivity to estimation errors; not a performance upper bound.
"""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/independent_source_reliability')

def worker(name,gpu):
    env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu));out=ROOT/'label_using_diagnosis'/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('w') as log:
        subprocess.run([PY,__file__,'--scene',name,'--root',str(ROOT)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([PY,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'typed_readout.pt'),'--evaluation',str(ROOT/f'{name}.evaluation.json'),'--output',str(out/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def diagnose(name):
    import torch
    from fit_independent_scope_reliability import samples
    from radio_gs.querying.typed_evidence import visible_operator,view_evidence
    torch.set_num_threads(4)
    scene_path=ROOT/'configs/adapter_vote'/f'{name}.scene.json';prior_path=ROOT/'evaluation/adapter_vote'/name/'query/category/typed_readout.pt'
    quality=samples(name,'cuda',scene_path,prior_path,ROOT/f'{name}.evaluation.json')['frame_quality']
    s=json.loads(scene_path.read_text());data=torch.load(prior_path,weights_only=False);observed=data['semantic_observed'].cuda();prior=data['native_logits'].cuda().sigmoid()
    records=[torch.load(v['identity'],weights_only=False) for v in s['source_evidence']['views']]
    square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for r in records:
        for j,x in enumerate(r['instances']):square[j]+=x['foreground_margin'].square().sum();count[j]+=len(x['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
    numerator=torch.zeros_like(prior);mass=torch.zeros_like(prior);visibility=torch.zeros_like(prior);old=torch.zeros_like(prior[0]);det=torch.zeros_like(old)
    for view,r in zip(s['source_evidence']['views'],records):
        h,w=r['raster_hw'];rows=r['instances'];op,evop,visible=visible_operator(torch.load(view['contributions'],weights_only=False),observed,h*w,False)
        probability=torch.stack([x['masks'].float().amax(0) if len(x['scores']) else torch.zeros(h,w) for x in rows],-1).reshape(h*w,-1).cuda()
        detector=torch.tensor([float(x['scores'].max()) if len(x['scores']) else 0 for x in rows],device='cuda')
        confidence=torch.tensor([float((x['scores']*(x['foreground_margin']/scale[j]).sigmoid()).max()) if len(x['scores']) else 0 for j,x in enumerate(rows)],device='cuda')
        e=view_evidence(op,evop,visible,probability,detector>0,view['frame'],h,w,confidence)
        known=(e.positive|e.negative).float();old+=(known*confidence).sum(0);det+=(known*detector).sum(0)
        q=quality[view['frame']].cuda();weight=detector*(e.positive*q[:,0]+e.negative*q[:,1])
        numerator+=torch.sparse.mm(e.adjoint,e.target*weight);mass+=torch.sparse.mm(e.adjoint,weight);visibility+=torch.sparse.mm(e.adjoint,e.visible.float())
    ratio=old/det.clamp_min(1e-8);numerator*=ratio;mass*=ratio;v=(numerator+.1*visibility*prior)/(mass+.1*visibility).clamp_min(1e-8)
    logits=torch.where(mass>0,torch.logit(v.clamp(1e-6,1-1e-6)),data['native_logits'].cuda()).cpu()
    data.update(category_logits=logits,scores=logits,semantic_labels_used_to_construct_predictions=True,diagnosis_only=True,method_performance=False)
    torch.save(data,ROOT/'label_using_diagnosis'/name/'typed_readout.pt')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();ROOT=a.root
    if a.scene:diagnose(a.scene)
    else:
        names=json.loads((ROOT/'split.json').read_text())['validation']
        with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(worker,names,range(3)))
        rows={n:json.loads((ROOT/'label_using_diagnosis'/n/'evaluation/report.json').read_text())['point_metrics'] for n in names}
        summary=dict(scenes=rows,macro={s:sum(r['source'][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']},labels_used_in_predictions=True,method_performance=False,upper_bound=False)
        (ROOT/'label_using_diagnosis/summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['macro'])
