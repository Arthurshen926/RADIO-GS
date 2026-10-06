"""LABEL-USING diagnostic; never a deployable map or automatic benchmark score.

Fit region categorical variables to evaluation-point labels with geometry and W
fixed. This is in-sample representability, not a proven IoU upper bound or test
accuracy. No learned variables or label-derived fields are saved.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/observation_representability')


def run_scene(name):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.scannet_constants import compute_split_metrics
    torch.set_num_threads(4)
    scene=json.loads(Path('configs/pipeline/scannet_observation_complete',name+'.scene.json').read_text())
    spec=json.loads(Path('configs/consolidated/scannet_native',name+'.evaluation.json').read_text())
    current=torch.load(Path('output/system_v3/scannet',name,'query/category/typed_readout.pt'),map_location='cpu',weights_only=False)
    field=torch.load(scene['native_observation_field'],map_location='cpu',weights_only=False)
    geometry=FrozenGaussianGeometry.load(scene['geometry']['path'],expected_sha256=scene['geometry']['sha256'],device='cpu')
    vertices=PlyData.read(spec['ground_truth'])['vertex'].data
    xyz=np.stack([vertices[k] for k in ['x','y','z']],1)
    index=cKDTree(geometry.xyz.numpy()).query(xyz,workers=4)[1]
    truth=np.asarray(vertices['label'])
    ids=np.asarray([q['class_id'] for q in current['queries']]);columns={int(c):i for i,c in enumerate(ids)}
    valid=np.isin(truth,ids)
    target_columns=np.asarray([columns[int(c)] for c in truth[valid]])
    counts=torch.bincount(torch.as_tensor(index[valid]*len(ids)+target_columns),minlength=len(geometry.xyz)*len(ids)).reshape(-1,len(ids)).float().cuda()
    supported=field['supported'].cuda()
    target=counts*supported[:,None]
    transfer=torch.sparse_csr_tensor(field['crow'].cuda(),field['columns'].cuda(),field['weights'].cuda(),size=field['shape'])
    native=100*(F.normalize(field['descriptors'].float().cuda(),dim=-1)@current['encodings'].cuda().T)
    def evaluate(probability):
        values=probability.detach().cpu().numpy()
        metrics={}
        for split,classes in spec['splits'].items():
            col=[columns[c] for c in classes]
            pred=np.asarray(classes)[values[:,col].argmax(1)]
            pred[~supported.cpu().numpy()]=0
            metrics[split]=compute_split_metrics(pred[index],truth,classes)
        return metrics
    with torch.no_grad():
        majority=evaluate(counts)
        voted=torch.sparse.mm(transfer,F.one_hot(native.argmax(1),len(ids)).float())
        reachable=(voted>0)
        reachable_points=float((target*reachable).sum().item()/counts.sum().item())
        region_counts=torch.sparse.mm(transfer.transpose(0,1),target)
        pooled=region_counts/region_counts.sum(1,keepdim=True).clamp_min(1e-8)
        pooled_metrics=evaluate(torch.sparse.mm(transfer,pooled))
        initializer=native.detach().clone()
    logits=torch.nn.Parameter(initializer)
    optimizer=torch.optim.Adam([logits],lr=.1)
    losses=[]
    for step in range(200):
        probability=torch.sparse.mm(transfer,torch.softmax(logits,dim=-1))
        loss=-(target*probability.clamp_min(1e-8).log()).sum()/target.sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step in [0,49,99,199]:
            losses.append(dict(step=step+1,nll=float(loss.detach())))
    with torch.no_grad():
        fitted=evaluate(torch.sparse.mm(transfer,torch.softmax(logits,dim=-1)))
    out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
    result=dict(label_using_diagnostic=True,deployable=False,in_sample=True,independent_test=False,
        gt_fitted_is_not_a_proven_iou_upper_bound=True,learned_variables_saved=False,
        geometry_and_sparse_weights_frozen=True,steps=200,learning_rate=.1,losses=losses,
        point_majority_max_accuracy=float(target.max(1).values.sum()/counts.sum()),
        any_native_region_top1_reachability=reachable_points,
        native_region_reachability_is_not_identity_correctness=True,
        metrics=dict(element_majority=majority,pooled_gt_region=pooled_metrics,fitted_gt_region=fitted))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(name,{k:{s:round(100*v['miou'],3) for s,v in x.items()} for k,x in result['metrics'].items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');a=p.parse_args()
    if a.scene:run_scene(a.scene);return
    scenes=sorted(Path('configs/pipeline/scannet_observation_complete').glob('*.scene.json'))
    gpus=['4','5']
    def worker(i):
        for path in scenes[i::len(gpus)]:
            name=path.name.removesuffix('.scene.json');out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
            with (out/'run.log').open('w') as log:subprocess.run([PYTHON,__file__,'--scene',name],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[i]),stdout=log,stderr=subprocess.STDOUT,check=True)
            print(name,'diagnostic complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(worker,range(len(gpus))))
    rows={p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/'report.json').read_text()) for p in scenes}
    macro={a:{s:sum(r['metrics'][a][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for a in ['element_majority','pooled_gt_region','fitted_gt_region']}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,label_using_diagnostic=True,not_automatic_method_performance=True),indent=2)+'\n');print(macro,flush=True)

if __name__=='__main__':main()
