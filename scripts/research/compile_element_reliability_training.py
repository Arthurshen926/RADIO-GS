"""Compile full-view supervised training rooms for consumed-element learning."""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.typed_evidence import visible_operator,view_evidence
from radio_gs.querying.supervised_scope_reliability import scope_features,native_identity_features

ROOT=Path('output/element_retrieval_reliability')

@torch.inference_mode()
def compile_scene(name,device):
    spec=json.loads(Path(f'configs/pipeline/paired_source_resolution/low/{name}.scene.json').read_text())
    readout=torch.load(f'output/adapter_observation_control/{name}/mask_adapter/query/typed_readout.pt',map_location='cpu')
    prior=readout['native_logits'].sigmoid().to(device);observed=readout['semantic_observed'].to(device)
    geo=FrozenGaussianGeometry.load(spec['geometry']['path'],expected_sha256=spec['geometry']['sha256'],device='cpu')
    evaluation=json.loads(Path(f'configs/consolidated/scannet_native/{name}.evaluation.json').read_text())
    vertex=PlyData.read(evaluation['ground_truth'])['vertex'].data
    xyz=np.stack([vertex[c] for c in ['x','y','z']],1);label=np.asarray(vertex['label'],dtype=np.int64)
    class_ids=[q['class_id'] for q in readout['queries']];n,c=prior.shape
    # Labels for quality auxiliaries use projected labelled mesh, as before.
    near=cKDTree(xyz).query(geo.xyz.numpy())[1]
    labels=torch.tensor(label[near],device=device)
    semantics=torch.cat([(labels[:,None]==torch.tensor(class_ids,device=device)).float(),(labels>0)[:,None].float()],1)
    # Actual downstream point loss follows the evaluation mapping direction.
    nearest=cKDTree(geo.xyz.numpy()).query(xyz)[1];counts=np.zeros((n,c),np.float32)
    for j,class_id in enumerate(class_ids):np.add.at(counts[:,j],nearest[label==class_id],1)
    records=[torch.load(v['identity'],map_location='cpu') for v in spec['source_evidence']['views']]
    square=torch.zeros(c);number=torch.zeros(c)
    for data in records:
        for j,r in enumerate(data['instances']):square[j]+=r['foreground_margin'].square().sum();number[j]+=len(r['scores'])
    scale=(square/number.clamp_min(1)).sqrt().clamp_min(1e-6)
    blocks=[];features=[];targets=[];valid_targets=[]
    visibility=torch.zeros_like(prior);budget=torch.zeros(c,device=device);detector_budget=torch.zeros_like(budget)
    for view,data in zip(spec['source_evidence']['views'],records):
        h,w=data['raster_hw'];rows=data['instances']
        op,evop,visible=visible_operator(torch.load(view['contributions'],map_location='cpu'),observed,h*w,False)
        probability=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in rows],-1).reshape(h*w,c).to(device)
        detector=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0 for r in rows],device=device)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0 for j,r in enumerate(rows)],device=device)
        evidence=view_evidence(op,evop,visible,probability,detector>0,view['frame'],h,w,confidence,boundary_radius=data.get('boundary_radius',1))
        x=scope_features(evidence,probability,detector,confidence,torch.tensor([len(r['scores']) for r in rows],device=device,dtype=torch.float32),torch.sparse.mm(op,prior))
        native=torch.load(f"output/category_union_adapter/{name}/mask_adapter/frame_{view['frame']:05d}.pt",map_location='cpu')
        native_ids=[native['queries'].index(q['text']) for q in readout['queries']]
        features.append(torch.cat([x,native_identity_features(native,native_ids,device)],1).cpu())
        gt=torch.sparse.mm(op,semantics);valid=visible.float()*gt[:,-1:];truth=visible*gt[:,:-1]
        selected=(probability>.5).float();tp=(selected*truth).sum(0);target_mass=truth.sum(0)
        precision=tp/(selected*valid).sum(0).clamp_min(1e-8)
        recall=torch.where(target_mass>1e-8,tp/target_mass.clamp_min(1e-8),torch.ones_like(tp))
        targets.append(torch.stack([precision,recall],1).clamp(0,1).cpu())
        valid_targets.append(((detector>0)&((selected*valid).sum(0)>0)).cpu())
        positive=evidence.positive.float()*detector;negative=evidence.negative.float()*detector
        coefficient=torch.stack([torch.sparse.mm(evidence.adjoint,probability*positive),torch.sparse.mm(evidence.adjoint,positive),
            torch.sparse.mm(evidence.adjoint,probability*negative),torch.sparse.mm(evidence.adjoint,negative)],-1).reshape(n,c,2,2)
        ids=coefficient[:,:,:,1].sum((1,2)).nonzero().flatten()
        blocks.append(dict(ids=ids.cpu(),coefficient=coefficient[ids].cpu()))
        visibility+=torch.sparse.mm(evidence.adjoint,evidence.visible.float())
        known=(evidence.positive|evidence.negative).float();budget+=(known*confidence).sum(0);detector_budget+=(known*detector).sum(0)
    return dict(scene=name,features=torch.stack(features),targets=torch.stack(targets),valid_targets=torch.stack(valid_targets),
        blocks=blocks,prior=prior.cpu(),visibility=visibility.cpu(),ratio=(budget/detector_budget.clamp_min(1e-8)).cpu(),
        counts=torch.from_numpy(counts),observed=observed.cpu(),source_views=len(records),class_ids=class_ids,
        ground_truth_training=True,geometry_and_teacher_frozen=True,element_mapping='GT vertices to nearest geometry centre')

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(3);scenes=json.loads((ROOT/'split.json').read_text())['fit'];out=ROOT/'training';out.mkdir(exist_ok=True)
    for name in scenes[a.shard::a.shards]:
        data=compile_scene(name,a.device);torch.save(data,out/f'{name}.pt')
        print(name,data['source_views'],len(data['prior']),sum(len(b['ids']) for b in data['blocks']),'compiled',flush=True)

if __name__=='__main__':main()
