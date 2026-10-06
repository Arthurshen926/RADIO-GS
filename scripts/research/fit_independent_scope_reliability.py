"""Fit two scope-quality heads on official training rooms, before validation.

Training targets use projected annotated mesh classes; validation labels are
never opened here. This is an explicitly supervised track, not zero-shot tuning.
"""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F
from radio_gs.querying.typed_evidence import visible_operator,view_evidence
from radio_gs.querying.supervised_scope_reliability import FEATURE_NAMES,scope_features
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry

ROOT=Path('output/independent_source_reliability')

@torch.inference_mode()
def samples(name,device,spec_path=None,readout_path=None,evaluation_path=None):
    spec=json.loads(Path(spec_path or f'configs/pipeline/paired_source_resolution/low/{name}.scene.json').read_text())
    readout=torch.load(readout_path or f'output/adapter_observation_control/{name}/mask_adapter/query/typed_readout.pt',map_location='cpu',weights_only=False)
    prior=readout['native_logits'].sigmoid().to(device);observed=readout['semantic_observed'].to(device)
    geo=FrozenGaussianGeometry.load(spec['geometry']['path'],expected_sha256=spec['geometry']['sha256'],device='cpu')
    evaluation=json.loads(Path(evaluation_path or f'configs/consolidated/scannet_native/{name}.evaluation.json').read_text())
    v=PlyData.read(evaluation['ground_truth'])['vertex'].data;xyz=np.stack([v[c] for c in ['x','y','z']],1)
    nearest=cKDTree(xyz).query(geo.xyz.numpy())[1];label=torch.as_tensor(np.array(v['label'][nearest],dtype=np.int64,copy=True),device=device)
    ids=torch.tensor([q['class_id'] for q in readout['queries']],device=device)
    semantics=torch.cat([(label[:,None]==ids).float(),(label>0)[:,None].float()],1)
    records=[torch.load(v['identity'],map_location='cpu',weights_only=False) for v in spec['source_evidence']['views']]
    square=torch.zeros(len(ids));count=torch.zeros_like(square)
    for data in records:
        for j,r in enumerate(data['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);features=[];targets=[];frame_quality={}
    for view,data in zip(spec['source_evidence']['views'],records):
        h,w=data['raster_hw'];rows=data['instances']
        op,evop,visible=visible_operator(torch.load(view['contributions'],map_location='cpu',weights_only=False),observed,h*w,False)
        probability=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in rows],-1).reshape(h*w,-1).to(device)
        detector=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0 for r in rows],device=device)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0 for j,r in enumerate(rows)],device=device)
        evidence=view_evidence(op,evop,visible,probability,detector>0,view['frame'],h,w,confidence)
        x=scope_features(evidence,probability,detector,confidence,torch.tensor([len(r['scores']) for r in rows],device=device,dtype=torch.float32),torch.sparse.mm(op,prior))
        gt=torch.sparse.mm(op,semantics);valid=visible.float()*gt[:,-1:];truth=gt[:,:-1]*visible
        selected=(probability>.5).float();tp=(selected*truth).sum(0)
        precision=tp/(selected*valid).sum(0).clamp_min(1e-8)
        true_mass=truth.sum(0);recall=torch.where(true_mass>1e-8,tp/true_mass.clamp_min(1e-8),torch.ones_like(tp))
        active=(detector>0)&((selected*valid).sum(0)>0)
        features.append(x[active].cpu());targets.append(torch.stack([precision,recall],1)[active].clamp(0,1).cpu())
        frame_quality[view['frame']]=torch.where(active[:,None],torch.stack([precision,recall],1).clamp(0,1),torch.ones(len(ids),2,device=device)).cpu()
    return dict(features=torch.cat(features),targets=torch.cat(targets),frame_quality=frame_quality,scene=name,source_views=len(records),target='visible labelled mesh projection precision and recall')

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--fit-only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    scenes=json.loads((ROOT/'split.json').read_text())['fit'];out=ROOT/'training';out.mkdir(parents=True,exist_ok=True)
    if not a.fit_only:
        for name in scenes[a.shard::a.shards]:
            data=samples(name,'cuda');torch.save(data,out/f'{name}.pt');print(name,len(data['features']),'training examples',flush=True)
        return
    torch.manual_seed(0);rows=[torch.load(out/f'{n}.pt',weights_only=False) for n in scenes]
    x=torch.cat([r['features'] for r in rows]);y=torch.cat([r['targets'] for r in rows]);weight=torch.cat([torch.full((len(r['features']),),1/len(r['features'])) for r in rows]);weight/=weight.sum()
    mean=x.mean(0);std=x.std(0).clamp_min(1e-4);x=(x-mean)/std
    model=torch.nn.Linear(x.shape[1],2);torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=(F.binary_cross_entropy_with_logits(model(x),y,reduction='none').mean(1)*weight).sum()+.001*model.weight.square().mean();loss.backward();return loss
    optimizer.step(closure)
    with torch.no_grad():mae=(model(x).sigmoid()-y).abs().mean(0).tolist()
    state=dict(weight=model.weight.detach(),bias=model.bias.detach(),mean=mean,std=std,feature_names=FEATURE_NAMES,fit_scenes=scenes,
        targets=['mask_precision','visible_category_recall'],class_ids_as_features=False,ground_truth_training=True,validation_labels_read=False,l2=.001,optimization='LBFGS100 fixed; no validation selection')
    torch.save(state,ROOT/'scope_reliability.pt')
    (ROOT/'fit.json').write_text(json.dumps(dict(scenes=scenes,examples=len(x),training_mae=mae,training_labels_used=True,validation_used=False,coefficients=model.weight.detach().tolist()),indent=2)+'\n');print('fit',len(x),mae,flush=True)

if __name__=='__main__':main()
