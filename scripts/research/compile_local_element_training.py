"""All-view local evidence on the fixed official-training subset."""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.local_element_fusion import compile_local_evidence

ROOT=Path('output/local_element_fusion')

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(3);scenes=json.loads((ROOT/'split.json').read_text())['fit'];out=ROOT/'training';out.mkdir(exist_ok=True)
    for name in scenes[a.shard::a.shards]:
        spec=json.loads(Path(f'configs/pipeline/paired_source_resolution/low/{name}.scene.json').read_text())
        first=torch.load(f'output/system_v5/scannet/{name}/query/category/typed_readout.pt',map_location='cpu')
        second=torch.load(f'output/adapter_observation_control/{name}/mask_adapter/query/typed_readout.pt',map_location='cpu')
        data=compile_local_evidence(first['native_logits'].cuda(),first['semantic_observed'].cuda(),spec,[q['text'] for q in first['queries']],second['native_logits'].cuda())
        replay=dict(max_logit_error=float((data['base_logits'].cpu()-first['category_logits']).abs().max()),
            element_class_disagreements=int((data['base_logits'].cpu().argmax(1)!=first['category_logits'].argmax(1)).sum()))
        geo=FrozenGaussianGeometry.load(spec['geometry']['path'],expected_sha256=spec['geometry']['sha256'],device='cpu')
        evaluation=json.loads(Path(f'configs/consolidated/scannet_native/{name}.evaluation.json').read_text())
        vertices=PlyData.read(evaluation['ground_truth'])['vertex'].data;xyz=np.stack([vertices[k] for k in ['x','y','z']],1)
        near=cKDTree(geo.xyz.numpy()).query(xyz)[1];classes=[q['class_id'] for q in first['queries']]
        counts=np.zeros((len(geo.xyz),len(classes)),np.float32)
        for j,c in enumerate(classes):np.add.at(counts[:,j],near[vertices['label']==c],1)
        payload={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        payload.update(counts=torch.from_numpy(counts),observed=first['semantic_observed'],class_ids=classes,scene=name,replay=replay)
        torch.save(payload,out/f'{name}.pt');print(name,data['source_views'],len(geo.xyz),replay,flush=True)

if __name__=='__main__':main()
