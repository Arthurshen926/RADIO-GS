"""Transfer only source-derived scope memberships onto the new native map."""
import argparse,json,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import torch
 from types import SimpleNamespace
 from radio_gs.querying.scannet_runtime import load,refine_from_declared_sources
 from radio_gs.querying.scope_composition import ScopeComposition
 torch.set_num_threads(4)
 root=Path('output/native_category_transfer')/a.scene
 data=load(root/'transferred/typed_readout.pt');scene=json.loads(Path(f'configs/consolidated/scannet/{a.scene}.scene.json').read_text());memory=load(scene['memory'])
 graph=ScopeComposition(memory['scope_hypergraph'],fixed_cores=memory.get('composition_fixed_cores',False));observed=data['semantic_observed'].cuda()
 model=SimpleNamespace(observed_for=lambda modality:observed)
 prior_logits=data['native_logits'].cuda();prior=prior_logits.sigmoid()*observed[:,None]
 with torch.inference_mode():
  refined=graph.refine(prior,data['encodings'].cuda(),model,'text')
  correction=(torch.logit(refined.clamp(1e-6,1-1e-6))-torch.logit(prior.clamp(1e-6,1-1e-6))).clamp(-1,1)
  logits=prior_logits+correction
  result,stats=refine_from_declared_sources(logits,observed,scene,data['texts'])
 data.update(native_logits=logits.cpu(),category_logits=result.cpu(),scores=result.cpu())
 out=root/'scope_transfer';out.mkdir(exist_ok=True);torch.save(data,out/'typed_readout.pt')
 (out/'result.json').write_text(json.dumps(dict(structure='source-derived nonnegative overlapping memberships; no old semantic field',scope_steps=80,max_logit_correction=1,source_solver=stats),indent=2)+'\n')
if __name__=='__main__':main()
