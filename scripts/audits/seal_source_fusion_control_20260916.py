import hashlib,json,math
from pathlib import Path
import torch
base=Path('/mnt/pool/sqy/results/RADIO-GS/output/component_retrieval_pipeline_20260915')
out={'schema':'source_fusion_initialization_iteration.v1','promote_joint_default':False,'benchmark_scope':'LERF labeled development; not independent final test','scenes':{}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for scene,n in [('figurines',2106),('ramen',3240)]:
 root=base/scene;bundle=torch.load(root/'source_fusion_initialization.pt',weights_only=False);record={'initialization_sha256':sha(root/'source_fusion_initialization.pt'),'runs':{}}
 for name in [f'fusion_init_trainable_{n}',f'fusion_init_fixed_decoder_checked_{n}',f'fusion_init_bounded002_{n}']:
  path=root/name/'field.pt';report=json.loads((root/name/'report.json').read_text());payload=torch.load(path,weights_only=False);state=payload['state_dict'];assert sha(path)==report['field_sha256'];assert report['source_fusion_initialization_sha256']==record['initialization_sha256'];assert payload['geometry_binding']==bundle['geometry_binding']
  r={'field_sha256':sha(path),'report_sha256':sha(root/name/'report.json'),'final_source_probe':report['initialization_trace'][-1],'readout_calibration':report['readout_calibration'],'geometry_binding_matches':True}
  r['cosine_threshold_at_probability_half']={m:-v['bias']/v['scale'] for m,v in r['readout_calibration'].items()};r['impossible_positive_modalities']=[m for m,t in r['cosine_threshold_at_probability_half'].items() if t>1]
  if report['freeze_native_decoder']:
   assert torch.equal(state['native_decoder.weight'],bundle['basis'].T);assert torch.count_nonzero(state['native_decoder.bias'])==0;r['decoder_exactly_preserved']=True
  if report.get('source_fusion_max_drift',0):
   valid=bundle['valid'];original=bundle['latent'][valid];delta=(state['latent'][valid]-original).norm(dim=-1)/original.norm(dim=-1);r['all_supported_elements_max_relative_drift']=float(delta.max());r['all_supported_elements_checked']=int(valid.sum());assert delta.max()<.020001
  record['runs'][name]=r
 for kind in ['text','reference']:
  path=Path(f'paper/artifacts/{scene}_fusion_init_bounded_{kind}_20260916.json');d=json.loads(path.read_text());record[kind]={'path':str(path),'sha256':sha(path),'runs':{k:{a:b for a,b in v.items() if a not in ['observations','per_category_iou','prompt_refinement','image_conditioning']} for k,v in d['runs'].items()}}
 path=Path(f'paper/artifacts/{scene}_bounded_fusion_ranking_20260916.json');d=json.loads(path.read_text());record['common_support_ranking']={'path':str(path),'sha256':sha(path),'macro':d['macro'],'shared_valid_elements':d['shared_valid_elements']}
 out['scenes'][scene]=record
out['training_executions']={'completed_and_exported':6,'rejected_by_old_frozen_head_gradient_validator':2,'rejected_names':['figurines/fusion_init_fixed_decoder_2106','ramen/fusion_init_fixed_decoder_3240']}
Path('paper/artifacts/source_fusion_initialization_iteration_20260916.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({s:{k:{x:v for x,v in r.items() if x in ['all_supported_elements_max_relative_drift','all_supported_elements_checked','impossible_positive_modalities','cosine_threshold_at_probability_half']} for k,r in d['runs'].items()} for s,d in out['scenes'].items()},indent=2))
