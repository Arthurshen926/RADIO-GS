"""Fresh unseen wordings: compiled field versus uncompiled observation operator.

No semantic labels are loaded; this verifies new-query execution/equivalence,
not recognition accuracy or four-query ability.
"""
import json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
out=Path('output/compiled_native_observations/fresh_queries');out.mkdir(parents=True,exist_ok=True)
original=json.loads(Path('configs/consolidated/scannet_native/scene0062_00.queries.json').read_text())
queries={'queries':[dict(id=str(i),modality='text',intent='category',text='a photograph of '+q['text']) for i,q in enumerate(original['queries'])]}
(out/'queries.json').write_text(json.dumps(queries,indent=2)+'\n')
for arm,path in [('compiled','output/compiled_native_observations/scene0062_00/scene.json'),('uncompiled','configs/pipeline/native_observation/scene0062_00.scene.json')]:
 scene=json.loads(Path(path).read_text());scene.pop('source_evidence',None);scene.pop('source_rgb_authority',None)
 if arm=='compiled':
  for key in ['native_field','native_region_directories','source_cameras']:scene.pop(key,None)
 (out/f'{arm}.scene.json').write_text(json.dumps(scene,indent=2)+'\n')
 with (out/f'{arm}.log').open('w') as log:
  subprocess.run([PYTHON,'-m','radio_gs.pipeline','--scene',str(out/f'{arm}.scene.json'),'--queries',str(out/'queries.json'),'--access','map-only','--output',str(out/arm),'--gpu','2'],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
import torch
a=torch.load(out/'compiled/category/typed_readout.pt',map_location='cpu',weights_only=False)
b=torch.load(out/'uncompiled/category/typed_readout.pt',map_location='cpu',weights_only=False)
observed=a['semantic_observed']
p,q=a['native_logits'].sigmoid(),b['native_logits'].sigmoid()
result=dict(query_count=len(queries['queries']),max_probability_difference=float((p[observed]-q[observed]).abs().max()),argmax_difference=int((p[observed].argmax(1)!=q[observed].argmax(1)).sum()),support_difference=int((observed!=b['semantic_observed']).sum()),ground_truth_read=False,accuracy_claim=False,compiled_source_assets_removed=True)
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
