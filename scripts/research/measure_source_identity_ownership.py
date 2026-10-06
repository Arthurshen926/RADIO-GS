import json,torch
from pathlib import Path
torch.set_num_threads(4)
rows=[]
for scene in sorted(Path('configs/pipeline/scannet_native_complete').glob('*.json')):
 d=json.loads(scene.read_text());total=multi=positive=borrowed=0;gap=0.
 records=[torch.load(v['identity'],map_location='cpu',weights_only=False) for v in d['source_evidence']['views']]
 square=torch.zeros(19);count=torch.zeros(19)
 for r in records:
  for j,x in enumerate(r['instances']):square[j]+=x['foreground_margin'].square().sum();count[j]+=len(x['scores'])
 scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
 for r in records:
  for j,x in enumerate(r['instances']):
   total+=1
   if not len(x['scores']):continue
   confidence=x['scores']*(x['foreground_margin']/scale[j]).sigmoid();prob,owner=x['masks'].float().flatten(1).max(0);valid=prob>.5
   positive+=int(valid.sum());multi+=int(len(confidence)>1)
   difference=(confidence.max()-confidence[owner])[valid]
   borrowed+=int((difference>1e-5).sum());gap+=float(difference.sum())
 rows.append(dict(scene=d['scene'],view_query_pairs=total,multi_proposal_pairs=multi,positive_pixels=positive,borrowed_positive_pixels=borrowed,mean_confidence_borrowed=gap/max(1,positive)))
 print(rows[-1],flush=True)
Path('output/native_evidence_contribution/identity_assignment_counts.json').write_text(json.dumps(rows,indent=2)+'\n')
