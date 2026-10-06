"""Read completed benchmark outputs; do not run prediction or inspect labels."""
import json
from pathlib import Path
roots={'full_source_video':'output/nvos_transfer_full_video_next72','video_only':'output/nvos_transfer_next72','joint_prior':'output/nvos_evidence_next73','prior_initialization':'output/nvos_completion_next73','unsupported_completion':'output/nvos_unsupported_next73'}
result={}
for name,root in roots.items():
 result[name]={}
 for carrier in ['old','new']:
  scenes={p.parent.name:json.loads(p.read_text())['metrics'] for p in sorted(Path(root,carrier).glob('*/report.json'))}
  result[name][carrier]={'completed':len(scenes),'scenes':scenes,'macro':{k:sum(v[k] for v in scenes.values())/len(scenes) for k in ['iou','average_precision']} if scenes else {}}
out=Path('output/next73');out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for name,arms in result.items():
 for carrier,r in arms.items():print(name,carrier,r['completed'],r['macro'])
