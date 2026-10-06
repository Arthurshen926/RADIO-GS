"""Aggregate completed mechanism controls, without opening benchmark labels."""
import json
from pathlib import Path
roots={'resolution4':'output/nvos_source_resolution4_next74','full_video':'output/nvos_transfer_full_video_next72','broad_replace':'output/reference_anchors_next74/eval_replace','broad_union':'output/reference_anchors_next74/eval_union','dominant_replace':'output/reference_surface_next74/eval_replace','dominant_union':'output/reference_surface_next74/eval_union','cover_replace':'output/reference_cover_next74/eval_replace','cover_union':'output/reference_cover_next74/eval_union','identity_union':'output/reference_identity_next74/evaluation','conditional_exclusion':'output/reference_exclusion_next74/evaluation','resolution2':'output/nvos_source_resolution_next74'}
results={}
for name,root in roots.items():
 results[name]={}
 for carrier in ['old','new']:
  scenes={p.parent.name:json.loads(p.read_text())['metrics'] for p in sorted(Path(root,carrier).glob('*/report.json'))}
  if not scenes:continue
  results[name][carrier]=dict(count=len(scenes),scenes=scenes,macro={k:sum(x[k] for x in scenes.values())/len(scenes) for k in ['iou','average_precision']})
out=Path('output/next74');out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
for k,v in results.items():
 for c,r in v.items():print(k,c,r['count'],r['macro'])
