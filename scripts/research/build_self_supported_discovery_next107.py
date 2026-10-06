"""Weight newly discovered scope by the detector's recovery of its own example.
This measures prompt compliance, not semantic truth. Preserve the old core;
never recursively certify discoveries. Optional unknown-expansions also withholds
negative supervision on proposed extensions after reducing their confidence.
"""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--unknown-expansions',action='store_true');a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;load=lambda p:torch.load(p,map_location='cpu',weights_only=False);out=Path('output/set_discovery_next107')/('self_supported_unknown' if a.unknown_expansions else 'self_supported')/key;out.mkdir(parents=True,exist_ok=True)
for source in sorted((Path('output/set_discovery_next107/box')/key).glob('*.pt')):
 qi=int(source.stem);t=load(source);original=load(Path('output/core_merged_next104/union')/key/source.name);rows=[]
 for f,rec in t['frames'].items():
  if 'detector_instances' not in rec:continue
  core=original['frames'][f]['probability'].float();mask=core>.5;instances=rec['detector_instances'].float();additional=instances.amax(0) if len(instances) else torch.zeros_like(core);compliance=float(additional[mask].mean());new=torch.maximum(core,additional*compliance);rec.update(probability=new.half(),present=bool((new>.5).any()),example_recovery=compliance);rows.append(dict(frame=f,example_recovery=compliance,core_pixels=int(mask.sum()),raw_added_pixels=int(((additional>.5)&~mask).sum()),weighted_added_pixels=int(((new>.5)&~mask).sum())))
  if a.unknown_expansions:
   rec['positive_allowed']=torch.ones_like(mask)
   rec['scope_reliability']=1.0
   rec['negative_allowed']=~((additional>.5)&~mask)
 t['source']='core-preserving visual discovery weighted by soft union recall on original positive example; not verified identity';torch.save(t,out/source.name);(out/f'{qi}.json').write_text(json.dumps(dict(query=t['query'],query_index=qi,rows=rows,source_only=True,unknown_expansions=a.unknown_expansions,thresholds_added=False),indent=2)+'\n')
print(key,'self-supported discovery built',flush=True)
