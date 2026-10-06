"""Encode only literal query words; no generated descriptions or aliases."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.native_query_encoders import load_siglip_text
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 for r in json.loads((Path('output/grounded_seeds_next88')/('lerf_'+scene)/'proposals.json').read_text()):rows.append(dict(scene=scene,query=r['query'],query_index=r['query_index'],terms=r['query'].split()))
field=load('output/taskwise_scope_composition_next38/lerf_ramen/field.pt');root=Path(field['root']);info=json.loads((root/'teachers.json').read_text());teacher=load(root/info['teacher_files']['siglip']['path']);model,tokenizer=load_siglip_text(Path(teacher['encoder']['path']),torch.device('cuda'));terms=sorted({t for r in rows for t in r['terms']});inputs=tokenizer(terms,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
with torch.inference_mode():features=F.normalize(model(**inputs).pooler_output.float(),dim=-1).cpu()
out=Path('output/literal_terms_next105');out.mkdir(exist_ok=True,parents=True);torch.save(dict(rows=rows,terms=terms,features=features),out/'terms.pt');(out/'terms.json').write_text(json.dumps(rows,indent=2)+'\n')
