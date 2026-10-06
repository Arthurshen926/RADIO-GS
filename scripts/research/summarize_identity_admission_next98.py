"""Counts of source-component teacher decisions, not ground-truth accuracy."""
import json
from pathlib import Path
rows=[]
for arm,base in [('direct','output/identity_tracks_next98'),('description','output/batched_description_identity_tracks_next98'),('category','output/category_identity_tracks_next98')]:
 for p in sorted(Path(base).glob('*/*.json')):
  d=json.loads(p.read_text());js=d['judgments'];rows.append(dict(arm=arm,key=p.parent.name,query=d['query'],query_index=d['query_index'],components=len(js),accepted=sum(j['answer']=='A' for j in js),rejected=sum(j['answer']=='B' for j in js),unknown=sum(j['answer']=='C' for j in js),accepted_pixels=sum(j['pixels'] for j in js if j['answer']=='A'),all_component_pixels=sum(j['pixels'] for j in js)))
Path('output/next98/identity_admission.json').write_text(json.dumps(rows,indent=2)+'\n')
for arm in ['direct','description','category']:
 r=[x for x in rows if x['arm']==arm];print(arm,len(r),{key:sum(x[key] for x in r) for key in ['components','accepted','rejected','unknown']})
