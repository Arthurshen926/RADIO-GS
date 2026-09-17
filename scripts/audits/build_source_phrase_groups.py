"""Materialize phrase groups without fitting the rejected membership codebook."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
root=a.root;path=root/'caption_region_evidence.pt';r=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==r['sha256']
e=torch.load(path,weights_only=False);assert e['source_only'] and not e['benchmark_data_used']
parent=root/'scale_affinity_v1_direct_control';pr=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==pr['field_sha256']
field=torch.load(parent/'field.pt',weights_only=False);assert field['geometry_binding']==e['geometry_binding'];observed=field['state_dict']['observed'];n=len(observed)
groups=[]
for qi,phrase in enumerate(e['phrases']):
    ids=[i for i,r in enumerate(e['records']) if qi in r['phrase_indices']]
    views=sorted({int(e['proposal_views'][i]) for i in ids})
    if len(views)<2:continue
    numerator=torch.zeros(n);denominator=torch.zeros(n)
    for vi in views:
        members=[i for i in ids if int(e['proposal_views'][i])==vi]
        numerator+=e['region_probability'][members].float().amax(0);denominator+=e['seen'][vi]
    membership=(numerator/denominator.clamp_min(1))*observed
    if int((membership>.5).sum())<20:continue
    groups.append(dict(phrase=phrase,source_query_index=qi,views=views,regions=ids))
if not groups:raise ValueError('no repeated source phrase with reliable carrier support')
out=root/'source_phrase_groups.json';assert not out.exists()
out.write_text(json.dumps(dict(groups=groups,evidence_sha256=r['sha256'],parent_field_sha256=pr['field_sha256'],source_only=True,benchmark_data_used=False,physical_instance_identity_verified=False,policy='Same source phrase in at least two views and >=20 positive observed carriers; category-union hypotheses, not instance identity.',implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
