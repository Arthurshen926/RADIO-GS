"""Measure native-to-learned query geometry distortion on training inputs."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4)
if a.output.exists():raise FileExistsError(a.output)
r=json.loads((a.root/a.run/'report.json').read_text());receipt=json.loads((a.root/'episodes.json').read_text())
if sha256_file(a.root/a.run/'field.pt')!=r['field_sha256'] or sha256_file(a.root/'episodes.pt')!=receipt['sha256']:raise ValueError('identity mismatch')
f=torch.load(a.root/a.run/'field.pt',weights_only=False);data=torch.load(a.root/'episodes.pt',weights_only=False)
info=json.loads((a.root/'teachers.json').read_text())
if sha256_file(a.root/'teachers.json')!=data['teacher_manifest_sha256']:raise ValueError('teacher mismatch')
trainviews=[info['frames'].index(x) for x in info['train_frames']]
regions=torch.where(torch.isin(data['proposal_views'],torch.tensor(trainviews)))[0]
rows={}
for name,q in [('text',data['queries']['text'].float()),('image',data['queries']['image'][regions].float())]:
 native=F.normalize(q,dim=-1);projected=F.normalize(F.linear(q,f['state_dict']['query_adapters.'+name+'.weight']),dim=-1)
 keep=torch.triu(torch.ones(len(q),len(q),dtype=torch.bool),diagonal=1)
 x=(native@native.T)[keep];y=(projected@projected.T)[keep]
 rows[name]={'queries':len(q),'native_pair_mean':float(x.mean()),'projected_pair_mean':float(y.mean()),'pair_cosine_mae':float((x-y).abs().mean()),'pair_pearson':float(torch.corrcoef(torch.stack([x,y]))[0,1])}
a.output.write_text(json.dumps({'field_sha256':r['field_sha256'],'episode_sha256':receipt['sha256'],'input_scope':'fixed source vocabulary and all training-view image queries; no validation queries','relations':rows},indent=2)+'\n');print(rows)
