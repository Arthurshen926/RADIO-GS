"""Provide the declared query-free source inventory to the residual experiment.

Same axes as caption evidence; provenance explicitly states there are no captions.
Only training-source proposals are included; benchmark prompts/GT are not read.
"""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
r=a.root;info=json.loads((r/'teachers.json').read_text());assert sha256_file(r/'episodes.pt')==json.loads((r/'episodes.json').read_text())['sha256']
e=torch.load(r/'episodes.pt',weights_only=False);assert e['teacher_manifest_sha256']==sha256_file(r/'teachers.json')
rec=info['teacher_files']['siglip'];assert sha256_file(r/rec['path'])==rec['sha256'];t=torch.load(r/rec['path'],weights_only=False)
rejected={r['region'] for r in e.get('prompt_admission',{}).get('rejected_regions',[]) if r['reason']=='missing_positive_support'}
ids=torch.tensor([i for i,v in enumerate(e['proposal_views']) if info['frames'][int(v)] in info['train_frames'] and i not in rejected],dtype=torch.long)
if not len(ids):raise ValueError('no source regions with positive geometry support')
bundle=dict(schema='source_region_residual_evidence.v1',evidence_origin='query-free SAM native crops; no caption discovery',source_only=True,benchmark_data_used=False,
 teacher_manifest_sha256=e['teacher_manifest_sha256'],geometry_binding=e['geometry_binding'],train_frames=info['train_frames'],
 masks=e['masks'][ids],proposal_views=e['proposal_views'][ids],region_probability=e['region_probability'][ids],seen=e['seen'],native_descriptors=t['descriptors'][ids],
 source_proposal_indices=ids,rejected_no_positive_geometry_regions=sorted(rejected),episode_sha256=sha256_file(r/'episodes.pt'),implementation_sha256=sha256_file(__file__))
out=r/'caption_region_evidence.pt';assert not out.exists();torch.save(bundle,out)
out.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256_file(out),evidence_origin=bundle['evidence_origin'],compatibility_filename_only=True),indent=2)+'\n')
