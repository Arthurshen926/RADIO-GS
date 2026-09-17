"""Rebuild a native anchor with equal source-view votes; retain the source basis.

Only immutable field-independent teacher/geometry evidence is linked. No field,
calibrator, membership head or old per-element anchor is copied as a model.
"""
import argparse,json,os,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.source_evidence_balancing import balance_source_views_and_origins
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
a.output.mkdir(parents=True,exist_ok=False)
files=['teachers.json','episodes.json','episodes.pt','responsibility.json','source_frames.json','native_query_encoder.json','queries.pt','cold_query_fixture.pt','cold_query_crop.png','grounded_text_supervision.pt','grounded_text_supervision.json','source_grounded_text.pt','source_grounded_text.json','caption_source_phrases.json','caption_grounded_regions.pt','caption_grounded_regions.json','caption_region_evidence.pt','caption_region_evidence.json']
info=json.loads((a.parent/'teachers.json').read_text());files += [r['path'] for r in info['teacher_files'].values()]
identities={}
for name in files:
 src=a.parent/name;identities[name]=sha256_file(src);os.link(src,a.output/name)
shutil.copytree(a.parent/'responsibility.json.views',a.output/'responsibility.json.views',copy_function=os.link)
ep=torch.load(a.parent/'episodes.pt',map_location='cpu',weights_only=False);assert ep['teacher_manifest_sha256']==sha256_file(a.parent/'teachers.json');assert identities['episodes.pt']==json.loads((a.parent/'episodes.json').read_text())['sha256']
sr=info['teacher_files']['siglip'];assert identities[sr['path']]==sr['sha256'];teacher=torch.load(a.parent/sr['path'],weights_only=False,map_location='cpu')
parentinit=a.parent/'source_fusion_initialization.pt';receipt=json.loads(parentinit.with_suffix('.json').read_text());assert sha256_file(parentinit)==receipt['sha256'];b=torch.load(parentinit,weights_only=False,map_location='cpu');assert b['episode_sha256']==identities['episodes.pt'] and b['siglip_teacher_sha256']==sr['sha256']
regions=torch.tensor(b['source_regions']);views=ep['proposal_views'][regions];assert all(info['frames'][int(v)] in info['train_frames'] for v in views)
compressed=F.normalize(teacher['descriptors'][regions].float(),dim=-1)@b['basis'].T
latent,mass=balance_source_views_and_origins(compressed,ep['region_probability'][regions].float(),views,torch.zeros_like(views));valid=b['valid'];assert (latent[valid].norm(dim=-1)>1e-8).all() and (mass[valid]>0).all()
new=dict(b,latent=latent*valid[:,None],fusion_policy='equal_available_source_views',source_initialization_parent_sha256=receipt['sha256'],basis_unchanged=True,implementation_sha256=sha256_file('radio_gs/training/source_evidence_balancing.py'))
path=a.output/'source_fusion_initialization.pt';torch.save(new,path);path.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256_file(path),valid_elements=int(valid.sum()),energy_retained=b['energy_retained'],benchmark_data_used=False,policy=new['fusion_policy'],basis_unchanged=True,source_parent_sha256=receipt['sha256']),indent=2)+'\n')
(a.output/'prepared_inputs.json').write_text(json.dumps(dict(parent=str(a.parent.resolve()),input_hashes=identities,field_independent_teacher_evidence_only=True,old_model_state_reused=False,basis_reused=True,benchmark_data_used=False,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
print(a.output,'view-balanced initialization prepared')
