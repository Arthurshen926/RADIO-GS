"""Change only source fusion; preserve geometry, basis, support and structural state."""
import argparse,json,os,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
a.output.mkdir(parents=True,exist_ok=False);info=json.loads((a.parent/'teachers.json').read_text());identities={}
# Immutable inputs only; never share a writable model directory.
files=['teachers.json','episodes.json','episodes.pt','responsibility.json','source_frames.json','native_query_encoder.json','queries.pt','cold_query_fixture.pt','cold_query_crop.png','grounded_text_supervision.pt','grounded_text_supervision.json','source_grounded_text.pt','source_grounded_text.json','caption_source_phrases.json','caption_grounded_regions.pt','caption_grounded_regions.json','caption_region_evidence.pt','caption_region_evidence.json','source_phrase_groups.json']
files += [r['path'] for r in info['teacher_files'].values()]
for name in sorted(set(files)):
 src=a.parent/name;identities[name]=sha256_file(src.resolve());dst=a.output/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src.resolve(),dst)
shutil.copytree(a.parent/'responsibility.json.views',a.output/'responsibility.json.views',copy_function=os.link)
def load(name,expected):
 assert sha256_file(a.parent/name)==expected
 return torch.load(a.parent/name,map_location='cpu',weights_only=False)
ep=load('episodes.pt',json.loads((a.parent/'episodes.json').read_text())['sha256']);assert ep['teacher_manifest_sha256']==identities['teachers.json']
auth=json.loads((a.parent/'responsibility.json').read_text());assert identities['responsibility.json']==info['authority_sha256']
tr=info['teacher_files']['siglip'];teacher=load(tr['path'],tr['sha256'])
parent=a.parent/'scale_affinity_v1_direct_control';pr=json.loads((parent/'report.json').read_text());payload=load('scale_affinity_v1_direct_control/field.pt',pr['field_sha256']);assert payload['geometry_binding']==ep['geometry_binding']
sd=payload['state_dict'];basis=sd['native_decoder.weight'].to(a.device);n=len(sd['latent']);desc=F.normalize(teacher['descriptors'].float().to(a.device),dim=-1)@basis
numerator=torch.zeros((n,basis.shape[1]),device=a.device);mass=torch.zeros(n,device=a.device)
for frame in info['train_frames']:
 vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256'])
 weights,_=source_pixel_evidence_weights(ep['masks'][ids].flatten(1).to(a.device),hit['pixel_ids'].long().to(a.device),hit['gaussian_ids'].long().to(a.device),hit['base_weights'].float().to(a.device),n,partition_overlap=True)
 numerator+=weights.T@desc[ids];mass+=weights.sum(0)
valid=sd['observed'].to(a.device);assert (mass[valid]>0).all();estimate=numerator/mass[:,None].clamp_min(1e-8);assert torch.isfinite(estimate).all()
state={k:v.clone() for k,v in sd.items()};state['latent'][valid.cpu(),:basis.shape[1]]=estimate[valid].cpu()
for k,v in sd.items():
 if k!='latent':assert torch.equal(v,state[k]),k
assert torch.equal(state['latent'][:,basis.shape[1]:],sd['latent'][:,basis.shape[1]:])
# Do not expose the old conditional-weight initialization as an initializer for this field.
out=a.output/'pixel_evidence96';out.mkdir();torch.save(dict(payload,state_dict=state),out/'field.pt')
r=dict(pr,field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],schema='pixel_evidence_fusion.v1',source_only=True,benchmark_data_used=False,geometry_unchanged=True,basis_unchanged=True,observed_unchanged=True,structural_suffix_exact=True,native_prefix_exactly_preserved=False,native_text_image_max_error=None,inherited_structural_metrics_are_parent_metrics=True,old_readout_heads_reused=False,inputs=identities,implementation_sha256=sha256_file(__file__),fusion_implementation_sha256=sha256_file('radio_gs/training/source_pixel_evidence.py'))
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');(a.output/'prepared_inputs.json').write_text(json.dumps(dict(input_hashes=identities,parent=str(a.parent),new_field_sha256=r['field_sha256'],old_initialization_not_used=True),indent=2)+'\n')
