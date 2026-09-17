"""Fresh native text/image/prompt encoding with all training caches forbidden."""
import argparse,json,sys
from pathlib import Path
import torch
from PIL import Image
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--field',type=Path,required=True);p.add_argument('--field-sha256',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True);p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256');p.add_argument('--prompt-frame',type=int);p.add_argument('--minimal-prompt-refinement',action='store_true')
p.add_argument('--require-concept-route-consistency',action='store_true')
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
def deny_training(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=str(args[0])
  if any(x in path for x in ('teacher_radio.pt','teacher_dino.pt','teacher_sam.pt','teacher_siglip.pt','episodes.pt','queries.pt','canonical_teacher_features','grounded_text_supervision','source_grounded_text','source_context_text','consistent_native_element_teacher')):
   raise RuntimeError('training feature/label access forbidden: '+path)
sys.addaudithook(deny_training)
f=torch.load(a.fixture,map_location='cpu',weights_only=False)
geometry=FrozenGaussianGeometry.load(f['geometry_path'],expected_sha256=f['geometry_sha256'],device=a.device)
field,_=load_element_retrieval_field(a.field,expected_sha256=a.field_sha256,geometry=geometry,expected_query_encoder=f['encoder_identity'],device=a.device)
encoder=NativeThreeQueryEncoder(f['encoder_identity'],a.device)
for key in ('image','prompt_image'):
 if sha256_file(f[key]['path'])!=f[key]['sha256']:raise ValueError('query image identity')
encoded={}
encoded['text']=encoder.encode_text([f['text']]);print('native text encoded',flush=True)
encoded['image']=encoder.encode_image(Image.open(f['image']['path']));print('native image encoded',flush=True)
# Free SigLIP before loading the independent native DINO prompt encoder.
encoder.vision=None;encoder.text_model=None
import gc
gc.collect();torch.cuda.empty_cache()
normal=f['prompt_world_normalization'];seedrows=f['seed_rows'].to(a.device)
encoded['prompt']=encoder.encode_prompt(f['prompt_image']['path'],f['seed_pixels'],geometry.xyz[seedrows],normal['minimum'],normal['extent'],f['height'],f['width']);print('native prompt encoded',flush=True)
results={}
with torch.no_grad():
 for m,q in encoded.items():
  q=q.half().float();reference=f['expected_encodings'][m].float().to(a.device)
  cosine=float(torch.nn.functional.cosine_similarity(q,reference).item())
  if cosine<.999:raise ValueError('native encoder differs from training encoder: '+m)
  scores=field.retrieve(q,m);baseline=field.retrieve(reference,m)
  if not torch.isfinite(scores['scores']).all():raise ValueError('nonfinite retrieval')
  results[m]={'native_vs_training_encoding_cosine':cosine,'elements':len(scores['scores']),'valid_elements':int(scores['valid'].sum()),'posterior_max_difference':float((scores['scores']-baseline['scores']).abs().max())}
  valid=scores['valid']
  changed=(scores['scores']>.5)!=(baseline['scores']>.5)
  results[m]['membership_changed_elements']=int(changed[valid].sum())
  if m=='text' and field.grounded_transfer is not None:
   online_route=field.grounded_transfer.route(q);cached_route=field.grounded_transfer.route(reference)
   consistent=all(torch.equal(online_route[i],cached_route[i]) for i in (0,1))
   results[m]['concept_route_consistent']=consistent
   results[m]['online_concept_route']={'index':online_route[0].tolist(),'accepted':online_route[1].tolist()}
   results[m]['cached_concept_route']={'index':cached_route[0].tolist(),'accepted':cached_route[1].tolist()}
   if a.require_concept_route_consistency and not consistent:raise ValueError('cold concept decision changes despite near-identical native encoding')
refined_report=None
if a.authority is not None:
 from radio_gs.models.signed_query_refinement import condition_signed_query,retrieve_latent_query
 if not a.authority_sha256 or sha256_file(a.authority)!=a.authority_sha256:raise ValueError('prompt authority identity')
 authority=json.loads(a.authority.read_text())
 geometry.assert_binding(authority['metadata']['frozen_geometry_binding'])
 if a.prompt_frame not in authority['frame_indices']:raise ValueError('registered prompt frame absent')
 if [f['height'],f['width']] != [authority['metadata']['feature_height'],authority['metadata']['feature_width']]:raise ValueError('prompt raster differs')
 record=authority['views'][authority['frame_indices'].index(a.prompt_frame)];path=a.authority.parent/record['relative_path']
 if sha256_file(path)!=record['sha256']:raise ValueError('source-pixel contribution identity')
 hit=torch.load(path,map_location=a.device,weights_only=False);rows=[];weights=[]
 for i,pixel in enumerate(f['seed_pixels'].tolist()):
  selected=hit['pixel_ids']==pixel;ids=hit['gaussian_ids'][selected].long();weight=hit['base_weights'][selected].float()
  if len(ids)==0 or int(ids[weight.argmax()])!=int(f['seed_rows'][i]):raise ValueError('signed input is registered to another camera/pixel')
  rows.append(ids);weights.append(weight)
 refinement=condition_signed_query(field,encoded['prompt'].half().float(),rows,weights,[1.,0.],stop_when_satisfied=a.minimal_prompt_refinement)
 with torch.no_grad():
  posterior=retrieve_latent_query(field,refinement['latent_query'])
  if not torch.isfinite(posterior['scores']).all():raise ValueError('nonfinite conditioned posterior')
 refined_report={k:v.tolist() if torch.is_tensor(v) else v for k,v in refinement.items() if k!='latent_query'}
 refined_report.update(geometry_authority_sha256=a.authority_sha256,prompt_frame=a.prompt_frame,elements=len(posterior['scores']),target_label_access=False)
 if any(parameter.grad is not None for parameter in field.parameters()):raise ValueError('query refinement wrote field gradients')
geometry.assert_binding(f['geometry_binding'])
report={'schema':'native_three_query_cold.v1','training_cache_reads_forbidden':True,'native_queries_encoded_online':True,'geometry_unchanged':True,'field_sha256':a.field_sha256,'fixture_sha256':sha256_file(a.fixture),'modalities':results,'quality_evaluation':False,'registered_prompt_refinement':refined_report}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
