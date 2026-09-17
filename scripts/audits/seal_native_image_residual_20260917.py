"""Seal current scoped candidate and all-metric retention against old control."""
import json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4)
out=Path('paper/artifacts/native_image_residual_seal_20260917.json');assert not out.exists()
snapshot=Path('/mnt/pool/sqy/results/RADIO-GS/output/native_image_residual_sources_20260917');snapshot.mkdir(exist_ok=False)
paths=['radio_gs/models/element_retrieval_field.py','radio_gs/models/native_decoder_readout.py','radio_gs/models/bounded_native_residual.py','radio_gs/models/bounded_extent_readout.py',
       'scripts/audits/fit_bounded_native_residual.py','scripts/audits/train_rendered_native_residual.py','scripts/audits/export_image_supervised_native_residual.py',
       'scripts/audits/calibrate_frozen_source_images.py','scripts/audits/evaluate_functional_queries.py','scripts/audits/evaluate_real_reference_queries.py','scripts/audits/evaluate_frozen_field_lerf_labels.py',
       'tests/test_bounded_native_residual.py']
sources={}
for p in paths:
 dest=snapshot/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[p]=dict(sha256=sha256_file(p),snapshot=str(dest))
scenes={}
for scene in ['figurines','ramen']:
 root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene
 old=root/'exact_render_query_threshold96';new=root/'image_supervised_native_residual'
 parent=torch.load(old/'field.pt',weights_only=False);candidate=torch.load(new/'field.pt',weights_only=False)
 for folder in (old,new):assert sha256_file(folder/'field.pt')==json.loads((folder/'report.json').read_text())['field_sha256']
 assert parent['geometry_binding']==candidate['geometry_binding'] and parent['query_encoder']==candidate['query_encoder']
 for k,v in parent['state_dict'].items():assert torch.equal(v,candidate['state_dict'][k]),k
 assert set(candidate['state_dict'])-set(parent['state_dict'])=={'native_residual_decoder.weight','native_residual_decoder.bias'}
 identities={};metrics={}
 for kind in ['reference','text']:
  op=Path(f'paper/artifacts/{scene}_exact_render_query_threshold_{kind}_20260916.json');np=Path(f'paper/artifacts/{scene}_image_supervised_native_residual_{kind}_20260917.json')
  before=json.loads(op.read_text())['runs']['exact_render_query_threshold96'];after=json.loads(np.read_text())['runs']['image_supervised_native_residual']
  assert after['field_sha256']==sha256_file(new/'field.pt')
  if kind=='text':assert before['observations']==after['observations']
  else:
   assert len(before['observations'])==len(after['observations'])
   for b,a in zip(before['observations'],after['observations']):
    assert (b['reference'],b['target_frame'],b['category'])==(a['reference'],a['target_frame'],a['category'])
    assert b['text']==a['text'] and b['prompt']==a['prompt']
   metrics=dict(before=before['macro']['image'],after=after['macro']['image'])
  identities[str(op)]=sha256_file(op);identities[str(np)]=sha256_file(np)
 scenes[scene]=dict(control_sha256=sha256_file(old/'field.pt'),candidate_sha256=sha256_file(new/'field.pt'),
  all_parent_tensors_exact=True,geometry_encoder_exact=True,full_text_observations_exact=True,reference_text_prompt_observations_exact=True,image=metrics,artifacts=identities)
result=dict(schema='native_image_residual_seal.v1',scenes=scenes,current_sources=sources,
 source_version_scope='Snapshots are final reusable source; earlier fit/evaluation receipts retain their own historical source hashes. Root/parent argument generalization and image-only scope followed shared controls.',
 acceptance='Scoped two-scene development candidate; independent joint benchmark not passed; retained controls preserved')
out.write_text(json.dumps(result,indent=2)+'\n');print(out)
