"""Check modality isolation and old-row retention for measured support extension."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--extension',required=True);p.add_argument('--geometry-report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
models=[];digests={}
geometry=None
if a.geometry_report is not None:
 from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
 authority=json.loads((a.root/'responsibility.json').read_text())
 assert sha256_file(a.geometry_report)==authority['metadata']['frozen_geometry_report_sha256']
 g=json.loads(a.geometry_report.read_text())['results']['independent_scale']
 geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cpu')

for name in ['image_supervised_native_residual',a.extension]:
 path=a.root/name/'field.pt';r=json.loads((path.parent/'report.json').read_text());assert sha256_file(path)==r['field_sha256'];payload=torch.load(path,weights_only=False)
 f=ElementRetrievalField(**payload['architecture']).eval().requires_grad_(False);f.load_state_dict(payload['state_dict'])
 if f.affinity is not None or f.membership is not None:
  if geometry is None:raise ValueError('registered/structural query requires --geometry-report')
  geometry.assert_binding(payload['geometry_binding'])
  if f.affinity is not None:f.affinity.bind_geometry(geometry.xyz,payload['query_context']['prompt_world_normalization'])
  if f.membership is not None:f.membership.bind_geometry(geometry.xyz,f.observed)
 models.append(f);digests[name]=r['field_sha256']
old,new=models;metrics={}
fixture=torch.load(a.root/'cold_query_fixture.pt',weights_only=False)
with torch.no_grad():
 for modality in ('text','image','prompt'):
  q=torch.cat([fixture['expected_encodings'][modality].float(),torch.randn(4,old.query_adapters[modality].in_features)],0)
  before=old.retrieve(q,modality);after=new.retrieve(q,modality)
  rows=old.observed if modality=='image' else torch.ones(len(old.latent),dtype=torch.bool)
  delta=float((before['scores'][rows]-after['scores'][rows]).abs().max());assert delta==0,(modality,delta)
  if modality!='image':assert torch.equal(before['valid'],after['valid'])
  metrics[modality]=dict(max_error=delta,checked_rows=int(rows.sum()),queries=len(q),new_supported_rows=int(after['valid'].sum()))
result=dict(schema='native_image_support_retention.v1',fields=digests,modalities=metrics,query_set='one sealed native fixture and four seeded random encoder vectors per modality',
 arbitrary_query_guarantee='structural parameter/row isolation; this numerical fixture is finite, not exhaustive',implementation_sha256=sha256_file(__file__),module_sha256=sha256_file('radio_gs/models/element_retrieval_field.py'))
assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n');print(metrics)
