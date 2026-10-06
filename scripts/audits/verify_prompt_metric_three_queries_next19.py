import json,gc,torch
from pathlib import Path
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4);base=Path('output/full_field_rebuild_next19/teatime/reconstruction/field');gp=json.loads(Path('output/depth_opacity_repair_20260918/teatime/geometry_report.json').read_text())['results']['independent_scale'];geo=FrozenGaussianGeometry.load(gp['geometry'],expected_sha256=gp['geometry_sha256'],device='cuda:0');ep=torch.load(base/'episodes.pt',map_location='cpu');queries={m:ep['queries'][m][:3].float() for m in ['text','image']};del ep
results={}
for label,path in [('parent',base/'bounded_fallback_admission/field.pt'),('regional',Path('output/prompt_metric_only_next19/teatime/prompt_metric_only/field.pt')),('deployment',Path('output/prompt_metric_deployscale_next19/teatime/prompt_metric_only/field.pt'))]:
 field,payload=load_element_retrieval_field(path,expected_sha256=sha256_file(path),geometry=geo,expected_query_encoder=json.loads((base/'native_query_encoder.json').read_text()),device='cuda:0');results[label]={}
 with torch.no_grad():
  for m,qs in queries.items():results[label][m]=torch.cat([field.retrieve(q[None].cuda(),m)['scores'].cpu() for q in qs],dim=1)
 del field,payload;gc.collect();torch.cuda.empty_cache()
for label,root in [('regional','prompt_metric_only_next19'),('deployment','prompt_metric_deployscale_next19')]:
 errors={m:float((results[label][m]-results['parent'][m]).abs().max()) for m in queries};assert max(errors.values())==0;Path('output',root,'teatime','three_query_score_verification.json').write_text(json.dumps(dict(queries_per_modality=3,sequential_batches=1,max_error=errors,all_exact=True),indent=2));print(label,errors)
