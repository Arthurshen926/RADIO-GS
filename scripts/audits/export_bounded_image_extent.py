"""Select one shared strength on source validation and export immutable controls."""
import json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 torch.set_num_threads(4)
 scenes=['figurines','ramen'];rootbase=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')
 sources={s:Path(f'paper/artifacts/{s}_bounded_image_extent_source_20260917.json') for s in scenes}
 fits={s:json.loads(p.read_text()) for s,p in sources.items()}
 grid=[0.,.5,1.,2.]
 eligible=[v for v in grid if all(fits[s]['macro'][str(v)][k]>=fits[s]['macro']['0.0'][k]-1e-10 for s in scenes for k in ['iou','average_precision'])]
 strength=max(eligible,key=lambda v:(sum(fits[s]['macro'][str(v)]['average_precision'] for s in scenes),-v))
 destination=Path('paper/artifacts/bounded_image_extent_selection_20260917.json')
 if destination.exists():raise FileExistsError(destination)
 result=dict(schema='bounded_image_extent_selection.v1',source_only=True,benchmark_data_used=False,
  policy='one shared strength maximizing equal-scene source-validation AP, requiring no source IoU/AP regression in either scene; zero is eligible; ties prefer smaller',
  selected_strength=strength,source_reports={s:sha256_file(p) for s,p in sources.items()},scenes={})
 for s in scenes:
  root=rootbase/s;fp=root/'exact_render_query_threshold96/field.pt';assert sha256_file(fp)==fits[s]['parent_field_sha256']
  payload=torch.load(fp,weights_only=False);name='bounded_image_extent96';out=root/name;out.mkdir(exist_ok=False)
  architecture=dict(payload['architecture'],bounded_extent={'image':strength})
  torch.save(dict(payload,architecture=architecture),out/'field.pt')
  auth=json.loads((root/'responsibility.json').read_text());gr=json.loads(Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914',s,'report.json').read_text())['results']['independent_scale']
  geometry=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cpu')
  old,_=load_element_retrieval_field(fp,expected_sha256=sha256_file(fp),geometry=geometry,expected_query_encoder=payload['query_encoder'],device='cpu')
  cold,_=load_element_retrieval_field(out/'field.pt',expected_sha256=sha256_file(out/'field.pt'),geometry=geometry,expected_query_encoder=payload['query_encoder'],device='cpu')
  for key,value in old.state_dict().items():assert torch.equal(value,cold.state_dict()[key]),key
  data=torch.load(root/'episodes.pt',weights_only=False)
  with torch.no_grad():
   unchanged={m:float((old.logits(data['queries'][m][:2].float(),m)-cold.logits(data['queries'][m][:2].float(),m)).abs().max()) for m in ['text','prompt']}
   assert all(v==0 for v in unchanged.values())
  report=dict(schema='bounded_image_extent_candidate.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=sha256_file(fp),
   source_only=True,benchmark_data_used=False,strength=strength,all_parent_tensors_unchanged=True,unchanged_response_errors=unchanged,
   source_validation_report_sha256=sha256_file(sources[s]),independent_test=False,
   implementation_sources={p:sha256_file(p) for p in ['radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_extent_readout.py','scripts/audits/export_bounded_image_extent.py']})
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');result['scenes'][s]=report;print(s,strength,unchanged,flush=True)
 destination.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
