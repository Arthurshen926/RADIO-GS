"""Seal image readout experiments and their unchanged-modality controls."""
import json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 torch.set_num_threads(4)
 out=Path('paper/artifacts/image_extent_stage_seal_20260917.json')
 if out.exists():raise FileExistsError(out)
 snapshot=Path('/mnt/pool/sqy/results/RADIO-GS/output/image_extent_stage_snapshot_20260917');snapshot.mkdir(exist_ok=False)
 sources=['radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_extent_readout.py','radio_gs/models/native_query_threshold.py','scripts/audits/fit_bounded_image_extent.py','scripts/audits/export_bounded_image_extent.py','scripts/audits/fit_native_image_query_threshold.py','scripts/audits/audit_image_query_score_ceiling.py','scripts/audits/evaluate_real_reference_queries.py','scripts/audits/seal_image_extent_stage.py','tests/test_bounded_extent_readout.py']
 bindings={}
 for name in sources:
  dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest);assert sha256_file(name)==sha256_file(dest);bindings[name]=dict(sha256=sha256_file(name),snapshot=str(dest))
 artifacts={};summaries={}
 for scene,identity in [('figurines','502469ad7f43febf39022aa2bc8d5c78aed57d0d662a14ee2090b994c86c7561'),('ramen','d86d5d324500c45ec74a7b35b55916be09da01cc5aa0e9bbbe89f684f7fa1856')]:
  root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene
  fp=root/'exact_render_query_threshold96/field.pt';assert sha256_file(fp)==identity
  parent=torch.load(fp,weights_only=False)['state_dict'];summaries[scene]={}
  for candidate,tag in [('bounded_image_extent96','bounded_image_extent'),('native_image_query_threshold96','native_image_threshold')]:
   cp=root/candidate/'field.pt';report=json.loads((cp.parent/'report.json').read_text());assert sha256_file(cp)==report['field_sha256']
   state=torch.load(cp,weights_only=False)['state_dict']
   for k,v in parent.items():assert torch.equal(v,state[k]),k
   resultpath=Path(f'paper/artifacts/{scene}_{tag}_reference_20260917.json');result=json.loads(resultpath.read_text());before=result['runs']['exact_render_query_threshold96'];after=result['runs'][candidate]
   assert before['field_sha256']==identity and after['field_sha256']==report['field_sha256']
   assert before['tasks_per_modality']==after['tasks_per_modality']
   for a,b in zip(before['observations'],after['observations']):
    for key in ['category','reference_frame','target_frame','text','prompt']:assert a[key]==b[key],key
   summaries[scene][candidate]=dict(before=before['macro']['image'],after=after['macro']['image'],observations=before['tasks_per_modality'])
   for p in [cp,cp.parent/'report.json',resultpath,resultpath.with_suffix('.protocol.json')]:artifacts[str(p.resolve())]=sha256_file(p)
  ceiling=Path(f'paper/artifacts/{scene}_image_query_score_ceiling_20260917.json');diag=json.loads(ceiling.read_text());assert not any(r['geometrically_unattainable_threshold'] for r in diag['rows'])
  cache=Path(diag['query_cache']);assert sha256_file(cache)==diag['query_cache_sha256']
  for p in [fp,ceiling,cache,Path(f'paper/artifacts/{scene}_bounded_image_extent_source_20260917.json')]:artifacts[str(p.resolve())]=sha256_file(p)
 selection=Path('paper/artifacts/bounded_image_extent_selection_20260917.json');artifacts[str(selection.resolve())]=sha256_file(selection)
 out.write_text(json.dumps(dict(schema='image_extent_stage_seal.v1',sources=bindings,artifacts=artifacts,metrics=summaries,
  tests_passed=35,complete_mainline_unchanged=True,text_prompt_observations_exactly_preserved=True,
  bounded_candidate='mixed development outcome; retained as opt-in only',image_threshold_candidate='rejected',
  scope_consistent_full_retraining_complete=False,independent_test=False),indent=2)+'\n')
 print(out,'sealed',len(artifacts),'artifacts',len(sources),'sources')

if __name__=='__main__':main()
