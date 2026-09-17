"""Seal source-evidence repair measurements without promoting a final model."""
import json,shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 artifacts=Path('paper/artifacts');output=artifacts/'source_inventory_repair_summary_20260916.json'
 if output.exists():raise FileExistsError(output)
 result={'schema':'source_inventory_repair.v1','final_model_promoted':False,'benchmark':'annotated LERF development','source_fixes':['preserve query-free parent regions when appending grounded evidence','gate normalized region probabilities by the same visibility threshold as seen'],'scenes':{}}
 for scene in ('figurines','ramen'):
  records={}
  for key,root in [('before','component_retrieval_pipeline_20260915'),('after','preserved_source_inventory_20260916')]:
   path=artifacts/f'{scene}_{root}_fusion_common_20260916.json';d=json.loads(path.read_text());records[key]={'path':str(path),'sha256':sha256_file(path),'fixed_support_sha256':d['fixed_support_sha256'],'support_elements':d['shared_valid_elements'],'source_regions':d['source_regions'],'metrics':d['macro']}
  assert records['before']['fixed_support_sha256']==records['after']['fixed_support_sha256']
  assert records['before']['support_elements']==records['after']['support_elements']
  delta=abs(records['before']['metrics']['learned_decoded_field']['average_precision']-records['after']['metrics']['learned_decoded_field']['average_precision'])
  assert delta<1e-5,delta
  records['unchanged_field_ap_error']=delta;result['scenes'][scene]=records
 integrity=artifacts/'preserved_source_inventory_integrity_20260916.json';result['integrity']={'path':str(integrity),'sha256':sha256_file(integrity)}
 files=['scripts/audits/prepare_component_source_root.py','radio_gs/scripts/compile_three_query_episodes.py','radio_gs/training/source_region_inventory.py','radio_gs/training/region_visibility.py','radio_gs/training/source_group_association.py','scripts/audits/build_source_group_diagnostic.py','scripts/audits/evaluate_source_group_diagnostic.py','scripts/audits/merge_source_group_diagnostic.py','scripts/audits/audit_actual_region_teacher_fusion.py']
 snapshots=Path('/mnt/pool/sqy/results/RADIO-GS/output/source_group_diagnostic_20260916/code');snapshots.mkdir(exist_ok=True);result['implementation']={}
 for name in files:
  path=Path(name);digest=sha256_file(path);snapshot=snapshots/(digest+'.py');shutil.copyfile(path,snapshot);result['implementation'][name]={'sha256':digest,'snapshot':str(snapshot)}
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['scenes'],indent=2))
if __name__=='__main__':main()
