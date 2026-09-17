"""Verify bounded benefit-admission controls and unchanged non-text retrieval."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 result=dict(schema='extent_benefit_controls.v1',independent_final_test=False,scenes={},implementation_sha256=sha256_file(__file__))
 for scene in ['figurines','ramen']:
  root=Path('output/pixel_evidence_rebuild_20260917')/scene;fieldroot=root/'field';parent=fieldroot/'joint_supervised_image_residual';parent_sha=sha256_file(parent/'field.pt');state=torch.load(parent/'field.pt',map_location='cpu',weights_only=False);row={}
  for name in ['source_extent_benefit_admission','counterfactual_extent_benefit_admission']:
   f=fieldroot/name;report=json.loads((f/'report.json').read_text());assert report['parent_field_sha256']==parent_sha and report['field_sha256']==sha256_file(f/'field.pt')
   for path,h in report['implementations'].items():assert sha256_file(f/'sources'/Path(path).name)==h
   for path,h in report['inputs'].items():assert sha256_file(path)==h
   checkpoint=torch.load(f/'field.pt',map_location='cpu',weights_only=False);assert checkpoint['geometry_binding']==state['geometry_binding']
   for key,v in state['state_dict'].items():assert torch.equal(v,checkpoint['state_dict'][key]),key
   assert set(checkpoint['state_dict'])-set(state['state_dict'])=={'extent_benefit_admission.center','extent_benefit_admission.scale','extent_benefit_admission.coefficient'}
   validation=root/(name+'_validation');execution=json.loads((validation/'execution.json').read_text());assert execution['finished'] and execution['image_prompt_observations_exact']
   item=dict(report_sha256=sha256_file(f/'report.json'),field_sha256=report['field_sha256'],source_selection=report['selected'],all_parent_state_exact=True,evaluations={})
   for task in ['text','reference']:
    path=validation/(task+'.json');x=json.loads(path.read_text());old=json.loads((root/(task+'_evaluation.json')).read_text());run=x['runs'][name];assert run['field_sha256']==report['field_sha256'];dispatch=json.loads(path.with_suffix('.dispatch.json').read_text());assert dispatch['result_sha256']==sha256_file(path) and dispatch['protocol_sha256']==sha256_file(path.with_suffix('.protocol.json'))
    protocols=[]
    for v in [old,x]:
     contract=v['protocol'].copy()
     for k in ['fields','field_sha256']:contract.pop(k,None)
     protocols.append(contract)
    assert protocols[0]==protocols[1]
    before=old['runs']['joint_supervised_image_residual'];keys=['frame','category'] if task=='text' else ['reference_frame','target_frame','category'];xs=[{tuple(o[k] for k in keys):o for o in r['observations']} for r in [before,run]];assert xs[0].keys()==xs[1].keys()
    if task=='reference':
     for key in xs[0]:
      for m in ['image','prompt']:assert xs[0][key][m]==xs[1][key][m]
     item['image_prompt_observations_exact']=True;item['evaluations'][task]=run['macro']
    else:item['evaluations'][task]={label:{k:v[k] for k in ['observation_miou','category_macro_miou','average_precision_macro']} for label,v in [('before',before),('after',run)]}
   if name.startswith('counterfactual'):
    cold=json.loads((validation/'cold.json').read_text());assert cold['field_sha256']==report['field_sha256'];assert cold['training_cache_reads_forbidden'] and cold['modalities']['text']['concept_route_consistent'];assert cold['modalities']['text']['posterior_max_difference']<.01
    item['cold']=cold;item['source_routed_validation_passed']=report['source_routed_validation_passed']
   row[name]=item
  evidence=torch.load(fieldroot/'caption_region_evidence.pt',map_location='cpu',weights_only=False)
  assert sha256_file(fieldroot/'teachers.json')==evidence['teacher_manifest_sha256']
  info=json.loads((fieldroot/'teachers.json').read_text());assert sha256_file(fieldroot/'responsibility.json')==info['authority_sha256']
  authority=json.loads((fieldroot/'responsibility.json').read_text())
  for rec in authority['views']:assert sha256_file(fieldroot/rec['relative_path'])==rec['sha256']
  row['source_identity']={'teacher_manifest_sha256':evidence['teacher_manifest_sha256'],'authority_sha256':info['authority_sha256'],'group_report_sha256':sha256_file(fieldroot/'grounded_semantic_transfer96/report.json')}
  del evidence
  base=Path('output/pixel_evidence_fusion_20260917')/scene
  row['area_balance']={}
  for mode in ['pixel_partition','area_balanced']:
   d=json.loads((base/(mode+'.json')).read_text());row['area_balance'][mode]=dict(sha256=sha256_file(base/(mode+'.json')),macro=d['macro'])
  result['scenes'][scene]=row
 a.output.write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'));print('Both scenes verified')
if __name__=='__main__':main()
