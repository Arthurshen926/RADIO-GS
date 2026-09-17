"""Verify paired complete recipes; keep observation and category aggregation distinct."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);result=dict(schema='pixel_evidence_rebuild_comparison.v1',independent_final_test=False,scenes={},implementation_sha256=sha256_file(__file__))
 for scene in ('figurines','ramen'):
  old=Path('output/lerf_pose_covered_pipeline_20260917')/scene;new=Path('output/pixel_evidence_rebuild_20260917')/scene;r=json.loads((new/'execution.json').read_text());assert r.get('finished')
  for s in r['stages']:
   assert s['status']=='complete'
   for f,h in s['outputs'].items():assert sha256_file(f)==h
   for f,h in s['implementations'].items():assert sha256_file(new/'sources'/s['name']/Path(f).name)==h
  row=dict(execution_sha256=sha256_file(new/'execution.json'),evaluations={},report_metadata_correction='Initial fusion export inherited parent native_text_image_max_error=0.0; it is not a measurement of changed native responses. Native prefix changed; prompt preservation independently verified here. Future exporter sets this inherited field to null.')
  x=torch.load(old/'field/scale_affinity_v1_direct_control/field.pt',map_location='cpu',weights_only=False);y=torch.load(new/'field/pixel_evidence96/field.pt',map_location='cpu',weights_only=False)
  assert x['geometry_binding']==y['geometry_binding'];assert x['architecture']==y['architecture']
  for k,v in x['state_dict'].items():
   if k!='latent':assert torch.equal(v,y['state_dict'][k]),k
  assert torch.equal(x['state_dict']['latent'][:,64:],y['state_dict']['latent'][:,64:])
  row['native_prefix_max_change']=float((x['state_dict']['latent'][:,:64]-y['state_dict']['latent'][:,:64]).abs().max());assert row['native_prefix_max_change']>0;del x,y
  prep=json.loads((new/'field/prepared_inputs.json').read_text())
  for f,h in prep['input_hashes'].items():assert sha256_file(old/'field'/f)==h and sha256_file(new/'field'/f)==h
  for task in ('text','reference','native_text'):
   paths=[new/'native_text_before.json',new/'native_text_after.json'] if task=='native_text' else [old/(task+'_evaluation.json'),new/(task+'_evaluation.json')]
   name='full_source_query_threshold_control' if task=='native_text' else 'joint_supervised_image_residual';results=[];contracts=[]
   for root,path in zip((old,new),paths):
    v=json.loads(path.read_text());d=json.loads(path.with_suffix('.dispatch.json').read_text());assert d['result_sha256']==sha256_file(path);assert d['protocol_sha256']==sha256_file(path.with_suffix('.protocol.json'))
    run=v['runs'][name];assert sha256_file(root/'field'/name/'field.pt')==run['field_sha256'];results.append(run)
    protocol=v['protocol'].copy()
    for k in ('fields','field_sha256'):protocol.pop(k,None)
    contracts.append(protocol)
   assert contracts[0]==contracts[1],(scene,task,'protocol mismatch')
   keys=('reference_frame','target_frame','category') if task=='reference' else ('frame','category')
   obs=[{tuple(o[k] for k in keys):o for o in v['observations']} for v in results];assert obs[0].keys()==obs[1].keys()
   if task=='reference':
    assert all(obs[0][k]['prompt']==obs[1][k]['prompt'] for k in obs[0]);row['prompt_observations_exact']=True
    row['evaluations'][task]={label:v['macro'] for label,v in zip(('before','after'),results)}
   else:row['evaluations'][task]={label:{k:v[k] for k in ('observation_miou','category_macro_miou','average_precision_macro')} for label,v in zip(('before','after'),results)}
  result['scenes'][scene]=row
 a.output.write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
