"""One registered source-only paired control for directional clamp gradients."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from radio_gs.utils.immutable_artifacts import sha256_file

BASE=Path('output/directional_scope_gradient_next32')
TRAINER='scripts/audits/train_directional_scope_gradient_next32.py'
PILOTS=[
 dict(key='lerf_figurines',root='output/full_field_rebuild_next19/figurines/reconstruction/field',
      geometry='output/depth_opacity_repair_20260918/figurines/geometry_report.json',guard='scripts/audits/run_lerf_source_stage.py'),
 dict(key='scannet_scene0000_00',root='output/scannet_expanded_budget_next29/scene0000_00/reconstruction/field',
      geometry='output/scannet_expanded_budget_next29/scene0000_00/geometry_report.json',guard='scripts/audits/run_scannet_source_stage.py'),
]
SOURCES=[__file__,TRAINER,'radio_gs/training/directional_scope_gradient.py',
 'radio_gs/training/selected_output_supervision.py','radio_gs/training/disjoint_query_competition.py',
 'radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py',
 'radio_gs/models/extent_benefit_gate.py','radio_gs/models/bounded_scope_evidence.py',
 'scripts/audits/train_rendered_extent.py','scripts/audits/run_lerf_source_stage.py',
 'scripts/audits/run_scannet_source_stage.py','tests/test_directional_scope_gradient_next32.py']


def main():
 p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--pilot',choices=[r['key'] for r in PILOTS]);a=p.parse_args()
 if a.register:
  BASE.mkdir(exist_ok=False);inputs={}
  for row in PILOTS:
   root=Path(row['root'])
   for path in [root/'bounded_fallback_admission/field.pt',root/'bounded_fallback_admission/report.json',root/'teachers.json',root/'responsibility.json',root/'caption_region_evidence.pt',root/'caption_region_evidence.json',Path(row['geometry'])]:
    inputs[str(path)]=sha256_file(path)
  registration=dict(schema='directional_scope_gradient.next32.registration.v1',pilots=PILOTS,
   modes=['visible_selected','directional_recovery'],steps=400,seed=42,lr=1e-4,
   source_only=True,benchmark_data_allowed=False,mainline_promoted=False,
   hypothesis='Recover wrong saturated concept-head gradient without changing any real inference forward or hard fallback.',
   biased_gradient_surrogate=True,surrogate='Old probability clamp forward exact; backward passes interior OR raw<1e-6 and grad<0 OR raw>1-1e-6 and grad>0. Outward gradients zero.',
   routing='Independent concept acceptance plus benefit admission; exact inference replay; replace only active concept gradients. Hard native fallback unchanged; no hard-gate STE.',
   preflight='Before any optimizer: both arms check all fixed source-validation queries for exact model-forward identity, finite raw/actual and active concept probability replay<2e-6.',
   training='Fresh paired400step baseline, identical visible-selected objective/source schedule/initial tensors. No pair-ranking loss, no target read.',
   validation='Last4 source views held out only from new head fitting; parent saw them. Fixed source-order ~20 masks/view; first20 eligible reciprocal disjoint query pairs/view with both supports present.',
   checkpoint='Both arms same: full source IoU no decrease from parent, hard-selected source IoU improves over chosen; step0 fallback. Pair ordering recorded but not used to choose checkpoint.',
   expansion_gate='Both pilots: chosen recovery strictly improves full and hard-selected IoU vs chosen fresh baseline, pair ordering accuracy no worse and pair hinge no worse. No benchmark otherwise.',
   implementations={path:sha256_file(path) for path in SOURCES},inputs=inputs)
  (BASE/'registration.json').write_text(json.dumps(registration,indent=2)+'\n');print('registered',flush=True);return
 registration=json.loads((BASE/'registration.json').read_text())
 for path,digest in registration['implementations'].items():assert sha256_file(path)==digest,path
 row=next(r for r in registration['pilots'] if r['key']==a.pilot)
 for path,digest in registration['inputs'].items():
  if path.startswith(row['root']) or path==row['geometry']:assert sha256_file(path)==digest,path
 out=BASE/a.pilot;out.mkdir(exist_ok=False)
 for mode in registration['modes']:
  command=[sys.executable,row['guard'],TRAINER,'--root',row['root'],'--geometry-report',row['geometry'],'--mode',mode,'--steps','400','--parent-name','bounded_fallback_admission','--device','cuda:0','--output',str(out/mode)]
  start=time.time()
  with (out/(mode+'.log')).open('x') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
  (out/(mode+'.execution.json')).write_text(json.dumps(dict(command=command,returncode=result.returncode,seconds=time.time()-start),indent=2)+'\n')
  assert result.returncode==0,(mode,'Preserve failed attempt; no silent rerun')
  print(a.pilot,mode,'complete',flush=True)
 def chosen(mode):
  result=json.loads((out/mode/'report.json').read_text())
  assert result['implementation_sha256']==registration['implementations'][TRAINER]
  assert sha256_file(out/mode/'field.pt')==result['field_sha256']
  return next(h['validation'] for h in result['history'] if h['step']==result['selected_step'])
 control=chosen('visible_selected');recovery=chosen('directional_recovery')
 gate={key:recovery[key]>control[key]+1e-6 for key in ('full_iou','selected_iou')}
 gate.update(query_order_accuracy=recovery['query_order_accuracy']>=control['query_order_accuracy']-1e-6,
             query_order_hinge=recovery['query_order_hinge']<=control['query_order_hinge']+1e-6)
 summary=dict(pilot=a.pilot,control=control,recovery=recovery,gate=gate,source_admitted=all(gate.values()),
  source_only=True,independent_test=False,biased_gradient_surrogate=True,benchmark_evaluation_launched=False,
  registration_sha256=sha256_file(BASE/'registration.json'),
  report_hashes={mode:sha256_file(out/mode/'report.json') for mode in registration['modes']})
 with (out/'summary.json').open('x') as file:json.dump(summary,file,indent=2);file.write('\n')
 print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
