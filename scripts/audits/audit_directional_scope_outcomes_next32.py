"""Reuse the sealed source-gradient diagnostic on both selected pilot heads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file

BASE=Path('output/directional_scope_gradient_next32')
DIAGNOSTIC=Path('scripts/audits/audit_scope_gradient_next32.py')


def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);a=p.parse_args()
 registration=json.loads((BASE/'registration.json').read_text())
 for path,digest in registration['implementations'].items():assert sha256_file(path)==digest,path
 seal=json.loads(Path('paper/artifacts/source_scope_gradient_audit_next32.json').read_text())
 assert sha256_file(DIAGNOSTIC)==seal['files'][str(DIAGNOSTIC)]
 row=next(r for r in registration['pilots'] if r['key']==a.pilot)
 src=Path(row['root']).resolve();authority=json.loads((src/'responsibility.json').read_text())
 folder=BASE/a.pilot;summary=json.loads((folder/'summary.json').read_text())
 assert summary['registration_sha256']==sha256_file(BASE/'registration.json')
 collected={}
 for mode in registration['modes']:
  checkpoint=folder/mode;report=json.loads((checkpoint/'report.json').read_text())
  assert sha256_file(checkpoint/'report.json')==summary['report_hashes'][mode]
  assert sha256_file(checkpoint/'field.pt')==report['field_sha256']
  overlay=folder/(mode+'_source_audit_inputs');overlay.mkdir(exist_ok=False)
  inputs=['teachers.json','responsibility.json','caption_region_evidence.pt']+[r['relative_path'] for r in authority['views']]
  for relative in dict.fromkeys(inputs):
   target=overlay/relative;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(src/relative)
  (overlay/'bounded_fallback_admission').symlink_to(checkpoint.resolve(),target_is_directory=True)
  mapping=dict(original_source_root=str(src),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=report['field_sha256'],
               source_diagnostic_sha256=sha256_file(DIAGNOSTIC),fixed_cohort='Identical sealed first2/last2 views and first8 scopes; no outcome-based selection.')
  (overlay/'overlay.json').write_text(json.dumps(mapping,indent=2)+'\n')
  output=folder/(mode+'_source_gradient_audit')
  command=[sys.executable,row['guard'],str(DIAGNOSTIC),'--root',str(overlay),'--device','cuda:0','--output',str(output)]
  with (folder/(mode+'_source_gradient_audit.log')).open('x') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
  assert result.returncode==0,mode
  diagnosis=json.loads((output/'report.json').read_text());assert diagnosis['implementation_sha256']==sha256_file(DIAGNOSTIC)
  collected[mode]=dict(summary=diagnosis['summary'],report_sha256=sha256_file(output/'report.json'))
  print(a.pilot,mode,'source-gradient diagnostic complete',flush=True)
 comparison=dict(pilot=a.pilot,source_only=True,benchmark_data_read=False,checkpoint_selection_changed=False,
  diagnostic_implementation_sha256=sha256_file(DIAGNOSTIC),outcome_audit_sha256=sha256_file(__file__),
  registration_sha256=sha256_file(BASE/'registration.json'),comparison=collected)
 with (folder/'source_gradient_comparison.json').open('x') as f:json.dump(comparison,f,indent=2);f.write('\n')


if __name__=='__main__':main()
