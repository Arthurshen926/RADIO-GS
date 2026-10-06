"""Verify and seal completed directional-gradient paired pilots, without GT."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

BASE=Path('output/directional_scope_gradient_next32')


def main():
 registration=json.loads((BASE/'registration.json').read_text());paths=[BASE/'registration.json',BASE/'tests.json',BASE/'independent_review.json']
 owned_tests=json.loads((BASE/'tests.json').read_text());independent=json.loads((BASE/'independent_review.json').read_text())
 assert owned_tests['exit_code']==0 and owned_tests['passed']==7
 assert owned_tests['sha256']==sha256_file('tests/test_directional_scope_gradient_next32.py')
 assert independent['test']['passed']==2 and not independent['blocking_issue_found']
 assert independent['test']['sha256']==sha256_file(independent['test']['file'])
 for path,digest in independent['reviewed'].items():assert sha256_file(path)==digest,path
 for path,digest in registration['implementations'].items():assert sha256_file(path)==digest,path
 paths += [Path(p) for p in registration['implementations']]
 paths += [Path('tests/test_directional_scope_gradient_loss_next32_independent.py'),Path(__file__),
           Path('scripts/audits/audit_directional_scope_outcomes_next32.py'),Path('scripts/audits/audit_directional_scope_outcomes_next32_v2.py'),
           BASE/'outcome_audit_v1_failure.json',Path('docs/experiments/2026-09-26-directional-scope-gradient-recovery.md')]
 summaries={};mechanism={}
 for row in registration['pilots']:
  folder=BASE/row['key'];summary=json.loads((folder/'summary.json').read_text())
  assert summary['registration_sha256']==sha256_file(BASE/'registration.json')
  assert not summary['benchmark_evaluation_launched'] and summary['source_only']
  reports={}
  for mode in registration['modes']:
   report_path=folder/mode/'report.json';report=json.loads(report_path.read_text())
   assert sha256_file(report_path)==summary['report_hashes'][mode]
   assert sha256_file(folder/mode/'field.pt')==report['field_sha256']
   assert report['implementation_sha256']==registration['implementations']['scripts/audits/train_directional_scope_gradient_next32.py']
   assert report['gradient_implementation_sha256']==registration['implementations']['radio_gs/training/directional_scope_gradient.py']
   assert report['steps']==400 and report['native_field_unchanged'] and report['geometry_unchanged'] and report['query_encoder_unchanged']
   assert report['source_only'] and not report['benchmark_data_used']
   assert all(item['actual_forward_bit_exact'] for item in report['gradient_preflight'])
   assert report['gradient_audit']['actual_forward_bit_exact']
   if mode=='directional_recovery':assert report['gradient_audit']['training_calls']==400
   reports[mode]=report
   paths += [report_path,folder/mode/'field.pt',folder/(mode+'.execution.json'),folder/(mode+'.log'),
             folder/(mode+'_source_gradient_audit_v2')/'report.json',folder/(mode+'_source_gradient_audit_v2.log'),
             folder/(mode+'_source_audit_inputs_v2')/'overlay.json']
  control=reports['visible_selected'];recovery=reports['directional_recovery']
  assert control['parent_field_sha256']==recovery['parent_field_sha256']
  assert control['training_frames']==recovery['training_frames'] and control['validation_frames']==recovery['validation_frames']
  for key in control['history'][0]['validation']:
   assert abs(control['history'][0]['validation'][key]-recovery['history'][0]['validation'][key])<1e-6,key
  comparison=json.loads((folder/'source_gradient_comparison_v2.json').read_text())
  assert comparison['registration_sha256']==sha256_file(BASE/'registration.json')
  for mode in registration['modes']:
   path=folder/(mode+'_source_gradient_audit_v2')/'report.json'
   assert sha256_file(path)==comparison['comparison'][mode]['report_sha256']
  summaries[row['key']]=summary;mechanism[row['key']]=comparison['comparison']
  paths += [folder/'summary.json',folder/'source_gradient_comparison_v2.json']
 summary=dict(schema='directional_scope_gradient.next32.complete.v1',complete=True,source_only=True,
  benchmark_data_read=False,benchmark_evaluation_launched=False,default_promoted=False,
  joint_source_admitted=all(s['source_admitted'] for s in summaries.values()),pilots=summaries,
  source_gradient_diagnostic=mechanism,tests_unique=9,biased_gradient_surrogate=True,
  frozen_inference_implementations_unchanged=True,registration_sha256=sha256_file(BASE/'registration.json'))
 output=BASE/'summary.json'
 with output.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
 paths.append(output)
 seal=dict(schema='directional_scope_gradient.next32.seal.v1',complete=True,default_promoted=False,
           files={str(p):sha256_file(p) for p in dict.fromkeys(paths)})
 with Path('paper/artifacts/directional_scope_gradient_next32.json').open('x') as f:json.dump(seal,f,indent=2);f.write('\n')
 print('PASS',len(seal['files']),'bindings; both400step paired pilots;9unique tests; no benchmark; admitted=',summary['joint_source_admitted'])


if __name__=='__main__':main()
