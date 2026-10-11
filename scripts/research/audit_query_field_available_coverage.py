"""Aggregate frozen construction gates, without interpreting them as accuracy."""
import json,subprocess
from pathlib import Path
from scripts.research.compile_query_field_available_coverage import ROOT,PROTOCOL
from scripts.research.compile_query_field_coverage_factors import CONFIG
from radio_gs.query_field.storage import sha256


def main():
    scenes={};conditions=0;maximum=0.;hard_maximum=0.;grad_maximum=0.
    files=['radio_gs/query_field/geometry.py','radio_gs/query_field/coverage_factors.py',
           'scripts/research/compile_query_field_coverage_factors.py',
           'scripts/research/compile_query_field_available_coverage.py',str(PROTOCOL)]
    for job in json.loads(CONFIG.read_text())['jobs']:
        root=ROOT/job['scene'];r=json.loads((root/'compilation.json').read_text())
        v=json.loads((root/'validation.json').read_text())
        old=json.loads((Path('output/query_field_20261010/compiled_coverage_factors_v1')/job['scene']/'compilation.json').read_text())
        assert r['complete'] and r['member_domain']=='available' and not r['benchmark_queries_used'] and not r['target_labels_used']
        assert v['passed'] and v['parent_tensors_exact'] and not v['denied_attempts'] and v['selection_disagreements']==0
        assert v['field_sha256']==r['field_sha256']
        assert r['protocol_sha256']==sha256(PROTOCOL)
        for f in files:
            assert subprocess.check_output(['git','show',r['implementation_commit']+':'+f])==Path(f).read_bytes(),f
        assert r['views']==old['views'] and r['factors']==old['factors']
        assert v['conditions']==r['views']*9
        for a,b in zip(r['records'],old['records']):
            assert a['frame']==b['frame'] and a['source_sha256']==b['source_sha256'] and a['factors']==b['factors']
            assert len(a['cases'])==9
            for c in a['cases']:
                assert c['max_alpha_error']<=1e-4 and c['selection_disagreements']==0
                maximum=max(maximum,c['max_alpha_error']);conditions+=1
                if c['condition']!='soft':
                    assert c['actual_subset_max_error']<=1e-4
                    hard_maximum=max(hard_maximum,c['actual_subset_max_error'])
            assert a['gradient_max_error']<=2e-4
            grad_maximum=max(grad_maximum,a['gradient_max_error'])
        scenes[job['scene']]={k:r[k] for k in ['views','factors','hits','eligible_rows','semantic_only_rows','additional_tensor_bytes','implementation_commit']}
        scenes[job['scene']].update(old_hits=old['hits'],old_additional_tensor_bytes=old['additional_tensor_bytes'],
            compilation_sha256=sha256(root/'compilation.json'),validation_sha256=sha256(root/'validation.json'))
    report=dict(passed=True,scenes=scenes,conditions=conditions,max_soft_alpha_error=maximum,
        max_actual_subset_error=hard_maximum,max_allrow_gradient_error=grad_maximum,
        all_parent_tensors_exact=True,all_cold_reloads_passed=True,source_labels_changed=False,
        sampled_construction_only=True,benchmark_accuracy_measured=False,model_promotion=False,whole_five_task=False)
    target=Path('docs/experiments/2026-10-11-available-coverage-audit.json')
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
