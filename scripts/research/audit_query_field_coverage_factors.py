"""Aggregate full cohort function checks; this does not score benchmarks."""
import json
from pathlib import Path
from radio_gs.query_field.storage import sha256
from scripts.research.compile_query_field_coverage_factors import ROOT,PROTOCOL


def main():
    scenes=['figurines','ramen','teatime','waldo_kitchen'];records={};baseline={}
    for scene in scenes:
        folder=ROOT/scene
        build=json.loads((folder/'compilation.json').read_text());cold=json.loads((folder/'validation.json').read_text())
        manifest=json.loads((folder/'field/manifest.json').read_text())
        assert build['complete'] and cold['passed'] and cold['parent_tensors_exact']
        assert build['protocol_sha256']==sha256(PROTOCOL)
        assert build['field_sha256']==cold['field_sha256']==manifest['files']['field.pt']['sha256']
        assert not build['benchmark_queries_used'] and not build['target_labels_used']
        assert cold['conditions']==7*build['views'] and cold['selection_disagreements']==0 and not cold['denied_attempts']
        assert all(c['max_alpha_error']<=1e-4 and c['selection_disagreements']==0 for v in build['records'] for c in v['cases'])
        assert all(v['gradient_max_error']<=2e-4 for v in build['records'])
        records[scene]=dict(views=build['views'],factors=build['factors'],hits=build['hits'],bytes=build['additional_tensor_bytes'],
            max_alpha_error=max(c['max_alpha_error'] for v in build['records'] for c in v['cases']),
            max_all_row_gradient_error=max(v['gradient_max_error'] for v in build['records']),
            cold=cold,compilation_sha256=sha256(folder/'compilation.json'),validation_sha256=sha256(folder/'validation.json'))
        for name,row in cold['fixed_visibility_diagnostic'].items():
            total=baseline.setdefault(name,dict(samples=0,absolute_error_sum=0.,max_error=0.,selection_disagreements=0))
            total['samples']+=row['samples'];total['absolute_error_sum']+=row['linear_absolute_error_sum']
            total['max_error']=max(total['max_error'],row['linear_max_error'])
            total['selection_disagreements']+=row['linear_selection_disagreements']
    for row in baseline.values():row['mean_absolute_error']=row['absolute_error_sum']/row['samples']
    report=dict(passed=True,views=sum(x['views'] for x in records.values()),factors=sum(x['factors'] for x in records.values()),
        hits=sum(x['hits'] for x in records.values()),additional_tensor_bytes=sum(x['bytes'] for x in records.values()),
        member_conditions=sum(x['views']*7 for x in records.values()),
        max_alpha_error=max(x['max_alpha_error'] for x in records.values()),
        max_all_row_gradient_error=max(x['max_all_row_gradient_error'] for x in records.values()),
        fixed_visibility_diagnostic=baseline,scenes=records,model_promotion=False,benchmark_scores_updated=False,
        sampled_construction_factors_only=True,membership_decoder_transferred=False,
        code_sha256=sha256(__file__))
    path=Path('docs/experiments/2026-10-10-coverage-factors-audit.json')
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))

if __name__=='__main__':main()
