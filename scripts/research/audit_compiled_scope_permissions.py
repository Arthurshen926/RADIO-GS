"""Provenance and complete-cohort audit of the query-free permission migration."""
import json,subprocess
from pathlib import Path
from radio_gs.query_field.storage import sha256
from scripts.research.compile_query_field_scope_permissions import ROOT,CONFIG


def main():
    reports=[]
    for job in json.loads(CONFIG.read_text())['jobs']:
        out=ROOT/job['scene'];c=json.loads((out/'compilation.json').read_text());v=json.loads((out/'validation.json').read_text())
        parent=json.loads((Path(job['field'])/'manifest.json').read_text());p=parent['provenance']
        assert p['region_construction_sha256']==sha256(job['construction'])
        assert p['observations_sha256']==sha256(Path(job['observations'])/'observations.json')
        assert [(x['frame'],x['sha256']) for x in c['records']]==[(x['frame'],x['sha256']) for x in p['source_records']]
        construction=json.loads(Path(job['construction']).read_text())
        assert [x['regions'] for x in construction['source_records']]==[x['regions'] for x in p['source_records']]
        exported=json.loads((out/'field/manifest.json').read_text())
        assert exported['external_scene_dependencies']==[] and set(exported['files'])=={'field.pt'}
        assert c['field_sha256']==v['field_sha256']==exported['files']['field.pt']['sha256']==sha256(out/'field/field.pt')
        assert v['passed'] and v['parent_tensors_exact'] and not v['denied_access_attempts']
        assert c['tested_conditions']==v['tested_conditions'] and c['compiled_without_queries']
        reports.append(dict(compilation=c,validation=v))
    report=dict(passed=True,implementation_commit='15382b4',scenes=reports,
        totals={key:sum(x['compilation'][key] for x in reports) for key in ['additional_tensor_bytes','source_views','regions','candidates','tested_conditions']},
        scope='candidate permission function; negative lifting and final membership not transferred',
        benchmark_queries_used=False,target_labels_used=False,new_benchmark_scores=False,model_promotion=False,
        current_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    path=Path('docs/experiments/2026-10-10-compiled-scope-permissions-audit.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['totals'],flush=True)

if __name__=='__main__':main()
