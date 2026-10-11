"""Audit all four construction receipts, view bindings and cold function checks."""
import json
from radio_gs.query_field.storage import sha256
from scripts.research.compile_query_field_support_footprint import ROOT,PARENT,CONFIG
from pathlib import Path


def run():
    implementation=json.loads((ROOT/'implementation.json').read_text())
    for path,digest in implementation['code_sha256'].items():
        assert sha256(path)==digest,path
    rows=[]
    for job in json.loads(CONFIG.read_text())['jobs'][:4]:
        scene=job['scene'];out=ROOT/scene
        report=json.loads((out/'compilation.json').read_text())
        cold=json.loads((out/'cold_validation.json').read_text())
        parent=json.loads((PARENT/scene/'field/manifest.json').read_text())
        child=json.loads((out/'field/manifest.json').read_text())
        assert report['complete'] and cold['complete'] and cold['parent_tensors_exact']
        prior_records=parent['provenance']['source_records']
        assert len(report['records'])==len(prior_records)==cold['views']
        for actual,reference in zip(report['records'],prior_records):
            assert actual['frame']==reference['frame']
            assert actual['source_sha256']==reference['source_sha256']
            assert actual['observed_reference_exact']
        assert child['provenance']['parent_field_sha256']==parent['files']['field.pt']['sha256']
        assert child['tensor_bytes']-parent['tensor_bytes']==report['additional_tensor_bytes']
        assert report['field_sha256']==child['files']['field.pt']['sha256']
        rows.append(dict(scene=scene,views=cold['views'],members=cold['members'],
            function_conditions=cold['function_conditions'],additional_tensor_bytes=report['additional_tensor_bytes'],
            observed_reference_exact=True,parent_tensors_exact=True))
    result=dict(complete=True,scenes=rows,views=sum(x['views'] for x in rows),
        additional_tensor_bytes=sum(x['additional_tensor_bytes'] for x in rows),
        benchmark_evaluated=False,full_extent_guaranteed=False,implementation=implementation)
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
