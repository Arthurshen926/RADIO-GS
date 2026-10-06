"""Read-only digest verification of completed next31/32 evidence snapshots."""
import json
import re
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    roots=sorted(p for p in Path('paper/artifacts').glob('*next3[12].json')
        if 'directional' not in p.name and 'core_evidence_review' not in p.name)
    cache={};checks=[];seals={};errors=[]
    def digest(path):
        canonical=str(Path(path).resolve())
        if canonical not in cache:cache[canonical]=sha256_file(path)
        return cache[canonical]
    def check(path,expected,owner):
        try:
            actual=digest(path);ok=actual==expected
            error=None if ok else 'digest mismatch'
        except Exception as exc:
            actual=None;ok=False;error=str(exc)
        record=dict(owner=owner,path=str(path),expected=expected,actual=actual,ok=ok)
        if error:record['error']=error;errors.append(record)
        checks.append(record)
    def walk(value,owner):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and re.fullmatch('[a-f0-9]{64}',str(value.get('sha256',''))):
                check(value['path'],value['sha256'],owner)
            for key,val in value.items():
                if isinstance(val,str) and '/' in key and re.fullmatch('[a-f0-9]{64}',val):check(key,val,owner)
                elif isinstance(val,(dict,list)):walk(val,owner)
        elif isinstance(value,list):
            for item in value:walk(item,owner)
    for path in roots:
        seals[str(path)]=digest(path);walk(json.loads(path.read_text()),str(path))
        print(path.name,'checked',flush=True)
    # Completed readout diagnostic has no seal yet: verify frozen registration,
    # complete reports and full-float render receipts independently.
    base=Path('output/lerf_readout_sensitivity_next32')
    for path in [base/'registration.json',base/'summary.json']:
        walk(json.loads(path.read_text()),str(path))
    summary=json.loads((base/'summary.json').read_text())
    for report_path in summary['reports']:
        path=Path(report_path);receipt=path.with_name('render_receipt.json')
        report=json.loads(path.read_text())
        check(receipt,report['receipt_sha256'],str(path))
        walk(json.loads(receipt.read_text()),str(receipt))
    output=Path('output/core_evidence_review_next32');output.mkdir(exist_ok=True)
    result=dict(schema='completed_child_seals.next32',gpu_used=False,read_only=True,
        unfinished_directional_gradient_excluded=True,seals=seals,checks=checks,
        reference_bindings_checked=len(checks),unique_files_hashed=len(cache),errors=errors,
        all_checked_bindings_match=not errors,implementation_sha256=sha256_file(__file__),
        scope='All currently completed next31/next32 top-level seals; direct path/digest bindings. Plus completed4-scene continuous readout registration/report/render receipts. Not every transitive repository asset.')
    (output/'completed_bindings.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','seals')},indent=2),flush=True)


if __name__=='__main__':main()
