"""Fresh-process field reload with source denial and sampled source-lift agreement."""
import argparse,builtins,io,json,os
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.compile_query_field_scope_constraints import ROOT,PARENT,CONFIG


def run(index):
    job=json.loads(CONFIG.read_text())['jobs'][index];out=ROOT/job['scene']
    fixtures=torch.load(out/'validation_conditions.pt',weights_only=True)
    parent,_=load_field(PARENT/job['scene']/'field');old=parent.state_dict();del parent
    allowed=(out/'field').resolve();attempts=[];originals=(builtins.open,io.open,os.open)
    def wrap(fn):
        def checked(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path)).resolve()
                if ('output' in p.parts or str(p).startswith('/mnt/')) and not p.is_relative_to(allowed):
                    attempts.append(str(p));raise PermissionError('source access denied')
            return fn(path,*args,**kwargs)
        return checked
    builtins.open,io.open,os.open=[wrap(x) for x in originals]
    try:
        field,manifest=load_field(allowed)
        for k,v in old.items():assert torch.equal(v,field.state_dict()[k]),k
        del old
        graph=field.scope_permissions;error=0.;tested=0
        for vi,fixture in enumerate(fixtures):
            selected=graph.accepted_views==vi
            for accepted,expected in zip(fixture['accepted'],fixture['expected']):
                query=torch.zeros_like(graph.accepted_views,dtype=torch.bool);query[selected]=accepted
                permission=graph(query)
                value=field.scope_constraints.views[vi](query,permission)[fixture['rows']]
                error=max(error,float((value-expected).abs().max()));assert error<=2e-6
                tested+=1
        assert not attempts
    finally:builtins.open,io.open,os.open=originals
    report=dict(passed=True,scene=job['scene'],parent_tensors_exact=True,conditions=tested,max_error=error,
        tolerance=2e-6,rows_per_view=min(256,len(field.xyz)),source_denial='Python open/io.open/os.open, not OS sandbox',
        denied_attempts=attempts,field_sha256=manifest['files']['field.pt']['sha256'],membership_parity_claim=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
