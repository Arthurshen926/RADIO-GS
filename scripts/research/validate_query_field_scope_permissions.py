"""Reload a standalone field and replay source-free conditional permissions."""
import argparse,builtins,io,json,os
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.compile_query_field_scope_permissions import ROOT,CONFIG


def run(index):
    job=json.loads(CONFIG.read_text())['jobs'][index];out=ROOT/job['scene']
    fixtures=torch.load(out/'validation_conditions.pt',weights_only=True)
    # Parent is read only to verify exact preservation, before source access denial.
    parent,_=load_field(job['field']);old={k:v.clone() for k,v in parent.state_dict().items()};del parent
    allowed=(out/'field').resolve();attempts=[]
    originals=(builtins.open,io.open,os.open)
    def wrap(fn):
        def checked(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path)).resolve()
                # Block all project data, except the actual two-file scene package.
                if ('output' in p.parts or str(p).startswith('/mnt/')) and not p.is_relative_to(allowed):
                    attempts.append(str(p));raise PermissionError('source access denied')
            return fn(path,*args,**kwargs)
        return checked
    builtins.open,io.open,os.open=[wrap(x) for x in originals]
    try:
        field,manifest=load_field(allowed)
        for k,v in old.items():assert torch.equal(v,field.state_dict()[k]),k
        del old
        graph=field.scope_permissions;tested=0
        for vi,fixture in enumerate(fixtures):
            positions=graph.accepted_views==vi;rows=graph.candidate_views==vi
            for accepted,expected in zip(fixture['accepted'],fixture['expected']):
                query=torch.zeros_like(graph.accepted_views,dtype=torch.bool);query[positions]=accepted
                result=graph(query)
                assert torch.equal(result[rows],expected) and not result[~rows].any()
                tested+=1
        assert not attempts
    finally:builtins.open,io.open,os.open=originals
    report=dict(passed=True,scene=job['scene'],parent_tensors_exact=True,tested_conditions=tested,
        source_denial='Python open/io.open/os.open guard, not an OS sandbox',denied_access_attempts=attempts,
        field_sha256=manifest['files']['field.pt']['sha256'],no_rasters_cameras_or_query_answers_in_graph=True,
        scope='candidate permission function only; not full native acceptance or Gaussian member decoding',model_promotion=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
