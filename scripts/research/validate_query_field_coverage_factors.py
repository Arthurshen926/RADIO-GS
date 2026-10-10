"""Cold reload of the sampled opacity function, no source observation access."""
import argparse,builtins,io,json,os
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.compile_query_field_coverage_factors import ROOT,PARENT,CONFIG,member_conditions


def fixed_visibility(operator, membership):
    # Matched ablation: freeze full-set contributions before selecting members.
    alpha=operator.alpha.double().clamp(max=.999)
    group=torch.repeat_interleave(torch.arange(len(operator.atom),device=alpha.device),operator.pointer.diff())
    log_step=torch.log1p(-alpha);prefix=log_step.cumsum(0)
    padded=torch.cat([prefix.new_zeros(1),prefix])
    inclusive=prefix-padded[operator.pointer[:-1]][group]
    weights=alpha*(inclusive-log_step).exp()*(inclusive.exp()>=1e-4)
    return torch.zeros(len(operator.atom),dtype=torch.float64,device=alpha.device).scatter_add(0,group,weights*membership[operator.index]).float()


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
        field,manifest=load_field(allowed,'cuda')
        for k,v in old.items():assert torch.equal(v,field.state_dict()[k].cpu()),k
        del old
        error=0.;gradient_error=0.;tested=0;disagreements=0;ablation={}
        for vi,fixture in enumerate(fixtures):
            operator=field.coverage_factors.views[vi]
            for (name,member),expected in zip(member_conditions(field,vi,fixture['back_half']),fixture['expected']):
                actual=operator(member).cpu()
                error=max(error,float((actual-expected).abs().max()))
                disagreements+=int(((actual>10.5/255)!=(expected>10.5/255)).sum())
                linear=fixed_visibility(operator,member).cpu()
                entry=ablation.setdefault(name,dict(samples=0,linear_absolute_error_sum=0.,linear_max_error=0.,linear_selection_disagreements=0))
                entry['samples']+=len(expected)
                entry['linear_absolute_error_sum']+=float((linear-expected).abs().sum())
                entry['linear_max_error']=max(entry['linear_max_error'],float((linear-expected).abs().max()))
                entry['linear_selection_disagreements']+=int(((linear>10.5/255)!=(expected>10.5/255)).sum())
                tested+=1
            member=member.detach().requires_grad_(True)
            gradient=torch.autograd.grad(operator(member).mean(),member)[0]
            gradient_error=max(gradient_error,float((gradient[fixture['gradient_rows'].cuda()].cpu()-fixture['gradient']).abs().max()))
        assert error<=1e-4 and gradient_error<=2e-4 and disagreements==0
        assert not attempts
    finally:builtins.open,io.open,os.open=originals
    report=dict(passed=True,scene=job['scene'],parent_tensors_exact=True,conditions=tested,max_alpha_error=error,
        max_sampled_gradient_error=gradient_error,selection_disagreements=disagreements,
        source_denial='Python open/io.open/os.open, not OS sandbox',denied_attempts=attempts,
        field_sha256=manifest['files']['field.pt']['sha256'],benchmark_queries_used=False,
        membership_decoder_transferred=False,sampled_construction_factors_only=True, fixed_visibility_diagnostic=ablation)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
