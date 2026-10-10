"""Query-free joint opacity factors with fixed functional acceptance tests."""
import argparse,json,time,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from radio_gs.query_field.storage import load_field,save_field,sha256
from radio_gs.query_field.coverage_factors import CoverageFactors,CoverageView,compile_coverage_view
from radio_gs.query_field.geometry import relaxed_subset_alpha
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.observations import erode
from scripts.research.compile_query_field_scope_constraints import ROOT as PARENT
from scripts.research.compile_query_field_scope_permissions import CONFIG
ROOT=Path('output/query_field_20261010/compiled_coverage_factors_v1')
PROTOCOL=Path('docs/experiments/2026-10-10-coverage-factors-protocol.md')


def member_conditions(field, vi, back):
    generator=torch.Generator(device=field.xyz.device).manual_seed(20261010+vi)
    n=len(field.xyz);device=field.xyz.device
    return [('empty',torch.zeros(n,device=device)),('full',torch.ones(n,device=device)),
        ('random15',(torch.rand(n,device=device,generator=generator)<.15).float()),
        ('random50',(torch.rand(n,device=device,generator=generator)<.5).float()),
        ('back_half',back.to(device).float()),
        ('x_half',(field.xyz[:,0]>field.xyz[:,0].median()).float()),
        ('soft',torch.rand(n,device=device,generator=generator))]


def run(index):
    start=time.monotonic();job=json.loads(CONFIG.read_text())['jobs'][index]
    out=ROOT/job['scene'];out.mkdir(parents=True,exist_ok=False)
    field,parent=load_field(PARENT/job['scene']/'field','cuda')
    source=SourceRelations(job['observations'],field.observed)
    construction=json.loads(Path(job['construction']).read_text())
    assert len(source.views)==len(field.scope_constraints.views)==len(construction['source_records'])
    state={};fixtures=[];checks=[]
    for vi,(v,r,constraint) in enumerate(zip(source.views,construction['source_records'],field.scope_constraints.views)):
        assert v['frame']==r['frame'] and v['original_regions']==r['regions']
        path=Path(job['observations'])/f"view_{v['frame']:05d}.pt";raw=torch.load(path,weights_only=True)
        masks=raw['masks'].bool();h,w=masks.shape[-2:];visible=raw['raw_mass'].reshape(h,w)>0
        interior=(erode(masks)&visible).flatten(1)
        packed=np.packbits(torch.cat([v['positive'],v['negative'],interior],0).T.numpy(),axis=1)
        _,first,inverse=np.unique(packed,axis=0,return_index=True,return_inverse=True)
        for key,ref in [('positive',v['positive']),('negative',v['negative']),('interior',interior)]:
            assert torch.equal(getattr(constraint,key).cpu(),ref[:,first].T)
        ys=((torch.arange(16)+.5)*h/16).long();xs=((torch.arange(16)+.5)*w/16).long()
        pixels=(ys[:,None]*w+xs[None,:]).flatten();atoms=torch.from_numpy(inverse[pixels.numpy()])
        compiled=compile_coverage_view(field,v['camera'],h,w,pixels,atoms)
        operator=CoverageView(compiled,len(field.xyz),len(first)).cuda()
        camera=v['camera'];depth=field.xyz@camera.world_to_camera[2,:3].cuda()+camera.world_to_camera[2,3].cuda()
        back=(depth>depth.median());cases=[];expected=[]
        for name,member in member_conditions(field,vi,back):
            with torch.no_grad():
                predicted=operator(member)
                reference=relaxed_subset_alpha(field,member,camera,h,w,field.world_units_per_metre)[pixels.cuda()]
            error=float((predicted-reference).abs().max());different=int(((predicted>10.5/255)!=(reference>10.5/255)).sum())
            cases.append(dict(condition=name,max_alpha_error=error,selection_disagreements=different))
            assert error<=1e-4 and different==0,(job['scene'],vi,cases[-1])
            expected.append(reference.cpu())
        member=member.detach().requires_grad_(True)
        actual_gradient=torch.autograd.grad(operator(member).mean(),member)[0]
        reference_gradient=torch.autograd.grad(relaxed_subset_alpha(field,member,camera,h,w,field.world_units_per_metre)[pixels.cuda()].mean(),member)[0]
        gradient_error=float((actual_gradient-reference_gradient).abs().max())
        assert gradient_error<=2e-4,(job['scene'],vi,gradient_error)
        rows=torch.linspace(0,len(field.xyz)-1,min(512,len(field.xyz)),device='cuda').long()
        fixtures.append(dict(back_half=back.cpu(),expected=torch.stack(expected),gradient_rows=rows.cpu(),gradient=reference_gradient[rows].cpu()))
        state.update({f'views.{vi}.'+k:value for k,value in compiled.items()})
        checks.append(dict(frame=v['frame'],source_sha256=sha256(path),factors=len(pixels),hits=len(compiled['index']),cases=cases,gradient_max_error=gradient_error))
        print('validated',job['scene'],vi,'hits',len(compiled['index']),'alpha',max(x['max_alpha_error'] for x in cases),'gradient',gradient_error,flush=True)
        del operator,actual_gradient,reference_gradient,member
    field.coverage_factors=CoverageFactors(state,len(field.xyz),field.scope_constraints).cuda()
    added=sum(x.numel()*x.element_size() for x in state.values())
    exported=save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        source_records=checks,benchmark_queries_used=False,target_labels_used=False,training_steps=0,
        default_decoder_changed=False,transferred_function='sampled selected opacity and gradient, not membership decoder'))
    assert exported['tensor_bytes']==parent['tensor_bytes']+added
    torch.save(fixtures,out/'validation_conditions.pt')
    report=dict(complete=True,scene=job['scene'],views=len(checks),factors=sum(x['factors'] for x in checks),hits=sum(x['hits'] for x in checks),
        additional_tensor_bytes=added,records=checks,field_sha256=exported['files']['field.pt']['sha256'],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha256=sha256(PROTOCOL),
        benchmark_queries_used=False,target_labels_used=False,model_promotion=False,whole_five_task=False,seconds=time.monotonic()-start)
    (out/'compilation.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
