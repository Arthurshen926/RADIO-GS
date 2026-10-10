"""Compile Gaussian union measures without query answers or runtime source rasters."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from radio_gs.query_field.storage import load_field,save_field,sha256
from radio_gs.query_field.scope_constraints import ScopeConstraints,compile_view
from radio_gs.query_field.scope_exclusion import union_exclusion
from radio_gs.query_field.observations import erode
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.element_relation_training import view_transfer
from scripts.research.compile_query_field_scope_permissions import ROOT as PARENT,CONFIG

ROOT=Path('output/query_field_20261010/compiled_scope_constraints_v1')


def run(index):
    job=json.loads(CONFIG.read_text())['jobs'][index];out=ROOT/job['scene'];out.mkdir(parents=True,exist_ok=False)
    field,parent=load_field(PARENT/job['scene']/'field');assert field.scope_constraints is None
    source=SourceRelations(job['observations'],field.observed)
    construction=json.loads(Path(job['construction']).read_text());assert len(source.views)==len(construction['source_records'])
    state={};checks=[];fixtures=[];offset=coffset=0;generator=torch.Generator().manual_seed(20261010)
    for vi,(v,r) in enumerate(zip(source.views,construction['source_records'])):
        assert v['frame']==r['frame'] and v['original_regions']==r['regions']
        path=Path(job['observations'])/f"view_{v['frame']:05d}.pt";raw=torch.load(path,weights_only=True)
        masks=raw['masks'].bool();visible=raw['raw_mass'].reshape(masks.shape[-2:])>0
        interior=(erode(masks)&visible).flatten(1);n=len(v['positive']);c=len(masks)
        assert torch.equal(field.scope_permissions.accepted_views[offset:offset+n],torch.full((n,),vi))
        assert torch.equal(field.scope_permissions.candidate_views[coffset:coffset+c],torch.full((c,),vi))
        full,peak=view_transfer(v['columns'],field.observed,minimum_peak=0.)
        transfer=(sparse.diags(np.minimum(1.,peak/.1))@full).tocsr()
        compiled=compile_view(transfer,v['positive'],v['negative'],interior,range(offset,offset+n),range(coffset,coffset+c))
        lift=sparse.csr_matrix((compiled['value'].numpy(),compiled['index'].numpy(),compiled['pointer'].numpy()),shape=(len(field.xyz),len(compiled['positive'])))
        errors={}
        for key in ['positive','negative']:
            actual=lift@sparse.csr_matrix(compiled[key].numpy().astype(np.float32))
            expected=transfer@sparse.csr_matrix(v[key].numpy().astype(np.float32)).T
            delta=actual-expected;error=float(abs(delta.data).max()) if delta.nnz else 0.
            assert error<=2e-6,(key,error);errors[key]=error
        conditions=torch.cat([torch.zeros(1,n,dtype=torch.bool),torch.rand(8,n,generator=generator)<.15,torch.ones(1,n,dtype=torch.bool)])
        sample=torch.linspace(0,len(field.xyz)-1,min(256,len(field.xyz))).long();expected_sample=[];joint_error=0.
        for a in conditions:
            accepted=torch.zeros_like(field.scope_permissions.accepted_views,dtype=torch.bool);accepted[offset:offset+n]=a
            permitted=field.scope_permissions(accepted)[coffset:coffset+c]
            full_a=torch.zeros(c,dtype=torch.bool);full_a[torch.tensor(r['regions'])]=a
            old=v['negative'][a].all(0) if a.any() else torch.zeros_like(v['negative'][0])
            negative,reference_permission=union_exclusion(masks,raw['scales_m'],full_a,visible,old.reshape(visible.shape))
            assert torch.equal(permitted,reference_permission)
            positive=v['positive'][a].any(0) if a.any() else torch.zeros_like(old)
            expected=transfer@torch.stack([positive,negative.flatten()],1).float().numpy()
            ap=compiled['positive'][:,a].any(1) if a.any() else torch.zeros(len(compiled['positive']),dtype=torch.bool)
            an=compiled['negative'][:,a].all(1) if a.any() else torch.zeros_like(ap)
            if permitted.any():an|=compiled['interior'][:,permitted].any(1)
            assert not (ap&an).any()
            actual=lift@torch.stack([ap,an],1).float().numpy()
            error=float(np.abs(actual-expected).max());assert error<=2e-6,error;joint_error=max(joint_error,error)
            expected_sample.append(torch.from_numpy(expected[sample.numpy()].copy()))
        fixtures.append(dict(accepted=conditions,rows=sample,expected=torch.stack(expected_sample)))
        state.update({f'views.{vi}.'+k:value for k,value in compiled.items()})
        checks.append(dict(frame=v['frame'],sha256=sha256(path),regions=n,candidates=c,atoms=len(compiled['positive']),
            single_region_errors=errors,joint_conditions=len(conditions),joint_max_error=joint_error))
        print('compiled',job['scene'],vi,checks[-1],flush=True);offset+=n;coffset+=c
    assert offset==len(field.scope_permissions.accepted_views) and coffset==len(field.scope_permissions.candidate_views)
    field.scope_constraints=ScopeConstraints(state,len(field.xyz),field.scope_permissions)
    added=sum(v.numel()*v.element_size() for v in state.values())
    exported=save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        source_records=checks,benchmark_queries_used=False,source_teacher_used=False,target_labels_used=False,
        training_steps=0,default_decoder_changed=False,transferred_function='union evidence lift, not joint membership solver'))
    assert exported['tensor_bytes']==parent['tensor_bytes']+added
    torch.save(fixtures,out/'validation_conditions.pt')
    report=dict(complete=True,scene=job['scene'],additional_tensor_bytes=added,regions=offset,candidates=coffset,
        atoms=sum(x['atoms'] for x in checks),tolerance=2e-6,records=checks,field_sha256=exported['files']['field.pt']['sha256'],
        benchmark_queries_used=False,target_labels_used=False,model_promotion=False)
    (out/'compilation.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
