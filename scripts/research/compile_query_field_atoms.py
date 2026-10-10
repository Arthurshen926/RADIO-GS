"""Query-free overlap factorization with exhaustive single-region replay."""
import numpy as np
import torch
from scipy import sparse
from radio_gs.query_field.compositional_evidence import compile_view
from radio_gs.query_field.element_relation_training import view_transfer
from radio_gs.query_field.relation_training import SourceRelations
from scripts.research.run_query_field_relations import read


def compile_scene(field,job):
    source=SourceRelations(job['observations'],field.observed)
    reference={}
    for key in ['positive','negative']:
        pointer,index,value=(x.cpu().numpy() for x in field.evidence.parts(key))
        reference[key]=sparse.csr_matrix((value,index,pointer),shape=(len(field.xyz),len(field.evidence.views)))
    construction=read(job['construction']);assert len(source.views)==len(construction['source_records'])
    views=[];rows=[];offset=0
    for vi,(view,record) in enumerate(zip(source.views,construction['source_records'])):
        assert view['frame']==record['frame'] and view['original_regions']==record['regions']
        full,peak=view_transfer(view['columns'],field.observed,minimum_peak=0.)
        transfer=(sparse.diags(np.minimum(1.,peak/.1))@full).tocsr()
        r=len(view['positive']);assert torch.equal(field.evidence.views[offset:offset+r],torch.full((r,),vi))
        assert torch.equal(view['scales'],field.evidence.scales[offset:offset+r])
        state=compile_view(transfer,view['positive'].numpy(),view['negative'].numpy(),range(offset,offset+r))
        lift=sparse.csr_matrix((state['value'].numpy(),state['index'].numpy(),state['pointer'].numpy()),shape=(len(field.xyz),len(state['positive'])))
        errors={}
        for key in ['positive','negative']:
            actual=(lift@sparse.csr_matrix(state[key].numpy().astype(np.float32))).tocsr()
            delta=actual-reference[key][:,offset:offset+r]
            error=float(np.abs(delta.data).max()) if delta.nnz else 0.
            assert error<=2e-6,('single-region preservation failed',view['frame'],key,error)
            errors[key]=error
        rows.append(dict(frame=view['frame'],regions=r,atoms=len(state['positive']),lift_nnz=lift.nnz,single_region_max_errors=errors))
        views.append(state);offset+=r
        print('compiled atoms',view['frame'],rows[-1],flush=True)
    assert offset==len(field.evidence.views)
    state=dict(count=len(field.xyz),region_count=offset,views=views)
    size=sum(t.numel()*t.element_size() for v in views for t in v.values())
    return state,dict(query_independent=True,teacher_acceptance_read=False,target_labels_opened=False,
        source_records=rows,tensor_bytes=size,regions=offset,atom_count=sum(x['atoms'] for x in rows),
        preserves_single_region_marginals=True,single_region_tolerance=2e-6,
        partial_relation_union_not_recomputed_morphology=True,not_integrated_into_deployed_field=True)
