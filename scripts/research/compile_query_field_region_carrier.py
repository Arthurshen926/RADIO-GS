"""Query-free binary regional scope compiled from the existing standalone field."""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from radio_gs.query_field.storage import load_field,save_field
from radio_gs.query_field.region_carrier import RegionCarrier
from scripts.research.run_query_field_relations import read,write,receipt
ROOT=Path('output/query_field_20261011/binary_region_carrier_v1')
PARENT=Path('output/query_field_20261011/member_energy_v2')
QUALITY=Path('output/query_field_20261011/region_quality_transfer_v1/quality.pt')


def binary_state(matrix,prefix):
    m=matrix.tocsr();m.sum_duplicates();m.eliminate_zeros();m.sort_indices()
    assert np.all(m.data==1)
    return {prefix+'_pointer':torch.from_numpy(m.indptr.astype(np.int64)),prefix+'_index':torch.from_numpy(m.indices.astype(np.int64))}


@torch.no_grad()
def run(index):
    j=read(PARENT/'prediction_config.json')['jobs'][index];out=ROOT/j['scene'];out.mkdir(parents=True,exist_ok=False)
    field,parent=load_field(j['field']);n=len(field.xyz)
    matrix=field.evidence.matrix('association')
    a=sparse.csr_matrix((matrix.values().numpy(),matrix.col_indices().numpy(),matrix.crow_indices().numpy()),shape=matrix.shape)
    b=(a>=.5).astype(np.float32).tocsr();b.eliminate_zeros()
    columns=[];replays=[]
    for vi,v in enumerate(field.scope_constraints.views):
        lift=sparse.csr_matrix((v.value.numpy(),v.index.numpy(),v.pointer.numpy()),shape=(n,len(v.positive)))
        confidence=np.asarray(lift.sum(1)).ravel()
        assert confidence.min()>=0 and confidence.max()<=1+2e-6
        # Existing lift stores min(1, peak responsibility/.1). Numerical
        # roundoff tolerance is fixed, not fitted against labels.
        visible=confidence>=1-2e-6
        local=b[:,field.evidence.views.numpy()==vi]
        supported=np.asarray(local.sum(1)).ravel()>0
        assert not np.any(supported&~visible), 'membership outside reliable view'
        columns.append(sparse.csr_matrix(visible.astype(np.float32)[:,None]))
        replays.append(dict(view=vi,visible=int(visible.sum()),member_rows=int(supported.sum()),confidence_max=float(confidence.max())))
    visibility=sparse.hstack(columns,format='csr')
    # Appearance uses the field's native local coordinates, not text semantics.
    # Aggregation differs from the original masked-image descriptor: recorded
    # as a distribution shift, never claimed to be feature equivalence.
    bt=b.T.tocsr();descriptor=np.zeros((b.shape[1],field.local.native_dim),np.float32)
    for start in range(0,n,8192):
        local=field.local.decode(torch.arange(start,min(n,start+8192)))
        local=torch.nn.functional.normalize(local,dim=1).numpy()
        descriptor+=bt[:,start:start+len(local)]@local
    area=np.asarray(b.sum(0)).ravel();descriptor/=np.maximum(area[:,None],1)
    descriptor/=np.maximum(np.linalg.norm(descriptor,axis=1,keepdims=True),1e-8)
    state=dict(frames=field.evidence.views.clone(),view_count=torch.tensor(field.evidence.view_count),descriptors=torch.from_numpy(descriptor),
        **binary_state(b,'membership'),**binary_state(visibility,'visibility'))
    state.update({'quality.'+k:v for k,v in torch.load(QUALITY,weights_only=True).items()})
    field.region_carrier=RegionCarrier(state,n)
    manifest=save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        quality_export=receipt(QUALITY),query_free=True,benchmark_queries_used=False,target_labels_used=False,
        additional_3d_supervision='historical quality model:24 independent ScanNet training rooms',
        feature_transfer='binary majority Gaussian association; reliable visibility; mean field local appearance',
        distribution_parity_proven=False,default_changed=False))
    # Cold replay of the new self-contained fields is separate from construction.
    write(out/'compilation.json',dict(complete=True,scene=j['scene'],regions=len(area),nonempty=int((area>0).sum()),
        members=b.nnz,visibility=visibility.nnz,views=replays,additional_bytes=manifest['tensor_bytes']-parent['tensor_bytes'],
        parent_manifest=receipt(Path(j['field'])/'manifest.json'),export_manifest=receipt(out/'field/manifest.json'),
        compiler=receipt(__file__),target_labels_used=False,source_observations_opened=False))
    print('compiled',j['scene'],'regions',len(area),'nnz',b.nnz,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
