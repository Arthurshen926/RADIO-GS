"""Query-independent overlap atoms preserve union before Gaussian lifting.

Research representation, not yet integrated into exported CapabilityField.
Atoms carry partial relation incidence, not RGB, cameras or query answers.
"""
import torch


def compile_view(transfer, positive, negative, regions):
    import numpy as np
    from scipy import sparse
    p=np.asarray(positive,dtype=bool);n=np.asarray(negative,dtype=bool)
    if p.ndim!=2 or n.shape!=p.shape or transfer.shape[1]!=p.shape[1] or len(regions)!=len(p):
        raise ValueError('unaligned partial relation factors')
    if np.any(p & n):raise ValueError('positive and negative overlap')
    packed=np.packbits(np.concatenate([p,n],axis=0).T,axis=1)
    _,first,inverse=np.unique(packed,axis=0,return_index=True,return_inverse=True)
    assignment=sparse.csr_matrix((np.ones(len(inverse),np.float32),(np.arange(len(inverse)),inverse)),shape=(len(inverse),len(first)))
    lift=(transfer@assignment).tocsr();lift.eliminate_zeros()
    return dict(pointer=torch.from_numpy(lift.indptr.astype(np.int64)),
        index=torch.from_numpy(lift.indices.astype(np.int64)),value=torch.from_numpy(lift.data.astype(np.float32)),
        positive=torch.from_numpy(p[:,first].T.copy()),negative=torch.from_numpy(n[:,first].T.copy()),
        regions=torch.as_tensor(regions,dtype=torch.long))


class CompositionalEvidence:
    def __init__(self,state,device='cpu'):
        self.count=int(state['count']);self.region_count=int(state['region_count'])
        self.views=[{k:v.to(device) for k,v in x.items()} for x in state['views']]
        joined=torch.cat([x['regions'] for x in self.views])
        if not torch.equal(joined,torch.arange(self.region_count,device=device)):
            raise ValueError('region partition differs')
        self.matrices=[]
        for x in self.views:
            if x['positive'].dtype!=torch.bool or x['negative'].dtype!=torch.bool or x['positive'].shape!=x['negative'].shape:
                raise ValueError('invalid atom relations')
            atoms,regions=x['positive'].shape
            if regions!=len(x['regions']) or x['pointer'].shape!=(self.count+1,) or int(x['pointer'][0])!=0 or int(x['pointer'][-1])!=len(x['value']):
                raise ValueError('unaligned atom lift')
            if bool((x['positive']&x['negative']).any()) or bool((x['pointer'].diff()<0).any()) or x['index'].shape!=x['value'].shape:
                raise ValueError('invalid atom structure')
            if not bool(torch.isfinite(x['value']).all()) or bool((x['value']<0).any()) or bool(((x['index']<0)|(x['index']>=atoms)).any()):
                raise ValueError('invalid atom mass')
            self.matrices.append(torch.sparse_csr_tensor(x['pointer'],x['index'],x['value'],size=(self.count,atoms),device=device))

    @torch.no_grad()
    def prepare(self,accepted):
        if accepted.dtype!=torch.bool or accepted.shape!=(self.region_count,):
            raise ValueError('binary region acceptance required')
        positive=[];negative=[];active=0
        for x,lift in zip(self.views,self.matrices):
            selected=accepted[x['regions']]
            if bool(selected.any()):
                active+=1
                labels=torch.stack([x['positive'][:,selected].any(1),x['negative'][:,selected].all(1)],1).to(lift.dtype)
                values=torch.sparse.mm(lift,labels)
            else:values=lift.values().new_zeros(self.count,2)
            positive.append(values[:,0]);negative.append(values[:,1])
        return torch.stack(positive,1),torch.stack(negative,1),dict(admitted_regions=int(accepted.sum()),
            active_views=active,confidence_sum=float(active),union_before_lifting=True,
            query_independent_atom_state=True,source_access=False,source_raster_access=False)
