"""Local boundary evidence and common soft or discrete set inference.

Only affirmative same-instance affinity smooths values. A missing/weak edge is
not a negative label, and nodes without query evidence are not background seeds.
Native semantic coordinates and the carrier geometry are never optimized here.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse.csgraph import maximum_flow, breadth_first_order


EDGE_FEATURES=['relative_distance','normal_disagreement','minimum_plane_offset',
    'maximum_plane_offset','minimum_curvature','maximum_curvature',
    'red_difference','green_difference','blue_difference','colour_distance',
    'native_appearance_distance']


def surface_neighbours(xyz,neighbours=12):
    distance,index=cKDTree(xyz).query(xyz,k=neighbours+1,workers=3)
    distance,index=distance[:,1:],index[:,1:]
    offset=xyz[index]-xyz[:,None]
    covariance=np.einsum('nki,nkj->nij',offset,offset)
    values,vectors=np.linalg.eigh(covariance)
    normal=vectors[:,:,0].astype(np.float32)
    curvature=(values[:,0]/np.maximum(values.sum(1),1e-12)).astype(np.float32)
    scale=np.maximum(distance[:,3],1e-6)
    left=np.repeat(np.arange(len(xyz),dtype=np.int32),neighbours)
    right=index.ravel().astype(np.int32)
    relative=distance.ravel()/np.sqrt(scale[left]*scale[right])
    keep=relative<=3
    return dict(left=left[keep],right=right[keep],relative=relative[keep].astype(np.float32),
                normal=normal,curvature=curvature)


def edge_features(xyz,rgb,appearance,surface,indices):
    left=surface['left'][indices];right=surface['right'][indices]
    delta=xyz[right]-xyz[left];length=np.maximum(np.linalg.norm(delta,axis=1),1e-8)
    normal=surface['normal'];curvature=surface['curvature']
    agreement=np.abs(np.sum(normal[left]*normal[right],axis=1))
    first=np.abs(np.sum(delta*normal[left],axis=1))/length
    second=np.abs(np.sum(delta*normal[right],axis=1))/length
    colour=np.abs(rgb[left]-rgb[right])
    cosine=np.sum(np.asarray(appearance[left],np.float32)*np.asarray(appearance[right],np.float32),axis=1)
    relative=surface['relative'][indices]
    return np.column_stack([relative/(1+relative),1-agreement,np.minimum(first,second),np.maximum(first,second),
        np.minimum(curvature[left],curvature[right]),np.maximum(curvature[left],curvature[right]),
        colour,np.linalg.norm(colour,axis=1),np.clip(1-cosine,0,2)]).astype(np.float32)


def affinity_graph(surface,weights,count):
    graph=sparse.csr_matrix((weights,(surface['left'],surface['right'])),shape=(count,count))
    graph=(graph+graph.T)*.5
    degree=np.asarray(graph.sum(1)).ravel()
    scale=sparse.diags(1/np.sqrt(np.maximum(degree,1e-8)))
    graph=(scale@graph@scale).tocsr();graph.eliminate_zeros()
    return graph


class SurfaceConsistency:
    def __init__(self,graph,strength,solver='laplacian',unknown_neutral=False,score_rule='unary'):
        self.graph=graph.tocsr();self.strength=float(strength)
        self.solver=solver
        self.unknown_neutral=unknown_neutral
        self.score_rule=score_rule

    def readout(self,values,vertices,anchor=None,threshold=.5):
        """Return ranking evidence and an explicit set as separate outputs.

        A binary MAP optimizer determines a set, not posterior marginals. Its
        local conditional margins must not silently replace the learned unary
        retrieval scores. The old score rule is an explicit replay control.
        """
        refined,detail=self.refine(values,vertices,anchor,threshold)
        selected=refined>threshold
        rank=values.copy() if self.score_rule=='unary' else refined
        detail.update(ranking_rule=self.score_rule,actual_selection_independent=True)
        return rank,selected,detail

    def refine(self,values,vertices,anchor=None,threshold=.5):
        """Combine unary evidence with a Laplacian or binary Potts objective.

        Restrict the graph to the caller's evidenced domain. Unknown outside it
        never becomes a zero-valued unary or an artificial negative prompt.
        The explicit positive point, when supplied, is a Dirichlet condition.
        """
        if self.strength==0:return values.copy(),dict(strength=0.,changed=False)
        if self.unknown_neutral:
            # Complete only on the offline-observed surface graph. A missing
            # query association supplies zero log-odds, not a background cost.
            observed=np.flatnonzero(self.graph.getnnz(axis=1)>0)
            unknown=np.setdiff1d(observed,vertices,assume_unique=True)
            expanded=np.union1d(vertices,observed)
            neutral=values.copy();neutral[unknown]=threshold
            graph=self.graph[expanded][:,expanded]
            result,detail=self.cut(neutral,expanded,graph,anchor,threshold)
            detail.update(unknown_policy='zero unary log-odds outside original evidence domain',
                original_evidenced_elements=len(vertices),unknown_elements=len(unknown))
            return result,detail
        graph=self.graph[vertices][:,vertices]
        if self.solver=='cut':return self.cut(values,vertices,graph,anchor,threshold)
        degree=np.asarray(graph.sum(1)).ravel()
        diagonal=1+self.strength*degree
        unary=np.asarray(values[vertices],np.float64)
        free=np.ones(len(vertices),bool)
        if anchor is not None:
            position=int(np.searchsorted(vertices,anchor));free[position]=False
            unary=unary+self.strength*graph[:,position].toarray().ravel()
        matrix=graph[free][:,free];diagonal=diagonal[free]
        operator=LinearOperator(matrix.shape,matvec=lambda x:diagonal*x-self.strength*(matrix@x))
        preconditioner=LinearOperator(matrix.shape,matvec=lambda x:x/diagonal)
        solution,status=cg(operator,unary[free],x0=values[vertices[free]],M=preconditioner,tol=1e-5,maxiter=100)
        output=values.copy();output[vertices[free]]=np.clip(solution,0,1)
        if anchor is not None:output[anchor]=1.
        return output,dict(strength=self.strength,solver_status=int(status),
            participating_elements=len(vertices),edges=matrix.nnz,
            negative_prompts_invented=False,native_coordinates_changed=False)

    def cut(self,values,vertices,graph,anchor,threshold):
        """Global binary Potts optimum; the pre-existing decision prior stays fixed.

        Returned scores are local conditional margins at the MAP configuration,
        not posterior marginals. Quantized MAP labels define the actual set.
        No assumption that a labelled instance is one connected component.
        """
        p=np.clip(values[vertices].astype(np.float64),1e-6,1-1e-6)
        prior=np.log(threshold)-np.log1p(-threshold)
        margin=np.log(p)-np.log1p(-p)-prior
        foreground=np.maximum(-margin,0);background=np.maximum(margin,0)
        if anchor is not None:
            position=int(np.searchsorted(vertices,anchor))
            background[position]=foreground[position]+2*self.strength*float(graph[position].sum())+1
        count=len(vertices);edges=graph.tocoo()
        rows=np.r_[edges.row,np.full(count,count),np.arange(count)]
        columns=np.r_[edges.col,np.arange(count),np.full(count,count+1)]
        capacity=np.rint(np.r_[self.strength*edges.data,background,foreground]*4096).astype(np.int64)
        network=sparse.csr_matrix((capacity,(rows,columns)),shape=(count+2,count+2))
        flow=maximum_flow(network,count,count+1,method='dinic')
        residual=network-flow.flow;residual.data=(residual.data>0).astype(np.int64);residual.eliminate_zeros()
        reachable=breadth_first_order(residual,count,directed=True,return_predecessors=False)
        selected=np.zeros(count,bool);selected[reachable[reachable<count]]=True
        conditional=margin+self.strength*(graph@(2*selected.astype(np.float64)-1))
        scores=(1/(1+np.exp(-(conditional+prior).clip(-30,30)))).astype(np.float32)
        # Finite integer capacities can leave sub-resolution ties. The MAP set,
        # not the floating approximation of its conditional score, is authoritative.
        scores=np.where(selected,np.maximum(scores,np.nextafter(np.float32(threshold),np.float32(1))),
                        np.minimum(scores,np.float32(threshold)))
        output=values.copy();output[vertices]=scores
        if anchor is not None:output[anchor]=1.
        return output,dict(strength=self.strength,solver='global_binary_potts',
            decision_prior=threshold,flow_integer_scale=4096,participating_elements=count,
            edges=graph.nnz,selected_elements=int(selected.sum()),
            score_kind='local conditional score at MAP; not calibrated marginal probability',
            negative_prompts_invented=False,connected_component_assumption=False)
