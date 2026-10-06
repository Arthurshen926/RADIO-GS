"""Frozen visual opinions pooled per element, with at most six camera groups.

Each directional target is in [0,1]. A group target is a nonnegative weighted
average of these targets, and its mass is in [0,1]. The final pool weights are
1/(1+sum mass) for the parent and mass/(1+sum mass) for each group: hence convex
and bounded. This is NOT independent Bayes; the parent shares source images.
"""
import torch
from radio_gs.querying.visual_scope_membership import VisualMembershipRuntime, VisualMembershipHead


def camera_groups(world_to_camera):
    """FPS from first source camera; normalized centers + unit forward vectors.

    Position scale is the source bounding-box diagonal. Equal poses have equal
    features and nearest-center assignments. Ties use first source order.
    """
    poses=torch.as_tensor(world_to_camera,dtype=torch.float64)
    inv=torch.linalg.inv(poses)
    centers=inv[:,:3,3];direction=torch.nn.functional.normalize(inv[:,:3,2],dim=1)
    scale=(centers.amax(0)-centers.amin(0)).norm().clamp_min(1e-12)
    features=torch.cat([(centers-centers[0])/scale,direction],1)
    distance=torch.cdist(features,features).square();chosen=[0]
    while len(chosen)<min(6,len(features)):
        nearest=distance[:,chosen].amin(1);index=int(nearest.argmax())
        if nearest[index]<=1e-24:break
        chosen.append(index)
    return distance[:,chosen].argmin(1),dict(centers=centers.tolist(),directions=direction.tolist(),position_scale=float(scale),fps_centers=chosen)


class ElementPosteriorRuntime(VisualMembershipRuntime):
    def __init__(self,bank,head,metadata,arm='B2',device='cpu'):
        super().__init__(bank,head,arm,device)
        self.metadata=metadata
        self.keep=metadata['keep'].to(device).bool()
        self.allowed=self.allowed & self.keep
        self.groups=metadata['region_groups'].to(device).long()
        assert len(torch.unique(self.groups))<=6
        indices=bank['indices'].long().to(device)
        weights=metadata['footprint_weights'].double().to(device)
        self.edges=[]
        for view in sorted(torch.unique(self.views).tolist()):
            select=(self.views[indices[1]]==view)&self.keep[indices[1]]
            self.edges.append((view,int(self.groups[self.views==view][0]),indices[0,select],indices[1,select],weights[select]))

    def membership_logits(self,q):
        # Fix GEMM shape per query. Tiny batch-dependent head rounding is
        # amplified by finite logit near probability endpoints.
        return torch.cat([self.head(self.features(q[j:j+1].contiguous())) for j in range(len(q))],1)

    def transport(self,membership_logits,parent,exclude_view=None):
        allowed=self.allowed if exclude_view is None else self.allowed & (self.views!=exclude_view)
        r=membership_logits.sigmoid().double()
        cp=(2*r-1).clamp_min(0)*allowed[:,None]
        competitor=torch.zeros_like(cp)
        # Max, not a sum: repeated competitors cannot inflate confidence.
        for k in range(len(r)):
            ids=self.disjoint[k]&allowed
            if ids.any():competitor[k]=cp[ids].amax(0)
        cn=(1-2*r).clamp_min(0)*competitor*allowed[:,None]
        p=parent.sigmoid().double();numerator=torch.zeros_like(p);total=torch.zeros_like(p)
        group_masses=[]
        for group in sorted(torch.unique(self.groups).tolist()):
            mass_sum=torch.zeros_like(p);displacement_sum=torch.zeros_like(p);mass_max=torch.zeros_like(p)
            for view,g,ii,kk,w in self.edges:
                if g!=group or view==exclude_view:continue
                mass=torch.zeros_like(p);value=torch.zeros_like(p)
                # Double accumulation gives stable rounding without float32 atomics.
                for start in range(0,len(ii),262144):
                    i=ii[start:start+262144];k=kk[start:start+262144];weight=w[start:start+262144,None]
                    a=weight*cp[k];b=weight*cn[k];diff=r[k]-p[i]
                    mass.index_add_(0,i,a+b)
                    value.index_add_(0,i,a*diff.clamp_min(0)+b*diff.clamp_max(0))
                mass_sum+=mass;displacement_sum+=value;mass_max=torch.maximum(mass_max,mass)
            gm=mass_max.clamp(max=1)
            displacement=displacement_sum/mass_sum.clamp_min(1e-30)
            numerator+=gm*displacement;total+=gm;group_masses.append(gm)
        change=numerator/(1+total);new=p+change
        # Clip only changed probabilities; untouched extreme categorical ranks
        # retain the original finite logits (e.g. -150), bit for bit.
        changed=(change!=0)&(new!=p)
        logits=parent.clone()
        logits[changed]=torch.logit(new[changed].clamp(torch.finfo(torch.float64).eps,1-torch.finfo(torch.float64).eps)).to(parent.dtype)
        return logits,dict(probability=r,positive=cp>0,negative=cn>0,allowed=allowed,group_mass=torch.stack(group_masses),probability_change=change,pooled_probability=new)

    def logits(self,q,parent,modality='text',exclude_view=None):
        if modality!='text':return parent
        return self.transport(self.membership_logits(q),parent,exclude_view)[0]

    def delta(self,q,parent,exclude_view=None):
        return self.logits(q,parent,exclude_view=exclude_view)-parent


def load_runtime(addon,device):
    from radio_gs.querying.query_region_evidence import load,digest
    metadata=load(addon['transport'])
    assert metadata['schema']=='element_posterior.next42.v3'
    assert digest(addon['bank'])==metadata['bank_sha256']
    saved=load(addon['checkpoint']);assert saved['schema']=='visual_membership.next42.v2'
    head=VisualMembershipHead();head.load_state_dict(saved['state_dict'])
    return ElementPosteriorRuntime(load(addon['bank']),head,metadata,addon['arm'],device)
