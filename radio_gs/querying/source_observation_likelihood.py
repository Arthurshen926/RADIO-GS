"""Cross-view estimation of noisy detection likelihoods, not membership targets.

A missing mask is an observation that a detector returned nothing, rather than
proof of object absence. Native identity anchors and source observation rates
both exclude the view currently decoded. This is a one-pass plug-in model,
not supervised calibration, independent truth, or iterative Dawid-Skene EM.
"""
import json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.querying.typed_evidence import visible_operator


class OtherViewNative:
    def __init__(self, field, query, layout):
        device=query.device
        self.n=field['shape'][0]
        response=(100*F.normalize(field['descriptors'].float().to(device),dim=-1)@query.T).softmax(-1)
        entropy=-(response*response.clamp_min(1e-30).log()).sum(1)
        information=(1-entropy/math.log(response.shape[1])).clamp(0,1)
        transfer=torch.sparse_csr_tensor(field['crow'].to(device),field['columns'].to(device),field['weights'].to(device),size=field['shape']).to_sparse_coo().coalesce()
        self.response=response
        self.total=torch.sparse.mm(transfer,response*information[:,None])
        self.mass=torch.sparse.mm(transfer,information[:,None])
        records=json.loads(Path(layout).read_text())['records']
        self.frames={r['frame']:i for i,r in enumerate(records)}
        self.offsets=[0]
        for r in records:self.offsets.append(self.offsets[-1]+r['regions'])
        rows,columns=transfer.indices();values=transfer.values()*information[columns]
        view=torch.bucketize(columns,torch.tensor(self.offsets[1:],device=device),right=True)
        order=view.argsort();self.rows=rows[order];self.columns=columns[order];self.values=values[order]
        self.cuts=torch.cat([torch.zeros(1,device=device,dtype=torch.long),torch.bincount(view,minlength=len(records)).cumsum(0)]).cpu().tolist()

    def excluding(self, frame):
        index=self.frames[frame];lo,hi=self.cuts[index:index+2];start,end=self.offsets[index:index+2]
        part=torch.sparse_coo_tensor(torch.stack([self.rows[lo:hi],self.columns[lo:hi]-start]),self.values[lo:hi],(self.n,end-start)).coalesce()
        own=torch.sparse.mm(part,self.response[start:end])
        mass=torch.sparse.mm(part,torch.ones(end-start,1,device=self.response.device))
        remaining=(self.mass-mass).clamp_min(0)
        value=(self.total-own).clamp_min(0)/remaining.clamp_min(1e-8)
        valid=remaining>1e-8
        value=torch.where(valid,value/value.sum(1,keepdim=True).clamp_min(1e-8),torch.zeros_like(value))
        return value,valid


def source_observation(view, texts, observed):
    record=torch.load(view['identity'],map_location='cpu')
    h,w=record['raster_hw'];device=observed.device
    masks=[]
    for text in texts:
        row=record['instances'][record['queries'].index(text)]
        masks.append(row['masks'].float().amax(0) if len(row['scores']) else torch.zeros(h,w))
    target=torch.stack(masks).to(device)
    binary=(target>.5).float()[:,None]
    positive=-F.max_pool2d(-binary,3,1,1)>.5
    negative=F.max_pool2d(binary,3,1,1)<.5
    known=(positive|negative)[:,0].flatten(1).T
    op,_,visible=visible_operator(torch.load(view['contributions'],map_location='cpu'),observed,h*w,False)
    return op,target.flatten(1).T,known&visible,visible


def rates(numerator,exposure,mode):
    if mode=='diagonal':
        # Each detector has its own sensitivity and false-positive rate, but
        # all alternative true classes share that detector's false-positive rate.
        diagonal=numerator.diag();denominator=exposure.diag()
        false_positive=(numerator.sum(0)-diagonal+.5)/(exposure.sum(0)-denominator+1)
        probability=false_positive[None].expand_as(numerator).clone()
        probability.diagonal().copy_((diagonal+.5)/(denominator+1))
        return probability
    return (numerator+.5)/(exposure+1)


@torch.inference_mode()
def fuse_observation_likelihood(prior_logits, observed, scene, texts, field, query):
    started=time.monotonic();device=prior_logits.device;c=len(texts)
    native=OtherViewNative(field,query,scene['observation_view_layout'])
    counts=[];exposures=[]
    for view in scene['source_evidence']['views']:
        op,target,known,visible=source_observation(view,texts,observed)
        other,support=native.excluding(view['frame'])
        q=torch.sparse.mm(op,other)
        covered=torch.sparse.mm(op,support.float())
        anchor=q/q.sum(1,keepdim=True).clamp_min(1e-8)
        weight=known.float()*covered
        counts.append(anchor.T@(target*weight));exposures.append(anchor.T@weight)
    count=sum(counts);exposure=sum(exposures);del native
    total=torch.zeros_like(prior_logits);mass=torch.zeros_like(prior_logits[:,:1])
    mode=scene['source_observation_model']
    for view,own,own_exposure in zip(scene['source_evidence']['views'],counts,exposures):
        probability=rates((count-own).clamp_min(0),(exposure-own_exposure).clamp_min(0),mode)
        op,target,known,visible=source_observation(view,texts,observed)
        # The complete detector menu enters before any query-memory batching.
        log_positive=probability.log();log_negative=torch.log1p(-probability)
        loglike=(target*known)@log_positive.T+((1-target)*known)@log_negative.T
        total+=torch.sparse.mm(op.T,loglike*visible)
        mass+=torch.sparse.mm(op.T,visible.float())
    # Average observations over visible cameras rather than multiplying an
    # arbitrarily long, strongly correlated video into infinite certainty.
    evidence=total/mass.clamp_min(1e-8)
    prior=prior_logits.sigmoid();prior=prior/prior.sum(1,keepdim=True).clamp_min(1e-30)
    posterior=(prior.clamp_min(1e-30).log()+evidence).softmax(-1)
    logits=torch.logit(posterior.clamp(torch.finfo(torch.float32).tiny,1-torch.finfo(torch.float32).eps))
    logits=torch.where(mass>0,logits,prior_logits)
    return logits,dict(method='crossfit_detector_likelihood',model=mode,views=len(counts),
        rate_matrix=rates(count,exposure,mode).cpu().tolist(),queries=texts,
        fit_native_anchor_excludes_whole_view=True,fit_observation_rates_exclude_decoded_view=True,
        pseudocount_positive=.5,pseudocount_negative=.5,iterations=0,
        absence_is_detector_event_not_membership_label=True,view_likelihood_aggregation='visibility weighted mean',
        complete_vocabulary_before_projection=True,learned_query_projection=False,
        ground_truth_read=False,source_rgb_read=False,new_teacher_calls=0,
        seconds=time.monotonic()-started)
